"""
trainer.py — PhaseTrainer
Main training loop: phase-aware curriculum, gradient accumulation,
periodic evaluation, phase gating, checkpoint management.

Integrates: LoRA+, OPLoRA, EWC, CrossCLR, CompositeLoss, CurriculumSampler.
"""

import logging
import random
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .model import setup_model, setup_optimizer, setup_scheduler, save_checkpoint
from .dataset import build_dataloader, build_val_loader
from .losses import CompositeLoss
from .ewc import EWCRegularizer
from .oplora import OPLoRAManager
from .crossclr import CrossCLRLoss
from .monitoring import TrainingMonitor, compute_grad_norm, quick_validate

log = logging.getLogger(__name__)


class PhaseTrainer:
    """
    Trains one phase of the curriculum.
    Call train_phase() for each phase sequentially.
    """

    def __init__(self, config: dict, run_name: str):
        self.config = config
        self.run_name = run_name
        self.phase = config.get("phase", 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        hw = config.get("hardware", {})
        self.batch_size = hw.get("batch_size", 3)
        self.grad_accum = hw.get("gradient_accumulation_steps", 8)
        self.max_grad_norm = hw.get("max_grad_norm", 1.0)
        self.max_total_len = hw.get("max_total_len", 1280)

        mon_cfg = config.get("monitoring", {})
        self.log_every = mon_cfg.get("log_every", 10)
        self.eval_every = mon_cfg.get("eval_every", 500)
        self.checkpoint_every = mon_cfg.get("checkpoint_every", 500)

        self.hdf5_path = config.get("data", {}).get("hdf5_path", "data/hdf5/engineering.h5")

        # Components (initialized in setup)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.ewc = None
        self.oplora = None
        self.crossclr = None
        self.composite_loss = None
        self.monitor = None

    def setup(self):
        """Initialize all components for this phase."""
        log.info(f"{'='*60}")
        log.info(f"Setting up Phase {self.phase} training")
        log.info(f"{'='*60}")

        # Model
        adapter_path = self.config.get("resume_from", None)
        self.model, self.tokenizer = setup_model(self.config, adapter_path=adapter_path)

        # EWC (Phase 2-3)
        self.ewc = EWCRegularizer()
        if self.phase >= 2:
            ewc_path = self.config.get("ewc", {}).get(
                "state_path", "checkpoints/phase1/final"
            )
            if not self.ewc.load(ewc_path):
                log.error("EWC state required for Phase 2+ but not found!")
                raise FileNotFoundError(f"EWC state not at {ewc_path}")

        # OPLoRA (Phase 2-3)
        self.oplora = None
        if self.config.get("oplora", {}).get("enabled", False):
            k = self.config.get("oplora", {}).get("k", 16)
            self.oplora = OPLoRAManager(k=k)
            self.oplora.compute_and_cache_svd(self.model)
            self.oplora.register_hooks(self.model)

        # CrossCLR (Phase 3)
        self.crossclr = None
        crossclr_weight = self.config.get("losses", {}).get("crossclr_weight", 0.0)
        if crossclr_weight > 0:
            crossclr_cfg = self.config.get("crossclr", {})
            hidden_dim = self.model.config.hidden_size  # 4096 for Mistral-7B
            self.crossclr = CrossCLRLoss(
                hidden_dim=hidden_dim,
                embed_dim=crossclr_cfg.get("embed_dim", 256),
                queue_size=crossclr_cfg.get("queue_size", 3000),
                tau=crossclr_cfg.get("tau", 0.03),
                lambda_intra=crossclr_cfg.get("lambda_intra", 0.75),
                kappa=crossclr_cfg.get("kappa", 3.5e-4),
            ).to(self.device)

        # Composite loss
        self.composite_loss = CompositeLoss(
            self.config, self.tokenizer,
            ewc=self.ewc,
            crossclr=self.crossclr,
        )

        # Monitor
        use_wandb = self.config.get("monitoring", {}).get("use_wandb", True)
        self.monitor = TrainingMonitor(
            self.run_name, self.config, use_wandb=use_wandb,
        )

        log.info("Setup complete.")

    def train_phase(self, replay_indices=None):
        """
        Execute training for the current phase.

        Args:
            replay_indices: list of paper training indices to replay (Phase 2-3)

        Returns:
            dict with results including global_step
        """
        self.setup()

        # Data loaders
        loaders = build_dataloader(
            self.hdf5_path, self.tokenizer,
            phase=self.phase, batch_size=self.batch_size,
            max_total_len=self.max_total_len,
            replay_indices=replay_indices,
        )
        val_loaders = build_val_loader(
            self.hdf5_path, self.tokenizer,
            batch_size=self.batch_size, max_total_len=self.max_total_len,
        )

        ratios = loaders["ratios"]

        # Calculate total steps
        epochs = self.config.get("epochs", 3)
        paper_loader = loaders["paper"]
        steps_per_epoch = len(paper_loader) // self.grad_accum
        if loaders.get("video"):
            steps_per_epoch = max(steps_per_epoch, len(loaders["video"]) // self.grad_accum)
        total_steps = steps_per_epoch * epochs

        # Optimizer
        self.optimizer = setup_optimizer(self.model, self.config)

        # Add CrossCLR projection head params to optimizer BEFORE scheduler
        if self.crossclr is not None:
            crossclr_lr = self.config.get("lora_plus", {}).get("lr_A", 1e-4)
            self.optimizer.add_param_group({
                "params": list(self.crossclr.parameters()),
                "lr": crossclr_lr,
                "weight_decay": 0.0,
            })

        self.scheduler = setup_scheduler(self.optimizer, total_steps, self.config)

        log.info(f"Phase {self.phase}: {epochs} epochs, {steps_per_epoch} steps/epoch, "
                 f"{total_steps} total steps")
        log.info(f"Curriculum: paper={ratios['paper']}, video={ratios['video']}, "
                 f"pair={ratios['pair']}, replay={ratios['replay']}")

        # Build cycling iterators for each loader
        iterators = {}
        iterators["paper"] = cycle(paper_loader)
        if loaders.get("video"):
            iterators["video"] = cycle(loaders["video"])
        if loaders.get("pair"):
            iterators["pair"] = cycle(loaders["pair"])
        if loaders.get("replay"):
            iterators["replay"] = cycle(loaders["replay"])

        # Training loop
        self.model.train()
        global_step = 0
        rng = random.Random(42)

        for epoch in range(epochs):
            log.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0

            for local_step in range(steps_per_epoch):
                # Accumulate gradients over grad_accum micro-batches
                accumulated_loss_dict = {}

                for micro in range(self.grad_accum):
                    # Select batch type based on curriculum ratios
                    r = rng.random()
                    if r < ratios["paper"] - ratios["replay"]:
                        batch_type = "paper"
                    elif r < ratios["paper"]:
                        batch_type = "replay" if "replay" in iterators else "paper"
                    elif r < ratios["paper"] + ratios["video"]:
                        batch_type = "video" if "video" in iterators else "paper"
                    else:
                        batch_type = "pair" if "pair" in iterators else "paper"

                    if batch_type == "pair":
                        if self.crossclr is not None:
                            loss, loss_dict = self._train_pair_step(iterators["pair"])
                        else:
                            # CrossCLR disabled — pair batches can't go through
                            # _train_step (wrong keys), fall back to paper
                            batch = next(iterators["paper"])
                            loss, loss_dict = self._train_step(batch)
                            batch_type = "paper"
                    elif batch_type in iterators:
                        batch = next(iterators[batch_type])
                        loss, loss_dict = self._train_step(batch)
                    else:
                        batch = next(iterators["paper"])
                        loss, loss_dict = self._train_step(batch)
                        batch_type = "paper"

                    # Scale loss for accumulation
                    scaled_loss = loss / self.grad_accum
                    scaled_loss.backward()

                    # Accumulate loss dict
                    for k, v in loss_dict.items():
                        accumulated_loss_dict[k] = accumulated_loss_dict.get(k, 0) + v / self.grad_accum

                # Gradient clipping + optimizer step
                grad_norm = compute_grad_norm(self.model)
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                global_step += 1
                epoch_loss += accumulated_loss_dict.get("total", 0)

                # Logging
                if global_step % self.log_every == 0:
                    lr_A = self.optimizer.param_groups[0]["lr"]
                    lr_B = self.optimizer.param_groups[1]["lr"] if len(self.optimizer.param_groups) > 1 else lr_A
                    self.monitor.log_step(
                        global_step, accumulated_loss_dict,
                        lr_A, lr_B, grad_norm, batch_type,
                    )

                # ρk monitoring (Phase 2-3, every 100 steps)
                if self.oplora and global_step % 100 == 0:
                    rho_values = self.oplora.compute_rho_k(self.model)
                    self.monitor.log_rho_k(global_step, rho_values)

                    # Hot-swap k if needed
                    threshold = self.config.get("oplora", {}).get("rho_k_threshold", 0.5)
                    self.oplora.maybe_upgrade_k(self.model, rho_values, threshold)

                # Periodic validation
                if global_step % self.eval_every == 0:
                    val_metrics = quick_validate(self.model, val_loaders, self.device)
                    self.monitor.log_eval(global_step, val_metrics)
                    self.model.train()

                # Periodic checkpoint
                if global_step % self.checkpoint_every == 0:
                    ckpt_path = f"checkpoints/phase{self.phase}/step_{global_step}"
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, global_step, ckpt_path,
                    )

            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            log.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

        # Final checkpoint
        final_path = f"checkpoints/phase{self.phase}/final"
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epochs, global_step, final_path,
        )

        # Compute Fisher at end of Phase 1
        if self.phase == 1:
            log.info("Computing Fisher Information Matrix...")
            self.ewc.compute_fisher(
                self.model, paper_loader, n_samples=1000, device=self.device,
            )
            self.ewc.save(final_path)

        self.monitor.log_phase_transition(self.phase, global_step)
        self.monitor.finish()

        return {
            "global_step": global_step,
            "final_checkpoint": final_path,
            "replay_indices": self._sample_replay_indices(loaders["paper_dataset"])
                              if self.phase == 1 else replay_indices,
        }

    def _train_step(self, batch):
        """Single forward pass + loss computation for paper/video batch."""
        batch_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = self.model(
            input_ids=batch_device["input_ids"],
            attention_mask=batch_device["attention_mask"],
            labels=batch_device["labels"],
        )

        total_loss, loss_dict = self.composite_loss.compute(
            self.model, outputs, batch_device,
        )
        return total_loss, loss_dict

    def _train_pair_step(self, pair_iter):
        """Forward pass for cross-modal pair batch (with CrossCLR)."""
        pair_batch = next(pair_iter)
        device = self.device

        # Forward both sides
        paper_out = self.model(
            input_ids=pair_batch["paper_input_ids"].to(device),
            attention_mask=pair_batch["paper_attention_mask"].to(device),
            labels=pair_batch["paper_labels"].to(device),
        )
        video_out = self.model(
            input_ids=pair_batch["video_input_ids"].to(device),
            attention_mask=pair_batch["video_attention_mask"].to(device),
            labels=pair_batch["video_labels"].to(device),
        )

        # CE loss from both sides
        ce_loss = (paper_out.loss + video_out.loss) / 2

        # CrossCLR loss
        crossclr_loss = None
        if self.crossclr is not None:
            paper_embeds = self.crossclr.get_embeddings(
                self.model, pair_batch["paper_input_ids"].to(device),
                pair_batch["paper_attention_mask"].to(device), "paper",
            )
            video_embeds = self.crossclr.get_embeddings(
                self.model, pair_batch["video_input_ids"].to(device),
                pair_batch["video_attention_mask"].to(device), "video",
            )
            crossclr_loss = self.crossclr(paper_embeds, video_embeds)

        # Dummy outputs object for composite loss
        class _Out:
            pass
        out = _Out()
        out.loss = ce_loss
        out.logits = paper_out.logits  # use paper side for diversity/term losses

        # Combine via composite loss
        dummy_batch = {
            "labels": pair_batch["paper_labels"].to(device),
        }
        total_loss, loss_dict = self.composite_loss.compute(
            self.model, out, dummy_batch, crossclr_loss=crossclr_loss,
        )

        return total_loss, loss_dict

    @staticmethod
    def _sample_replay_indices(paper_dataset, ratio=0.10, seed=42):
        """Sample 10% of paper training indices for replay buffer."""
        n = len(paper_dataset)
        k = max(1, int(n * ratio))
        rng = random.Random(seed)
        indices = rng.sample(range(n), k)
        log.info(f"Replay buffer: {k} samples ({ratio*100:.0f}% of {n})")
        return indices