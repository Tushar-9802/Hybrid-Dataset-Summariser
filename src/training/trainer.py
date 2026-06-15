"""
trainer.py — PhaseTrainer
Main training loop: phase-aware curriculum, gradient accumulation,
periodic evaluation, phase gating, checkpoint management.

Integrates: LoRA+, OPLoRA, EWC, CrossCLR, CompositeLoss, CurriculumSampler.
"""

import logging
import random
import shutil
import time
from itertools import cycle
from pathlib import Path

import numpy as np
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
        self.ckpt_root = Path("checkpoints") / run_name
        self.phase = config.get("phase", 1)
        self.seed = config.get("seed", 42)
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
        self.checkpoint_interval_sec = mon_cfg.get("checkpoint_interval_sec", 3600)

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
        self._curriculum_rng = None

    def setup(self):
        """Initialize all components for this phase."""
        log.info(f"{'='*60}")
        log.info(f"Setting up Phase {self.phase} training")
        log.info(f"{'='*60}")

        # Model — resume from this run's in-progress checkpoint if one exists,
        # otherwise from the configured phase hand-off (resume_from).
        adapter_path = self.config.get("resume_from", None)
        resume_dir = self.ckpt_root / "latest"
        if (resume_dir / "adapter_config.json").exists():
            log.info(f"Resumable checkpoint found — loading adapter from {resume_dir}")
            adapter_path = str(resume_dir)
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
            seed=self.seed,
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

        # Curriculum batch-type RNG (instance attr so checkpoints can snapshot
        # it). Phase offset keeps phases 1/2/3 from sharing an identical stream.
        self._curriculum_rng = random.Random(self.seed + self.phase)

        # Resume from this run's in-progress checkpoint if one exists.
        start_epoch, start_global_step, start_local_step = 0, 0, 0
        resumed = self._load_resume_state(steps_per_epoch)
        if resumed is not None:
            start_epoch, start_global_step, start_local_step = resumed

        # Training loop
        self.model.train()
        global_step = start_global_step
        last_ckpt_time = time.time()

        for epoch in range(start_epoch, epochs):
            log.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            first_local = start_local_step if epoch == start_epoch else 0

            for local_step in range(first_local, steps_per_epoch):
                # Accumulate gradients over grad_accum micro-batches
                accumulated_loss_dict = {}

                for micro in range(self.grad_accum):
                    # Select batch type based on curriculum ratios
                    r = self._curriculum_rng.random()
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

                    # Skip micro-batches with non-finite loss (NaN/inf) rather
                    # than propagating corrupt gradients into the optimizer step.
                    if not torch.isfinite(loss):
                        log.warning(
                            f"Non-finite loss ({loss.item()}) at step {global_step}, "
                            f"micro {micro}, batch_type={batch_type} — skipping micro-batch"
                        )
                        continue

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

                # Hourly resumable checkpoint
                if time.time() - last_ckpt_time >= self.checkpoint_interval_sec:
                    self._write_resume_checkpoint(epoch, global_step)
                    last_ckpt_time = time.time()

            steps_run = max(steps_per_epoch - first_local, 1)
            avg_loss = epoch_loss / steps_run
            log.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

            # Epoch-boundary checkpoint
            self._write_resume_checkpoint(epoch + 1, global_step)

        # Final checkpoint
        final_path = str(self.ckpt_root / "final")
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epochs, global_step, final_path,
        )
        # Run complete — drop the resume checkpoint so a re-run starts fresh.
        latest_dir = self.ckpt_root / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir, ignore_errors=True)

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
            "replay_indices": self._sample_replay_indices(loaders["paper_dataset"], seed=self.seed)
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

    def _write_resume_checkpoint(self, epoch: int, global_step: int):
        """Write a full resumable checkpoint to checkpoints/{run}/latest.

        Captures model adapter + optimizer + scheduler (via save_checkpoint)
        plus every RNG state and the CrossCLR momentum buffers, so an
        interrupted run can continue without losing progress.
        """
        extra = {
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            "rng_curriculum": self._curriculum_rng.getstate(),
        }
        if self.crossclr is not None:
            extra["crossclr_state"] = self.crossclr.state_dict()
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, global_step, str(self.ckpt_root / "latest"), extra=extra,
        )

    def _load_resume_state(self, steps_per_epoch: int):
        """Restore optimizer/scheduler/RNG/CrossCLR state from this run's
        latest checkpoint, if present.

        Returns (start_epoch, start_global_step, start_local_step), or None
        when there is nothing to resume. The model adapter itself is already
        reloaded in setup() from the same directory.
        """
        state_path = self.ckpt_root / "latest" / "training_state.pt"
        if not state_path.exists():
            return None

        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        # Optimizer state loads onto CPU — move it to the training device so
        # the first post-resume step does not hit a device mismatch.
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if "rng_python" in state:
            random.setstate(state["rng_python"])
            np.random.set_state(state["rng_numpy"])
            torch.set_rng_state(state["rng_torch"])
            if state.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["rng_cuda"])
            self._curriculum_rng.setstate(state["rng_curriculum"])
        if self.crossclr is not None and "crossclr_state" in state:
            self.crossclr.load_state_dict(state["crossclr_state"])

        gstep = int(state["global_step"])
        start_epoch = gstep // steps_per_epoch
        start_local_step = gstep % steps_per_epoch
        log.info(f"RESUMING {self.run_name}: global_step={gstep}, "
                 f"epoch={start_epoch + 1}, local_step={start_local_step}")
        return start_epoch, gstep, start_local_step