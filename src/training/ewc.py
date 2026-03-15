"""
ewc.py — Elastic Weight Consolidation
Fisher Information weighted quadratic penalty preventing catastrophic forgetting.

L_EWC = λ_ewc · Σ_i [F_i · (θ_i - θ*_i)²]

Computed at end of Phase 1 over LoRA parameters only.
λ_ewc = 200 (Phase 2), 400 (Phase 3).
"""

import logging
from pathlib import Path

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


class EWCRegularizer:
    """Elastic Weight Consolidation for LoRA parameters."""

    def __init__(self):
        self.fisher = {}       # {param_name: Fisher diagonal}
        self.theta_star = {}   # {param_name: parameter snapshot}
        self.ready = False
    def compute_fisher(
        self,
        model,
        dataloader,
        n_samples: int = 1000,
        device: str = "cuda",
    ):
        log.info(f"Computing Fisher Information ({n_samples} samples)...")

        # Keep gradient checkpointing ON — disabling it causes OOM
        model.train()
    # def compute_fisher(
    #     self,
    #     model,
    #     dataloader,
    #     n_samples: int = 1000,
    #     device: str = "cuda",
    # ):
        """
        Compute diagonal Fisher Information Matrix over LoRA params.

        Must temporarily disable gradient checkpointing for clean backward pass.
        """
        log.info(f"Computing Fisher Information ({n_samples} samples)...")

        # Temporarily disable gradient checkpointing for clean backward
        had_grad_ckpt = getattr(model.config, '_gradient_checkpointing', False)
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        # had_grad_ckpt = getattr(model.config, '_gradient_checkpointing', False)
        # if hasattr(model, 'gradient_checkpointing_disable'):
        #     model.gradient_checkpointing_disable()

        model.train()

        # Initialize Fisher accumulators
        fisher = {}
        theta_star = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
                theta_star[name] = param.data.clone().detach()

        count = 0
        for batch in tqdm(dataloader, total=min(n_samples, len(dataloader)), desc="Fisher"):
            if count >= n_samples:
                break

            model.zero_grad()
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            count += batch_device["input_ids"].shape[0]

        # Normalize
        for name in fisher:
            fisher[name] /= max(count, 1)

        self.fisher = fisher
        self.theta_star = theta_star
        self.ready = True

        # Re-enable gradient checkpointing
        # if had_grad_ckpt or True:  # always re-enable for training
        #     if hasattr(model, 'gradient_checkpointing_enable'):
        #         model.gradient_checkpointing_enable()

        log.info(f"Fisher computed over {count} samples, "
                 f"{len(fisher)} parameter groups")

    def penalty(self, model) -> torch.Tensor:
        """
        Compute EWC penalty: Σ_i [F_i · (θ_i - θ*_i)²]

        Returns scalar tensor (caller multiplies by λ_ewc).
        """
        if not self.ready:
            return torch.tensor(0.0, requires_grad=True)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                fisher_diag = self.fisher[name].to(param.device)
                theta_ref = self.theta_star[name].to(param.device)
                loss = loss + (fisher_diag * (param - theta_ref) ** 2).sum()

        return loss

    def save(self, path: str):
        """Save Fisher + θ* to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "fisher": {k: v.cpu() for k, v in self.fisher.items()},
            "theta_star": {k: v.cpu() for k, v in self.theta_star.items()},
        }, save_path / "ewc_state.pt")
        log.info(f"EWC state saved to {save_path / 'ewc_state.pt'}")

    def load(self, path: str):
        """Load Fisher + θ* from disk."""
        state_path = Path(path) / "ewc_state.pt"
        if not state_path.exists():
            log.error(f"EWC state not found: {state_path}")
            return False

        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.fisher = state["fisher"]
        self.theta_star = state["theta_star"]
        self.ready = True
        log.info(f"EWC state loaded ({len(self.fisher)} params)")
        return True