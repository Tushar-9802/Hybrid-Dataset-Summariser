"""
monitoring.py — Training Diagnostics
WandB + local JSON logging for training metrics, ρk, gradient norms, VRAM.
"""

import json
import logging
import time
from pathlib import Path

import torch

log = logging.getLogger(__name__)


class TrainingMonitor:
    """Centralized training diagnostics."""

    def __init__(
        self,
        run_name: str,
        config: dict,
        log_dir: str = "logs",
        use_wandb: bool = True,
    ):
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.local_log_path = self.log_dir / f"{run_name}_metrics.jsonl"
        self.step_times = []
        self.start_time = time.time()

        # WandB
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project="hybrid-summariser",
                    name=run_name,
                    config=config,
                    reinit=True,
                )
                log.info(f"WandB initialized: {run_name}")
            except Exception as e:
                log.warning(f"WandB unavailable ({e}), using local logging only")

    def log_step(self, step: int, loss_dict: dict, lr_A: float, lr_B: float,
                 grad_norm: float, batch_type: str = "paper"):
        """Log per-step metrics."""
        metrics = {
            "step": step,
            "batch_type": batch_type,
            "lr_A": lr_A,
            "lr_B": lr_B,
            "grad_norm": grad_norm,
            **{f"loss/{k}": v for k, v in loss_dict.items()},
        }

        # VRAM
        if torch.cuda.is_available():
            metrics["vram_gb"] = torch.cuda.memory_allocated() / 1e9

        # Throughput
        now = time.time()
        self.step_times.append(now)
        if len(self.step_times) > 10:
            recent = self.step_times[-10:]
            steps_per_sec = len(recent) / (recent[-1] - recent[0] + 1e-8)
            metrics["steps_per_sec"] = steps_per_sec

        # Log to WandB
        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

        # Log to local JSONL
        with open(self.local_log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def log_eval(self, step: int, metrics: dict, prefix: str = "val"):
        """Log evaluation metrics."""
        tagged = {f"{prefix}/{k}": v for k, v in metrics.items()}
        tagged["step"] = step

        if self.wandb_run:
            try:
                import wandb
                wandb.log(tagged, step=step)
            except Exception:
                pass

        with open(self.local_log_path, "a") as f:
            f.write(json.dumps(tagged) + "\n")

        # Console summary
        parts = [f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)]
        log.info(f"[Step {step}] {prefix}: {', '.join(parts)}")

    def log_rho_k(self, step: int, rho_values: dict):
        """Log OPLoRA ρk values."""
        avg_rho = sum(rho_values.values()) / max(len(rho_values), 1)
        max_rho = max(rho_values.values()) if rho_values else 0.0

        metrics = {
            "rho_k/mean": avg_rho,
            "rho_k/max": max_rho,
            "step": step,
        }

        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

        with open(self.local_log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        log.info(f"[Step {step}] ρk: mean={avg_rho:.4f}, max={max_rho:.4f}")

    def log_phase_transition(self, phase: int, step: int, d_gap: float = None):
        """Log phase transition event."""
        msg = f"Phase {phase} complete at step {step}"
        if d_gap is not None:
            msg += f" (D_gap={d_gap:+.4f})"

        log.info(f"{'='*60}")
        log.info(msg)
        log.info(f"{'='*60}")

        metrics = {"phase": phase, "step": step}
        if d_gap is not None:
            metrics["d_gap"] = d_gap

        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self):
        """Finalize logging."""
        elapsed = time.time() - self.start_time
        log.info(f"Training complete. Total time: {elapsed/3600:.1f} hours")

        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


def compute_grad_norm(model) -> float:
    """Compute total gradient norm across all trainable parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def quick_validate(model, val_loaders, device="cuda", max_batches=50):
    """
    Quick validation pass — compute average CE loss on paper + video val splits.
    Used for periodic checks during training (not full evaluation).
    """
    model.eval()
    results = {}

    for modality, loader in val_loaders.items():
        if loader is None:
            continue

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break

                batch_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = model(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                    labels=batch_device["labels"],
                )
                total_loss += outputs.loss.item()
                count += 1

        if count > 0:
            results[f"{modality}_loss"] = total_loss / count

    model.train()
    return results