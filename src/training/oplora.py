"""
oplora.py — Orthogonal Projection LoRA (OPLoRA)
Preserves top-k singular directions of pre-trained weights during LoRA fine-tuning.

Implementation:
  1. One-time SVD of (dequantized) base weights -> cache Uk, Vk
  2. Forward hooks project LoRA updates: DW' = PL . DW . PR
  3. rho_k monitoring for subspace interference detection

Source: OPLoRA (AAAI 2026) -- k=16 default, hot-swap to 128 if rho_k > 0.5
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class OPLoRAManager:
    """Manages SVD caching, projection hooks, and rho_k monitoring."""

    def __init__(self, k: int = 16, cache_dir: str = "checkpoints/svd_cache"):
        self.k = k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.projections = {}  # module_name -> (Uk, Vk)
        self.hooks = []

    def compute_and_cache_svd(self, model, force: bool = False):
        """
        Compute truncated SVD for each LoRA target module's base weight.
        Dequantizes 4-bit weights one module at a time to limit peak RAM.
        """
        cache_file = self.cache_dir / f"svd_k{self.k}.pt"
        if cache_file.exists() and not force:
            log.info(f"Loading cached SVD from {cache_file}")
            self.projections = torch.load(cache_file, map_location="cpu", weights_only=False)
            log.info(f"Loaded SVD for {len(self.projections)} modules (k={self.k})")
            return

        log.info(f"Computing SVD for LoRA modules (k={self.k})...")
        self.projections = {}

        for name, module in model.named_modules():
            # Find LoRA layers (PEFT wraps original linear as base_layer)
            if not hasattr(module, 'lora_A'):
                continue
            if not hasattr(module, 'base_layer'):
                continue

            base = module.base_layer
            log.info(f"  SVD: {name} ({base.weight.shape})")

            # Dequantize 4-bit weight
            try:
                import bitsandbytes as bnb
                if hasattr(base, 'weight') and hasattr(base.weight, 'quant_state'):
                    W = bnb.functional.dequantize_4bit(
                        base.weight.data, base.weight.quant_state
                    ).float()
                else:
                    W = base.weight.data.float()
            except Exception:
                W = base.weight.data.float()

            # Truncated SVD
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            Uk = U[:, :self.k].contiguous()    # (out, k)
            Vk = Vh[:self.k, :].contiguous()   # (k, in)

            self.projections[name] = (Uk.cpu(), Vk.cpu())

            # Free immediately
            del W, U, S, Vh
            torch.cuda.empty_cache()

        torch.save(self.projections, cache_file)
        log.info(f"SVD cached to {cache_file} ({len(self.projections)} modules)")

    def register_hooks(self, model):
        """
        Register forward hooks on LoRA B layers to project updates.

        PL . (BA) . PR = (BA) - Uk(Uk^T BA) - (BA Vk^T)Vk + Uk(Uk^T BA Vk^T)Vk
        """
        self.remove_hooks()

        for name, module in model.named_modules():
            if name not in self.projections:
                continue
            if not hasattr(module, 'lora_A'):
                continue

            Uk, Vk = self.projections[name]
            device = next(module.parameters()).device

            # Use float16 explicitly — do NOT inherit dtype from quantized
            # params (which may be uint8 under QLoRA / bitsandbytes 4-bit).
            Uk_d = Uk.to(device=device, dtype=torch.float16)
            Vk_d = Vk.to(device=device, dtype=torch.float16)

            # Hook the lora_B forward to project its output
            # PEFT computes: result = base(x) + scaling * lora_B(lora_A(dropout(x)))
            # We intercept lora_B's output and project it
            for adapter_name in module.lora_B:
                lora_B_module = module.lora_B[adapter_name]

                def make_hook(Uk_local, Vk_local, lora_A_mod):
                    def hook_fn(mod, input, output):
                        # output shape: (batch, seq, out_features) from lora_B
                        # Uk shape: (out_features, k)
                        # Left projection: out - Uk @ (Uk^T @ out)
                        #   = remove component along top-k singular directions

                        # Cast to float32 for safe matmul, then cast back
                        orig_dtype = output.dtype
                        out = output.float()           # (B, S, D)
                        Uk = Uk_local.float()          # (D, k)

                        # Project: coeffs = out @ Uk -> (B, S, k)
                        # Reconstruct: out @ Uk @ Uk^T via (B,S,k) @ (k,D) -> (B,S,D)
                        coeffs = out @ Uk              # (B, S, k)
                        proj = out - coeffs @ Uk.T     # (B, S, D)
                        return proj.to(orig_dtype)
                    return hook_fn

                lora_A_mod = module.lora_A[adapter_name]
                h = lora_B_module.register_forward_hook(
                    make_hook(Uk_d, Vk_d, lora_A_mod)
                )
                self.hooks.append(h)

        log.info(f"Registered {len(self.hooks)} OPLoRA projection hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_rho_k(self, model) -> dict:
        """
        Compute subspace interference metric for each LoRA layer.

        rho_k = ||Qk . DW||^2_F / ||DW||^2_F
        where Qk = Uk @ Uk^T, DW = B @ A
        """
        rho_values = {}

        for name, module in model.named_modules():
            if name not in self.projections:
                continue
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            Uk, Vk = self.projections[name]
            device = next(module.parameters()).device
            Uk_d = Uk.to(device=device, dtype=torch.float32)

            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.data.float()  # (r, in)
                B = module.lora_B[adapter_name].weight.data.float()  # (out, r)
                delta_W = B @ A  # (out, in)

                Qk_delta = Uk_d @ (Uk_d.T @ delta_W)  # projection onto top-k subspace
                rho = (Qk_delta.norm() ** 2) / (delta_W.norm() ** 2 + 1e-8)
                rho_values[name] = float(rho.item())

                del A, B, delta_W, Qk_delta

        return rho_values

    def maybe_upgrade_k(self, model, rho_values: dict, threshold: float = 0.5):
        """If any rho_k exceeds threshold, recompute SVD with k=128."""
        max_rho = max(rho_values.values()) if rho_values else 0.0
        if max_rho > threshold and self.k < 128:
            log.warning(f"rho_k={max_rho:.4f} > {threshold} -- upgrading k: {self.k} -> 128")
            self.k = 128
            self.compute_and_cache_svd(model, force=True)
            self.register_hooks(model)
            return True
        return False