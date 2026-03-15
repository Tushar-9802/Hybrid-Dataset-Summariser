"""
losses.py — Composite Loss Orchestrator
Phase-aware loss computation combining CE, diversity, terminology, EWC, CrossCLR.

Phase 1: CE only
Phase 2: CE + diversity (0.2) + terminology (0.4) + EWC (λ=200)
Phase 3: CE + diversity (0.2) + terminology (0.4) + EWC (λ=400) + CrossCLR (0.3)
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Domain terminology for CS/engineering (loaded once)
_TERM_IDS = None


def load_term_vocabulary(tokenizer, term_file: str = None) -> torch.Tensor:
    """
    Load domain term vocabulary as token IDs.
    Falls back to hardcoded CS terms if no file provided.
    """
    global _TERM_IDS
    if _TERM_IDS is not None:
        return _TERM_IDS

    terms = [
        "algorithm", "neural", "gradient", "optimization", "training",
        "inference", "embedding", "attention", "transformer", "convolution",
        "classification", "regression", "clustering", "segmentation",
        "detection", "recognition", "reinforcement", "supervised",
        "unsupervised", "generative", "discriminative", "encoder", "decoder",
        "tokenizer", "backpropagation", "architecture", "hyperparameter",
        "regularization", "normalization", "activation", "dropout",
        "overfitting", "underfitting", "convergence", "divergence",
        "benchmark", "baseline", "ablation", "evaluation", "metric",
        "precision", "recall", "accuracy", "latency", "throughput",
        "scalability", "robustness", "generalization", "representation",
        "feature", "kernel", "pooling", "stride", "batch", "epoch",
        "learning", "network", "layer", "weight", "bias", "parameter",
        "loss", "objective", "function", "distribution", "probability",
        "posterior", "likelihood", "entropy", "information", "mutual",
        "contrastive", "adversarial", "augmentation", "preprocessing",
        "pipeline", "framework", "implementation", "deployment",
        "quantization", "pruning", "distillation", "fine-tuning",
        "pre-training", "transfer", "domain", "adaptation", "modality",
        "multimodal", "cross-modal", "summarization", "generation",
        "extraction", "retrieval", "similarity", "distance", "cosine",
        "vector", "matrix", "tensor", "dimension", "subspace",
        "projection", "decomposition", "factorization", "singular",
        "eigenvalue", "orthogonal", "sparse", "dense", "linear",
        "nonlinear", "recurrent", "convolutional", "residual",
        "curriculum", "catastrophic", "forgetting", "preservation",
    ]

    if term_file and Path(term_file).exists():
        with open(term_file) as f:
            terms = [line.strip() for line in f if line.strip()]

    # Tokenize each term, collect unique token IDs
    all_ids = set()
    for term in terms:
        ids = tokenizer.encode(term, add_special_tokens=False)
        all_ids.update(ids)

    _TERM_IDS = torch.tensor(sorted(all_ids), dtype=torch.long)
    log.info(f"Loaded {len(terms)} domain terms → {len(_TERM_IDS)} unique token IDs")
    return _TERM_IDS


class CompositeLoss:
    """
    Phase-aware composite loss.

    Components:
      - CE: standard cross-entropy (from model output)
      - Diversity: negative entropy of predictions (encourages varied vocabulary)
      - Terminology: log-probability of domain terms in reference
      - EWC: Fisher-weighted quadratic penalty
      - CrossCLR: contrastive alignment (pair batches only)
    """

    def __init__(self, config: dict, tokenizer, ewc=None, crossclr=None):
        self.config = config
        losses_cfg = config.get("losses", {})

        self.ce_weight = losses_cfg.get("ce_weight", 1.0)
        self.div_weight = losses_cfg.get("diversity_weight", 0.0)
        self.term_weight = losses_cfg.get("terminology_weight", 0.0)
        self.ewc_lambda = losses_cfg.get("ewc_lambda", 0)
        self.crossclr_weight = losses_cfg.get("crossclr_weight", 0.0)

        self.ewc = ewc
        self.crossclr = crossclr

        # Load term vocabulary if terminology loss is active
        if self.term_weight > 0:
            self.term_ids = load_term_vocabulary(tokenizer)
        else:
            self.term_ids = None

        log.info(f"CompositeLoss: CE={self.ce_weight}, div={self.div_weight}, "
                 f"term={self.term_weight}, EWC_λ={self.ewc_lambda}, "
                 f"CrossCLR={self.crossclr_weight}")

    def compute(self, model, outputs, batch, crossclr_loss=None):
        """
        Compute total loss.

        Args:
            model: the PEFT model (needed for EWC)
            outputs: model forward outputs (has .loss and .logits)
            batch: input batch dict
            crossclr_loss: pre-computed CrossCLR loss (or None)

        Returns:
            (total_loss, loss_dict) where loss_dict has per-component values
        """
        loss_dict = {}
        total = torch.tensor(0.0, device=outputs.loss.device)

        # 1. Cross-entropy (from model)
        ce_loss = outputs.loss
        loss_dict["ce"] = ce_loss.item()
        total = total + self.ce_weight * ce_loss

        # 2. Diversity loss (negative entropy → maximize vocabulary diversity)
        if self.div_weight > 0 and outputs.logits is not None:
            div_loss = self._diversity_loss(outputs.logits, batch["labels"])
            loss_dict["diversity"] = div_loss.item()
            total = total + self.div_weight * div_loss

        # 3. Terminology preservation
        if self.term_weight > 0 and self.term_ids is not None:
            term_loss = self._terminology_loss(
                outputs.logits, batch["labels"], self.term_ids.to(outputs.logits.device)
            )
            loss_dict["terminology"] = term_loss.item()
            total = total + self.term_weight * term_loss

        # 4. EWC penalty
        if self.ewc_lambda > 0 and self.ewc is not None and self.ewc.ready:
            ewc_loss = self.ewc.penalty(model)
            loss_dict["ewc"] = ewc_loss.item()
            total = total + self.ewc_lambda * ewc_loss

        # 5. CrossCLR (pre-computed, only on pair batches)
        if self.crossclr_weight > 0 and crossclr_loss is not None:
            loss_dict["crossclr"] = crossclr_loss.item()
            total = total + self.crossclr_weight * crossclr_loss

        loss_dict["total"] = total.item()
        return total, loss_dict

    @staticmethod
    def _diversity_loss(logits, labels):
        """
        Negative entropy of predicted distribution over label positions.
        Lower entropy = more peaked = less diverse vocabulary → penalize.
        """
        # Only compute over positions where we have real labels
        label_mask = labels != -100
        if not label_mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Get logits at label positions
        masked_logits = logits[label_mask]  # (N, vocab)
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # (N,)

        # We want HIGH entropy (diverse predictions)
        # Loss = -mean(entropy) → minimizing this maximizes entropy
        return -entropy.mean()

    @staticmethod
    def _terminology_loss(logits, labels, term_ids):
        """
        Average negative log-probability of domain terms when they appear in reference.
        Penalizes low probability assigned to domain vocabulary.
        """
        # Find positions where labels are domain terms
        label_flat = labels.reshape(-1)
        logits_flat = logits.reshape(-1, logits.shape[-1])

        # Create mask for domain term positions
        term_mask = torch.isin(label_flat, term_ids) & (label_flat != -100)
        if not term_mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Log probability of the correct term at each position
        log_probs = F.log_softmax(logits_flat, dim=-1)
        term_log_probs = log_probs[term_mask].gather(
            -1, label_flat[term_mask].unsqueeze(-1)
        ).squeeze(-1)

        # Maximize log probability → minimize negative
        return -term_log_probs.mean()