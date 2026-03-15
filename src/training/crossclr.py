"""
crossclr.py — Cross-Modal Contrastive Learning (CrossCLR)
Inter + intra-modality contrastive loss with connectivity-based false negative pruning.

Source: CrossCLR (ICCV 2021)
  τ = 0.03 (temperature)
  λ_intra = 0.75 (intra-modality weight)
  κ = 3.5e-4 (connectivity threshold)
  queue_size = 3000 (momentum memory bank)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Linear projection from LM hidden dim to contrastive embedding space."""

    def __init__(self, hidden_dim: int = 4096, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class CrossCLRLoss(nn.Module):
    """
    Cross-modal contrastive loss with momentum queue.

    L(xi) = -log[δ(xi,yi) / (δ(xi,yi) + Σ_inter + λ·Σ_intra)]
    where δ(a,b) = exp(a·b / τ)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        embed_dim: int = 256,
        queue_size: int = 3000,
        tau: float = 0.03,
        lambda_intra: float = 0.75,
        kappa: float = 3.5e-4,
    ):
        super().__init__()
        self.tau = tau
        self.lambda_intra = lambda_intra
        self.kappa = kappa
        self.embed_dim = embed_dim

        # Projection heads (one per modality)
        self.paper_proj = ProjectionHead(hidden_dim, embed_dim)
        self.video_proj = ProjectionHead(hidden_dim, embed_dim)

        # Momentum queue
        self.register_buffer("queue", torch.randn(queue_size, embed_dim))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer("queue_modality", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    @torch.no_grad()
    def _enqueue(self, embeddings, modality_id):
        """Add embeddings to FIFO queue."""
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = embeddings.detach()
            self.queue_modality[ptr:ptr + batch_size] = modality_id
        else:
            # Wrap around
            overflow = (ptr + batch_size) - self.queue_size
            fit = batch_size - overflow
            self.queue[ptr:ptr + fit] = embeddings[:fit].detach()
            self.queue_modality[ptr:ptr + fit] = modality_id
            self.queue[:overflow] = embeddings[fit:].detach()
            self.queue_modality[:overflow] = modality_id

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def get_embeddings(self, model, input_ids, attention_mask, modality: str):
        """
        Extract mean-pooled embeddings from last hidden state.

        Args:
            model: the causal LM
            input_ids, attention_mask: standard inputs
            modality: "paper" or "video" (selects projection head)
        """
        with torch.no_grad():
            model.eval()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            model.train()

        # Last hidden state
        hidden = outputs.hidden_states[-1]  # (B, T, D)

        # Mean pool over non-padding positions
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Project
        proj = self.paper_proj if modality == "paper" else self.video_proj
        return proj(pooled.float())  # (B, embed_dim), L2-normalized

    def forward(self, paper_embeds, video_embeds):
        """
        Compute CrossCLR loss for a batch of paper-video pairs.

        Args:
            paper_embeds: (B, embed_dim) normalized
            video_embeds: (B, embed_dim) normalized

        Returns:
            scalar loss
        """
        B = paper_embeds.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=paper_embeds.device, requires_grad=True)

        # Inter-modality: paper[i] should be close to video[i]
        # Similarity matrix between papers and videos
        queue_snapshot = self.queue.clone().detach()
        sim_pv = (paper_embeds @ video_embeds.T) / self.tau  # (B, B)

        # Positive pairs are on the diagonal
        positives = torch.diag(sim_pv)  # (B,)

        # Inter-modality negatives: all off-diagonal pairs
        inter_neg = sim_pv.exp().sum(dim=1) - positives.exp()  # (B,)

        # Queue negatives (mixed modalities)
        sim_p_queue = (paper_embeds @ queue_snapshot.T) / self.tau  # (B, Q)
        queue_neg = sim_p_queue.exp().sum(dim=1)  # (B,)

        # Intra-modality negatives (paper vs paper, with connectivity weighting)
        sim_pp = (paper_embeds @ paper_embeds.T) / self.tau  # (B, B)
        # Connectivity: average cosine similarity
        connectivity = (paper_embeds @ paper_embeds.T).mean(dim=1)  # (B,)
        weights = (connectivity / self.kappa).exp()  # (B,)

        intra_neg = (sim_pp.exp() * weights.unsqueeze(0)).sum(dim=1)  # (B,)
        # Remove self-similarity
        intra_neg = intra_neg - (torch.diag(sim_pp).exp() * weights)

        # Loss: -log(positive / (positive + inter + λ * intra + queue))
        denominator = positives.exp() + inter_neg + self.lambda_intra * intra_neg + queue_neg
        loss = -torch.log(positives.exp() / denominator.clamp(min=1e-8))

        # Update queue with both modalities
        self._enqueue(paper_embeds, modality_id=0)
        self._enqueue(video_embeds, modality_id=1)

        return loss.mean()