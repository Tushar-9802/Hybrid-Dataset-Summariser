# Hybrid-Dataset Summariser

Cross-modal learning framework preventing catastrophic forgetting in video summarization when fine-tuning language models on domain-specific text. Follow-up to our IMPACT 2025 (Springer) paper documenting 14-50% performance degradation in cross-modal LoRA transfer.

## Results

| System | Video R-1 | Video R-2 | Video PVR% | CMC | Paper R-1 |
|--------|-----------|-----------|------------|-----|-----------|
| Zero-shot Mistral-7B | 0.263 | 0.032 | 10.0 | 0.356 | 0.320 |
| Vanilla LoRA (no protection) | 0.315 | 0.060 | 18.4 | 0.470 | 0.451 |
| IMPACT 2025 naive LoRA | 0.272 | 0.060 | 31.4 | — | 0.333 |
| **Phase 1** (LoRA+) | 0.305 | 0.052 | 15.9 | 0.474 | **0.429** |
| **Phase 2** (+OPLoRA, +EWC) | 0.381 | 0.101 | **10.2** | 0.473 | 0.296 |
| **Phase 3** (full framework) | **0.417** | **0.119** | 14.1 | **0.531** | 0.309 |

All improvements statistically significant (p < 0.005, paired t-test). Video R-1 +58%, R-2 +272% vs baseline. PVR held at 14.1% vs 31.4% catastrophic (IMPACT 2025).

### Component Ablations

| Configuration | Video R-1 | Video R-2 | BERTScore | PVR% | Nom.% |
|--------------|-----------|-----------|-----------|------|-------|
| Full framework | 0.417 | 0.119 | 0.151 | 14.1 | 6.5 |
| - EWC | **0.438** | **0.137** | **0.169** | **11.0** | 6.5 |
| - OPLoRA | 0.389 | 0.114 | 0.104 | 9.8 | 10.5 |
| Vanilla LoRA | 0.315 | 0.060 | 0.091 | 18.4 | — |

**Key finding:** Curriculum phasing is the primary mechanism (~80% of improvement over vanilla LoRA). OPLoRA provides semantic quality preservation (BERTScore +45%) and nominalization control (6.5% vs 10.5%). EWC at lambda=400 over-constrains — removing it improves all video metrics.

## Problem

Fine-tuning Mistral-7B with LoRA on 25,000 academic papers improved paper summarization (+6-8% ROUGE-1) but degraded video summarization by 14-50% (ROUGE-2 worst hit at -36-50%). Error analysis showed adapters entangled domain knowledge with academic style conventions (passive voice +112%, nominalization +85%) in the rank-16 subspace, producing outputs like "this paper presents" for conversational video content.

Published: *Cross-Modal Transfer Learning in Domain-Adaptive Video Summarization*, IMPACT 2025 (Springer), presented December 6, 2025.

## Solution

Curriculum-based framework with progressive method activation:

**Phase 1** (3 epochs, 100% papers): LoRA+ asymmetric learning rates (eta_B/eta_A = 8) for domain knowledge acquisition. Fisher Information Matrix computed at exit.

**Phase 2** (1 epoch, 50/40/10 papers/videos/pairs): OPLoRA orthogonal projection (k=16) preserving base model subspace + EWC Fisher regularization (lambda=200) + diversity and terminology auxiliary losses + 10% replay buffer.

**Phase 3** (1 epoch, 30/60/10): CrossCLR contrastive alignment (tau=0.03) + EWC lambda=400 + full composite loss. Video-focused specialization.

Phase transitions gated by D_gap composite metric on video validation split. Ablations show curriculum phasing is the dominant intervention; protection mechanisms provide secondary gains.

## Dataset

Locked to **CS/Engineering** domain (worst degradation in IMPACT 2025).

| Source | Count | Details |
|--------|-------|---------|
| arXiv CS papers | 2,368 | cs.AI, cs.CL, cs.LG, cs.CV, cs.RO, cs.SE, cs.DS |
| YouTube CS videos | 738 | Lectures, conference talks, tutorials |
| Cross-modal pairs | 1,218 | SBERT-mined (threshold >= 0.55) |
| **Total** | **4,324** | 80/10/10 train/val/test split |

| Modality | Train | Val | Test | Source Tokens (mean) | Label Tokens (mean) |
|----------|-------|-----|------|---------------------|-------------------|
| Papers | 1,894 | 236 | 238 | 1,022 (truncated from 12,383) | 211 |
| Videos | 590 | 73 | 75 | 1,018 (truncated from 5,471) | 152 |
| Pairs | 974 | 121 | 123 | — | — |

Paper source: article body, label: full abstract. Video source: Whisper transcript, label: GPT-4o-mini summary (length-adaptive, 80-180 words). Cross-modal pairs: SBERT cosine similarity, mean 0.64, range 0.55-0.87.

## Data Pipeline

```
[ccdv/arxiv-summarization] --SBERT filter (>=0.35)--> papers/
[jamescalam/ai-arxiv]      --category filter--------> papers/
[YouTube search queries]   --yt-dlp + Whisper--------> videos/
                           --GPT-4o-mini-------------> video summaries

papers/ + videos/ --SBERT cosine similarity--> cross_modal_pairs/
                  --Mistral tokenizer--------> manifest.json
                  --pad + truncate-----------> engineering.h5
```

## Training

```
Phase 1: python scripts/train.py --config configs/phase1.yaml --run-name phase1
Phase 2: python scripts/train.py --config configs/phase2.yaml --run-name phase2
Phase 3: python scripts/train.py --config configs/phase3.yaml --run-name phase3
All:     python scripts/train.py --all --run-name full_run
```

## Evaluation

```
python scripts/evaluate.py --mode base --run-name baseline
python scripts/evaluate.py --mode adapter --checkpoint checkpoints/phase3/final \
    --run-name phase3 --compare baseline
```

Metrics: ROUGE-1/2/L, BERTScore F1, passive voice %, nominalization %, type-token ratio, cross-modal consistency (SBERT cosine), D_gap composite with phase-gating thresholds. Statistical tests: paired t-test, Cohen's d, Wilcoxon, bootstrap 95% CI.

## Ablations

```
python scripts/train.py --config configs/ablations/vanilla_lora.yaml --run-name abl_vanilla
python scripts/train.py --config configs/ablations/no_ewc.yaml --run-name abl_no_ewc
python scripts/train.py --config configs/ablations/no_oplora.yaml --run-name abl_no_oplora
python scripts/train.py --config configs/ablations/no_crossclr.yaml --run-name abl_no_crossclr
python scripts/train.py --config configs/ablations/no_lora_plus.yaml --run-name abl_no_loraplus
python scripts/train.py --config configs/ablations/no_curriculum.yaml --run-name abl_no_curriculum
```

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5070 Ti (16GB VRAM, Blackwell sm_120) |
| CPU | AMD Ryzen 7 7800X3D (8C/16T) |
| RAM | 32GB DDR5 |
| Peak VRAM | 11.3 GB (Phase 3) |

## Technical Configuration

```yaml
model: Mistral-7B-v0.1 (4-bit NF4, double quant, bfloat16 compute)

lora:
  rank: 32
  alpha: 64
  dropout: 0.1
  targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  trainable: 83.9M params (1.16% of 7.24B total)

optimizer: 8-bit AdamW (bitsandbytes)
batch: micro=3, grad_accum=8, effective=24

phases:
  1: lr_A=1e-4, lr_B=8e-4, 3 epochs papers-only
  2: lr_A=5e-5, lr_B=4e-4, 1 epoch, +OPLoRA(k=16), +EWC(lambda=200)
  3: lr_A=2.5e-5, lr_B=2e-4, 1 epoch, +CrossCLR(tau=0.03), EWC(lambda=400)

losses:
  ce_weight: 1.0
  crossclr_weight: 0.3
  diversity_weight: 0.2
  term_preservation_weight: 0.4
```

## Project Structure

```
Hybrid-Dataset-Summariser/
├── configs/
│   ├── base.yaml, phase1.yaml, phase2.yaml, phase3.yaml
│   └── ablations/
│       ├── vanilla_lora.yaml, no_ewc.yaml, no_oplora.yaml
│       ├── no_crossclr.yaml, no_lora_plus.yaml, no_curriculum.yaml
├── data/
│   ├── raw/papers/, raw/videos/
│   ├── processed/manifest.json, cross_modal_pairs/
│   └── hdf5/engineering.h5
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation harness
│   ├── generate_figures.py      # Publication figures (fig1-fig5)
│   └── generate_fig6.py         # Ablation comparison figure
├── src/
│   ├── training/
│   │   ├── dataset.py           # HDF5 DataLoader + CurriculumSampler
│   │   ├── model.py             # QLoRA + LoRA+ optimizer
│   │   ├── trainer.py           # PhaseTrainer with gating
│   │   ├── losses.py            # Composite loss orchestrator
│   │   ├── oplora.py            # Orthogonal projection hooks
│   │   ├── ewc.py               # Fisher computation + penalty
│   │   ├── crossclr.py          # Contrastive loss + momentum queue
│   │   └── monitoring.py        # WandB + local logging
│   ├── processing/              # Data pipeline scripts
│   └── data/                    # Collection + transcription
├── results/
│   ├── baseline/, phase1/, phase2/, phase3/
│   ├── abl_vanilla/, abl_no_ewc/, abl_no_oplora/
├── checkpoints/
│   ├── phase1/final/, phase2/final/, phase3/final/
│   └── svd_cache/
└── docs/RESEARCH_NOTES.md
```

## Models & Data

| Resource | Link |
|----------|------|
| Phase 3 adapter (best video quality) | [HuggingFace](https://huggingface.co/Tushar9802/hybrid-summariser-crossmodal-lora) |
| Phase 2 adapter (best style balance) | [HuggingFace](https://huggingface.co/Tushar9802/hybrid-summariser-phase2-lora) |
| Dataset (4,324 samples, HDF5) | [Kaggle](https://www.kaggle.com/datasets/tusharjaju/hybrid-dataset-summariser-crossmodal) |
| Prior work (IMPACT 2025) | [GitHub](https://github.com/Tushar-9802/YouTube-Transcript-Summarizer) |

## Publications

1. T. Jaju, T. Saharawat, S. Bhatia, S. Rastogi, "Cross-Modal Transfer Learning in Domain-Adaptive Video Summarization," *IMPACT 2025*, Springer (presented Dec 2025; proceedings forthcoming)
2. T. Jaju, T. Saharawat, S. Bhatia, S. Rastogi, "Preventing Catastrophic Forgetting in Cross-Modal Summarization: A Curriculum-Based Approach with Orthogonal Subspace Preservation," *in preparation*

## Research Foundation

| Paper | Venue | Contribution | Key Parameters |
|-------|-------|-------------|----------------|
| LoRA+ | ICML 2024 | Asymmetric learning rates | eta_B/eta_A = 8 |
| OPLoRA | AAAI 2026 | Orthogonal subspace preservation | k=16/128 |
| EWC | PNAS 2017 | Fisher-weighted regularization | lambda=200/400 |
| CrossCLR | ICCV 2021 | Cross-modal contrastive learning | tau=0.03, lambda_intra=0.75 |
| MoNA | ICML 2024 | Modality gap formalization | D_gap monitoring |
| LfVS | CVPR 2024 | LLM video summarization benchmarks | Quality filters |
| VISTA | 2024 | Academic video summarization | Dataset reference |
| CoMM | 2024 | Bidirectional cross-modal transfer | Curriculum validation |

## Environment Setup

```bash
conda create -n hybrid-video python=3.11 -y && conda activate hybrid-video
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
winget install DenoLand.Deno  # Required by yt-dlp for YouTube
```

## License

MIT License

---

Last updated: March 21, 2026