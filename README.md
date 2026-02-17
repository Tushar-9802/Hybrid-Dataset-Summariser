
# Hybrid-Dataset Summariser

Cross-modal learning framework addressing catastrophic forgetting in video summarization when fine-tuning language models on domain-specific text. Follow-up to our IMPACT 2025 (Springer) paper documenting 14-50% performance degradation in cross-modal LoRA transfer.

## Problem

Fine-tuning Mistral-7B with LoRA on 25,000 academic papers improved paper summarization (+6-8% ROUGE-1) but degraded video summarization by 14-50% (ROUGE-2 worst hit at -36-50%). Error analysis showed adapters entangled domain knowledge with academic style conventions (passive voice +112%, nominalization +85%) in the rank-16 subspace, producing outputs like "this paper presents" for conversational video content.

Published:  *Cross-Modal Transfer Learning in Domain-Adaptive Video Summarization* , IMPACT 2025 (Springer), presented December 6, 2025.

## Solution

Six synergistic techniques targeting the identified failure modes:

**OPLoRA** -- Orthogonal SVD projection (architectural, not loss penalty) preserving base model subspace. Prevents domain knowledge / style entanglement that caused the original failure.

**LoRA+** -- Asymmetric learning rates (eta_B/eta_A = 8) for faster convergence. Sweep planned over {4, 8, 16}.

**CrossCLR** -- Contrastive loss aligning paper and video embeddings (tau=0.03, lambda_intra=0.7).

**EWC** -- Elastic Weight Consolidation preventing catastrophic forgetting (lambda=200 to 400 progressive).

**Curriculum Learning** -- 3-phase progressive mixing with D_gap gating and replay buffer.

**Monitoring** -- Real-time rho_k subspace interference tracking with adaptive k-scaling (16 to 128).

## Domain & Dataset

Locked to **Engineering/CS** (single domain). No medical or scientific data.

Papers collected from two sources: ccdv/arxiv-summarization (SBERT-filtered, threshold >= 0.35) and jamescalam/ai-arxiv (real arXiv categories). All papers verified as CS-domain via semantic similarity filtering against CS-domain anchor descriptions.

| Source            | Count           | Details                                         |
| ----------------- | --------------- | ----------------------------------------------- |
| arXiv CS papers   | 2,368           | cs.AI, cs.CL, cs.LG, cs.CV, cs.RO, cs.SE, cs.DS |
| YouTube CS videos | 738             | Lectures, conference talks, tutorials           |
| Cross-modal pairs | 1,218           | SBERT-mined (threshold >= 0.55)                 |
| **Total**   | **4,324** | 80/10/10 train/val/test split                   |

### Dataset Statistics

| Modality | Train | Val | Test | Source Tokens (mean)          | Label Tokens (mean) |
| -------- | ----- | --- | ---- | ----------------------------- | ------------------- |
| Papers   | 1,894 | 236 | 238  | 1,022 (truncated from 12,383) | 211                 |
| Videos   | 590   | 73  | 75   | 1,018 (truncated from 5,471)  | 152                 |
| Pairs    | 974   | 121 | 123  | -                             | -                   |

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

## Training Schedule

```
Phase 1 (3 epochs): 100% papers
  -> Establish domain vocabulary
  -> Compute Fisher Information Matrix at end
  -> Gate Phase 2 entry on D_gap < 0.7

Phase 2 (1 epoch): 50% papers + 40% videos + 10% cross-modal pairs
  -> Activate CrossCLR, EWC (lambda=200), diversity, term preservation
  -> 10% replay buffer from Phase 1
  -> Monitor rho_k; hot-swap to k=128 if > 0.5

Phase 3 (1 epoch): 30% papers + 60% videos + 10% cross-modal pairs
  -> EWC lambda -> 400
  -> Final specialization
```

## Targets

| Metric          | Baseline (IMPACT 2025)               | Target           |
| --------------- | ------------------------------------ | ---------------- |
| Video ROUGE-1   | 0.347 (base) / 0.272 (degraded LoRA) | >= 0.37          |
| Paper ROUGE-1   | 0.333 (LoRA)                         | >= 0.33 (retain) |
| Passive voice % | 31.4% (LoRA) / 14.8% (base)          | <= 16%           |

## Hardware

| Component | Spec                                             |
| --------- | ------------------------------------------------ |
| GPU       | NVIDIA RTX 5070 Ti (16GB VRAM, Blackwell sm_120) |
| CPU       | AMD Ryzen 7 7800X3D (8C/16T)                     |
| RAM       | 32GB DDR5                                        |
| Storage   | 1TB NVMe SSD                                     |

**VRAM Budget (BF16 training):**

```
Mistral-7B (BF16):          14.0 GB
LoRA adapters (r=32):        0.85 GB
Gradients (checkpointed):    0.40 GB
8-bit AdamW (bitsandbytes):  0.85 GB
Batch data (batch=3):        0.15 GB
Total:                      ~16.25 GB
```

8-bit AdamW from bitsandbytes is required. Standard AdamW overflows 16GB. Gradient checkpointing mandatory.

## Environment Setup

**Python:** 3.11 (Miniconda)
**PyTorch:** Nightly with CUDA 12.8 (RTX 5070 Ti requires sm_120)
**JS Runtime:** Deno (required by yt-dlp since 2025.11.12 for YouTube downloads)

```bash
conda create -n hybrid-video python=3.11 -y
conda activate hybrid-video

pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt

# Deno for YouTube data collection
winget install DenoLand.Deno
```

## Project Structure

```
Hybrid-Dataset-Summariser/
├── configs/
│   └── phase1_engineering.yaml
├── data/
│   ├── raw/
│   │   ├── papers/                # arXiv CS papers (2,368 JSON)
│   │   └── videos/                # YouTube audio + meta.json (738)
│   ├── processed/
│   │   ├── cross_modal_pairs/     # SBERT-mined pairs (1,218)
│   │   └── manifest.json          # Quality-filtered sample manifest
│   ├── hdf5/
│   │   └── engineering.h5         # Training-ready tokenized dataset (11.3 MB)
│   └── quarantine/                # Rejected papers (non-CS)
├── logs/
├── src/
│   ├── data/
│   │   ├── hf_arxiv_dataset.py
│   │   └── yt_audio_collector.py
│   └── processing/
│       ├── collect_cs_papers.py   # SBERT-filtered CS paper collection
│       ├── collect_ai_arxiv.py    # Supplemental papers (jamescalam/ai-arxiv)
│       ├── filter_cs_papers.py    # Non-CS paper detection and quarantine
│       ├── pair_miner.py          # Cross-modal SBERT pair mining
│       ├── quality_filter.py      # Token-level quality gates + split assignment
│       ├── dataset_builder.py     # HDF5 packaging with Mistral tokenizer
│       ├── validate_dataset.py    # 12-check dataset validation
│       ├── transcribe.py          # Whisper transcription
│       ├── summary_generator.py   # GPT-4o-mini video summaries
│       ├── validate_transcripts.py
│       └── validate_summaries.py
├── cookies.txt                    # YouTube auth (gitignored)
├── rebuild_overnight.bat          # Full pipeline rebuild script
├── requirements.txt
└── README.md
```

## Technical Configuration

```yaml
lora:
  rank: 32
  alpha: 64
  dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

oplora:
  implementation: projection
  projection_rank: 16           # Fallback to 128 if rho_k > 0.5

lora_plus:
  lr_A: 1.0e-4
  lr_B: 8.0e-4                 # ratio = 8

losses:
  ce_weight: 1.0
  crossclr_weight: 0.3         # tau=0.03, gamma=0.9
  diversity_weight: 0.2
  term_preservation_weight: 0.4
  # OPLoRA is architectural (forward pass), not a loss term
  # EWC added in Phase 2 (lambda=200) and Phase 3 (lambda=400)
```

## Publication Target

**Venue:** Springer Multimedia Systems or IEEE Transactions on Multimedia
**Format:** 10-14 page journal paper with 8-10 ablation configurations
**Timeline:** Submit when ready (no hard deadline)

## Research Foundation

| Paper                | Contribution                                    | Key Parameters                        |
| -------------------- | ----------------------------------------------- | ------------------------------------- |
| CrossCLR (ICCV 2021) | Cross-modal contrastive learning                | tau=0.03, lambda_intra=0.7, gamma=0.9 |
| OPLoRA (2024)        | Orthogonal projection for subspace preservation | k=16/128                              |
| LoRA+ (2024)         | Asymmetric learning rates                       | eta_B/eta_A = 8                       |
| EWC on Gemma2 (2025) | Forgetting prevention for LLMs                  | lambda=200-400                        |
| MoNA (ICML 2024)     | Modality gap formalization                      | D_gap monitoring                      |
| LfVS (CVPR 2024)     | LLM-based video summarization benchmarks        | Quality filters                       |

## License

MIT License

---

Last updated: February 17, 2026
