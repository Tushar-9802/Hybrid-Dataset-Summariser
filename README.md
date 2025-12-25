# Hybrid-Dataset Summariser

Cross-modal learning framework addressing catastrophic forgetting in video summarization when fine-tuning language models on domain-specific text.

## Problem Statement

Fine-tuning Mistral-7B with LoRA on academic papers improves paper summarization performance but catastrophically degrades video summarization by 14-50%. This occurs due to the modality gap between academic text and video transcripts, causing domain-specific training to overwrite general video understanding capabilities.

## Solution Approach

3-phase curriculum learning with cross-modal alignment to bridge the modality gap while preventing catastrophic forgetting through:

* Enhanced LoRA configuration (rank 32, asymmetric learning rates)
* Composite loss function (cross-entropy + contrastive + diversity + term preservation)
* Elastic Weight Consolidation with replay buffer
* Progressive curriculum: Papers (100%) -> Mixed (50/40/10) -> Videos (30/60/10)

## Hardware Requirements

**Tested Configuration:**

* GPU: NVIDIA RTX 5070 Ti (16GB VRAM, Blackwell architecture, sm_120)
* CPU: AMD Ryzen 7 7800X3D (8C/16T)
* RAM: 32GB DDR5
* Storage: 1TB NVMe SSD

**Minimum Requirements:**

* GPU: 16GB VRAM (RTX 4080, RTX 5070 Ti, A100-40GB)
* CPU: 8+ cores
* RAM: 32GB
* Storage: 500GB SSD

**Memory Budget (16GB VRAM):**

```
Mistral-7B (BF16):        14.0 GB
LoRA Adapters (r=32):      0.85 GB
LoRA Gradients:            0.85 GB
AdamW Optimizer:           1.70 GB
Batch Data (batch=3):      0.15 GB
Gradient Checkpointing:   -0.60 GB
Total:                    ~16.95 GB

Training Config:
- batch_size: 3
- gradient_accumulation_steps: 6
- effective_batch_size: 18
```

## Environment Setup

### Current Installation

**Platform:** Windows 11

**Python:** 3.11.14 (via Miniconda3)

**Environment Manager:** Conda

**PyTorch:** 2.7.0.dev (nightly build)

**CUDA:** 13.0

### Why These Specific Versions

**Python 3.11:** Performance improvements (10-60% faster than 3.10), better error messages, full ML library support.

**PyTorch Nightly with CUDA 13.0:** RTX 5070 Ti (Blackwell architecture) requires compute capability sm_120, which is only supported in PyTorch nightly builds with CUDA 12.8+. Stable PyTorch releases (as of December 2025) do not include sm_120 support.

**Conda:** Provides Python version isolation and package management. Note that PyTorch itself is installed via pip due to nightly build requirements.

### Installation Instructions

**Step 1: Install Miniconda**
Download from: https://docs.conda.io/en/latest/miniconda.html

**Step 2: Create Environment**

```bash
conda create -n hybrid-video python=3.11 -y
conda activate hybrid-video
```

**Step 3: Install PyTorch Nightly**

```bash
# RTX 5070 Ti requires CUDA 13.0 for sm_120 support
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

**Step 4: Verify GPU**

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:

```
PyTorch: 2.7.0.dev20250310+cu130
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

**Step 5: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Download NLTK Data**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Project Structure

```
Hybrid-Dataset-Summariser/
├── configs/              # Training configurations (Phase 1/2/3)
├── src/
│   ├── models/           # LoRA configuration wrappers
│   ├── data/             # Curriculum dataloader
│   ├── losses/           # Composite loss implementation
│   ├── training/         # Training loop with EWC
│   └── evaluation/       # Metrics and statistical tests
├── scripts/              # Data collection and preprocessing
├── data/                 # Datasets (gitignored)
├── checkpoints/          # Model checkpoints (gitignored)
├── results/              # Evaluation outputs
└── docs/                 # Documentation
```

## Technical Specifications

### LoRA Configuration

```yaml
rank: 32
alpha: 64
dropout: 0.1
target_modules:
  - q_proj, k_proj, v_proj, o_proj  # Attention
  - gate_proj, up_proj, down_proj   # MLP
learning_rates:
  lora_A: 1e-4   # Asymmetric (LoRA+)
  lora_B: 2e-4
trainable_params: ~85M (<1% of Mistral-7B)
```

### Loss Function

```
L_total = L_CE + 0.3*L_contrastive + 0.2*L_diversity + 0.4*L_term

Where:
- L_CE: Cross-entropy for generation
- L_contrastive: CrossCLR-inspired cross-modal alignment
- L_diversity: Shannon entropy (mode collapse prevention)
- L_term: Technical term preservation (domain knowledge retention)
```

### Training Phases

```
Phase 1 (Epoch 0.0-0.5): 100% papers
- Domain vocabulary establishment
- Fisher Information Matrix computation

Phase 2 (Epoch 0.5-1.5): 50% papers + 40% videos + 10% cross-modal
- Cross-modal alignment with EWC (lambda=400)
- 10% replay buffer

Phase 3 (Epoch 1.5-2.5): 30% papers + 60% videos + 10% cross-modal
- Video specialization
- Continued EWC + replay
```

## Usage

### Training

```bash
# Phase 1: Domain establishment
python train.py --config configs/phase1_medical.yaml

# Phase 2: Cross-modal alignment
python train.py --config configs/phase2_medical.yaml

# Phase 3: Video specialization
python train.py --config configs/phase3_medical.yaml
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/phase3/medical/final \
                   --test-set data/test/medical.h5
```

## Research Foundation

This work builds on:

* **CrossCLR** (ICCV 2021): Cross-modal contrastive learning methodology
* **MoNA** (ICML 2024): Modality gap formalization and meta-learning
* **EWC on Gemma2** (2025): Catastrophic forgetting prevention for LLMs
* **LoRA+** (2024): Asymmetric learning rates for parameter-efficient fine-tuning
* **LfVS** (CVPR 2024): LLM-based video summarization benchmarks

Full bibliography available in `docs/PAPER_LINKS.md`

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Out of Memory

Reduce batch size in configuration:

```yaml
batch_size: 2  # from 3
gradient_accumulation_steps: 9  # from 6
```

### RTX 5070 Ti Compatibility

The RTX 5070 Ti requires PyTorch nightly with CUDA 13.0 (or 12.8+). Stable PyTorch releases do not support compute capability sm_120 as of December 2025.

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.

---

Last updated: December 25, 2025
