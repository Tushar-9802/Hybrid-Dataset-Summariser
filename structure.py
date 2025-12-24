#!/usr/bin/env python3
"""
structure.py - Project Directory Structure Creator
Creates complete folder structure for Hybrid-Dataset Summariser project
"""

import os
from pathlib import Path
from typing import List, Dict

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_directory(path: Path, description: str = "") -> None:
    """Create directory if it doesn't exist"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}"
        desc = f" - {description}" if description else ""
        print(f"{status} {path}{desc}")
    except Exception as e:
        print(f"{Colors.FAIL}✗{Colors.ENDC} {path} - Error: {e}")

def create_file(path: Path, content: str = "") -> None:
    """Create file with optional content"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content)
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} {path}")
        else:
            print(f"{Colors.WARNING}⊙{Colors.ENDC} {path} (already exists)")
    except Exception as e:
        print(f"{Colors.FAIL}✗{Colors.ENDC} {path} - Error: {e}")

def create_project_structure():
    """Create complete project directory structure"""
    
    base_dir = Path.cwd()
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}Creating Hybrid-Dataset Summariser Project Structure{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Base directory: {base_dir}{Colors.ENDC}\n")
    
    # Directory structure with descriptions
    directories = {
        # Configs
        "configs": "Training configuration files",
        "configs/ablations": "Ablation study configurations",
        
        # Scripts
        "scripts": "Execution and preprocessing scripts",
        
        # Source code
        "src": "Core implementation modules",
        "src/models": "Model wrappers and LoRA configurations",
        "src/data": "Dataset loaders and preprocessing",
        "src/losses": "Loss function implementations",
        "src/training": "Training loop and utilities",
        "src/evaluation": "Evaluation metrics and tests",
        "src/utils": "Helper utilities",
        
        # Data directories
        "data": "Data storage root",
        "data/raw": "Raw downloaded data",
        "data/raw/youtube": "Downloaded YouTube videos",
        "data/raw/papers": "Downloaded papers (PDFs/texts)",
        "data/processed": "Processed data",
        "data/processed/transcripts": "YouTube transcripts",
        "data/processed/summaries": "GPT-4o-mini generated summaries",
        "data/processed/embeddings": "SBERT embeddings for pair mining",
        "data/validated": "Quality-validated data",
        "data/validated/youtube": "Quality-checked YouTube summaries",
        "data/validated/cross_modal": "Cross-modal pairs",
        "data/datasets": "Final HDF5 datasets",
        "data/test": "Test sets",
        "data/lexicons": "Domain-specific term dictionaries",
        
        # Checkpoints
        "checkpoints": "Model checkpoints",
        "checkpoints/phase1": "Phase 1 checkpoints",
        "checkpoints/phase1/medical": "Medical domain Phase 1",
        "checkpoints/phase1/engineering": "Engineering domain Phase 1",
        "checkpoints/phase1/scientific": "Scientific domain Phase 1",
        "checkpoints/phase2": "Phase 2 checkpoints",
        "checkpoints/phase2/medical": "Medical domain Phase 2",
        "checkpoints/phase2/engineering": "Engineering domain Phase 2",
        "checkpoints/phase2/scientific": "Scientific domain Phase 2",
        "checkpoints/phase3": "Phase 3 checkpoints",
        "checkpoints/phase3/medical": "Medical domain Phase 3",
        "checkpoints/phase3/engineering": "Engineering domain Phase 3",
        "checkpoints/phase3/scientific": "Scientific domain Phase 3",
        
        # Results
        "results": "Evaluation results",
        "results/baselines": "Baseline model results",
        "results/phase1": "Phase 1 evaluation results",
        "results/phase2": "Phase 2 evaluation results",
        "results/phase3": "Phase 3 evaluation results",
        "results/ablations": "Ablation study results",
        "results/paper": "Paper figures and tables",
        
        # Templates
        "templates": "Prompt templates",
        
        # Paper
        "paper": "Workshop/conference paper",
        "paper/figures": "Paper figures",
        "paper/sections": "LaTeX sections",
        
        # Documentation
        "docs": "Project documentation",
        
        # Logs
        "logs": "Training and evaluation logs",
        "logs/wandb": "Weights & Biases logs",
    }
    
    print(f"{Colors.BOLD}Creating directories...{Colors.ENDC}\n")
    for dir_path, description in directories.items():
        create_directory(base_dir / dir_path, description)
    
    # Create __init__.py files for Python packages
    print(f"\n{Colors.BOLD}Creating Python package files...{Colors.ENDC}\n")
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/losses/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
    ]
    
    for init_file in init_files:
        create_file(base_dir / init_file, '"""Package initialization"""\n')
    
    # Create essential configuration files
    print(f"\n{Colors.BOLD}Creating configuration files...{Colors.ENDC}\n")
    
    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints
*.ipynb

# Data files (too large for git)
data/raw/**
data/processed/**
data/validated/**
data/datasets/*.h5
!data/lexicons/
!data/test/.gitkeep

# Model checkpoints (too large)
checkpoints/**/*.bin
checkpoints/**/*.pt
checkpoints/**/*.safetensors
!checkpoints/.gitkeep

# Results
results/**/*.png
results/**/*.pdf
results/**/*.csv

# Logs
logs/**
wandb/**
*.log

# OS
.DS_Store
Thumbs.db

# Secrets
.env
*.key
*.pem
api_keys.txt

# Large files
*.mp4
*.avi
*.mov
*.mkv
"""
    create_file(base_dir / ".gitignore", gitignore_content)
    
    # README.md
    readme_content = """# Hybrid-Dataset Summariser

Cross-modal learning framework for video summarization that prevents domain-specific training degradation through hybrid curriculum learning.

## 🎯 Project Overview

This project addresses a critical problem: domain-specific LoRA training on academic papers degrades video summarization performance by 14-50%. We solve this through:

- **3-Phase Curriculum Learning**: Papers → Mixed → Videos
- **Modality-Aware Loss**: Contrastive + Diversity + Term Preservation
- **Catastrophic Forgetting Prevention**: EWC + Replay Buffer
- **Enhanced LoRA**: Rank 32, asymmetric learning rates (LoRA+)

## 📊 Results

Target performance:
- Video ROUGE-1: ≥0.37 (+42% vs degraded baseline)
- Paper ROUGE-1: ≥0.35 (100% retention)
- Deployable on 16GB VRAM (RTX 5070 Ti)

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Hybrid-Dataset-Summariser.git
cd Hybrid-Dataset-Summariser

# 2. Create project structure
python structure.py

# 3. Set up environment
conda create -n hybrid-video python=3.10
conda activate hybrid-video
pip install -r requirements.txt

# 4. Download data (Week 3-4)
python scripts/download_youtube.py --config configs/data_collection.yaml

# 5. Train (Week 6-8)
python train.py --config configs/phase1_medical.yaml
```

## 📁 Project Structure

```
Hybrid-Dataset-Summariser/
├── configs/          # Training configurations
├── scripts/          # Data collection & preprocessing
├── src/              # Core implementation
│   ├── models/       # LoRA configurations
│   ├── data/         # Dataset loaders
│   ├── losses/       # Composite loss functions
│   ├── training/     # Training loop & EWC
│   └── evaluation/   # Metrics & statistical tests
├── data/             # Datasets (not in git)
├── checkpoints/      # Model checkpoints (not in git)
├── results/          # Evaluation results
└── docs/             # Documentation
```

## 🛠️ Hardware Requirements

**Minimum**:
- GPU: 16GB VRAM (RTX 4080, RTX 5070 Ti, or equivalent)
- CPU: 8 cores / 16 threads
- RAM: 32GB DDR4/DDR5
- Storage: 500GB SSD

**Optimized For**:
- GPU: RTX 5070 Ti (16GB VRAM)
- CPU: AMD Ryzen 7 7800X3D (16 threads)
- RAM: 32GB DDR5
- Storage: NVMe SSD

## 📚 Documentation

- [Complete Project Plan](docs/PROJECT_PLAN_PHASE0.md)
- [Hardware-Optimized Architecture](docs/HARDWARE_OPTIMIZED_ARCHITECTURE.md)
- [Week 1-2 Action Plan](docs/WEEK1-2_ACTION_PLAN.md)
- [Paper Links](docs/PAPER_LINKS.md)

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{hybrid-dataset-summariser-2025,
  title={Cross-Modal Transfer Learning for Abstractive Summarization: When Domain Knowledge Hurts Video Understanding},
  author={Your Name},
  booktitle={NeurIPS Workshop on Negative Results},
  year={2025}
}
```

## 🙏 Acknowledgments

This project builds upon:
- [CrossCLR](https://arxiv.org/abs/2109.14910) (ICCV 2021)
- [MoNA](https://arxiv.org/abs/2406.18864) (ICML 2024)
- [LfVS](https://arxiv.org/abs/2404.03398) (CVPR 2024)

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

## 🔗 Links

- [Paper](link-to-paper)
- [Dataset](link-to-huggingface)
- [Demo](link-to-demo)
"""
    create_file(base_dir / "README.md", readme_content)
    
    # requirements.txt
    requirements_content = """# Core ML
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

# Transformers & PEFT
transformers==4.36.0
peft==0.8.0
accelerate==0.26.0
bitsandbytes==0.42.0

# Data
datasets==2.16.0
h5py==3.10.0
pandas==2.1.4
numpy==1.26.3

# Evaluation
rouge-score==0.1.2
bert-score==0.3.13
sacrebleu==2.3.1
nltk==3.8.1
spacy==3.7.2

# APIs
openai==1.10.0
youtube-transcript-api==0.6.1
yt-dlp==2024.1.15
openai-whisper==20231117

# Embeddings
sentence-transformers==2.3.1
faiss-cpu==1.7.4

# Utilities
tqdm==4.66.1
wandb==0.16.2
tensorboard==2.15.1
matplotlib==3.8.2
seaborn==0.13.1
pyyaml==6.0.1

# Testing
pytest==7.4.4
pytest-cov==4.1.0
"""
    create_file(base_dir / "requirements.txt", requirements_content)
    
    # environment.yml
    environment_content = """name: hybrid-video
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - cudatoolkit=12.1
  - pip:
      - -r requirements.txt
"""
    create_file(base_dir / "environment.yml", environment_content)
    
    # Create placeholder .gitkeep files for empty directories
    print(f"\n{Colors.BOLD}Creating .gitkeep placeholders...{Colors.ENDC}\n")
    gitkeep_dirs = [
        "data/raw/youtube",
        "data/raw/papers",
        "data/processed/transcripts",
        "data/processed/summaries",
        "data/processed/embeddings",
        "data/validated/youtube",
        "data/validated/cross_modal",
        "data/datasets",
        "data/test",
        "checkpoints",
        "logs",
    ]
    
    for dir_path in gitkeep_dirs:
        create_file(base_dir / dir_path / ".gitkeep", "")
    
    # Create sample config file
    print(f"\n{Colors.BOLD}Creating sample configuration...{Colors.ENDC}\n")
    sample_config = """# Phase 1 Medical Domain Configuration
model: mistralai/Mistral-7B-v0.1
dtype: bfloat16
phase: phase1
domain: medical

# LoRA Configuration
lora_config:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
  task_type: CAUSAL_LM

# Training Configuration
batch_size: 3
gradient_accumulation_steps: 6
gradient_checkpointing: true
max_seq_length: 4096
num_epochs: 0.5

# Optimizer
optimizer:
  type: AdamW
  learning_rate: 2e-4
  lora_A_lr: 1e-4  # LoRA+ asymmetric
  lora_B_lr: 2e-4
  betas: [0.9, 0.999]
  weight_decay: 0.01

# Scheduler
lr_scheduler:
  type: cosine
  warmup_steps: 100
  num_training_steps: 2000

# Data
data_config:
  dataset_path: data/datasets/medical.h5
  data_mix:
    papers: 1.0
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2

# Logging
logging:
  wandb_project: hybrid-video-sum
  wandb_entity: null
  log_steps: 50
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3

# Output
output_dir: checkpoints/phase1/medical
"""
    create_file(base_dir / "configs/phase1_medical.yaml", sample_config)
    
    # Create LICENSE
    license_content = """MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    create_file(base_dir / "LICENSE", license_content)
    
    # Summary
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}✓ Project structure created successfully!{Colors.ENDC}\n")
    print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
    print(f"  1. Review README.md")
    print(f"  2. Initialize git: {Colors.OKCYAN}git init{Colors.ENDC}")
    print(f"  3. Create GitHub repo (see instructions below)")
    print(f"  4. Install dependencies: {Colors.OKCYAN}pip install -r requirements.txt{Colors.ENDC}")
    print(f"  5. Start reading papers (see docs/PAPER_LINKS.md)")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

if __name__ == "__main__":
    create_project_structure()