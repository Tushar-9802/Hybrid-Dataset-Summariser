"""
train.py — Training Entry Point
Runs one or all phases of the curriculum training pipeline.

Run from repo root:
    python scripts/train.py --config configs/phase1.yaml --run-name phase1
    python scripts/train.py --config configs/phase2.yaml --run-name phase2
    python scripts/train.py --config configs/phase3.yaml --run-name phase3
    python scripts/train.py --all --run-name full_run           # All 3 phases
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Must be set before the CUDA context initializes for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import PhaseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config with inheritance."""
    path = Path(config_path)
    if not path.exists():
        log.error(f"Config not found: {path}")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "inherits" in config:
        parent_path = path.parent / config.pop("inherits")
        with open(parent_path) as f:
            parent = yaml.safe_load(f)
        # Child overrides parent
        merged = deep_merge(parent, config)
        return merged

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def seed_everything(seed: int):
    """Seed all RNGs and enable deterministic kernels (functional determinism).

    Non-deterministic GPU ops fall back with a warning rather than erroring
    (warn_only=True) — sufficient for multi-seed variance estimation, without
    forcing CPU fallbacks that would change runtimes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    log.info(f"Seeded all RNGs with seed={seed} (cudnn deterministic, warn_only)")


def run_single_phase(config_path: str, run_name: str, seed: int = 42,
                     checkpoint_interval_sec: int = 3600):
    """Run a single training phase."""
    config = load_config(config_path)
    config["seed"] = seed
    config.setdefault("monitoring", {})["checkpoint_interval_sec"] = checkpoint_interval_sec
    phase = config.get("phase", 1)
    config["data"] = config.get("data", {})
    config["data"]["hdf5_path"] = config["data"].get("hdf5_path", "data/hdf5/engineering.h5")

    log.info(f"Starting Phase {phase}: {run_name}")
    log.info(f"Config: {config_path}")

    trainer = PhaseTrainer(config, run_name)

    # Load replay indices for Phase 2-3
    replay_indices = None
    if phase >= 2:
        replay_path = Path(config.get("resume_from", "checkpoints/phase1/final")) / "training_state.pt"
        if replay_path.exists():
            state = torch.load(replay_path, map_location="cpu", weights_only=False)
            replay_indices = state.get("replay_indices", None)
            if replay_indices:
                log.info(f"Loaded {len(replay_indices)} replay indices")

    result = trainer.train_phase(replay_indices=replay_indices)

    log.info(f"Phase {phase} complete. Checkpoint: {result['final_checkpoint']}")
    log.info(f"Next: python scripts/evaluate.py --mode adapter "
             f"--checkpoint {result['final_checkpoint']} "
             f"--run-name {run_name} --compare baseline")

    return result


def run_all_phases(run_name: str, seed: int = 42, checkpoint_interval_sec: int = 3600):
    """Run all 3 phases sequentially."""
    configs_dir = Path("configs")

    # Phase 1
    log.info("="*60)
    log.info("PHASE 1: Paper-only training")
    log.info("="*60)
    result1 = run_single_phase(str(configs_dir / "phase1.yaml"), f"{run_name}_phase1", seed,
                               checkpoint_interval_sec)
    replay_indices = result1.get("replay_indices")

    # Save replay indices into the checkpoint for Phase 2 to find
    if replay_indices:
        import json
        replay_path = Path(result1["final_checkpoint"]) / "replay_indices.json"
        with open(replay_path, "w") as f:
            json.dump(replay_indices, f)
        # Also save into training_state.pt
        state_path = Path(result1["final_checkpoint"]) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            state["replay_indices"] = replay_indices
            torch.save(state, state_path)

    # Phase 2
    log.info("="*60)
    log.info("PHASE 2: Mixed training with EWC + OPLoRA")
    log.info("="*60)
    result2 = run_single_phase(str(configs_dir / "phase2.yaml"), f"{run_name}_phase2", seed,
                               checkpoint_interval_sec)

    # Phase 3
    log.info("="*60)
    log.info("PHASE 3: Video-focused with CrossCLR")
    log.info("="*60)
    result3 = run_single_phase(str(configs_dir / "phase3.yaml"), f"{run_name}_phase3", seed,
                               checkpoint_interval_sec)

    log.info("="*60)
    log.info("ALL PHASES COMPLETE")
    log.info(f"Final checkpoint: {result3['final_checkpoint']}")
    log.info(f"Run full eval: python scripts/evaluate.py --mode adapter "
             f"--checkpoint {result3['final_checkpoint']} "
             f"--run-name {run_name}_final --compare baseline")
    log.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid-Dataset Summariser (phase-based curriculum)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --config configs/phase1.yaml --run-name phase1
  python scripts/train.py --all --run-name full_run
  python scripts/train.py --config configs/ablations/no_ewc.yaml --run-name ablation_no_ewc
        """,
    )
    parser.add_argument("--config", type=str, help="Path to phase YAML config")
    parser.add_argument("--run-name", type=str, required=True, help="Name for this run")
    parser.add_argument("--all", action="store_true", help="Run all 3 phases sequentially")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-interval-sec", type=int, default=3600,
                        help="Seconds between resumable checkpoints (default 3600 = 1h)")
    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed)

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    if args.all:
        run_all_phases(args.run_name, args.seed, args.checkpoint_interval_sec)
    elif args.config:
        run_single_phase(args.config, args.run_name, args.seed, args.checkpoint_interval_sec)
    else:
        parser.error("Provide --config or --all")


if __name__ == "__main__":
    main()