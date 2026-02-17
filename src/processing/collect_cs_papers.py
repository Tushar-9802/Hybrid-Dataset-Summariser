"""
collect_cs_papers.py — Re-collect CS papers from ccdv/arxiv-summarization
Uses SBERT similarity to CS-domain anchors as primary filter (not keyword regex).

Streams the full HF dataset (~200K papers), encodes abstracts in batches,
keeps papers scoring >threshold against CS anchors. Stops at target count.

Features:
  - GPU-accelerated SBERT encoding (batched)
  - Auto-resume: skips papers already saved
  - Progress logging with ETA
  - Category assignment from best-matching CS anchor
  - Same output schema as original hf_arxiv_dataset.py

Run from repo root:
    python src/processing/collect_cs_papers.py                    # Full run
    python src/processing/collect_cs_papers.py --target 100       # Test run
    python src/processing/collect_cs_papers.py --threshold 0.45   # Stricter

Output: data/raw/papers/*.json (one file per paper)
"""

import json
import hashlib
import logging
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

PAPERS_DIR   = Path("data/raw/papers")
LOG_DIR      = Path("logs")
SBERT_MODEL  = "all-mpnet-base-v2"
BATCH_SIZE   = 256       # Abstracts to accumulate before SBERT encoding
TARGET       = 2500
THRESHOLD    = 0.35      # Verified: borderline at 0.35 are genuine CS papers

# Category anchors — one per CS subfield + one general
# Index maps directly to category label
CS_ANCHORS = [
    # 0: cs.AI
    "artificial intelligence reasoning knowledge representation planning search heuristics "
    "intelligent agents constraint satisfaction game playing expert systems",
    # 1: cs.CL
    "natural language processing text classification sentiment analysis machine translation "
    "language models tokenization parsing named entity recognition word embeddings",
    # 2: cs.CV
    "computer vision image classification object detection semantic segmentation "
    "convolutional neural networks image processing feature extraction visual recognition",
    # 3: cs.DS
    "data structures algorithms graph algorithms sorting searching dynamic programming "
    "computational complexity approximation algorithms hash tables binary trees",
    # 4: cs.LG
    "machine learning deep learning neural networks supervised learning unsupervised learning "
    "reinforcement learning gradient descent backpropagation optimization training",
    # 5: cs.RO
    "robotics motion planning path planning SLAM simultaneous localization and mapping "
    "robot kinematics manipulation grasping autonomous navigation mobile robots",
    # 6: cs.SE
    "software engineering code review testing debugging refactoring continuous integration "
    "software architecture design patterns version control agile development",
    # 7: general CS (fallback)
    "programming algorithm software computation database distributed systems "
    "operating systems computer networks cybersecurity cloud computing",
]

ANCHOR_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CV", "cs.DS", "cs.LG", "cs.RO", "cs.SE", "cs.AI"
    # Index 7 (general CS) falls back to cs.AI
]

LOG_DIR.mkdir(exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "collect_cs_papers.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


# ── TEXT UTILITIES ────────────────────────────────────────────────────────────

def is_english(text: str, threshold: float = 0.90) -> bool:
    ascii_count = sum(c.isascii() for c in text)
    return (ascii_count / max(len(text), 1)) >= threshold


def make_summary(abstract: str, max_words: int = 65) -> str:
    """Truncated abstract as reference summary (legacy compatibility)."""
    words = abstract.split()
    candidate = " ".join(words[:max_words])
    for punct in [".", "!", "?"]:
        last = candidate.rfind(punct)
        if last > len(candidate) * 0.5:
            return candidate[:last + 1].strip()
    return candidate.strip()


def stable_id(raw: dict) -> str:
    """Generate stable ID from article_id or abstract hash."""
    art_id = (raw.get("article_id") or "").strip()
    if art_id:
        # Clean arXiv ID to be filesystem-safe
        import re
        return re.sub(r"[^A-Za-z0-9._-]", "_", art_id)
    return hashlib.md5((raw.get("abstract") or "").encode()).hexdigest()[:12]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect CS papers with SBERT filtering")
    parser.add_argument("--target", type=int, default=TARGET,
                        help=f"Number of CS papers to collect (default: {TARGET})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"SBERT similarity threshold (default: {THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"SBERT batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    # ── Resume: check existing papers ─────────────────────────────────
    existing = {f.stem for f in PAPERS_DIR.glob("*.json") if not f.stem.startswith("_")}
    log.info(f"Existing papers in {PAPERS_DIR}: {len(existing)}")

    if len(existing) >= args.target:
        log.info(f"Already have {len(existing)} >= target {args.target}. Done.")
        return

    # ── Load SBERT ────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    import torch

    log.info(f"Loading SBERT: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    log.info(f"Device: {device}")

    # Encode CS anchors once
    anchor_embs = model.encode(
        CS_ANCHORS, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )

    # ── Stream HF dataset ─────────────────────────────────────────────
    from datasets import load_dataset

    log.info("Loading ccdv/arxiv-summarization (streaming, all splits)...")
    ds_dict = load_dataset(
        "ccdv/arxiv-summarization",
        "document",
        streaming=True,
    )

    # Chain all splits together for maximum coverage
    from itertools import chain
    ds = chain(
        ds_dict["train"],
        ds_dict["validation"],
        ds_dict["test"],
    )

    collected = len(existing)
    scanned = 0
    skipped_short = 0
    skipped_lang = 0
    skipped_dup = 0
    skipped_no_source = 0
    cat_counts = Counter()

    # Batch accumulator
    batch_raws = []
    batch_abstracts = []

    start_time = time.time()

    def flush_batch():
        """Encode accumulated batch, filter, save passing papers."""
        nonlocal collected, batch_raws, batch_abstracts

        if not batch_abstracts:
            return

        # SBERT encode batch
        embs = model.encode(
            batch_abstracts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Compute similarity to anchors
        sims = embs @ anchor_embs.T  # (batch, n_anchors)
        max_sims = sims.max(axis=1)
        best_anchors = sims.argmax(axis=1)

        for i, raw in enumerate(batch_raws):
            if max_sims[i] < args.threshold:
                continue

            paper_id = stable_id(raw)
            if paper_id in existing:
                continue

            abstract = raw.get("abstract", "").strip()
            article = raw.get("article", "").strip()
            summary = make_summary(abstract)
            anchor_idx = int(best_anchors[i])
            primary_cat = ANCHOR_CATEGORIES[anchor_idx]

            # Secondary categories: all anchors above threshold
            all_cats = [
                ANCHOR_CATEGORIES[j]
                for j in range(len(CS_ANCHORS))
                if sims[i, j] >= args.threshold and j < len(ANCHOR_CATEGORIES)
            ]
            all_cats = list(dict.fromkeys(all_cats))  # deduplicate, preserve order

            wc_source = len(article.split())
            wc_summary = len(summary.split())

            record = {
                "id": paper_id,
                "abstract": abstract,
                "source": article,
                "summary": summary,
                "domain": "engineering",
                "modality": "paper",
                "primary_category": primary_cat,
                "categories": all_cats,
                "cs_similarity": round(float(max_sims[i]), 4),
                "word_count_source": wc_source,
                "word_count_summary": wc_summary,
                "token_count_source": int(wc_source * 1.3),
                "token_count_summary": int(wc_summary * 1.3),
            }

            path = PAPERS_DIR / f"{paper_id}.json"
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                            encoding="utf-8")

            existing.add(paper_id)
            collected += 1
            cat_counts[primary_cat] += 1

            if collected >= args.target:
                break

        # Clear batch
        batch_raws = []
        batch_abstracts = []

    # ── Main streaming loop ───────────────────────────────────────────
    log.info(f"Streaming dataset — target: {args.target}, threshold: {args.threshold}")

    for raw in ds:
        if collected >= args.target:
            break

        scanned += 1

        abstract = (raw.get("abstract") or "").strip()
        article = (raw.get("article") or "").strip()

        # Basic quality gates (fast, before SBERT)
        if len(abstract.split()) < 80 or len(abstract.split()) > 400:
            skipped_short += 1
            continue

        if len(article.split()) < 200:
            skipped_no_source += 1
            continue

        if not is_english(abstract):
            skipped_lang += 1
            continue

        # Check for duplicates
        pid = stable_id(raw)
        if pid in existing:
            skipped_dup += 1
            continue

        # Add to batch
        batch_raws.append(raw)
        batch_abstracts.append(abstract)

        # Flush when batch is full
        if len(batch_abstracts) >= args.batch_size:
            flush_batch()

            # Progress logging every 5 batches
            elapsed = time.time() - start_time
            rate = collected / max(elapsed, 1)
            remaining = (args.target - collected) / max(rate, 0.01)
            log.info(
                f"Progress: {collected}/{args.target} collected | "
                f"{scanned:,} scanned | "
                f"{rate:.1f} papers/sec | "
                f"ETA: {remaining/60:.0f} min"
            )

    # Final flush
    if batch_abstracts and collected < args.target:
        flush_batch()

    elapsed = time.time() - start_time

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"CS PAPER COLLECTION REPORT")
    print(f"{'='*65}")
    print(f"Scanned:          {scanned:,}")
    print(f"Collected:        {collected}")
    print(f"Target:           {args.target}")
    print(f"Threshold:        {args.threshold}")
    print(f"Time:             {elapsed/60:.1f} min")
    print()
    print(f"SKIP REASONS")
    print(f"  Short/long abstract: {skipped_short:,}")
    print(f"  No article body:     {skipped_no_source:,}")
    print(f"  Non-English:         {skipped_lang:,}")
    print(f"  Duplicate:           {skipped_dup:,}")
    print(f"  Below threshold:     {scanned - collected - skipped_short - skipped_no_source - skipped_lang - skipped_dup:,}")
    print()
    print(f"CATEGORY DISTRIBUTION")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat}: {cat_counts[cat]}")
    print()

    if collected >= args.target:
        print(f"STATUS: OK - Target reached ({collected}/{args.target})")
    else:
        print(f"STATUS: WARNING - Only {collected}/{args.target} -- dataset exhausted")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()