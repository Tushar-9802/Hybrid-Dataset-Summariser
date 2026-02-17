"""
quality_filter.py — Dataset Quality Gates
Final validation + filtering before HDF5 packaging.

Loads all papers, videos, and pairs. Applies token-level and content-level
quality gates. Outputs a clean manifest with split assignments.

Run from repo root:
    python src/processing/quality_filter.py              # Full run
    python src/processing/quality_filter.py --stats-only # Report without saving

Output:
    data/processed/manifest.json   — clean sample list with split assignments
"""

import json
import logging
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

PAPERS_DIR   = Path("data/raw/papers")
VIDEOS_DIR   = Path("data/raw/videos")
PAIRS_FILE   = Path("data/processed/cross_modal_pairs/pairs.json")
OUTPUT_FILE  = Path("data/processed/manifest.json")
LOG_DIR      = Path("logs")

# Token limits (Mistral tokenizer)
SOURCE_MIN_TOKENS  = 64
SOURCE_MAX_TOKENS  = 2048
LABEL_MIN_TOKENS   = 15
LABEL_MAX_TOKENS   = 256

# Content limits
PAPER_SOURCE_MIN_WORDS   = 200   # Article body should be substantial
PAPER_ABSTRACT_MIN_WORDS = 80    # Abstract used as label
VIDEO_SUMMARY_MIN_WORDS  = 30

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

SEED = 42

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "quality_filter.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


# ── TOKENIZER ─────────────────────────────────────────────────────────────────

_tokenizer = None

def get_tokenizer():
    """Lazy-load Mistral tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        log.info("Loading Mistral tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            use_fast=True,
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        log.info(f"Tokenizer loaded. Vocab size: {_tokenizer.vocab_size}")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens using Mistral tokenizer."""
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_papers() -> list[dict]:
    papers = []
    for f in sorted(PAPERS_DIR.glob("*.json")):
        if f.stem.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        cat = (data.get("primary_category")
               or (data.get("categories", [None])[0]
                   if isinstance(data.get("categories"), list) else None)
               or "unknown")

        source = data.get("source", "").strip()    # Article body = input
        abstract = data.get("abstract", "").strip()  # Full abstract = label

        # Skip papers without article body (source field)
        if not source or not abstract:
            continue

        papers.append({
            "id": data.get("id", f.stem),
            "source": source,
            "summary": abstract,
            "modality": "paper",
            "category": cat,
        })
    return papers


def load_videos() -> list[dict]:
    videos = []
    for vid_dir in sorted(VIDEOS_DIR.iterdir()):
        if not vid_dir.is_dir():
            continue

        tp = vid_dir / "transcript.json"
        mp = vid_dir / "meta.json"
        if not tp.exists():
            continue

        try:
            tdata = json.loads(tp.read_text(encoding="utf-8"))
        except Exception:
            continue

        summary = tdata.get("summary", "").strip()
        transcript = tdata.get("text", "").strip()
        if not summary or not transcript:
            continue

        cat = "unknown"
        if mp.exists():
            try:
                cat = json.loads(mp.read_text(encoding="utf-8")).get("category", "unknown")
            except Exception:
                pass

        videos.append({
            "id": vid_dir.name,
            "source": transcript,     # Transcript = source
            "summary": summary,       # GPT-4o-mini summary = label
            "modality": "video",
            "category": cat,
        })
    return videos


def load_pairs() -> list[dict]:
    if not PAIRS_FILE.exists():
        log.warning(f"Pairs file not found: {PAIRS_FILE}")
        return []

    data = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    return data.get("pairs", [])


# ── QUALITY GATES ─────────────────────────────────────────────────────────────

def filter_sample(sample: dict) -> tuple[bool, list[str]]:
    """
    Apply quality gates to a single sample.
    Returns (passed, [rejection_reasons]).
    """
    issues = []

    source = sample.get("source", "")
    summary = sample.get("summary", "")
    modality = sample.get("modality", "")

    # Empty field check
    if not source:
        issues.append("empty_source")
    if not summary:
        issues.append("empty_summary")
    if issues:
        return False, issues

    # UTF-8 cleanliness (check for replacement chars)
    if "\ufffd" in source or "\ufffd" in summary:
        issues.append("utf8_dirty")

    # Word count checks
    source_wc = len(source.split())
    summary_wc = len(summary.split())

    if modality == "paper":
        if source_wc < PAPER_SOURCE_MIN_WORDS:
            issues.append(f"paper_source_short({source_wc}w)")
        if summary_wc < PAPER_ABSTRACT_MIN_WORDS:
            issues.append(f"paper_abstract_short({summary_wc}w)")
    elif modality == "video":
        if summary_wc < VIDEO_SUMMARY_MIN_WORDS:
            issues.append(f"video_summary_short({summary_wc}w)")

    # Token count checks (expensive — only if no issues yet)
    if not issues:
        src_tokens = count_tokens(source)
        sum_tokens = count_tokens(summary)

        sample["source_tokens"] = src_tokens
        sample["summary_tokens"] = sum_tokens

        if src_tokens < SOURCE_MIN_TOKENS:
            issues.append(f"source_too_few_tokens({src_tokens})")
        if src_tokens > SOURCE_MAX_TOKENS:
            # Not a rejection — will be truncated in Dataset Builder. Just flag.
            sample["source_truncated"] = True
        if sum_tokens < LABEL_MIN_TOKENS:
            issues.append(f"summary_too_few_tokens({sum_tokens})")
        if sum_tokens > LABEL_MAX_TOKENS:
            # Not a rejection — will be truncated in Dataset Builder. Just flag.
            sample["summary_truncated"] = True

    return len(issues) == 0, issues


# ── SPLIT ASSIGNMENT ──────────────────────────────────────────────────────────

def assign_splits(samples: list[dict], seed: int = SEED) -> list[dict]:
    """
    Stratified split by modality. Each modality independently split 80/10/10.
    """
    rng = random.Random(seed)

    by_modality = defaultdict(list)
    for s in samples:
        by_modality[s["modality"]].append(s)

    result = []
    for modality, items in by_modality.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        for i, s in enumerate(items):
            if i < n_train:
                s["split"] = "train"
            elif i < n_train + n_val:
                s["split"] = "val"
            else:
                s["split"] = "test"
            result.append(s)

    return result


def assign_pair_splits(pairs: list[dict], seed: int = SEED) -> list[dict]:
    """Split pairs 80/10/10."""
    rng = random.Random(seed + 1)  # Different seed to avoid correlation
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    for i, p in enumerate(pairs):
        if i < n_train:
            p["split"] = "train"
        elif i < n_train + n_val:
            p["split"] = "val"
        else:
            p["split"] = "test"

    return pairs


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset quality filtering")
    parser.add_argument("--stats-only", action="store_true",
                        help="Report stats without saving manifest")
    args = parser.parse_args()

    # Load
    log.info("Loading data...")
    papers = load_papers()
    videos = load_videos()
    pairs = load_pairs()
    log.info(f"Loaded: {len(papers)} papers, {len(videos)} videos, {len(pairs)} pairs")

    # ── Filter papers and videos ──────────────────────────────────────
    all_samples = papers + videos
    passed = []
    rejected = []
    rejection_reasons = Counter()

    log.info("Running quality gates (tokenizing — may take a few minutes)...")
    from tqdm import tqdm
    for sample in tqdm(all_samples, desc="Quality filtering"):
        ok, issues = filter_sample(sample)
        if ok:
            passed.append(sample)
        else:
            rejected.append({"id": sample["id"], "modality": sample["modality"],
                             "issues": issues})
            for issue in issues:
                rejection_reasons[issue] += 1

    # ── Validate pairs (check that both sides still exist) ────────────
    passed_ids = set(s["id"] for s in passed)
    valid_pairs = []
    orphan_pairs = 0
    for p in pairs:
        if p["paper_id"] in passed_ids and p["video_id"] in passed_ids:
            valid_pairs.append(p)
        else:
            orphan_pairs += 1

    # ── Assign splits ─────────────────────────────────────────────────
    passed = assign_splits(passed)
    valid_pairs = assign_pair_splits(valid_pairs)

    # ── Statistics ────────────────────────────────────────────────────
    papers_passed = [s for s in passed if s["modality"] == "paper"]
    videos_passed = [s for s in passed if s["modality"] == "video"]

    split_counts = defaultdict(lambda: defaultdict(int))
    for s in passed:
        split_counts[s["modality"]][s["split"]] += 1

    pair_split_counts = Counter(p["split"] for p in valid_pairs)

    # Token stats
    src_tokens = [s["source_tokens"] for s in passed if "source_tokens" in s]
    sum_tokens = [s["summary_tokens"] for s in passed if "summary_tokens" in s]
    truncated_src = sum(1 for s in passed if s.get("source_truncated"))
    truncated_sum = sum(1 for s in passed if s.get("summary_truncated"))

    paper_src_tokens = [s["source_tokens"] for s in papers_passed if "source_tokens" in s]
    paper_sum_tokens = [s["summary_tokens"] for s in papers_passed if "summary_tokens" in s]
    video_src_tokens = [s["source_tokens"] for s in videos_passed if "source_tokens" in s]
    video_sum_tokens = [s["summary_tokens"] for s in videos_passed if "summary_tokens" in s]

    # Category distribution in passed
    paper_cats = Counter(s["category"] for s in papers_passed)
    video_cats = Counter(s["category"] for s in videos_passed)

    print(f"\n{'='*65}")
    print(f"A5 — DATASET QUALITY FILTER REPORT")
    print(f"{'='*65}")

    print(f"\nINPUT / OUTPUT")
    print(f"  Papers:  {len(papers):>5d} in → {len(papers_passed):>5d} passed "
          f"({len(papers)-len(papers_passed)} rejected)")
    print(f"  Videos:  {len(videos):>5d} in → {len(videos_passed):>5d} passed "
          f"({len(videos)-len(videos_passed)} rejected)")
    print(f"  Pairs:   {len(pairs):>5d} in → {len(valid_pairs):>5d} valid "
          f"({orphan_pairs} orphaned)")
    print(f"  Total:   {len(all_samples):>5d} in → {len(passed):>5d} passed")

    if rejection_reasons:
        print(f"\nREJECTION REASONS")
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason}: {count}")
    else:
        print(f"\nREJECTION REASONS: None — all samples passed")

    print(f"\nSPLIT ASSIGNMENT")
    print(f"  {'':12s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Total':>7s}")
    print(f"  {'-'*42}")
    for mod in ["paper", "video"]:
        sc = split_counts[mod]
        total = sc["train"] + sc["val"] + sc["test"]
        print(f"  {mod:<12s} {sc['train']:>7d} {sc['val']:>7d} {sc['test']:>7d} {total:>7d}")
    print(f"  {'pairs':<12s} {pair_split_counts['train']:>7d} "
          f"{pair_split_counts['val']:>7d} {pair_split_counts['test']:>7d} "
          f"{len(valid_pairs):>7d}")

    print(f"\nTOKEN STATISTICS (Mistral tokenizer)")
    if src_tokens:
        print(f"  Source tokens (all):")
        print(f"    Mean={sum(src_tokens)//len(src_tokens)}, "
              f"Median={sorted(src_tokens)[len(src_tokens)//2]}, "
              f"Min={min(src_tokens)}, Max={max(src_tokens)}")
        print(f"    Will be truncated to 1024: {truncated_src}")
    if sum_tokens:
        print(f"  Summary tokens (all):")
        print(f"    Mean={sum(sum_tokens)//len(sum_tokens)}, "
              f"Median={sorted(sum_tokens)[len(sum_tokens)//2]}, "
              f"Min={min(sum_tokens)}, Max={max(sum_tokens)}")
        print(f"    Will be truncated to 256: {truncated_sum}")

    if paper_src_tokens:
        print(f"  Paper source:  mean={sum(paper_src_tokens)//len(paper_src_tokens)}, "
              f"range=[{min(paper_src_tokens)}, {max(paper_src_tokens)}]")
    if paper_sum_tokens:
        print(f"  Paper summary: mean={sum(paper_sum_tokens)//len(paper_sum_tokens)}, "
              f"range=[{min(paper_sum_tokens)}, {max(paper_sum_tokens)}]")
    if video_src_tokens:
        print(f"  Video source:  mean={sum(video_src_tokens)//len(video_src_tokens)}, "
              f"range=[{min(video_src_tokens)}, {max(video_src_tokens)}]")
    if video_sum_tokens:
        print(f"  Video summary: mean={sum(video_sum_tokens)//len(video_sum_tokens)}, "
              f"range=[{min(video_sum_tokens)}, {max(video_sum_tokens)}]")

    print(f"\nCATEGORY DISTRIBUTION (passed)")
    print(f"  Papers: {dict(sorted(paper_cats.items()))}")
    print(f"  Videos: {dict(sorted(video_cats.items()))}")

    # ── Validation checks ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"VALIDATION")
    checks = [
        (f"Papers passed >= 2200 (have {len(papers_passed)})",
         len(papers_passed) >= 2200),
        (f"Videos passed >= 700 (have {len(videos_passed)})",
         len(videos_passed) >= 700),
        (f"Pairs valid >= 350 (have {len(valid_pairs)})",
         len(valid_pairs) >= 350),
        ("All 7 categories in papers",
         len(paper_cats) >= 7),
        ("All 7 categories in videos",
         len(video_cats) >= 7),
        ("Train/val/test splits assigned",
         all(s.get("split") for s in passed)),
        ("Rejection rate < 10%",
         len(rejected) / max(len(all_samples), 1) < 0.10),
    ]

    all_ok = True
    for msg, ok in checks:
        print(f"  [{'✓' if ok else '✗'}] {msg}")
        if not ok:
            all_ok = False

    status = "✓ PASS — ready for HDF5 packaging" if all_ok else "⚠ Review issues"
    print(f"\n  STATUS: {status}")
    print(f"{'='*65}")

    # ── Save manifest ─────────────────────────────────────────────────
    if not args.stats_only:
        # Strip source text from manifest to keep file size reasonable
        # Dataset Builder will reload from original files using IDs
        manifest_samples = []
        for s in passed:
            manifest_samples.append({
                "id": s["id"],
                "modality": s["modality"],
                "category": s["category"],
                "split": s["split"],
                "source_tokens": s.get("source_tokens", 0),
                "summary_tokens": s.get("summary_tokens", 0),
                "source_truncated": s.get("source_truncated", False),
                "summary_truncated": s.get("summary_truncated", False),
            })

        manifest_pairs = []
        for p in valid_pairs:
            manifest_pairs.append({
                "pair_id": p["pair_id"],
                "paper_id": p["paper_id"],
                "video_id": p["video_id"],
                "category": p["category"],
                "similarity": p["similarity"],
                "split": p["split"],
                "synthetic": p.get("synthetic", False),
            })

        manifest = {
            "metadata": {
                "total_samples": len(passed),
                "papers": len(papers_passed),
                "videos": len(videos_passed),
                "pairs": len(valid_pairs),
                "rejected": len(rejected),
                "splits": {
                    "paper": dict(split_counts["paper"]),
                    "video": dict(split_counts["video"]),
                    "pairs": dict(pair_split_counts),
                },
            },
            "samples": manifest_samples,
            "pairs": manifest_pairs,
        }

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Saved manifest: {OUTPUT_FILE} ({len(passed)} samples, {len(valid_pairs)} pairs)")
    else:
        log.info("Stats-only mode — no manifest saved")


if __name__ == "__main__":
    main()