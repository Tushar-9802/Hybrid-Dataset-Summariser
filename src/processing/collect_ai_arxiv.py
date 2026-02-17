"""
collect_ai_arxiv.py — Supplement CS papers from jamescalam/ai-arxiv
Pre-filtered AI/CS dataset with full article text and real arXiv categories.

Adapts schema to match our pipeline:
    content       -> source  (article body)
    summary       -> abstract (full abstract)
    primary_category preserved (real arXiv categories)

Run from repo root:
    python src/processing/collect_ai_arxiv.py
"""

import json
import re
import logging
from pathlib import Path
from collections import Counter

PAPERS_DIR = Path("data/raw/papers")
LOG_DIR = Path("logs")

# Only keep papers in our target CS categories
TARGET_CATEGORIES = {
    "cs.AI", "cs.CL", "cs.CV", "cs.DS", "cs.LG", "cs.RO", "cs.SE",
    # Also accept closely related categories
    "cs.IR", "cs.NE", "cs.MA", "cs.HC", "cs.CR", "cs.DC",
    "cs.PL", "cs.SI", "cs.SD", "cs.AR", "cs.CY",
    "stat.ML",  # Often cross-listed with cs.LG
}

# Map related categories to our 7 core categories
CATEGORY_MAP = {
    "cs.AI": "cs.AI", "cs.CL": "cs.CL", "cs.CV": "cs.CV",
    "cs.DS": "cs.DS", "cs.LG": "cs.LG", "cs.RO": "cs.RO",
    "cs.SE": "cs.SE",
    "cs.IR": "cs.AI", "cs.NE": "cs.LG", "cs.MA": "cs.AI",
    "cs.HC": "cs.AI", "cs.CR": "cs.SE", "cs.DC": "cs.DS",
    "cs.PL": "cs.SE", "cs.SI": "cs.DS", "cs.SD": "cs.AI",
    "cs.AR": "cs.DS", "cs.CY": "cs.AI",
    "stat.ML": "cs.LG",
}

LOG_DIR.mkdir(exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "collect_ai_arxiv.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


def safe_id(arxiv_id: str) -> str:
    """Convert arXiv ID to filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id.strip())


def is_english(text: str, threshold: float = 0.90) -> bool:
    ascii_count = sum(c.isascii() for c in text)
    return (ascii_count / max(len(text), 1)) >= threshold


def make_summary(abstract: str, max_words: int = 65) -> str:
    """Legacy compatibility: truncated abstract."""
    words = abstract.split()
    candidate = " ".join(words[:max_words])
    for punct in [".", "!", "?"]:
        last = candidate.rfind(punct)
        if last > len(candidate) * 0.5:
            return candidate[:last + 1].strip()
    return candidate.strip()


def main():
    from datasets import load_dataset

    existing = {f.stem for f in PAPERS_DIR.glob("*.json") if not f.stem.startswith("_")}
    log.info(f"Existing papers: {len(existing)}")

    log.info("Loading jamescalam/ai-arxiv (streaming)...")
    ds = load_dataset("jamescalam/ai-arxiv", streaming=True, split="train")

    collected = 0
    skipped_cat = 0
    skipped_dup = 0
    skipped_short = 0
    skipped_lang = 0
    cat_counts = Counter()

    for raw in ds:
        pcat = (raw.get("primary_category") or "").strip()

        # Category filter
        if pcat not in TARGET_CATEGORIES:
            skipped_cat += 1
            continue

        mapped_cat = CATEGORY_MAP.get(pcat, "cs.AI")

        # Get fields
        arxiv_id = (raw.get("id") or "").strip()
        abstract = (raw.get("summary") or "").strip()
        content = (raw.get("content") or "").strip()

        if not arxiv_id or not abstract or not content:
            skipped_short += 1
            continue

        paper_id = safe_id(arxiv_id)

        # Dedup against existing
        if paper_id in existing:
            skipped_dup += 1
            continue

        # Quality gates
        if len(abstract.split()) < 80 or len(abstract.split()) > 400:
            skipped_short += 1
            continue

        if len(content.split()) < 200:
            skipped_short += 1
            continue

        if not is_english(abstract):
            skipped_lang += 1
            continue

        # All categories from the record
        all_cats_raw = raw.get("categories") or []
        if isinstance(all_cats_raw, str):
            all_cats_raw = [c.strip() for c in all_cats_raw.split(",")]
        all_cats = [CATEGORY_MAP.get(c.strip(), mapped_cat) for c in all_cats_raw
                    if c.strip() in TARGET_CATEGORIES]
        all_cats = list(dict.fromkeys([mapped_cat] + all_cats))

        summary = make_summary(abstract)
        wc_source = len(content.split())
        wc_summary = len(summary.split())

        record = {
            "id": paper_id,
            "abstract": abstract,
            "source": content,
            "summary": summary,
            "domain": "engineering",
            "modality": "paper",
            "primary_category": mapped_cat,
            "categories": all_cats,
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
        cat_counts[mapped_cat] += 1

    # Report
    print(f"\n{'='*65}")
    print(f"AI-ARXIV COLLECTION REPORT")
    print(f"{'='*65}")
    print(f"Collected:    {collected}")
    print(f"Skipped:")
    print(f"  Wrong category: {skipped_cat}")
    print(f"  Duplicate:      {skipped_dup}")
    print(f"  Too short/long: {skipped_short}")
    print(f"  Non-English:    {skipped_lang}")
    print()
    print(f"CATEGORY DISTRIBUTION")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat}: {cat_counts[cat]}")
    print()
    print(f"Total papers in {PAPERS_DIR}: {len(existing)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()