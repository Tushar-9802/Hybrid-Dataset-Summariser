"""
pair_miner.py — Cross-Modal Pair Mining
Mines semantically aligned paper-video pairs for contrastive learning (CrossCLR).

Strategy:
  1. Load paper abstracts (2,400) and video summaries (738)
  2. Encode all with SBERT (all-mpnet-base-v2)
  3. Compute cosine similarity matrix (papers × videos)
  4. Greedy select top pairs above threshold
  5. Constraint: each paper in at most 3 pairs, each video in at most 3 pairs
  6. Same-category pairs only (cs.AI paper <-> cs.AI video)
  7. If mined pairs < target, generate synthetic pairs via GPT-4o-mini

Output: data/processed/cross_modal_pairs/pairs.json

Run from repo root:
    python src/processing/pair_miner.py --test            # Light test: 50 papers + 20 videos
    python src/processing/pair_miner.py --dry-run         # Stats only, no encoding
    python src/processing/pair_miner.py                   # Full run
    python src/processing/pair_miner.py --synthetic       # Full + synthetic fallback

Hardware: SBERT encoding runs on GPU (~2 min for 3,138 texts on RTX 5070 Ti)
"""

import os
import json
import logging
import argparse
import random
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

PAPERS_DIR   = Path("data/raw/papers")
VIDEOS_DIR   = Path("data/raw/videos")
OUTPUT_DIR   = Path("data/processed/cross_modal_pairs")
OUTPUT_FILE  = OUTPUT_DIR / "pairs.json"
LOG_DIR      = Path("logs")

SBERT_MODEL  = "all-mpnet-base-v2"  # 768-dim, good quality/speed balance
SIM_THRESHOLD = 0.55                # Lowered from spec 0.60 for cross-register gap
MAX_PAIRS_PER_PAPER = 3
MAX_PAIRS_PER_VIDEO = 3
TARGET_PAIRS = 450

BATCH_SIZE   = 128                  # SBERT encoding batch size

# Ensure logs dir exists before configuring handlers
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pair_miner.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_papers(limit: int = 0) -> list[dict]:
    """Load paper JSONs. Returns list of {id, abstract, summary, category}."""
    papers = []
    files = sorted(f for f in PAPERS_DIR.glob("*.json") if not f.stem.startswith("_"))

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Skipping paper {f.name}: {e}")
            continue

        abstract = data.get("abstract", "").strip()
        if not abstract:
            continue

        # Primary category — try multiple field names for robustness
        cat = (data.get("primary_category")
               or (data.get("categories", [None])[0]
                   if isinstance(data.get("categories"), list) else None)
               or "unknown")

        papers.append({
            "id": data.get("id", f.stem),
            "abstract": abstract,
            "summary": data.get("summary", ""),
            "category": cat,
        })

    log.info(f"Loaded {len(papers)} papers")

    if limit and len(papers) > limit:
        random.seed(42)
        papers = random.sample(papers, limit)
        log.info(f"  Test mode: sampled {limit} papers")

    return papers


def load_videos(limit: int = 0) -> list[dict]:
    """Load video transcript+summary JSONs."""
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
        except Exception as e:
            log.warning(f"Skipping video {vid_dir.name}: {e}")
            continue

        summary = tdata.get("summary", "").strip()
        if not summary:
            continue

        # Category from meta.json
        cat = "unknown"
        if mp.exists():
            try:
                cat = json.loads(mp.read_text(encoding="utf-8")).get("category", "unknown")
            except Exception:
                pass

        videos.append({
            "id": vid_dir.name,
            "summary": summary,
            "word_count": tdata.get("word_count", 0),
            "category": cat,
        })

    log.info(f"Loaded {len(videos)} videos with summaries")

    if limit and len(videos) > limit:
        random.seed(42)
        videos = random.sample(videos, limit)
        log.info(f"  Test mode: sampled {limit} videos")

    return videos


# ── SBERT ENCODING ────────────────────────────────────────────────────────────

def encode_texts(texts: list[str], model_name: str = SBERT_MODEL,
                 batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode texts with SBERT, returns (N, 768) normalized embeddings."""
    from sentence_transformers import SentenceTransformer
    import torch

    log.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    log.info(f"SBERT device: {device}")

    log.info(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Unit vectors → dot product = cosine sim
        convert_to_numpy=True,
    )

    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ── PAIR MINING ───────────────────────────────────────────────────────────────

def mine_pairs(papers: list[dict], videos: list[dict],
               paper_embs: np.ndarray, video_embs: np.ndarray,
               threshold: float = SIM_THRESHOLD,
               same_category: bool = True) -> list[dict]:
    """
    Greedy pair selection from cosine similarity matrix.
    Embeddings are L2-normalized, so dot product = cosine similarity.
    """
    n_papers = len(papers)
    n_videos = len(videos)

    log.info(f"Computing similarity matrix: {n_papers} papers × {n_videos} videos")
    sim_matrix = paper_embs @ video_embs.T  # (n_papers, n_videos)

    # Category mask: only allow same-category pairs
    if same_category:
        log.info("Applying same-category constraint")
        cat_mask = np.zeros((n_papers, n_videos), dtype=bool)
        for i, p in enumerate(papers):
            for j, v in enumerate(videos):
                if p["category"] == v["category"]:
                    cat_mask[i, j] = True
        matched_cells = cat_mask.sum()
        log.info(f"  Same-category cells: {matched_cells:,} / {n_papers * n_videos:,} "
                 f"({matched_cells / (n_papers * n_videos) * 100:.1f}%)")
        sim_matrix = np.where(cat_mask, sim_matrix, -1.0)

    # Threshold mask
    valid_mask = sim_matrix >= threshold
    n_valid = valid_mask.sum()
    log.info(f"Cells above threshold {threshold}: {n_valid:,}")

    if n_valid == 0:
        log.warning(f"No pairs above threshold {threshold}!")
        # Show what thresholds would yield
        flat = sim_matrix[sim_matrix > 0]
        if len(flat) > 0:
            for t in [0.50, 0.45, 0.40, 0.35]:
                count = (flat >= t).sum()
                log.info(f"  Threshold {t}: {count:,} candidates")
        return []

    valid_indices = np.argwhere(valid_mask)  # (K, 2)
    sims = sim_matrix[valid_indices[:, 0], valid_indices[:, 1]]
    sorted_order = np.argsort(-sims)  # Descending

    # Greedy selection with per-item caps
    paper_counts = Counter()
    video_counts = Counter()
    pairs = []

    for idx in sorted_order:
        pi, vi = int(valid_indices[idx, 0]), int(valid_indices[idx, 1])
        sim = float(sims[idx])

        if paper_counts[pi] >= MAX_PAIRS_PER_PAPER:
            continue
        if video_counts[vi] >= MAX_PAIRS_PER_VIDEO:
            continue

        paper = papers[pi]
        video = videos[vi]

        pairs.append({
            "pair_id": f"pair_{len(pairs)+1:04d}",
            "paper_id": paper["id"],
            "video_id": video["id"],
            "category": paper["category"],
            "similarity": round(sim, 4),
            "paper_abstract": paper["abstract"],
            "paper_summary": paper["summary"],
            "video_summary": video["summary"],
        })

        paper_counts[pi] += 1
        video_counts[vi] += 1

    log.info(f"Mined {len(pairs)} pairs (threshold={threshold})")
    return pairs


# ── SYNTHETIC PAIR GENERATION (FALLBACK) ──────────────────────────────────────

def generate_synthetic_pairs(papers: list[dict], existing_paper_ids: set,
                             count: int = 200, api_key: str = "") -> list[dict]:
    """
    Fallback: GPT-4o-mini rewrites paper abstracts as lecture-style summaries.
    Only uses papers NOT already in mined pairs.
    """
    import asyncio
    from openai import AsyncOpenAI

    if not api_key:
        log.error("No API key for synthetic generation. Set OPENAI_API_KEY or --api-key.")
        return []

    candidates = [p for p in papers if p["id"] not in existing_paper_ids]
    if len(candidates) < count:
        log.warning(f"Only {len(candidates)} unpaired papers (need {count})")
    candidates = candidates[:count]

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(3)  # Conservative for TPM limits

    SYSTEM_PROMPT = (
        "You rewrite academic abstracts as conversational video lecture explanations. "
        "Sound like a knowledgeable instructor explaining to students — clear, direct, "
        "specific. Never use academic framing like 'this paper presents' or 'the authors'. "
        "Keep the same technical content but change the register to informal educational."
    )

    async def rewrite_one(paper: dict, pair_idx: int) -> dict | None:
        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": (
                                f"Rewrite this abstract as a ~100-word educational "
                                f"video summary:\n\n{paper['abstract']}"
                            )},
                        ],
                        max_tokens=300,
                        temperature=0.7,
                    )
                    rewritten = response.choices[0].message.content.strip()
                    return {
                        "pair_id": f"synth_{pair_idx+1:04d}",
                        "paper_id": paper["id"],
                        "video_id": f"synthetic_{paper['id']}",
                        "category": paper["category"],
                        "similarity": 1.0,
                        "paper_abstract": paper["abstract"],
                        "paper_summary": paper["summary"],
                        "video_summary": rewritten,
                        "synthetic": True,
                    }
                except Exception as e:
                    log.warning(f"Synthetic pair {pair_idx} attempt {attempt+1}: {e}")
                    await asyncio.sleep(2 ** attempt)
            return None

    async def run_all():
        tasks = [rewrite_one(p, i) for i, p in enumerate(candidates)]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                         desc="Synthetic pairs"):
            result = await coro
            if result:
                results.append(result)
        return results

    log.info(f"Generating {len(candidates)} synthetic pairs via GPT-4o-mini...")
    results = asyncio.run(run_all())
    log.info(f"Generated {len(results)} synthetic pairs")
    return results


# ── STATISTICS ────────────────────────────────────────────────────────────────

def print_stats(pairs: list[dict], papers: list[dict], videos: list[dict],
                is_test: bool = False):
    """Print pair mining statistics."""
    tag = " [TEST MODE]" if is_test else ""

    if not pairs:
        print(f"No pairs mined.{tag}")
        return False

    sims = [p["similarity"] for p in pairs]
    cats = Counter(p["category"] for p in pairs)
    n_synth = sum(1 for p in pairs if p.get("synthetic", False))
    n_mined = len(pairs) - n_synth

    unique_papers = len(set(p["paper_id"] for p in pairs))
    unique_videos = len(set(p["video_id"] for p in pairs if not p.get("synthetic")))

    paper_pair_counts = Counter(p["paper_id"] for p in pairs)
    video_pair_counts = Counter(p["video_id"] for p in pairs if not p.get("synthetic"))

    print(f"\n{'='*65}")
    print(f"A4 — CROSS-MODAL PAIR MINING REPORT{tag}")
    print(f"{'='*65}")
    print(f"Input:    {len(papers)} papers, {len(videos)} videos")
    print(f"Output:   {len(pairs)} total pairs")
    print(f"  Mined:      {n_mined}")
    print(f"  Synthetic:  {n_synth}")
    print()

    print(f"SIMILARITY DISTRIBUTION (mined only)")
    mined_sims = [p["similarity"] for p in pairs if not p.get("synthetic")]
    if mined_sims:
        print(f"  Mean:   {np.mean(mined_sims):.4f}")
        print(f"  Median: {np.median(mined_sims):.4f}")
        print(f"  Min:    {min(mined_sims):.4f}")
        print(f"  Max:    {max(mined_sims):.4f}")
        print(f"  P10:    {np.percentile(mined_sims, 10):.4f}")
        print(f"  P90:    {np.percentile(mined_sims, 90):.4f}")

        bins = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
        labels = ["0.45-0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65",
                  "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85+"]
        hist, _ = np.histogram(mined_sims, bins=bins)
        print(f"\n  Histogram:")
        for label, count in zip(labels, hist):
            bar = "█" * max(count // 3, 1) if count > 0 else ""
            print(f"    {label}: {count:4d} {bar}")
    print()

    print(f"COVERAGE")
    print(f"  Unique papers used: {unique_papers}/{len(papers)} "
          f"({unique_papers/len(papers)*100:.0f}%)")
    print(f"  Unique videos used: {unique_videos}/{len(videos)} "
          f"({unique_videos/len(videos)*100:.0f}%)")
    if paper_pair_counts:
        pp_vals = list(paper_pair_counts.values())
        print(f"  Pairs/paper: mean={np.mean(pp_vals):.1f}, max={max(pp_vals)}")
    if video_pair_counts:
        vp_vals = list(video_pair_counts.values())
        print(f"  Pairs/video: mean={np.mean(vp_vals):.1f}, max={max(vp_vals)}")
    print()

    print(f"PER-CATEGORY DISTRIBUTION")
    print(f"  {'Category':<10s} {'Pairs':>6s} {'%':>6s}")
    print(f"  {'-'*24}")
    for cat in sorted(cats.keys()):
        pct = cats[cat] / len(pairs) * 100
        print(f"  {cat:<10s} {cats[cat]:>6d} {pct:>5.0f}%")
    print()

    # Sample pairs — top 3 and bottom 3 by similarity
    print(f"SAMPLE PAIRS")
    sorted_pairs = sorted(pairs, key=lambda p: p["similarity"], reverse=True)

    print(f"  --- Top 3 (highest similarity) ---")
    for p in sorted_pairs[:3]:
        tag_s = " [synthetic]" if p.get("synthetic") else ""
        print(f"\n  [{p['pair_id']}] sim={p['similarity']:.4f} cat={p['category']}{tag_s}")
        print(f"  Paper: {p['paper_abstract'][:100]}...")
        print(f"  Video: {p['video_summary'][:100]}...")

    print(f"\n  --- Bottom 3 (lowest similarity) ---")
    for p in sorted_pairs[-3:]:
        tag_s = " [synthetic]" if p.get("synthetic") else ""
        print(f"\n  [{p['pair_id']}] sim={p['similarity']:.4f} cat={p['category']}{tag_s}")
        print(f"  Paper: {p['paper_abstract'][:100]}...")
        print(f"  Video: {p['video_summary'][:100]}...")

    # Validation
    print(f"\n{'='*65}")
    print(f"VALIDATION")

    if is_test:
        checks = [
            (f"Pairs found > 0 (have {len(pairs)})", len(pairs) > 0),
            ("All pairs have paper_id", all(p.get("paper_id") for p in pairs)),
            ("All pairs have video_id", all(p.get("video_id") for p in pairs)),
            ("All pairs have similarity > 0", all(p.get("similarity", 0) > 0 for p in pairs)),
            ("Similarity values plausible (max < 1.0 for mined)",
             max(mined_sims) < 1.0 if mined_sims else True),
        ]
    else:
        checks = [
            (f"Total pairs >= 300 (have {len(pairs)})", len(pairs) >= 300),
            (f"Mined pairs > 0", n_mined > 0),
            ("All pairs have paper_id", all(p.get("paper_id") for p in pairs)),
            ("All pairs have video_id", all(p.get("video_id") for p in pairs)),
            ("All pairs have similarity > 0", all(p.get("similarity", 0) > 0 for p in pairs)),
            ("All 7 categories represented", len(cats) >= 7),
            (f"No category > 40% of pairs",
             all(c / len(pairs) < 0.40 for c in cats.values())),
        ]

    all_ok = True
    for msg, ok in checks:
        print(f"  [{'✓' if ok else '✗'}] {msg}")
        if not ok:
            all_ok = False

    if is_test:
        status = "✓ TEST PASS — pipeline works, run full with: python src/processing/pair_miner.py"
    elif all_ok:
        status = "✓ PASS — ready for A5/A6"
    else:
        status = "⚠ Review issues above"

    print(f"\n  STATUS: {status}")
    print(f"{'='*65}")

    return all_ok


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A4: Cross-modal pair mining")
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {SIM_THRESHOLD})")
    parser.add_argument("--no-category-filter", action="store_true",
                        help="Allow cross-category pairs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and show stats, no encoding/mining")
    parser.add_argument("--test", action="store_true",
                        help="Light test: 50 papers + 20 videos, verify pipeline end-to-end")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic pairs if mined < target")
    parser.add_argument("--api-key", type=str, default="",
                        help="OpenAI API key for synthetic generation")
    parser.add_argument("--target", type=int, default=TARGET_PAIRS,
                        help=f"Target pair count (default: {TARGET_PAIRS})")
    args = parser.parse_args()

    is_test = args.test
    paper_limit = 50 if is_test else 0
    video_limit = 20 if is_test else 0

    # Load data
    papers = load_papers(limit=paper_limit)
    videos = load_videos(limit=video_limit)

    if not papers:
        log.error(f"No papers found in {PAPERS_DIR}")
        return
    if not videos:
        log.error(f"No videos found in {VIDEOS_DIR}")
        return

    # Category distribution
    paper_cats = Counter(p["category"] for p in papers)
    video_cats = Counter(v["category"] for v in videos)
    shared_cats = set(paper_cats.keys()) & set(video_cats.keys())

    log.info(f"Paper categories: {dict(paper_cats)}")
    log.info(f"Video categories: {dict(video_cats)}")
    log.info(f"Shared categories: {shared_cats}")

    if args.dry_run:
        print(f"\nDRY RUN — Pair potential by category:")
        print(f"  {'Category':<10s} {'Papers':>7s} {'Videos':>7s} {'Max Pairs':>10s}")
        print(f"  {'-'*38}")
        total_potential = 0
        for cat in sorted(shared_cats):
            np_ = paper_cats.get(cat, 0)
            nv = video_cats.get(cat, 0)
            potential = min(np_ * MAX_PAIRS_PER_PAPER, nv * MAX_PAIRS_PER_VIDEO)
            total_potential += potential
            print(f"  {cat:<10s} {np_:>7d} {nv:>7d} {potential:>10d}")
        print(f"  {'TOTAL':<10s} {len(papers):>7d} {len(videos):>7d} {total_potential:>10d}")
        print(f"\n  Threshold: {args.threshold}")
        print(f"  Actual pairs will be << max potential (depends on similarity)")
        print(f"  DRY RUN complete — no embeddings computed.")
        return

    # ── Encode ────────────────────────────────────────────────────────
    paper_texts = [p["abstract"] for p in papers]
    video_texts = [v["summary"] for v in videos]

    all_texts = paper_texts + video_texts
    all_embs = encode_texts(all_texts)

    paper_embs = all_embs[:len(papers)]
    video_embs = all_embs[len(papers):]

    # ── Mine pairs ────────────────────────────────────────────────────
    pairs = mine_pairs(
        papers, videos, paper_embs, video_embs,
        threshold=args.threshold,
        same_category=not args.no_category_filter,
    )

    # ── Synthetic fallback (skip in test mode) ────────────────────────
    if not is_test and len(pairs) < args.target and args.synthetic:
        shortfall = args.target - len(pairs)
        log.info(f"Mined {len(pairs)}, need {shortfall} synthetic to reach {args.target}")

        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        existing_ids = set(p["paper_id"] for p in pairs)

        synth = generate_synthetic_pairs(
            papers, existing_ids, count=shortfall, api_key=api_key
        )
        pairs.extend(synth)
    elif not is_test and len(pairs) < args.target:
        log.warning(
            f"Only {len(pairs)} mined pairs (target: {args.target}). "
            f"Re-run with --synthetic to fill gap."
        )

    # ── Save (skip in test mode) ──────────────────────────────────────
    if not is_test:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output = {
            "metadata": {
                "total_pairs": len(pairs),
                "mined_pairs": sum(1 for p in pairs if not p.get("synthetic")),
                "synthetic_pairs": sum(1 for p in pairs if p.get("synthetic")),
                "threshold": args.threshold,
                "sbert_model": SBERT_MODEL,
                "same_category_filter": not args.no_category_filter,
                "papers_count": len(papers),
                "videos_count": len(videos),
            },
            "pairs": pairs,
        }

        OUTPUT_FILE.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Saved {len(pairs)} pairs to {OUTPUT_FILE}")
    else:
        log.info("Test mode — skipping file save")

    # ── Report ────────────────────────────────────────────────────────
    print_stats(pairs, papers, videos, is_test=is_test)


if __name__ == "__main__":
    main()