"""
filter_cs_papers.py — Remove non-CS papers from dataset
Uses SBERT similarity to CS-domain anchor descriptions.
Papers that score below threshold against ALL CS anchors are flagged as non-CS.

Run from repo root:
    python src/processing/filter_cs_papers.py              # Report only
    python src/processing/filter_cs_papers.py --fix        # Move non-CS to quarantine

Output:
    Moves non-CS papers to data/quarantine/papers/
    Logs which papers were removed and why
"""

import json
import logging
import argparse
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

PAPERS_DIR = Path("data/raw/papers")
QUARANTINE_DIR = Path("data/quarantine/papers")
LOG_DIR = Path("logs")
SBERT_MODEL = "all-mpnet-base-v2"
BATCH_SIZE = 128

# Threshold: max similarity to ANY CS anchor must be above this
CS_THRESHOLD = 0.35

# CS domain anchor descriptions — one per subfield, written to capture core concepts
CS_ANCHORS = [
    # cs.AI
    "artificial intelligence reasoning knowledge representation planning search heuristics "
    "intelligent agents constraint satisfaction game playing expert systems",

    # cs.CL
    "natural language processing text classification sentiment analysis machine translation "
    "language models tokenization parsing named entity recognition word embeddings",

    # cs.CV
    "computer vision image classification object detection semantic segmentation "
    "convolutional neural networks image processing feature extraction visual recognition",

    # cs.DS
    "data structures algorithms graph algorithms sorting searching dynamic programming "
    "computational complexity approximation algorithms hash tables binary trees",

    # cs.LG
    "machine learning deep learning neural networks supervised learning unsupervised learning "
    "reinforcement learning gradient descent backpropagation optimization training",

    # cs.RO
    "robotics motion planning path planning SLAM simultaneous localization and mapping "
    "robot kinematics manipulation grasping autonomous navigation mobile robots",

    # cs.SE
    "software engineering code review testing debugging refactoring continuous integration "
    "software architecture design patterns version control agile development",

    # General CS terms (broad net)
    "programming algorithm software computation database distributed systems "
    "operating systems computer networks cybersecurity cloud computing",
]

# Hard negative keywords — terms that strongly indicate non-CS papers
# Only flag papers that ALSO have low SBERT CS similarity (< 0.45)
NON_CS_KEYWORDS = [
    # Physics (unambiguous)
    r"\bquark\b", r"\bhadron\b", r"\bboson\b", r"\bfermion\b", r"\bgluon\b",
    r"\bsuperconductor", r"\bsuperconducti", r"\bmagnetic susceptib",
    r"\bstellar\b", r"\bgalax", r"\bcosmolog", r"\bastrophys",
    r"\bneutron star", r"\bblack hole", r"\bgravitational wave",
    r"\bquantum field theory\b", r"\bstring theory\b", r"\bdark matter\b",
    r"@xmath",  # LaTeX math markers common in physics arxiv papers
    r"\bspin glass\b", r"\blattice gauge\b",

    # Chemistry / Materials (unambiguous)
    r"\bcatalys", r"\bpolymer\b", r"\bnanoparticle", r"\bcrystallog",
    r"\belectrochem", r"\bphotocatalys",

    # Biology / Medicine (unambiguous — removed therapy, treatment, diagnos, patient, genome, mortality)
    r"\bclinical trial", r"\bcarcinoma\b", r"\btumor\b",
    r"\bprotein fold", r"\bDNA sequenc", r"\bamino acid",
    r"\bepidemiolog", r"\bsurgical\b",
    r"\boncolog", r"\bpatholog",

    # Math pure (unambiguous — removed Hamiltonian, Lagrangian, Ising, manifold, topology,
    # eigenvalue, Hilbert space as these appear in ML/optimization/quantum computing)
    r"\bhomomorphism", r"\bisomorphism\b",
    r"\bRiemannian\b", r"\bBanach space\b",

    # Economics / Finance (unambiguous — removed GDP as it appears in NLP/data science)
    r"\bmonetary policy\b", r"\bfiscal\b",
    r"\binflation rate\b", r"\bcentral bank\b",
]

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "filter_cs_papers.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


def load_papers() -> list[dict]:
    papers = []
    for f in sorted(PAPERS_DIR.glob("*.json")):
        if f.stem.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        papers.append({
            "id": data.get("id", f.stem),
            "abstract": data.get("abstract", "").strip(),
            "category": data.get("primary_category", "unknown"),
            "path": f,
        })
    return papers


def check_hard_negatives(abstract: str) -> list[str]:
    """Check for hard negative keywords that strongly indicate non-CS."""
    import re
    found = []
    for pattern in NON_CS_KEYWORDS:
        if re.search(pattern, abstract, re.IGNORECASE):
            found.append(pattern)
    return found


def main():
    parser = argparse.ArgumentParser(description="Filter non-CS papers")
    parser.add_argument("--fix", action="store_true",
                        help="Move non-CS papers to quarantine")
    parser.add_argument("--threshold", type=float, default=CS_THRESHOLD,
                        help=f"CS similarity threshold (default: {CS_THRESHOLD})")
    args = parser.parse_args()

    papers = load_papers()
    log.info(f"Loaded {len(papers)} papers")

    if not papers:
        print("No papers found.")
        return

    # ── SBERT encoding ────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    import torch

    log.info(f"Loading SBERT: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    log.info(f"Device: {device}")

    # Encode anchors
    anchor_embs = model.encode(
        CS_ANCHORS, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )

    # Encode paper abstracts
    abstracts = [p["abstract"] for p in papers]
    log.info(f"Encoding {len(abstracts)} paper abstracts...")
    paper_embs = model.encode(
        abstracts, batch_size=BATCH_SIZE, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )

    # Compute max similarity to any CS anchor
    sim_matrix = paper_embs @ anchor_embs.T  # (n_papers, n_anchors)
    max_sims = sim_matrix.max(axis=1)        # (n_papers,)
    best_anchors = sim_matrix.argmax(axis=1)

    # ── Hard negative check ───────────────────────────────────────────
    flagged_sbert = []
    flagged_keyword = []
    clean = []

    for i, paper in enumerate(papers):
        sim = float(max_sims[i])
        paper["cs_similarity"] = sim
        paper["best_anchor"] = int(best_anchors[i])

        hard_neg = check_hard_negatives(paper["abstract"])
        paper["hard_negatives"] = hard_neg

        # Hard negatives only flag if SBERT also shows weak CS relevance.
        # Papers with keywords like "tumor" but high CS similarity (>0.45)
        # are CS papers applied to other domains — keep them.
        if hard_neg and sim < 0.45:
            flagged_keyword.append(paper)
        elif not hard_neg and sim < args.threshold:
            flagged_sbert.append(paper)
        elif hard_neg and sim >= 0.45:
            clean.append(paper)  # CS paper applied to non-CS domain
        else:
            clean.append(paper)

    total_flagged = len(flagged_keyword) + len(flagged_sbert)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"CS PAPER FILTER REPORT")
    print(f"{'='*65}")
    print(f"Total papers:         {len(papers)}")
    print(f"Clean (CS-relevant):  {len(clean)}")
    print(f"Flagged (non-CS):     {total_flagged}")
    print(f"  By keyword:         {len(flagged_keyword)}")
    print(f"  By SBERT (<{args.threshold}):   {len(flagged_sbert)}")
    print()

    # Similarity distribution
    print(f"CS SIMILARITY DISTRIBUTION")
    print(f"  Mean:   {np.mean(max_sims):.4f}")
    print(f"  Median: {np.median(max_sims):.4f}")
    print(f"  Min:    {np.min(max_sims):.4f}")
    print(f"  Max:    {np.max(max_sims):.4f}")
    print(f"  P5:     {np.percentile(max_sims, 5):.4f}")
    print(f"  P25:    {np.percentile(max_sims, 25):.4f}")

    bins = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.01]
    labels = ["<0.20", "0.20-0.25", "0.25-0.30", "0.30-0.35",
              "0.35-0.40", "0.40-0.45", "0.45-0.50", "0.50-0.60", "0.60+"]
    hist, _ = np.histogram(max_sims, bins=bins)
    print(f"\n  Histogram:")
    for label, count in zip(labels, hist):
        marker = " ← threshold" if label == f"{args.threshold:.2f}-{args.threshold+0.05:.2f}"[:9] else ""
        bar = "█" * (count // 10) if count > 0 else ""
        print(f"    {label}: {count:5d} {bar}{marker}")
    print()

    # Category breakdown of flagged papers
    flagged_cats = Counter(p["category"] for p in flagged_keyword + flagged_sbert)
    clean_cats = Counter(p["category"] for p in clean)
    print(f"CATEGORY IMPACT")
    print(f"  {'Category':<10s} {'Before':>7s} {'After':>7s} {'Removed':>8s}")
    print(f"  {'-'*37}")
    all_cats = sorted(set(list(flagged_cats.keys()) + list(clean_cats.keys())))
    for cat in all_cats:
        before = clean_cats.get(cat, 0) + flagged_cats.get(cat, 0)
        after = clean_cats.get(cat, 0)
        removed = flagged_cats.get(cat, 0)
        print(f"  {cat:<10s} {before:>7d} {after:>7d} {removed:>8d}")
    print()

    # Sample flagged papers
    print(f"SAMPLE FLAGGED PAPERS (keyword)")
    for p in flagged_keyword[:5]:
        print(f"  [{p['id']}] cat={p['category']} sim={p['cs_similarity']:.3f}")
        print(f"    Keywords: {p['hard_negatives'][:3]}")
        print(f"    Abstract: {p['abstract'][:120]}...")
        print()

    print(f"SAMPLE FLAGGED PAPERS (low SBERT similarity)")
    for p in sorted(flagged_sbert, key=lambda x: x["cs_similarity"])[:5]:
        print(f"  [{p['id']}] cat={p['category']} sim={p['cs_similarity']:.3f}")
        print(f"    Abstract: {p['abstract'][:120]}...")
        print()

    # Borderline cases (just above threshold)
    borderline = sorted(clean, key=lambda x: x["cs_similarity"])[:5]
    print(f"BORDERLINE PAPERS (lowest clean)")
    for p in borderline:
        print(f"  [{p['id']}] cat={p['category']} sim={p['cs_similarity']:.3f}")
        print(f"    Abstract: {p['abstract'][:120]}...")
        print()

    # ── Fix mode ──────────────────────────────────────────────────────
    if args.fix and total_flagged > 0:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p in flagged_keyword + flagged_sbert:
            src = p["path"]
            dst = QUARANTINE_DIR / src.name
            if src.exists():
                shutil.move(str(src), str(dst))
                moved += 1

        print(f"MOVED {moved} non-CS papers to {QUARANTINE_DIR}")
        print(f"Remaining papers: {len(clean)}")
        print(f"\nRe-run pipeline: quality_filter.py → dataset_builder.py → validate_dataset.py")
    elif total_flagged > 0:
        print(f"{total_flagged} papers flagged. Run with --fix to quarantine them.")
    else:
        print(f"All papers appear CS-relevant. No action needed.")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()