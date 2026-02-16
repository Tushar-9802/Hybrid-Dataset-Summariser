"""
hf_arxiv_dataset.py  [A1]
Pulls engineering/CS papers from ccdv/arxiv-summarization (HuggingFace).
No API key. No rate limits. Streaming — low RAM usage.

Run from repo root:
    python hf_arxiv_dataset.py
    python hf_arxiv_dataset.py --target 2400 --output data/raw/papers

Output:
    data/raw/papers/{id}.json       one file per paper
    data/raw/papers/_manifest.json  run statistics

Prereqs (already in requirements.txt):
    pip install datasets tqdm
"""

import json, re, hashlib, logging, argparse
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET       = 2400
OUTPUT_DIR   = Path("data/raw/papers")
LOG_DIR      = Path("logs")
ABS_MIN_W    = 80    # abstract word floor
ABS_MAX_W    = 500   # abstract word ceiling
SRC_MIN_W    = 300   # full article word floor

# ── CATEGORY KEYWORD PATTERNS ─────────────────────────────────────────────────
# ccdv/arxiv-summarization has no category labels — we infer from text.
# Each category gets high-precision patterns; a record can match multiple.
PATTERNS = {
    "cs.AI": [
        r"\bartificial intelligence\b", r"\bintelligent agent", r"\bknowledge graph",
        r"\breasoning\b", r"\bplanning\b", r"\bsearch algorithm",
        r"\bgame playing\b", r"\bexpert system", r"\bontolog", r"\bsymbolic AI",
    ],
    "cs.CL": [
        r"\bnatural language\b", r"\bNLP\b", r"\blanguage model",
        r"\btext classif", r"\bmachine translation", r"\bsentiment\b",
        r"\bnamed entity", r"\bword embedding", r"\btransformer\b",
        r"\bBERT\b", r"\bGPT\b", r"\bsummariz", r"\bdialogue\b",
        r"\bquestion answer", r"\bcoreference",
    ],
    "cs.LG": [
        r"\bmachine learning\b", r"\bdeep learning\b", r"\bneural network",
        r"\bgradient descent", r"\bbackpropagation\b", r"\boverfit",
        r"\btransfer learning\b", r"\bmeta.?learning\b", r"\bfew.?shot\b",
        r"\breinforcement learning\b", r"\bpolicy gradient", r"\bQ.?learning",
        r"\bGAN\b", r"\bvariational autoencoder\b", r"\bdiffusion model\b",
        r"\battention mechanism\b",
    ],
    "cs.CV": [
        r"\bcomputer vision\b", r"\bimage classif", r"\bobject detect",
        r"\bsemantic segment", r"\binstance segment", r"\bpose estimation",
        r"\bdepth estimation\b", r"\boptical flow\b", r"\bimage generat",
        r"\bconvolutional\b", r"\bCNN\b", r"\bResNet\b", r"\bViT\b",
        r"\bpoint cloud\b", r"\b3D reconstruct",
    ],
    "cs.RO": [
        r"\brobotics\b", r"\brobot\b", r"\bautonomous\b", r"\bself.?driving\b",
        r"\bmotion planning\b", r"\bpath planning\b", r"\bSLAM\b",
        r"\blocalization\b", r"\bmanipulation\b", r"\bgrasp",
        r"\bnavigation\b", r"\bdrone\b", r"\bUAV\b", r"\bhuman.?robot\b",
    ],
    "cs.SE": [
        r"\bsoftware engineering\b", r"\bcode generat", r"\bprogram synthes",
        r"\bbug detect", r"\bfault local", r"\btest generat",
        r"\brefactor", r"\bstatic analys", r"\bprogram analys",
        r"\bdevops\b", r"\bmicroservice", r"\bAPI design\b",
        r"\bcode review\b", r"\btechnical debt",
    ],
    "cs.DS": [
        r"\bdata struct", r"\bgraph algorithm", r"\bsorting\b",
        r"\bsearch tree\b", r"\bhashing\b", r"\bdynamic programming\b",
        r"\bcomputational complex", r"\bapproximat\s+algorithm",
        r"\bstreaming algorithm", r"\bparallel algorithm",
        r"\bdistributed algorithm", r"\bdatabase\b", r"\bquery optim",
    ],
}

COMPILED = {cat: re.compile("|".join(pats), re.IGNORECASE) for cat, pats in PATTERNS.items()}

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "hf_arxiv_dataset.log"),
    ]
)
log = logging.getLogger(__name__)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def detect_categories(abstract: str) -> list[str]:
    return [cat for cat, pat in COMPILED.items() if pat.search(abstract)]


def is_english(text: str, threshold: float = 0.90) -> bool:
    return (sum(c.isascii() for c in text) / max(len(text), 1)) >= threshold


def make_summary(abstract: str, max_words: int = 65) -> str:
    """Truncate abstract to ~65 words ending at sentence boundary."""
    words = abstract.split()
    candidate = " ".join(words[:max_words])
    for punct in [".", "!", "?"]:
        last = candidate.rfind(punct)
        if last > len(candidate) * 0.5:
            return candidate[: last + 1].strip()
    return candidate.strip()


def stable_id(raw: dict) -> str:
    art_id = (raw.get("article_id") or "").strip()
    if art_id:
        return re.sub(r"[^A-Za-z0-9._-]", "_", art_id)
    return hashlib.md5((raw.get("abstract") or "").encode()).hexdigest()[:12]


def build_record(raw: dict, primary: str, all_cats: list[str]) -> dict:
    abstract = raw.get("abstract", "").strip()
    article  = raw.get("article",  "").strip()
    summary  = make_summary(abstract)
    wc_src   = len(article.split())
    wc_sum   = len(summary.split())
    return {
        "id":                  stable_id(raw),
        "abstract":            abstract,
        "source":              article,
        "summary":             summary,       # reference summary for evaluation
        "domain":              "engineering",
        "modality":            "paper",
        "primary_category":    primary,
        "categories":          all_cats,
        "word_count_source":   wc_src,
        "word_count_summary":  wc_sum,
        "token_count_source":  int(wc_src * 1.3),
        "token_count_summary": int(wc_sum * 1.3),
    }


# ── STREAM + FILTER ───────────────────────────────────────────────────────────

def stream_papers() -> Iterator[dict]:
    """Stream filtered records from HF dataset."""
    log.info("Loading ccdv/arxiv-summarization (streaming=True)...")
    ds = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    rejected = {k: 0 for k in ("no_category", "short_abstract",
                                "long_abstract", "short_source", "non_english")}

    for raw in ds:
        abstract = (raw.get("abstract") or "").strip()
        article  = (raw.get("article")  or "").strip()

        cats = detect_categories(abstract)
        if not cats:
            rejected["no_category"] += 1
            continue

        wc = len(abstract.split())
        if wc < ABS_MIN_W:
            rejected["short_abstract"] += 1
            continue
        if wc > ABS_MAX_W:
            rejected["long_abstract"] += 1
            continue
        if len(article.split()) < SRC_MIN_W:
            rejected["short_source"] += 1
            continue
        if not is_english(abstract):
            rejected["non_english"] += 1
            continue

        yield build_record(raw, cats[0], cats)

    log.info(f"Rejection breakdown: {rejected}")


# ── STATS ─────────────────────────────────────────────────────────────────────

def print_stats(out_dir: Path) -> bool:
    files   = [f for f in out_dir.glob("*.json") if not f.stem.startswith("_")]
    papers  = [json.loads(f.read_text()) for f in files]
    if not papers:
        log.warning("No papers found.")
        return False

    wcs  = [p["word_count_source"]  for p in papers]
    wcsm = [p["word_count_summary"] for p in papers]
    cat_dist: dict[str, int] = {}
    for p in papers:
        c = p.get("primary_category", "unknown")
        cat_dist[c] = cat_dist.get(c, 0) + 1

    print("\n" + "=" * 64)
    print("hf_arxiv_dataset.py — STATISTICS")
    print("=" * 64)
    print(f"Total papers    : {len(papers)}")
    print(f"Source words    : min={min(wcs)}, mean={sum(wcs)//len(wcs)}, max={max(wcs)}")
    print(f"Summary words   : min={min(wcsm)}, mean={sum(wcsm)//len(wcsm)}, max={max(wcsm)}")
    print("\nCategory distribution:")
    for cat in sorted(cat_dist):
        bar = "█" * (cat_dist[cat] // 15)
        print(f"  {cat:10s}  {cat_dist[cat]:4d}  {bar}")

    checks = [
        (f"Count ≥ 2300 (have {len(papers)})",     len(papers) >= 2300),
        ("All have abstract",                       all(p["abstract"] for p in papers)),
        ("All have summary",                        all(p["summary"]  for p in papers)),
        ("All have source",                         all(p["source"]   for p in papers)),
        ("domain=engineering on all",               all(p["domain"] == "engineering" for p in papers)),
        ("modality=paper on all",                   all(p["modality"] == "paper"     for p in papers)),
    ]

    print("\nValidation:")
    all_ok = True
    for msg, ok in checks:
        print(f"  [{'✓' if ok else '✗'}] {msg}")
        if not ok:
            all_ok = False

    (out_dir / "_manifest.json").write_text(json.dumps({
        "total": len(papers), "category_distribution": cat_dist,
        "mean_source_words": sum(wcs)//len(wcs),
        "mean_summary_words": sum(wcsm)//len(wcsm),
        "all_checks_passed": all_ok,
    }, indent=2))

    tag = "✓ Ready for transcribe.py" if all_ok else "✗ Fix issues before next step"
    print(f"\n{tag}\n")
    return all_ok


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",     type=int,  default=TARGET)
    ap.add_argument("--output",     type=str,  default=str(OUTPUT_DIR))
    ap.add_argument("--stats-only", action="store_true",
                    help="Print stats for already-collected papers and exit")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stats_only:
        print_stats(out_dir)
        return

    existing = {f.stem for f in out_dir.glob("*.json") if not f.stem.startswith("_")}
    log.info(f"Already collected: {len(existing)} papers")

    need = args.target - len(existing)
    if need <= 0:
        log.info("Target already met.")
        print_stats(out_dir)
        return

    log.info(f"Collecting {need} more papers...")
    saved = 0
    with tqdm(total=need, unit="paper") as bar:
        for record in stream_papers():
            if record["id"] in existing:
                continue
            (out_dir / f"{record['id']}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False)
            )
            existing.add(record["id"])
            saved += 1
            bar.update(1)
            if saved >= need:
                break

    log.info(f"Saved {saved} new papers.")
    print_stats(out_dir)


if __name__ == "__main__":
    main()