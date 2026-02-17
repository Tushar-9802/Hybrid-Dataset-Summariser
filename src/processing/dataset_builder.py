"""
dataset_builder.py — HDF5 Dataset Packaging
Tokenizes all samples with Mistral tokenizer and packages into HDF5.

Reads manifest from A5 (data/processed/manifest.json) to get filtered IDs
and split assignments, then reloads source text from original files.

Run from repo root:
    python src/processing/dataset_builder.py
    python src/processing/dataset_builder.py --dry-run    # Show what would be built

Output:
    data/hdf5/engineering.h5

Hardware: ~2-4 GB RAM for tokenization. No GPU needed.
"""

import json
import logging
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import h5py
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

MANIFEST_FILE = Path("data/processed/manifest.json")
PAPERS_DIR    = Path("data/raw/papers")
VIDEOS_DIR    = Path("data/raw/videos")
PAIRS_FILE    = Path("data/processed/cross_modal_pairs/pairs.json")
OUTPUT_DIR    = Path("data/hdf5")
OUTPUT_FILE   = OUTPUT_DIR / "engineering.h5"
LOG_DIR       = Path("logs")

MAX_SEQ_LEN   = 1024   # Source token limit
MAX_LABEL_LEN = 256    # Summary/label token limit

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dataset_builder.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


# ── TOKENIZER ─────────────────────────────────────────────────────────────────

_tokenizer = None

def get_tokenizer():
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
        log.info(f"Tokenizer loaded. Vocab size: {_tokenizer.vocab_size}, "
                 f"pad_token_id: {_tokenizer.pad_token_id}")
    return _tokenizer


def tokenize_and_pad(text: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenize, truncate to max_len, pad to max_len.
    Returns (input_ids, attention_mask) as int32 arrays.
    """
    tok = get_tokenizer()
    encoded = tok(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="np",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"][0].astype(np.int32)
    attention_mask = encoded["attention_mask"][0].astype(np.int32)
    return input_ids, attention_mask


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}. Run quality_filter.py first.")
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def load_paper_text(paper_id: str) -> tuple[str, str]:
    """Load source (article body) and summary (abstract) for a paper."""
    for suffix in [".json"]:
        path = PAPERS_DIR / f"{paper_id}{suffix}"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("source", ""), data.get("abstract", "")
    return "", ""


def load_video_text(video_id: str) -> tuple[str, str]:
    """Load source (transcript) and summary for a video."""
    tp = VIDEOS_DIR / video_id / "transcript.json"
    if tp.exists():
        data = json.loads(tp.read_text(encoding="utf-8"))
        return data.get("text", ""), data.get("summary", "")
    return "", ""


def load_pairs_data() -> dict:
    """Load pairs file, return dict keyed by pair_id."""
    if not PAIRS_FILE.exists():
        return {}
    data = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    return {p["pair_id"]: p for p in data.get("pairs", [])}


# ── HDF5 BUILDING ────────────────────────────────────────────────────────────

def build_hdf5(manifest: dict, dry_run: bool = False):
    """
    Build HDF5 dataset from manifest.

    Structure:
    /papers/
        input_ids      (N, 1024)
        attention_mask  (N, 1024)
        labels          (N, 256)
        label_mask      (N, 256)
        modality        (N,)          # 0 = paper
        split           (N,)          # 0=train, 1=val, 2=test
        category        (N,)          # string stored as int index
        sample_id       (N,)          # string stored via special dtype
    /videos/
        input_ids      (N, 1024)
        attention_mask  (N, 1024)
        labels          (N, 256)
        label_mask      (N, 256)
        modality        (N,)          # 1 = video
        split           (N,)
        category        (N,)
        sample_id       (N,)
    /cross_modal/
        paper_input_ids     (N, 1024)
        paper_attention_mask(N, 1024)
        paper_labels        (N, 256)
        video_input_ids     (N, 1024)
        video_attention_mask(N, 1024)
        video_labels        (N, 256)
        similarity          (N,)
        split               (N,)
        pair_id             (N,)
    /metadata/
        categories     list of category strings
        max_seq_len    1024
        max_label_len  256
        tokenizer      "mistralai/Mistral-7B-v0.1"
        pad_token_id   int
    """
    samples = manifest["samples"]
    pair_manifest = manifest["pairs"]

    papers = [s for s in samples if s["modality"] == "paper"]
    videos = [s for s in samples if s["modality"] == "video"]

    # Category index mapping
    all_cats = sorted(set(s["category"] for s in samples))
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}

    # Split mapping
    split_map = {"train": 0, "val": 1, "test": 2}

    log.info(f"Building HDF5: {len(papers)} papers, {len(videos)} videos, "
             f"{len(pair_manifest)} pairs")
    log.info(f"Categories: {all_cats}")

    if dry_run:
        print(f"\nDRY RUN — Would build:")
        print(f"  /papers/:       {len(papers)} samples × (1024 + 1024 + 256 + 256) int32")
        print(f"  /videos/:       {len(videos)} samples × (1024 + 1024 + 256 + 256) int32")
        print(f"  /cross_modal/:  {len(pair_manifest)} pairs × 2 × (1024 + 1024 + 256) int32")

        paper_bytes = len(papers) * (1024 + 1024 + 256 + 256) * 4
        video_bytes = len(videos) * (1024 + 1024 + 256 + 256) * 4
        pair_bytes = len(pair_manifest) * 2 * (1024 + 1024 + 256) * 4
        total_mb = (paper_bytes + video_bytes + pair_bytes) / (1024 * 1024)

        print(f"  Estimated file size: {total_mb:.0f} MB")
        print(f"  Output: {OUTPUT_FILE}")
        return

    # Load pair source data
    pairs_data = load_pairs_data()
    pair_manifest_ids = set(p["pair_id"] for p in pair_manifest)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = get_tokenizer()
    dt_str = h5py.special_dtype(vlen=str)

    with h5py.File(str(OUTPUT_FILE), "w") as hf:

        # ── Papers group ──────────────────────────────────────────────
        log.info("Tokenizing papers...")
        n_papers = len(papers)
        p_input_ids = np.zeros((n_papers, MAX_SEQ_LEN), dtype=np.int32)
        p_attn_mask = np.zeros((n_papers, MAX_SEQ_LEN), dtype=np.int32)
        p_labels    = np.zeros((n_papers, MAX_LABEL_LEN), dtype=np.int32)
        p_label_mask = np.zeros((n_papers, MAX_LABEL_LEN), dtype=np.int32)
        p_modality  = np.zeros(n_papers, dtype=np.int32)
        p_split     = np.zeros(n_papers, dtype=np.int32)
        p_cat       = np.zeros(n_papers, dtype=np.int32)
        p_ids       = []

        skipped = 0
        w = 0  # Write pointer — only advances on successful writes
        for sample in tqdm(papers, desc="Papers"):
            source, summary = load_paper_text(sample["id"])
            if not source or not summary:
                skipped += 1
                continue

            ids, mask = tokenize_and_pad(source, MAX_SEQ_LEN)
            lab_ids, lab_mask = tokenize_and_pad(summary, MAX_LABEL_LEN)

            p_input_ids[w] = ids
            p_attn_mask[w] = mask
            p_labels[w] = lab_ids
            p_label_mask[w] = lab_mask
            p_modality[w] = 0
            p_split[w] = split_map.get(sample["split"], 0)
            p_cat[w] = cat_to_idx.get(sample["category"], 0)
            p_ids.append(sample["id"])
            w += 1

        if skipped:
            log.warning(f"Skipped {skipped} papers with missing text")

        # Truncate arrays to actual count
        p_input_ids = p_input_ids[:w]
        p_attn_mask = p_attn_mask[:w]
        p_labels = p_labels[:w]
        p_label_mask = p_label_mask[:w]
        p_modality = p_modality[:w]
        p_split = p_split[:w]
        p_cat = p_cat[:w]

        grp = hf.create_group("papers")
        grp.create_dataset("input_ids", data=p_input_ids, compression="gzip", compression_opts=4)
        grp.create_dataset("attention_mask", data=p_attn_mask, compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=p_labels, compression="gzip", compression_opts=4)
        grp.create_dataset("label_mask", data=p_label_mask, compression="gzip", compression_opts=4)
        grp.create_dataset("modality", data=p_modality)
        grp.create_dataset("split", data=p_split)
        grp.create_dataset("category", data=p_cat)
        grp.create_dataset("sample_id", data=p_ids, dtype=dt_str)
        log.info(f"Papers written: {n_papers - skipped}")

        # Free memory
        del p_input_ids, p_attn_mask, p_labels, p_label_mask

        # ── Videos group ──────────────────────────────────────────────
        log.info("Tokenizing videos...")
        n_videos = len(videos)
        v_input_ids = np.zeros((n_videos, MAX_SEQ_LEN), dtype=np.int32)
        v_attn_mask = np.zeros((n_videos, MAX_SEQ_LEN), dtype=np.int32)
        v_labels    = np.zeros((n_videos, MAX_LABEL_LEN), dtype=np.int32)
        v_label_mask = np.zeros((n_videos, MAX_LABEL_LEN), dtype=np.int32)
        v_modality  = np.ones(n_videos, dtype=np.int32)
        v_split     = np.zeros(n_videos, dtype=np.int32)
        v_cat       = np.zeros(n_videos, dtype=np.int32)
        v_ids       = []

        skipped = 0
        for i, sample in enumerate(tqdm(videos, desc="Videos")):
            source, summary = load_video_text(sample["id"])
            if not source or not summary:
                skipped += 1
                continue

            ids, mask = tokenize_and_pad(source, MAX_SEQ_LEN)
            lab_ids, lab_mask = tokenize_and_pad(summary, MAX_LABEL_LEN)

            v_input_ids[i] = ids
            v_attn_mask[i] = mask
            v_labels[i] = lab_ids
            v_label_mask[i] = lab_mask
            v_split[i] = split_map.get(sample["split"], 0)
            v_cat[i] = cat_to_idx.get(sample["category"], 0)
            v_ids.append(sample["id"])

        if skipped:
            log.warning(f"Skipped {skipped} videos with missing text")

        grp = hf.create_group("videos")
        grp.create_dataset("input_ids", data=v_input_ids, compression="gzip", compression_opts=4)
        grp.create_dataset("attention_mask", data=v_attn_mask, compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=v_labels, compression="gzip", compression_opts=4)
        grp.create_dataset("label_mask", data=v_label_mask, compression="gzip", compression_opts=4)
        grp.create_dataset("modality", data=v_modality)
        grp.create_dataset("split", data=v_split)
        grp.create_dataset("category", data=v_cat)
        grp.create_dataset("sample_id", data=v_ids, dtype=dt_str)
        log.info(f"Videos written: {n_videos - skipped}")

        del v_input_ids, v_attn_mask, v_labels, v_label_mask

        # ── Cross-modal pairs group ───────────────────────────────────
        log.info("Tokenizing cross-modal pairs...")
        valid_pairs = [p for p in pair_manifest if p["pair_id"] in pairs_data]
        n_pairs = len(valid_pairs)

        cm_p_input_ids = np.zeros((n_pairs, MAX_SEQ_LEN), dtype=np.int32)
        cm_p_attn_mask = np.zeros((n_pairs, MAX_SEQ_LEN), dtype=np.int32)
        cm_p_labels    = np.zeros((n_pairs, MAX_LABEL_LEN), dtype=np.int32)
        cm_v_input_ids = np.zeros((n_pairs, MAX_SEQ_LEN), dtype=np.int32)
        cm_v_attn_mask = np.zeros((n_pairs, MAX_SEQ_LEN), dtype=np.int32)
        cm_v_labels    = np.zeros((n_pairs, MAX_LABEL_LEN), dtype=np.int32)
        cm_sim         = np.zeros(n_pairs, dtype=np.float32)
        cm_split       = np.zeros(n_pairs, dtype=np.int32)
        cm_ids         = []

        skipped = 0
        for i, pm in enumerate(tqdm(valid_pairs, desc="Pairs")):
            pair = pairs_data[pm["pair_id"]]

            # Paper side — load article body from disk as source, abstract as label
            paper_source, paper_label = load_paper_text(pair["paper_id"])

            # If paper file is gone (quarantined/deleted), skip this pair entirely
            if not paper_source or not paper_label:
                skipped += 1
                continue

            # Video side — load transcript from disk, summary from pairs.json
            video_source, _ = load_video_text(pair["video_id"])
            video_summary = pair.get("video_summary", "")

            if not paper_source or not video_summary:
                skipped += 1
                continue

            # If video_source missing (synthetic pair), use video_summary as source
            if not video_source:
                video_source = video_summary

            p_ids_enc, p_mask = tokenize_and_pad(paper_source, MAX_SEQ_LEN)
            p_lab, _ = tokenize_and_pad(paper_label, MAX_LABEL_LEN)
            v_ids_enc, v_mask = tokenize_and_pad(video_source, MAX_SEQ_LEN)
            v_lab, _ = tokenize_and_pad(video_summary, MAX_LABEL_LEN)

            cm_p_input_ids[i] = p_ids_enc
            cm_p_attn_mask[i] = p_mask
            cm_p_labels[i] = p_lab
            cm_v_input_ids[i] = v_ids_enc
            cm_v_attn_mask[i] = v_mask
            cm_v_labels[i] = v_lab
            cm_sim[i] = pair.get("similarity", 0.0)
            cm_split[i] = split_map.get(pm["split"], 0)
            cm_ids.append(pm["pair_id"])

        if skipped:
            log.warning(f"Skipped {skipped} pairs with missing text")

        grp = hf.create_group("cross_modal")
        grp.create_dataset("paper_input_ids", data=cm_p_input_ids, compression="gzip", compression_opts=4)
        grp.create_dataset("paper_attention_mask", data=cm_p_attn_mask, compression="gzip", compression_opts=4)
        grp.create_dataset("paper_labels", data=cm_p_labels, compression="gzip", compression_opts=4)
        grp.create_dataset("video_input_ids", data=cm_v_input_ids, compression="gzip", compression_opts=4)
        grp.create_dataset("video_attention_mask", data=cm_v_attn_mask, compression="gzip", compression_opts=4)
        grp.create_dataset("video_labels", data=cm_v_labels, compression="gzip", compression_opts=4)
        grp.create_dataset("similarity", data=cm_sim)
        grp.create_dataset("split", data=cm_split)
        grp.create_dataset("pair_id", data=cm_ids, dtype=dt_str)
        log.info(f"Pairs written: {n_pairs - skipped}")

        del cm_p_input_ids, cm_p_attn_mask, cm_p_labels
        del cm_v_input_ids, cm_v_attn_mask, cm_v_labels

        # ── Metadata group ────────────────────────────────────────────
        meta = hf.create_group("metadata")
        meta.create_dataset("categories", data=all_cats, dtype=dt_str)
        meta.attrs["max_seq_len"] = MAX_SEQ_LEN
        meta.attrs["max_label_len"] = MAX_LABEL_LEN
        meta.attrs["tokenizer"] = "mistralai/Mistral-7B-v0.1"
        meta.attrs["pad_token_id"] = tok.pad_token_id
        meta.attrs["n_papers"] = n_papers
        meta.attrs["n_videos"] = n_videos
        meta.attrs["n_pairs"] = n_pairs
        meta.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── Report ────────────────────────────────────────────────────────
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print(f"\n{'='*65}")
    print(f"HDF5 DATASET BUILD REPORT")
    print(f"{'='*65}")
    print(f"Output: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"  /papers/:      {n_papers} samples")
    print(f"  /videos/:      {n_videos} samples")
    print(f"  /cross_modal/: {n_pairs} pairs")
    print(f"  /metadata/:    {len(all_cats)} categories")
    print(f"\n  max_seq_len:   {MAX_SEQ_LEN}")
    print(f"  max_label_len: {MAX_LABEL_LEN}")
    print(f"  tokenizer:     mistralai/Mistral-7B-v0.1")
    print(f"  pad_token_id:  {tok.pad_token_id}")

    # Verification — reopen and check shapes
    print(f"\nVERIFICATION (reopening file)")
    with h5py.File(str(OUTPUT_FILE), "r") as hf:
        checks = []
        for group in ["papers", "videos"]:
            g = hf[group]
            checks.append((f"{group}/input_ids shape",
                           g["input_ids"].shape[1] == MAX_SEQ_LEN))
            checks.append((f"{group}/labels shape",
                           g["labels"].shape[1] == MAX_LABEL_LEN))
            checks.append((f"{group}/sample_id count matches",
                           len(g["sample_id"]) == g["input_ids"].shape[0]))

        g = hf["cross_modal"]
        checks.append(("cross_modal/paper_input_ids shape",
                        g["paper_input_ids"].shape[1] == MAX_SEQ_LEN))
        checks.append(("cross_modal/video_labels shape",
                        g["video_labels"].shape[1] == MAX_LABEL_LEN))

        # Check no all-zero rows (indicate skipped samples)
        paper_nonzero = np.sum(hf["papers"]["attention_mask"][:], axis=1) > 0
        video_nonzero = np.sum(hf["videos"]["attention_mask"][:], axis=1) > 0
        checks.append((f"Papers: no all-zero rows ({paper_nonzero.sum()}/{n_papers})",
                        paper_nonzero.sum() == n_papers))
        checks.append((f"Videos: no all-zero rows ({video_nonzero.sum()}/{n_videos})",
                        video_nonzero.sum() == n_videos))

        all_ok = True
        for msg, ok in checks:
            print(f"  [{'✓' if ok else '✗'}] {msg}")
            if not ok:
                all_ok = False

    status = "✓ PASS — HDF5 ready for training" if all_ok else "⚠ Check issues above"
    print(f"\n  STATUS: {status}")
    print(f"{'='*65}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A6: HDF5 dataset packaging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be built, no actual tokenization")
    args = parser.parse_args()

    manifest = load_manifest()
    log.info(f"Manifest: {manifest['metadata']['total_samples']} samples, "
             f"{manifest['metadata']['pairs']} pairs")

    build_hdf5(manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()