"""
validate_videos.py — Audio Dataset Validator
Validates downloaded YouTube audio before Whisper transcription.

Checks:
  1. Quantity (target: 1200 ± 100)
  2. WAV file existence and integrity (non-zero, readable by ffprobe)
  3. Duration within spec (5-90 min)
  4. meta.json completeness (required fields present)
  5. Duplicate detection (by video ID and by title similarity)
  6. Category distribution balance
  7. Channel diversity (max 10 per channel)

Run from repo root:
    python src/data/validate_videos.py
    python src/data/validate_videos.py --fix     # Remove invalid entries
"""

import json
import argparse
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path("data/raw/videos")
MIN_DUR = 5 * 60
MAX_DUR = 90 * 60
MIN_WAV_BYTES = 50_000          # 50KB — anything smaller is corrupt/empty
REQUIRED_META_FIELDS = {"id", "title", "channel_id", "duration_seconds", "domain", "modality", "category"}


def get_wav_duration(wav_path: Path) -> float | None:
    """Get duration in seconds via ffprobe. Returns None if unreadable."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav_path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def title_normalize(title: str) -> str:
    """Normalize title for duplicate comparison."""
    import re
    t = title.lower().strip()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def validate():
    parser = argparse.ArgumentParser(description="Validate video audio dataset")
    parser.add_argument("--fix", action="store_true",
                        help="Remove invalid/duplicate entries")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist")
        return

    # ── Collect all entries ──
    entries = []
    for meta_path in sorted(DATA_DIR.glob("*/meta.json")):
        vid_dir = meta_path.parent
        vid_id = vid_dir.name
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        entries.append({
            "vid_id": vid_id,
            "vid_dir": vid_dir,
            "meta_path": meta_path,
            "wav_path": vid_dir / "audio.wav",
            "meta": meta,
        })

    total = len(entries)
    print(f"{'='*60}")
    print(f"VIDEO DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Total entries found: {total}")
    print()

    # ── Check categories ──
    issues = {
        "missing_wav": [],
        "corrupt_wav": [],
        "too_small_wav": [],
        "duration_out_of_range": [],
        "missing_meta_fields": [],
        "duplicate_ids": [],
        "duplicate_titles": [],
        "channel_overflow": [],
    }

    valid_entries = []
    seen_ids = set()
    seen_titles = {}
    channel_counts = defaultdict(list)
    cat_counts = Counter()
    durations = []

    for entry in entries:
        vid_id = entry["vid_id"]
        meta = entry["meta"]
        wav = entry["wav_path"]
        problems = []

        # 1. Duplicate ID
        if vid_id in seen_ids:
            issues["duplicate_ids"].append(vid_id)
            problems.append("duplicate_id")
        seen_ids.add(vid_id)

        # 2. WAV existence
        if not wav.exists():
            issues["missing_wav"].append(vid_id)
            problems.append("missing_wav")
        elif wav.stat().st_size < MIN_WAV_BYTES:
            issues["too_small_wav"].append(vid_id)
            problems.append("too_small_wav")
        else:
            # 3. WAV integrity — check with ffprobe
            dur = get_wav_duration(wav)
            if dur is None:
                issues["corrupt_wav"].append(vid_id)
                problems.append("corrupt_wav")
            else:
                durations.append(dur)
                # 4. Duration check (use WAV actual duration, not meta)
                if not (MIN_DUR <= dur <= MAX_DUR):
                    issues["duration_out_of_range"].append(
                        f"{vid_id} ({dur/60:.1f}m)"
                    )
                    problems.append("duration_out_of_range")

        # 5. Meta completeness
        missing = REQUIRED_META_FIELDS - set(meta.keys())
        if missing:
            issues["missing_meta_fields"].append(f"{vid_id}: missing {missing}")
            problems.append("missing_meta_fields")

        # 6. Duplicate title detection
        title = meta.get("title", "")
        norm_title = title_normalize(title)
        if norm_title and len(norm_title) > 10:
            if norm_title in seen_titles:
                issues["duplicate_titles"].append(
                    f"{vid_id} ≈ {seen_titles[norm_title]} ('{title[:50]}')"
                )
                problems.append("duplicate_title")
            else:
                seen_titles[norm_title] = vid_id

        # 7. Channel tracking
        chan = meta.get("channel_id", "unknown")
        channel_counts[chan].append(vid_id)

        # Category tracking
        cat = meta.get("category", "unknown")
        cat_counts[cat] += 1

        entry["problems"] = problems
        if not problems:
            valid_entries.append(entry)

    # Channel overflow check
    for chan, vids in channel_counts.items():
        if len(vids) > 10:
            issues["channel_overflow"].append(
                f"{chan}: {len(vids)} videos (max 10)"
            )

    # ── Report ──
    print("QUANTITY CHECK")
    print(f"  Total entries:  {total}")
    print(f"  Valid entries:  {len(valid_entries)}")
    print(f"  Target:         1200 ± 100")
    status = "PASS" if 1100 <= len(valid_entries) <= 1300 else "FAIL"
    print(f"  Status:         {status}")
    print()

    print("CATEGORY DISTRIBUTION")
    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        bar = "█" * (count // 5)
        print(f"  {cat:8s}: {count:4d} {bar}")
    print()

    if durations:
        avg_dur = sum(durations) / len(durations)
        min_dur = min(durations)
        max_dur = max(durations)
        print("DURATION STATS (from WAV files)")
        print(f"  Count:    {len(durations)}")
        print(f"  Mean:     {avg_dur/60:.1f} min")
        print(f"  Min:      {min_dur/60:.1f} min")
        print(f"  Max:      {max_dur/60:.1f} min")
        print()

    print("CHANNEL DIVERSITY")
    print(f"  Unique channels: {len(channel_counts)}")
    top_channels = sorted(channel_counts.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for chan, vids in top_channels:
        print(f"  {chan[:20]:20s}: {len(vids)} videos")
    print()

    print("ISSUES FOUND")
    any_issues = False
    for issue_name, issue_list in issues.items():
        if issue_list:
            any_issues = True
            print(f"  {issue_name}: {len(issue_list)}")
            for item in issue_list[:5]:
                print(f"    - {item}")
            if len(issue_list) > 5:
                print(f"    ... and {len(issue_list) - 5} more")
    if not any_issues:
        print("  None — dataset is clean")
    print()

    # ── Fix mode ──
    if args.fix:
        to_remove = set()
        for entry in entries:
            if entry["problems"]:
                to_remove.add(entry["vid_id"])

        if not to_remove:
            print("Nothing to fix — all entries valid.")
        else:
            print(f"FIXING: Removing {len(to_remove)} invalid entries...")
            for vid_id in to_remove:
                vid_dir = DATA_DIR / vid_id
                if vid_dir.exists():
                    shutil.rmtree(vid_dir)
                    print(f"  Removed {vid_id}")
            remaining = total - len(to_remove)
            print(f"  Done. {remaining} entries remaining.")
    else:
        bad_count = total - len(valid_entries)
        if bad_count > 0:
            print(f"Run with --fix to remove {bad_count} invalid entries.")

    print(f"{'='*60}")


if __name__ == "__main__":
    validate()