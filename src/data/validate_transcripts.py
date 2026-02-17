"""
validate_transcripts.py — Transcript Quality Validator
Validates Whisper transcript.json files before GPT-4o-mini summary generation.

Checks:
  1. Missing transcripts (audio exists, no transcript)
  2. Orphan dirs (no transcript — prune candidates)
  3. Empty/near-empty (word_count < 100)
  4. Below spec floor (word_count < 500)
  5. Above spec ceiling (word_count > 15,000)
  6. Whisper hallucination detection (CONSECUTIVE identical sentences only)
  7. Non-English / garbled output (non-ASCII ratio, low lexical diversity)
  8. Word count distribution + per-category stats

v2 fixes: Hallucination detector no longer flags normal lecture speech patterns
   like "we are going to", "I'm going to", etc. Only flags ACTUAL Whisper
   failure modes (stuck loops producing identical consecutive output).

Run from repo root:
    python src/data/validate_transcripts.py
    python src/data/validate_transcripts.py --fix      # Quarantine bad transcripts
    python src/data/validate_transcripts.py --prune    # Delete untranscribed dirs
    python src/data/validate_transcripts.py --fix --prune
"""

import json
import argparse
import shutil
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("data/raw/videos")
QUARANTINE_DIR = Path("data/quarantine")

# Thresholds
MIN_WORDS = 100          # Below this = likely corrupt/silent
SPEC_FLOOR = 500         # Spec [A5] minimum for training
SPEC_CEIL = 15_000       # Spec [A5] maximum for training
NON_ASCII_THRESHOLD = 0.15  # >15% non-ASCII chars = likely non-English
LEXICAL_DIV_THRESHOLD = 0.10  # unique/total words below this = garbled


def detect_hallucinations(text: str) -> list[str]:
    """
    Detect ACTUAL Whisper hallucination patterns, NOT normal lecture speech.

    Real Whisper hallucinations:
    - Exact same sentence repeated back-to-back 4+ times (stuck decoding loop)
    - Known Whisper filler phrases at abnormally high frequency
    - Music note symbols looping

    NOT hallucinations (normal in lectures):
    - "I'm going to", "we have to", "let's look at" scattered throughout
    - Common pedagogical phrases repeated naturally over 30-60 minutes
    - Indian English patterns like "is called as", "we are going to discuss"
    """
    issues = []

    # === CHECK 1: Consecutive identical sentences ===
    # Split on sentence boundaries and look for runs of identical ones.
    # This catches Whisper's actual failure mode: getting stuck in a loop.
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 15]

    if len(sentences) > 5:
        max_run = 1
        current_run = 1
        run_text = ""
        for i in range(1, len(sentences)):
            if sentences[i] == sentences[i - 1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    run_text = sentences[i]
            else:
                current_run = 1

        if max_run >= 4:
            issues.append(
                f"Identical sentence repeated {max_run}x consecutively: "
                f"\"{run_text[:60]}\""
            )

    # === CHECK 2: Very long verbatim blocks ===
    # Only flag 10+ word phrases appearing an absurd number of times.
    # Scale threshold by transcript length. A 10K word transcript naturally
    # has more repetition than a 1K word one.
    words = text.lower().split()
    if len(words) > 500:
        for ngram_size in [10, 15]:
            if len(words) < ngram_size * 20:
                continue
            ngram_counts = Counter()
            for i in range(len(words) - ngram_size + 1):
                ngram = " ".join(words[i:i + ngram_size])
                ngram_counts[ngram] += 1

            for ngram, count in ngram_counts.most_common(1):
                # Threshold: 0.3% of total words must be this exact phrase
                threshold = max(20, len(words) // 300)
                if count >= threshold:
                    issues.append(
                        f"Repeated {ngram_size}-gram {count}x "
                        f"(threshold {threshold}): \"{ngram[:70]}\""
                    )

    # === CHECK 3: Known Whisper hallucination fillers ===
    # These are phrases Whisper inserts when it hallucinates on silence/music.
    # Must be at abnormal frequency to flag.
    hallucination_fillers = [
        ("thank you for watching", 8),
        ("please subscribe", 8),
        ("like and subscribe", 5),
        ("thanks for watching", 8),
        ("see you in the next video", 5),
    ]
    text_lower = text.lower()
    total_words = len(words) if words else 1
    for phrase, min_count in hallucination_fillers:
        count = text_lower.count(phrase)
        phrase_word_frac = (len(phrase.split()) * count) / total_words
        if count >= min_count and phrase_word_frac > 0.01:
            issues.append(
                f"Whisper filler \"{phrase}\" appears {count}x "
                f"({phrase_word_frac:.1%} of transcript)"
            )

    # === CHECK 4: Music note loops ===
    music_count = text.count("\u266a")
    if music_count >= 10:
        issues.append(f"Music note symbol appears {music_count}x")

    return issues


def check_language_quality(text: str) -> list[str]:
    """Check for non-English or garbled output."""
    issues = []

    # Non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    if ratio > NON_ASCII_THRESHOLD:
        issues.append(
            f"High non-ASCII ratio: {ratio:.1%} "
            f"(threshold {NON_ASCII_THRESHOLD:.0%})"
        )

    # Lexical diversity (unique words / total words)
    words = text.lower().split()
    if len(words) > 50:
        unique = len(set(words))
        diversity = unique / len(words)
        if diversity < LEXICAL_DIV_THRESHOLD:
            issues.append(
                f"Very low lexical diversity: {diversity:.3f} "
                f"({unique} unique / {len(words)} total)"
            )

    return issues


def validate():
    parser = argparse.ArgumentParser(description="Validate Whisper transcripts")
    parser.add_argument("--fix", action="store_true",
                        help="Quarantine bad transcripts to data/quarantine/")
    parser.add_argument("--prune", action="store_true",
                        help="Delete directories with no transcript.json")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist")
        return

    # Collect all video directories
    all_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    has_transcript = []
    no_transcript = []

    for d in all_dirs:
        if (d / "transcript.json").exists():
            has_transcript.append(d)
        else:
            no_transcript.append(d)

    print(f"{'=' * 65}")
    print(f"TRANSCRIPT VALIDATION REPORT")
    print(f"{'=' * 65}")
    print(f"Total video directories:  {len(all_dirs)}")
    print(f"With transcript.json:     {len(has_transcript)}")
    print(f"Without transcript.json:  {len(no_transcript)}")
    print()

    # -- Prune untranscribed dirs --
    if no_transcript:
        has_audio_count = sum(1 for d in no_transcript if (d / "audio.wav").exists())
        empty_count = len(no_transcript) - has_audio_count
        print(f"UNTRANSCRIBED DIRECTORIES: {len(no_transcript)}")
        print(f"  With audio.wav (failed transcription): {has_audio_count}")
        print(f"  Empty / no audio:                      {empty_count}")
        for d in no_transcript[:5]:
            has_audio = (d / "audio.wav").exists()
            has_meta = (d / "meta.json").exists()
            print(f"  {d.name}  audio={'Y' if has_audio else 'N'}  meta={'Y' if has_meta else 'N'}")
        if len(no_transcript) > 5:
            print(f"  ... and {len(no_transcript) - 5} more")
        print()

        if args.prune:
            print(f"PRUNING: Deleting {len(no_transcript)} untranscribed directories...")
            deleted = 0
            bytes_freed = 0
            for d in no_transcript:
                try:
                    dir_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    bytes_freed += dir_size
                    shutil.rmtree(d)
                    deleted += 1
                except Exception as e:
                    print(f"  Failed to delete {d.name}: {e}")
            print(f"  Deleted {deleted} directories ({bytes_freed / 1e9:.2f} GB freed)")
            print()
        else:
            print(f"  Run with --prune to delete these {len(no_transcript)} directories")
            print()

    # -- Validate transcripts --
    issues = {
        "empty": [],           # word_count < 100
        "below_floor": [],     # word_count < 500
        "above_ceil": [],      # word_count > 15000
        "hallucination": [],   # actual Whisper stuck loops
        "language": [],        # non-English / garbled
        "parse_error": [],     # can't read JSON
    }

    cat_counts = defaultdict(int)
    cat_word_counts = defaultdict(list)
    all_word_counts = []
    valid_count = 0
    to_quarantine = set()

    for vid_dir in has_transcript:
        vid_id = vid_dir.name
        transcript_path = vid_dir / "transcript.json"
        meta_path = vid_dir / "meta.json"

        # Load transcript
        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
            text = data.get("text", "")
            word_count = data.get("word_count", len(text.split()))
        except Exception as e:
            issues["parse_error"].append(f"{vid_id}: {e}")
            to_quarantine.add(vid_id)
            continue

        # Load category from meta
        cat = "unknown"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cat = meta.get("category", "unknown")
            except Exception:
                pass

        cat_counts[cat] += 1
        cat_word_counts[cat].append(word_count)
        all_word_counts.append(word_count)

        problems = []

        # Check 1: Empty / near-empty
        if word_count < MIN_WORDS:
            issues["empty"].append(f"{vid_id}: {word_count} words")
            problems.append("empty")

        # Check 2: Below spec floor (warn only, don't quarantine)
        elif word_count < SPEC_FLOOR:
            issues["below_floor"].append(f"{vid_id}: {word_count} words")

        # Check 3: Above spec ceiling (warn only, truncate at training time)
        if word_count > SPEC_CEIL:
            issues["above_ceil"].append(f"{vid_id}: {word_count} words")

        # Check 4: Hallucination detection (consecutive repetitions only)
        if text:
            hall_issues = detect_hallucinations(text)
            if hall_issues:
                issues["hallucination"].append(f"{vid_id}: {'; '.join(hall_issues)}")
                problems.append("hallucination")

        # Check 5: Language quality
        if text:
            lang_issues = check_language_quality(text)
            if lang_issues:
                issues["language"].append(f"{vid_id}: {'; '.join(lang_issues)}")
                problems.append("language")

        # Quarantine ONLY for critical failures
        if "empty" in problems or "hallucination" in problems or "language" in problems:
            to_quarantine.add(vid_id)
        else:
            valid_count += 1

    # -- Report --
    print(f"TRANSCRIPT QUALITY")
    print(f"  Valid transcripts:    {valid_count}")
    print(f"  Quarantine candidates: {len(to_quarantine)}")
    print()

    # Word count stats
    if all_word_counts:
        all_word_counts.sort()
        n = len(all_word_counts)
        mean_wc = sum(all_word_counts) / n
        median_wc = all_word_counts[n // 2]
        p5 = all_word_counts[int(n * 0.05)]
        p95 = all_word_counts[int(n * 0.95)]

        print(f"WORD COUNT DISTRIBUTION (n={n})")
        print(f"  Mean:     {mean_wc:,.0f}")
        print(f"  Median:   {median_wc:,.0f}")
        print(f"  Min:      {min(all_word_counts):,}")
        print(f"  Max:      {max(all_word_counts):,}")
        print(f"  P5:       {p5:,}")
        print(f"  P95:      {p95:,}")
        print()

        # Histogram
        bins = [0, 100, 500, 1000, 2000, 5000, 10000, 15000, 999999]
        labels = ["<100", "100-500", "500-1K", "1K-2K", "2K-5K",
                  "5K-10K", "10K-15K", ">15K"]
        hist = Counter()
        for wc in all_word_counts:
            for i in range(len(bins) - 1):
                if bins[i] <= wc < bins[i + 1]:
                    hist[labels[i]] += 1
                    break

        print(f"  HISTOGRAM")
        for label in labels:
            count = hist[label]
            bar = "\u2588" * (count // 3) if count > 0 else ""
            flag = " \u26a0" if label in ("<100", ">15K") else ""
            print(f"    {label:>8s}: {count:4d} {bar}{flag}")
        print()

    # Per-category breakdown
    print(f"PER-CATEGORY STATS")
    print(f"  {'Category':<10s} {'Count':>6s} {'Mean WC':>8s} {'Min':>6s} {'Max':>6s}")
    print(f"  {'-' * 40}")
    for cat in sorted(cat_counts.keys()):
        wcs = cat_word_counts[cat]
        mean_c = sum(wcs) / len(wcs) if wcs else 0
        min_c = min(wcs) if wcs else 0
        max_c = max(wcs) if wcs else 0
        print(f"  {cat:<10s} {cat_counts[cat]:>6d} {mean_c:>8,.0f} {min_c:>6,} {max_c:>6,}")
    print()

    # Issue details
    print(f"ISSUES FOUND")
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
        print(f"  None \u2014 all transcripts clean")
    print()

    # -- Fix mode --
    if args.fix and to_quarantine:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"QUARANTINING: Moving {len(to_quarantine)} entries to {QUARANTINE_DIR}/...")
        moved = 0
        for vid_id in sorted(to_quarantine):
            src = DATA_DIR / vid_id
            dst = QUARANTINE_DIR / vid_id
            if src.exists():
                try:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
                    moved += 1
                except Exception as e:
                    print(f"  Failed to move {vid_id}: {e}")
        print(f"  Moved {moved} directories to quarantine")
        remaining = len(has_transcript) - moved
        print(f"  Remaining: {remaining}")
        print()
    elif to_quarantine:
        print(f"Run with --fix to quarantine {len(to_quarantine)} bad transcripts")
        print()

    # -- Final summary --
    print(f"{'=' * 65}")
    print(f"SUMMARY")
    print(f"  Transcribed videos:       {len(has_transcript)}")
    print(f"  Untranscribed (prunable): {len(no_transcript)}")
    print(f"  Valid for training:       {valid_count}")
    if valid_count > 0:
        print(f"  Training split (80/10/10):")
        print(f"    Train: {int(valid_count * 0.8)}")
        print(f"    Val:   {int(valid_count * 0.1)}")
        print(f"    Test:  {int(valid_count * 0.1)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    validate()