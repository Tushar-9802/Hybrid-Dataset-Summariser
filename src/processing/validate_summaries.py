"""
validate_summaries.py — Summary Quality Validation
Extensive quality checks on GPT-4o-mini generated summaries before pair mining.

Folder: src/processing/

Checks:
  1. Coverage: how many transcripts have summaries vs missing
  2. Word count distribution vs length tier targets
  3. Academic framing detection (expanded marker list)
  4. Hallucination heuristics (names/entities not in transcript)
  5. Degenerate output detection (copy-paste from prompt, boilerplate)
  6. Specificity scoring (does the summary name concrete topics?)
  7. Tone consistency (conversational vs formal register signals)
  8. Per-category quality breakdown
  9. Duplicate/near-duplicate summary detection
  10. Sample output for manual inspection

Run from repo root:
    python src/processing/validate_summaries.py
    python src/processing/validate_summaries.py --samples 10   # Print more samples
    python src/processing/validate_summaries.py --fix           # Clear invalid summaries for re-generation
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("data/raw/videos")

# ── THRESHOLDS ──────────────────────────────────────────────────────────────

# Length tiers matching summary_generator.py
LENGTH_TIERS = [
    (2000,  30,  100,  "short"),
    (8000,  60,  160,  "medium"),
    (99999, 90,  220,  "long"),
]

# Academic framing — expanded list
ACADEMIC_MARKERS = [
    "this paper", "the authors", "we propose", "et al.", "[1]", "[2]",
    "contributions include", "this work presents", "the study",
    "the proposed method", "the paper", "our approach", "we present",
    "this study", "the researchers", "our method", "herein",
    "aforementioned", "the manuscript", "in this work", "we introduce",
    "the proposed framework", "our contribution", "this article",
]

# Boilerplate / degenerate output markers
BOILERPLATE_MARKERS = [
    "i cannot summarize",
    "i'm unable to",
    "as an ai",
    "i don't have access",
    "the transcript is",
    "here is a summary",
    "here's a summary",
    "sure, here",
    "certainly,",
    "of course,",
]

# Vagueness markers — summaries that are too generic
VAGUENESS_MARKERS = [
    "various topics",
    "various concepts",
    "several aspects",
    "different things",
    "many things",
    "a number of topics",
    "a range of",
    "various techniques",
    "different approaches",
    "multiple subjects",
    "various methods",
    "some concepts",
]

# Conversational tone signals (positive — we want these)
CONVERSATIONAL_SIGNALS = [
    "walks through", "covers", "dives into", "breaks down",
    "explains", "shows how", "goes over", "demonstrates",
    "walks you through", "touches on", "gets into",
    "explores", "tackles", "lays out", "works through",
]

# Formal tone signals (negative — we don't want these dominating)
FORMAL_SIGNALS = [
    "elucidates", "delineates", "postulates", "furthermore",
    "henceforth", "thereby", "wherein", "thus,",
    "moreover,", "consequently,", "in conclusion,",
    "it is evident that", "it should be noted",
]


def get_tier(word_count: int) -> tuple[int, int, str]:
    """Return (min_summary, max_summary, tier_name) for transcript word count."""
    for max_wc, min_s, max_s, name in LENGTH_TIERS:
        if word_count <= max_wc:
            return min_s, max_s, name
    return LENGTH_TIERS[-1][1], LENGTH_TIERS[-1][2], LENGTH_TIERS[-1][3]


def check_specificity(summary: str) -> tuple[float, list[str]]:
    """
    Score how specific a summary is (0-1).
    Looks for named algorithms, tools, concepts, numbers.
    """
    indicators = []

    # Named technical terms (capitalized multi-word or known patterns)
    tech_terms = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b', summary)
    tech_terms += re.findall(r'\b[A-Z]{2,}\b', summary)  # Acronyms like CNN, LSTM, API
    if tech_terms:
        indicators.append(f"tech_terms: {', '.join(list(set(tech_terms))[:5])}")

    # Specific numbers or measurements
    numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|x|GB|MB|ms|fps)?\b', summary)
    if numbers:
        indicators.append(f"numbers: {', '.join(numbers[:3])}")

    # Named tools/frameworks (lowercase but specific)
    tools = re.findall(
        r'\b(?:python|pytorch|tensorflow|docker|kubernetes|github|git|'
        r'numpy|pandas|matplotlib|scikit|opencv|ros|arduino|react|'
        r'javascript|sql|mongodb|kafka|spark|hadoop)\b',
        summary.lower()
    )
    if tools:
        indicators.append(f"tools: {', '.join(list(set(tools)))}")

    # Algorithm/method names with common patterns
    methods = re.findall(
        r'\b(?:gradient descent|backpropagation|convolution|attention|'
        r'regression|classification|clustering|sorting|searching|'
        r'dynamic programming|recursion|dijkstra|quicksort|mergesort|'
        r'binary search|breadth.first|depth.first|reinforcement learning|'
        r'neural network|decision tree|random forest|svm|'
        r'transformer|encoder|decoder|embedding|tokeniz)\w*\b',
        summary.lower()
    )
    if methods:
        indicators.append(f"methods: {', '.join(list(set(methods))[:5])}")

    # Score: more indicators = more specific
    raw_score = min(len(indicators) / 2.0, 1.0)  # 2+ indicator types = 1.0
    return raw_score, indicators


def check_hallucination_heuristic(summary: str, transcript: str) -> list[str]:
    """
    Simple heuristic: find capitalized proper nouns in summary that
    don't appear anywhere in the transcript. Not foolproof, but catches
    obvious fabrications.
    """
    issues = []

    # Extract capitalized words from summary (potential proper nouns)
    summary_caps = set(re.findall(r'\b[A-Z][a-z]{3,}\b', summary))
    # Remove common sentence starters
    common = {"The", "This", "That", "These", "Those", "Here", "There",
              "What", "How", "When", "Where", "Which", "Some", "Each",
              "Using", "Building", "Setting", "Starting", "Working",
              "Making", "Going", "Looking", "Getting", "Taking"}
    summary_caps -= common

    transcript_lower = transcript.lower()
    for word in summary_caps:
        if word.lower() not in transcript_lower:
            issues.append(word)

    return issues


def find_near_duplicates(summaries: list[tuple[str, str]], threshold: int = 20) -> list[tuple[str, str, int]]:
    """
    Find summaries that share suspiciously many words in sequence.
    Returns list of (vid_id_a, vid_id_b, shared_word_count).
    """
    dupes = []
    texts = [(vid_id, set(s.lower().split())) for vid_id, s in summaries]

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            vid_a, words_a = texts[i]
            vid_b, words_b = texts[j]
            overlap = len(words_a & words_b)
            min_len = min(len(words_a), len(words_b))
            if min_len > 0 and overlap / min_len > 0.85:
                dupes.append((vid_a, vid_b, overlap))

    return dupes


def validate():
    parser = argparse.ArgumentParser(description="Validate generated summaries")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of sample summaries to print per tier")
    parser.add_argument("--fix", action="store_true",
                        help="Clear invalid summaries so they can be re-generated")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist")
        return

    # ── Collect data ────────────────────────────────────────────────────
    all_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    has_summary = []
    missing_summary = []
    parse_errors = []

    for vid_dir in all_dirs:
        tp = vid_dir / "transcript.json"
        if not tp.exists():
            continue
        try:
            data = json.loads(tp.read_text(encoding="utf-8"))
        except Exception as e:
            parse_errors.append(f"{vid_dir.name}: {e}")
            continue

        if data.get("summary"):
            has_summary.append((vid_dir.name, data))
        else:
            missing_summary.append(vid_dir.name)

    total = len(has_summary) + len(missing_summary)

    print(f"{'='*65}")
    print(f"SUMMARY VALIDATION REPORT")
    print(f"{'='*65}")
    print(f"Total transcripts:    {total}")
    print(f"With summary:         {len(has_summary)}")
    print(f"Missing summary:      {len(missing_summary)}")
    if parse_errors:
        print(f"Parse errors:         {len(parse_errors)}")
    print()

    if not has_summary:
        print("No summaries to validate.")
        return

    # ── Check 1: Word count distribution by tier ───────────────────────
    tier_stats = defaultdict(lambda: {"counts": [], "violations": []})
    all_summary_wcs = []
    all_summaries_for_dupes = []

    issues = {
        "too_short": [],
        "too_long": [],
        "academic_framing": [],
        "boilerplate": [],
        "vague": [],
        "formal_tone": [],
        "low_specificity": [],
        "possible_hallucination": [],
        "already_flagged": [],
    }

    cat_stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": 0})

    for vid_id, data in has_summary:
        summary = data["summary"]
        transcript_wc = data.get("word_count", 0)
        summary_wc = len(summary.split())
        all_summary_wcs.append(summary_wc)
        all_summaries_for_dupes.append((vid_id, summary))

        min_s, max_s, tier_name = get_tier(transcript_wc)
        tier_stats[tier_name]["counts"].append(summary_wc)

        # Category
        meta_path = DATA_DIR / vid_id / "meta.json"
        cat = "unknown"
        if meta_path.exists():
            try:
                cat = json.loads(meta_path.read_text(encoding="utf-8")).get("category", "unknown")
            except Exception:
                pass
        cat_stats[cat]["total"] += 1

        vid_issues = []

        # Word count range
        if summary_wc < min_s:
            issues["too_short"].append(f"{vid_id}: {summary_wc}w (min {min_s}, tier={tier_name})")
            vid_issues.append("short")
        if summary_wc > max_s:
            issues["too_long"].append(f"{vid_id}: {summary_wc}w (max {max_s}, tier={tier_name})")
            vid_issues.append("long")

        summary_lower = summary.lower()

        # Academic framing
        found_academic = [m for m in ACADEMIC_MARKERS if m in summary_lower]
        if found_academic:
            issues["academic_framing"].append(f"{vid_id}: {found_academic}")
            vid_issues.append("academic")

        # Boilerplate
        found_boilerplate = [m for m in BOILERPLATE_MARKERS if m in summary_lower]
        if found_boilerplate:
            issues["boilerplate"].append(f"{vid_id}: {found_boilerplate}")
            vid_issues.append("boilerplate")

        # Vagueness
        found_vague = [m for m in VAGUENESS_MARKERS if m in summary_lower]
        if found_vague:
            issues["vague"].append(f"{vid_id}: {found_vague}")
            vid_issues.append("vague")

        # Formal tone
        found_formal = [m for m in FORMAL_SIGNALS if m in summary_lower]
        if len(found_formal) >= 2:
            issues["formal_tone"].append(f"{vid_id}: {found_formal}")
            vid_issues.append("formal")

        # Specificity
        spec_score, spec_indicators = check_specificity(summary)
        if spec_score < 0.5:
            issues["low_specificity"].append(f"{vid_id}: score={spec_score:.2f}")
            vid_issues.append("vague_spec")

        # Hallucination heuristic
        transcript_text = data.get("text", "")
        if transcript_text:
            hall_words = check_hallucination_heuristic(summary, transcript_text)
            if len(hall_words) >= 2:
                issues["possible_hallucination"].append(
                    f"{vid_id}: unknown terms: {', '.join(hall_words[:5])}"
                )
                vid_issues.append("hallucination")

        # Already flagged by generator
        if not data.get("summary_valid", True):
            issues["already_flagged"].append(
                f"{vid_id}: {data.get('summary_issues', [])}"
            )

        if vid_issues:
            cat_stats[cat]["issues"] += 1
        else:
            cat_stats[cat]["valid"] += 1

    # ── Report: Word count distribution ────────────────────────────────
    print(f"SUMMARY WORD COUNT DISTRIBUTION (n={len(all_summary_wcs)})")
    all_summary_wcs.sort()
    n = len(all_summary_wcs)
    print(f"  Mean:   {sum(all_summary_wcs)/n:.0f}")
    print(f"  Median: {all_summary_wcs[n//2]}")
    print(f"  Min:    {min(all_summary_wcs)}")
    print(f"  Max:    {max(all_summary_wcs)}")
    print(f"  P5:     {all_summary_wcs[int(n*0.05)]}")
    print(f"  P95:    {all_summary_wcs[int(n*0.95)]}")
    print()

    # Histogram
    bins = [0, 30, 60, 80, 100, 130, 160, 200, 999]
    labels = ["<30", "30-60", "60-80", "80-100", "100-130", "130-160", "160-200", ">200"]
    hist = Counter()
    for wc in all_summary_wcs:
        for i in range(len(bins) - 1):
            if bins[i] <= wc < bins[i + 1]:
                hist[labels[i]] += 1
                break

    print(f"  HISTOGRAM")
    for label in labels:
        count = hist.get(label, 0)
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"    {label:>8s}: {count:4d} {bar}")
    print()

    # Per-tier stats
    print(f"PER-TIER WORD COUNT STATS")
    print(f"  {'Tier':<8s} {'N':>5s} {'Mean':>6s} {'Min':>5s} {'Max':>5s} {'Target Range':>14s}")
    print(f"  {'-'*48}")
    for tier_name in ["short", "medium", "long"]:
        counts = tier_stats[tier_name]["counts"]
        if not counts:
            continue
        min_s, max_s = 0, 0
        for max_wc, ms, mxs, tn in LENGTH_TIERS:
            if tn == tier_name:
                min_s, max_s = ms, mxs
                break
        print(f"  {tier_name:<8s} {len(counts):>5d} {sum(counts)/len(counts):>6.0f} "
              f"{min(counts):>5d} {max(counts):>5d} {min_s:>5d}-{max_s:<5d}")
    print()

    # ── Report: Issues ─────────────────────────────────────────────────
    print(f"QUALITY ISSUES")
    total_issues = 0
    for issue_name, issue_list in issues.items():
        if issue_list:
            total_issues += len(issue_list)
            print(f"  {issue_name}: {len(issue_list)}")
            for item in issue_list[:3]:
                print(f"    - {item}")
            if len(issue_list) > 3:
                print(f"    ... and {len(issue_list) - 3} more")
    if total_issues == 0:
        print(f"  None — all summaries clean")
    print()

    # ── Report: Conversational tone check ──────────────────────────────
    conv_count = 0
    for _, data in has_summary:
        s_lower = data["summary"].lower()
        if any(sig in s_lower for sig in CONVERSATIONAL_SIGNALS):
            conv_count += 1

    conv_pct = conv_count / len(has_summary) * 100
    print(f"TONE ANALYSIS")
    print(f"  Conversational signals present: {conv_count}/{len(has_summary)} ({conv_pct:.0f}%)")
    formal_count = len(issues["formal_tone"])
    print(f"  Overly formal:                 {formal_count}/{len(has_summary)}")
    print()

    # ── Report: Near-duplicate detection ───────────────────────────────
    print(f"DUPLICATE DETECTION")
    dupes = find_near_duplicates(all_summaries_for_dupes)
    if dupes:
        print(f"  Near-duplicate pairs: {len(dupes)}")
        for a, b, overlap in dupes[:5]:
            print(f"    - {a} <-> {b} ({overlap} shared words)")
        if len(dupes) > 5:
            print(f"    ... and {len(dupes) - 5} more")
    else:
        print(f"  No near-duplicates found")
    print()

    # ── Report: Per-category breakdown ─────────────────────────────────
    print(f"PER-CATEGORY QUALITY")
    print(f"  {'Category':<10s} {'Total':>6s} {'Valid':>6s} {'Issues':>7s} {'Valid%':>7s}")
    print(f"  {'-'*40}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        pct = s["valid"] / s["total"] * 100 if s["total"] > 0 else 0
        flag = " ⚠" if pct < 90 else ""
        print(f"  {cat:<10s} {s['total']:>6d} {s['valid']:>6d} {s['issues']:>7d} {pct:>6.0f}%{flag}")
    print()

    # ── Sample outputs for manual inspection ───────────────────────────
    print(f"SAMPLE SUMMARIES (for manual inspection)")
    print(f"{'='*65}")

    import random
    random.seed(42)

    for tier_name in ["short", "medium", "long"]:
        tier_vids = [
            (vid_id, data) for vid_id, data in has_summary
            if get_tier(data.get("word_count", 0))[2] == tier_name
        ]
        if not tier_vids:
            continue

        sample = random.sample(tier_vids, min(args.samples, len(tier_vids)))
        print(f"\n--- {tier_name.upper()} TIER (transcript <={'2K' if tier_name=='short' else '8K' if tier_name=='medium' else '8K+'} words) ---")

        for vid_id, data in sample:
            wc = data.get("word_count", 0)
            swc = len(data["summary"].split())
            cat = "?"
            mp = DATA_DIR / vid_id / "meta.json"
            if mp.exists():
                try:
                    cat = json.loads(mp.read_text(encoding="utf-8")).get("category", "?")
                except Exception:
                    pass

            spec_score, _ = check_specificity(data["summary"])
            print(f"\n  [{vid_id}] cat={cat} transcript={wc:,}w summary={swc}w spec={spec_score:.1f}")
            print(f"  {data['summary']}")
    print()

    # ── Fix mode ───────────────────────────────────────────────────────
    # Collect all vid_ids with critical issues
    critical_ids = set()
    for issue_name in ["boilerplate", "academic_framing", "possible_hallucination"]:
        for entry in issues[issue_name]:
            vid_id = entry.split(":")[0].strip()
            critical_ids.add(vid_id)
    for entry in issues["too_short"]:
        vid_id = entry.split(":")[0].strip()
        critical_ids.add(vid_id)

    if args.fix and critical_ids:
        print(f"FIX MODE: Clearing {len(critical_ids)} summaries for re-generation...")
        cleared = 0
        for vid_id in critical_ids:
            tp = DATA_DIR / vid_id / "transcript.json"
            if tp.exists():
                data = json.loads(tp.read_text(encoding="utf-8"))
                for key in ["summary", "summary_word_count", "summary_valid", "summary_issues"]:
                    data.pop(key, None)
                tp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                cleared += 1
        print(f"  Cleared {cleared} summaries. Re-run summary_generator.py to regenerate.")
        print()
    elif critical_ids:
        print(f"  {len(critical_ids)} summaries have critical issues.")
        print(f"  Run with --fix to clear them for re-generation.")
        print()

    # ── Final verdict ──────────────────────────────────────────────────
    valid_count = len(has_summary) - len(critical_ids)
    valid_pct = valid_count / len(has_summary) * 100 if has_summary else 0

    print(f"{'='*65}")
    print(f"VERDICT")
    print(f"  Total summaries:     {len(has_summary)}")
    print(f"  Critical issues:     {len(critical_ids)}")
    print(f"  Clean for training:  {valid_count} ({valid_pct:.0f}%)")

    if valid_pct >= 95:
        print(f"  STATUS: ✓ PASS — proceed to pair mining (A4)")
    elif valid_pct >= 85:
        print(f"  STATUS: ~ MARGINAL — consider --fix and re-generating")
    else:
        print(f"  STATUS: ✗ FAIL — run --fix, then re-generate, then re-validate")
    print(f"{'='*65}")


if __name__ == "__main__":
    validate()