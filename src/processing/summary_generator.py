"""
summary_generator.py — GPT-4o-mini Video Summary Generation
Generates reference summaries for validated video transcripts.

Folder: src/processing/

Design decisions:
  - Length-adaptive: summary length scales with transcript length
  - Three-window truncation: first 40% + middle 30% + last 30% (captures
    lecture intro, core content, and conclusion)
  - Few-shot examples anchor conversational tone across CS subdomains
  - Quality validation: word count range, no academic framing markers
  - Summaries written back into each transcript.json as "summary" field
  - Resume-safe: skips videos that already have a summary

Cost: ~$0.70-0.80 for 738 transcripts via GPT-4o-mini

Prerequisites:
    pip install openai

Run from repo root:
    set OPENAI_API_KEY=sk-...
    python src/processing/summary_generator.py --dry-run
    python src/processing/summary_generator.py
    python src/processing/summary_generator.py --concurrency 5
"""

import json
import argparse
import asyncio
import logging
import time
import sys
import os
from pathlib import Path
from typing import Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw/videos")
LOG_DIR = Path("logs")

MAX_TRANSCRIPT_WORDS = 6000

# Length tiers: (max_transcript_words, length_instruction, min_summary, max_summary)
LENGTH_TIERS = [
    (2000,  "Write 2-3 sentences (40-80 words).",   30,  100),
    (8000,  "Write 3-5 sentences (80-130 words).",   60,  160),
    (99999, "Write 5-7 sentences (120-180 words).",  90,  220),
]

# Academic framing markers — reject summaries containing these
ACADEMIC_MARKERS = [
    "this paper", "the authors", "we propose", "et al.", "[1]",
    "contributions include", "this work presents", "the study",
    "the proposed method", "the paper", "our approach", "we present",
    "this study", "the researchers", "our method",
]

# Pricing (GPT-4o-mini, per token)
PRICE_INPUT = 0.15 / 1_000_000
PRICE_OUTPUT = 0.60 / 1_000_000

# ── PROMPTS ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You summarize educational video lectures concisely. Your summaries "
    "sound like how a knowledgeable person would describe a video to a "
    "colleague — clear, direct, and specific about what was covered. "
    "Never use academic framing."
)

USER_PROMPT_TEMPLATE = """\
Summarize this educational video transcript. Be specific about the topics \
and techniques covered. Use a natural, conversational tone as if describing \
the video to a colleague.

Example 1 (ML/AI topic):
Transcript excerpt: "...today we're going to look at how transformers handle \
long sequences. The key idea is sparse attention — instead of attending to \
every token, you only attend to a subset..."
Summary: "The video walks through how transformers deal with long sequences, \
focusing on sparse attention mechanisms. It covers the computational savings \
from attending to token subsets rather than full sequences, and shows \
benchmark comparisons on document-length inputs."

Example 2 (Systems/SE topic):
Transcript excerpt: "...so the whole point of continuous integration is that \
every time you push code, the pipeline runs your tests automatically. We'll \
set up GitHub Actions with a Docker container..."
Summary: "This covers setting up a CI/CD pipeline using GitHub Actions and \
Docker. The walkthrough goes from writing a basic workflow YAML file to \
containerizing tests, with practical tips on caching dependencies to speed \
up build times."

Rules:
- {length_instruction}
- Focus on WHAT was taught — name specific algorithms, methods, or concepts
- No academic framing ("this paper", "the authors", "we propose", "the study")
- Be specific, not vague — "covers sorting algorithms including quicksort \
and mergesort" is better than "covers various algorithms"

Transcript:
{transcript}"""


# ── HELPERS ─────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "summary_generation.log"),
    ],
)
log = logging.getLogger(__name__)


def get_length_tier(word_count: int) -> tuple[str, int, int]:
    """Return (length_instruction, min_words, max_words) for transcript length."""
    for max_wc, instruction, min_s, max_s in LENGTH_TIERS:
        if word_count <= max_wc:
            return instruction, min_s, max_s
    return LENGTH_TIERS[-1][1], LENGTH_TIERS[-1][2], LENGTH_TIERS[-1][3]


def truncate_transcript(text: str, max_words: int = MAX_TRANSCRIPT_WORDS) -> str:
    """
    Three-window truncation: first 40% + middle 30% + last 30%.
    Captures lecture intro, core content, and wrap-up.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    head_size = int(max_words * 0.40)
    mid_size = int(max_words * 0.30)
    tail_size = max_words - head_size - mid_size

    total = len(words)
    mid_start = (total - mid_size) // 2

    head = words[:head_size]
    middle = words[mid_start:mid_start + mid_size]
    tail = words[-tail_size:]

    return (
        " ".join(head)
        + "\n\n[...]\n\n"
        + " ".join(middle)
        + "\n\n[...]\n\n"
        + " ".join(tail)
    )


def validate_summary(summary: str, min_words: int, max_words: int) -> tuple[bool, list[str]]:
    """Validate summary against quality gates."""
    issues = []
    wc = len(summary.split())

    if wc < min_words:
        issues.append(f"Too short: {wc} words (min {min_words})")
    if wc > max_words:
        issues.append(f"Too long: {wc} words (max {max_words})")

    summary_lower = summary.lower()
    for marker in ACADEMIC_MARKERS:
        if marker in summary_lower:
            issues.append(f"Academic framing: '{marker}'")

    return len(issues) == 0, issues


async def generate_summary(
    client: AsyncOpenAI,
    vid_id: str,
    transcript: str,
    transcript_word_count: int,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[str, Optional[str], float, float, bool, list[str]]:
    """
    Generate and validate summary for one video.
    Returns (vid_id, summary, input_tokens, output_tokens, is_valid, issues).
    """
    length_instruction, min_words, max_words = get_length_tier(transcript_word_count)
    truncated = truncate_transcript(transcript)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        length_instruction=length_instruction,
        transcript=truncated,
    )

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )

                summary = response.choices[0].message.content.strip()
                if summary.startswith('"') and summary.endswith('"'):
                    summary = summary[1:-1].strip()

                in_tok = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens

                is_valid, issues = validate_summary(summary, min_words, max_words)
                return vid_id, summary, in_tok, out_tok, is_valid, issues

            except Exception as e:
                wait = 2 ** (attempt + 1)
                log.warning(f"  {vid_id} attempt {attempt+1}: {e}. Retry in {wait}s")
                await asyncio.sleep(wait)

    return vid_id, None, 0, 0, False, ["API failure after retries"]


async def run(model: str, concurrency: int, dry_run: bool):
    if not DATA_DIR.exists():
        log.error(f"{DATA_DIR} does not exist")
        return

    all_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    to_process = []
    already_done = 0

    for vid_dir in all_dirs:
        transcript_path = vid_dir / "transcript.json"
        if not transcript_path.exists():
            continue

        data = json.loads(transcript_path.read_text(encoding="utf-8"))

        if data.get("summary"):
            already_done += 1
            continue

        text = data.get("text", "")
        word_count = data.get("word_count", len(text.split()))
        if not text or word_count < 100:
            continue

        to_process.append((vid_dir.name, text, word_count))

    log.info(f"Already summarized: {already_done}")
    log.info(f"To process:         {len(to_process)}")

    if not to_process:
        log.info("Nothing to do.")
        return

    # Length tier distribution
    tier_counts = [0, 0, 0]
    for _, _, wc in to_process:
        if wc <= 2000:
            tier_counts[0] += 1
        elif wc <= 8000:
            tier_counts[1] += 1
        else:
            tier_counts[2] += 1
    log.info(f"Length tiers: short(<2K)={tier_counts[0]}  "
             f"medium(2-8K)={tier_counts[1]}  long(>8K)={tier_counts[2]}")

    # Cost estimate
    est_input = sum(min(wc, MAX_TRANSCRIPT_WORDS) * 1.3 + 350 for _, _, wc in to_process)
    est_output = sum(
        80 if wc <= 2000 else 130 if wc <= 8000 else 180
        for _, _, wc in to_process
    )
    est_cost = est_input * PRICE_INPUT + est_output * PRICE_OUTPUT
    log.info(f"Estimated cost: ${est_cost:.2f}  "
             f"(~{est_input/1e6:.2f}M in, ~{est_output/1e6:.3f}M out)")

    if dry_run:
        log.info("DRY RUN — no API calls.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    log.info(f"Generating (model={model}, concurrency={concurrency})...")
    start = time.time()

    total_in = 0
    total_out = 0
    succeeded = 0
    failed = 0
    invalid = 0
    failed_ids = []

    chunk_size = 50
    for cs in range(0, len(to_process), chunk_size):
        chunk = to_process[cs:cs + chunk_size]

        tasks = [
            generate_summary(client, vid_id, text, wc, model, semaphore)
            for vid_id, text, wc in chunk
        ]
        results = await asyncio.gather(*tasks)

        for vid_id, summary, in_tok, out_tok, is_valid, issues in results:
            total_in += in_tok
            total_out += out_tok

            if summary is None:
                failed += 1
                failed_ids.append(vid_id)
                continue

            transcript_path = DATA_DIR / vid_id / "transcript.json"
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
            data["summary"] = summary
            data["summary_word_count"] = len(summary.split())
            data["summary_valid"] = is_valid
            if issues:
                data["summary_issues"] = issues
                invalid += 1

            transcript_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            succeeded += 1

        done = cs + len(chunk)
        elapsed = time.time() - start
        cost = total_in * PRICE_INPUT + total_out * PRICE_OUTPUT
        rate = done / elapsed * 60 if elapsed > 0 else 0
        log.info(
            f"  [{done}/{len(to_process)}] "
            f"{rate:.0f}/min  ${cost:.3f}  "
            f"ok={succeeded} fail={failed} invalid={invalid}"
        )

    total_time = (time.time() - start) / 60
    total_cost = total_in * PRICE_INPUT + total_out * PRICE_OUTPUT

    log.info(f"{'='*55}")
    log.info(f"COMPLETE")
    log.info(f"  Succeeded:      {succeeded}")
    log.info(f"  Failed:         {failed}")
    log.info(f"  Quality issues: {invalid}")
    log.info(f"  Time:           {total_time:.1f} min")
    log.info(f"  Tokens:         {total_in:,} in / {total_out:,} out")
    log.info(f"  Cost:           ${total_cost:.3f}")
    log.info(f"{'='*55}")

    if failed_ids:
        log.info(f"Failed IDs: {failed_ids[:20]}")
    if invalid:
        log.info(
            f"{invalid} summaries flagged (saved but summary_valid=false). "
            f"Inspect and re-run if needed."
        )


def main():
    parser = argparse.ArgumentParser(description="[A3] Generate video summaries")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost, no API calls")
    args = parser.parse_args()

    asyncio.run(run(args.model, args.concurrency, args.dry_run))


if __name__ == "__main__":
    main()