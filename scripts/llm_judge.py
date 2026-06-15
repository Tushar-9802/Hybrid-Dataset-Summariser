"""
llm_judge.py — Blind pairwise LLM-as-judge for summary quality.

Compares two evaluation runs' generated summaries head-to-head. For each test
sample the judge sees the reference summary and two anonymised candidates
("Candidate A" / "Candidate B") and picks the better one.

Design (counters known LLM-judge failure modes):
  - Blind        : the judge never sees model/run identity, only A vs B.
  - Both orders  : every pair is judged twice with the candidates swapped;
                    a winner counts only if both orders agree, else TIE.
                    This neutralises pairwise position bias (arXiv:2406.12319).
  - Dual judge   : GPT-4o and Claude vote independently; agreement is reported.
  - Resumable    : append-only JSONL + seen-keys index — safe to interrupt.

Inputs are the `generations.jsonl` sidecars written by evaluate.py.

Usage:
    # offline dry-run — stub judge, no API key needed
    python scripts/llm_judge.py --run-a results/baseline --run-b results/phase3 \
        --judges stub --out results/judge/base_vs_phase3.jsonl --dry-run

    # real run (Sprint 5) — needs OPENAI_API_KEY and/or ANTHROPIC_API_KEY
    python scripts/llm_judge.py --run-a results/baseline --run-b results/phase3 \
        --judges gpt-4o,claude --out results/judge/base_vs_phase3.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OPENAI_MODEL_DEFAULT = "gpt-4o"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-6"


# ━━ PROMPT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JUDGE_PROMPT = """\
You are evaluating two candidate summaries of the same source document.
A reference (gold) summary is provided for guidance.

Reference summary:
{reference}

Candidate A:
{cand_a}

Candidate B:
{cand_b}

Which candidate is the better summary — more faithful to the reference, more \
fluent, and more informative? Answer with exactly one token: A, B, or TIE."""


def build_prompt(reference: str, cand_a: str, cand_b: str) -> str:
    return JUDGE_PROMPT.format(
        reference=reference.strip() or "(empty)",
        cand_a=cand_a.strip() or "(empty)",
        cand_b=cand_b.strip() or "(empty)",
    )


def parse_verdict(text: str) -> str:
    """Map a raw judge response to A / B / TIE / PARSE_ERROR."""
    t = (text or "").strip().upper()
    if t.startswith("TIE") or t == "T":
        return "TIE"
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    # last resort: scan for a standalone token
    for tok in ("TIE", "A", "B"):
        if tok in t.split():
            return tok
    return "PARSE_ERROR"


# ━━ JUDGES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _retry(fn, attempts: int = 3, base_delay: float = 2.0):
    """Call fn() with exponential backoff on any exception."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — transient API errors vary by SDK
            if i == attempts - 1:
                raise
            delay = base_delay * (2 ** i)
            log.warning(f"Judge call failed ({e}); retry {i + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


class StubJudge:
    """Offline judge — no API. Picks the candidate with higher unigram overlap
    with the reference, so a dry-run produces a non-degenerate distribution."""

    name = "stub"

    @staticmethod
    def _overlap(cand: str, reference: str) -> float:
        c, r = set(cand.lower().split()), set(reference.lower().split())
        if not c or not r:
            return 0.0
        return len(c & r) / len(c | r)

    def vote(self, reference: str, cand_a: str, cand_b: str) -> str:
        oa = self._overlap(cand_a, reference)
        ob = self._overlap(cand_b, reference)
        if abs(oa - ob) < 1e-6:
            return "TIE"
        return "A" if oa > ob else "B"


class OpenAIJudge:
    """GPT-4o judge via the OpenAI SDK. Needs OPENAI_API_KEY."""

    def __init__(self, model: str = OPENAI_MODEL_DEFAULT):
        from openai import OpenAI  # lazy — dry-run must not require the SDK
        self.model = model
        self.name = model
        self.client = OpenAI()

    def vote(self, reference: str, cand_a: str, cand_b: str) -> str:
        prompt = build_prompt(reference, cand_a, cand_b)

        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            return resp.choices[0].message.content

        return parse_verdict(_retry(_call))


class AnthropicJudge:
    """Claude judge via the Anthropic SDK. Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str = ANTHROPIC_MODEL_DEFAULT):
        import anthropic  # lazy
        self.model = model
        self.name = model
        self.client = anthropic.Anthropic()

    def vote(self, reference: str, cand_a: str, cand_b: str) -> str:
        prompt = build_prompt(reference, cand_a, cand_b)

        def _call():
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=8,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        return parse_verdict(_retry(_call))


def build_judges(spec: str, dry_run: bool,
                 openai_model: str, anthropic_model: str) -> list:
    """Resolve a --judges spec ('stub', 'gpt-4o,claude', ...) to judge objects."""
    judges = []
    for name in [s.strip().lower() for s in spec.split(",") if s.strip()]:
        if name == "stub":
            judges.append(StubJudge())
        elif name in ("gpt-4o", "gpt4o", "openai"):
            if dry_run:
                log.warning("--dry-run: skipping OpenAI judge")
                continue
            judges.append(OpenAIJudge(openai_model))
        elif name in ("claude", "anthropic"):
            if dry_run:
                log.warning("--dry-run: skipping Anthropic judge")
                continue
            judges.append(AnthropicJudge(anthropic_model))
        else:
            raise ValueError(f"Unknown judge: {name!r} (expected stub|gpt-4o|claude)")
    if not judges:
        raise ValueError("No judges resolved — for --dry-run use --judges stub")
    return judges


# ━━ DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_generations(run_path: Path) -> dict[str, dict]:
    """Load a run's generations.jsonl into {sample_id: {reference, generation, modality}}."""
    jsonl = run_path / "generations.jsonl" if run_path.is_dir() else run_path
    if not jsonl.exists():
        raise FileNotFoundError(
            f"No generations.jsonl at {jsonl} — run evaluate.py for that config first."
        )
    out = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["sample_id"]] = r
    return out


def load_seen(out_path: Path) -> set[tuple]:
    """Seen-keys index: (sample_id, judge) pairs already judged in out_path."""
    if not out_path.exists():
        return set()
    seen = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        seen.add((r["sample_id"], r["judge"]))
    return seen


# ━━ JUDGING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resolve_winner(v1: str, v2: str, run_a: str, run_b: str) -> tuple[str, bool]:
    """Translate the two ordered verdicts to a run-space winner.

    Order 1: Candidate A = run_a's generation.
    Order 2: Candidate A = run_b's generation (swapped).
    A winner counts only if both orders agree; otherwise TIE (position bias).
    """
    w1 = {"A": run_a, "B": run_b}.get(v1, "TIE")
    w2 = {"A": run_b, "B": run_a}.get(v2, "TIE")
    if w1 == w2:
        return w1, True
    return "TIE", False


def judge_pair(judge, sample_a: dict, sample_b: dict,
               run_a: str, run_b: str) -> dict:
    """Run one sample through one judge in both candidate orders."""
    reference = sample_a.get("reference", "")
    gen_a, gen_b = sample_a["generation"], sample_b["generation"]

    v1 = judge.vote(reference, gen_a, gen_b)          # A = run_a
    v2 = judge.vote(reference, gen_b, gen_a)          # A = run_b (swapped)
    winner, consistent = resolve_winner(v1, v2, run_a, run_b)

    return {
        "sample_id": sample_a["sample_id"],
        "modality": sample_a.get("modality", ""),
        "judge": judge.name,
        "run_a": run_a,
        "run_b": run_b,
        "verdict_order1": v1,
        "verdict_order2": v2,
        "winner": winner,
        "consistent": consistent,
    }


def aggregate(out_path: Path, run_a: str, run_b: str):
    """Print per-judge win counts and cross-judge agreement."""
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        log.info("No verdicts to aggregate.")
        return

    judges = sorted({r["judge"] for r in rows})
    print("\n" + "=" * 60)
    print(f"LLM-AS-JUDGE SUMMARY  --  {run_a}  vs  {run_b}")
    print("=" * 60)

    by_judge_sample = {}  # judge -> {sample_id: winner}
    for j in judges:
        jr = [r for r in rows if r["judge"] == j]
        wins_a = sum(r["winner"] == run_a for r in jr)
        wins_b = sum(r["winner"] == run_b for r in jr)
        ties = sum(r["winner"] == "TIE" for r in jr)
        inconsistent = sum(not r["consistent"] for r in jr)
        n = len(jr)
        print(f"\n[{j}]  n={n}")
        print(f"  {run_a:<28} wins: {wins_a:4d}  ({wins_a / n:.1%})")
        print(f"  {run_b:<28} wins: {wins_b:4d}  ({wins_b / n:.1%})")
        print(f"  {'TIE / inconsistent':<28}     : {ties:4d}  "
              f"({ties / n:.1%})  [{inconsistent} order-inconsistent]")
        by_judge_sample[j] = {r["sample_id"]: r["winner"] for r in jr}

    if len(judges) >= 2:
        j1, j2 = judges[0], judges[1]
        common = set(by_judge_sample[j1]) & set(by_judge_sample[j2])
        if common:
            agree = sum(by_judge_sample[j1][s] == by_judge_sample[j2][s] for s in common)
            print(f"\nCross-judge agreement ({j1} vs {j2}): "
                  f"{agree}/{len(common)}  ({agree / len(common):.1%})")
    print("=" * 60 + "\n")


# ━━ MAIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Blind pairwise LLM-as-judge")
    parser.add_argument("--run-a", required=True,
                        help="Run dir (or generations.jsonl path) for config A")
    parser.add_argument("--run-b", required=True,
                        help="Run dir (or generations.jsonl path) for config B")
    parser.add_argument("--out", required=True, help="Output verdicts JSONL path")
    parser.add_argument("--judges", default="gpt-4o,claude",
                        help="Comma list: stub | gpt-4o | claude")
    parser.add_argument("--openai-model", default=OPENAI_MODEL_DEFAULT)
    parser.add_argument("--anthropic-model", default=ANTHROPIC_MODEL_DEFAULT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Judge only the first N samples (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stub judge only — no API calls")
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    run_a_name = run_a.name if run_a.is_dir() else run_a.stem
    run_b_name = run_b.name if run_b.is_dir() else run_b.stem

    gens_a = load_generations(run_a)
    gens_b = load_generations(run_b)
    common_ids = sorted(set(gens_a) & set(gens_b))
    if not common_ids:
        log.error("No overlapping sample_ids between the two runs.")
        sys.exit(1)
    if args.limit > 0:
        common_ids = common_ids[:args.limit]
    log.info(f"{len(common_ids)} samples to judge ({run_a_name} vs {run_b_name})")

    judges = build_judges(args.judges, args.dry_run,
                          args.openai_model, args.anthropic_model)
    log.info(f"Judges: {[j.name for j in judges]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(out_path)
    if seen:
        log.info(f"Resuming: {len(seen)} (sample, judge) verdicts already done")

    n_written = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for sid in common_ids:
            for judge in judges:
                if (sid, judge.name) in seen:
                    continue
                row = judge_pair(judge, gens_a[sid], gens_b[sid],
                                 run_a_name, run_b_name)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()  # survive interruption mid-run
                n_written += 1

    log.info(f"Wrote {n_written} new verdicts to {out_path}")
    aggregate(out_path, run_a_name, run_b_name)


if __name__ == "__main__":
    main()
