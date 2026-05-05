"""Rank all 75 test-split videos by adapter R-1 lift (already-computed in
results/baseline/ + results/phase3/), then regenerate the top N with
beam=4 (paper-faithful) to confirm they're loop-artifact-free, and emit
a curated picks list."""

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import app as demo  # noqa: E402

TOP_N = 12      # regenerate top N by R-1 lift
BEAM = 4        # match scripts/evaluate.py default
MAX_TOKENS = 200


def load_csv(path):
    return {r["sample_id"]: r for r in csv.DictReader(open(path))
            if r["modality"] == "video"}


def has_loop_artifact(text: str) -> bool:
    """Detect VISIBLE token-loop artifacts only.

    - Char-level subpiece loop: e.g. 'tokenizationizationization' — same 3+ char
      sequence repeated 2+ extra times with no separator between repeats.
    - Word-level loop: same 3+ char word repeated 3+ times in a row (with
      whitespace), to avoid flagging natural repeats like 'in in in' (2 char) or
      'on on' (2 occurrences only).
    - Total collapse: 't' repeated 4+ times alone.
    """
    if re.search(r"([A-Za-z]{3,})\1{2,}", text):  # 'tion' x3+ no space
        return True
    if re.search(r"\b([A-Za-z]{3,})\b(?:\s+\1\b){2,}", text):  # 'word word word'
        return True
    if re.search(r"\btt+t+\b", text):  # totot collapse
        return True
    return False


def safe_tail(text: str, n: int = 100) -> str:
    """Return last n chars of text, ASCII-safe for cp1252 console."""
    return text[-n:].encode("ascii", "replace").decode("ascii").replace("\n", " ")


def main():
    base = load_csv(REPO / "results/baseline/per_sample.csv")
    adp = load_csv(REPO / "results/phase3/per_sample.csv")
    common = sorted(set(base) & set(adp))

    ranked = []
    for sid in common:
        b1 = float(base[sid]["rouge1"])
        a1 = float(adp[sid]["rouge1"])
        ranked.append({"id": sid, "b_r1": b1, "a_r1": a1, "lift": a1 - b1})
    ranked.sort(key=lambda r: r["lift"], reverse=True)
    top = ranked[:TOP_N]

    print(f"Regenerating top {TOP_N} by R-1 lift "
          f"(paper-faithful: beam={BEAM}, max_new_tokens={MAX_TOKENS})\n")

    samples = demo.load_video_samples(n=128)  # full test split
    by_id = {s["id"]: s for s in samples}

    available = tuple(k for k, p in demo.ADAPTERS.items() if p.exists())
    model, tok, _ = demo.load_model_and_adapters(available)

    # Monkey-patch generate to use beam search (paper-faithful)
    def beam_generate(model, tokenizer, source, *, use_adapter, max_new_tokens):
        import torch
        from contextlib import nullcontext
        from peft import PeftModel
        prompt = demo.PROMPT_TEMPLATE.format(source=source)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        target = next(model.parameters()).device
        inputs = {k: v.to(target) for k, v in inputs.items()}

        is_peft = isinstance(model, PeftModel)
        if is_peft and use_adapter is not None:
            model.set_adapter(use_adapter)
        ctx = model.disable_adapter() if (is_peft and use_adapter is None) else nullcontext()

        import time as _t
        t0 = _t.perf_counter()
        with torch.no_grad(), ctx:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=BEAM,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = _t.perf_counter() - t0
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip(), elapsed

    out_md = ["# Top adapter wins — paper-faithful regeneration (beam=4)", ""]
    out_md.append(f"Ranked top-{TOP_N} test-split videos by R-1 lift (adapter - base) "
                  f"using cached eval CSVs, then regenerated with beam={BEAM} to verify "
                  "loop-artifact-free output.")
    out_md.append("")

    clean_winners = []
    for i, r in enumerate(top, 1):
        sid = r["id"]
        if sid not in by_id:
            print(f"[{i:2d}] {sid}: missing from HDF5 sample window — skipping")
            continue
        s = by_id[sid]

        base_out, base_t = beam_generate(model, tok, s["source"], use_adapter=None, max_new_tokens=MAX_TOKENS)
        adp_out, adp_t = beam_generate(model, tok, s["source"], use_adapter="phase3", max_new_tokens=MAX_TOKENS)

        rb = demo.compute_rouge(base_out, s["reference"])
        ra = demo.compute_rouge(adp_out, s["reference"])
        sb = demo.style_metrics(base_out)
        sa = demo.style_metrics(adp_out)
        loop = has_loop_artifact(adp_out)

        flag = "X loop" if loop else "OK"
        print(f"[{i:2d}] {sid}  csv-lift={r['lift']:+.3f}  "
              f"now: base R1={rb['rouge1']:.3f} -> adp R1={ra['rouge1']:.3f}  "
              f"({adp_t:.1f}s, {flag})")
        if loop:
            print(f"      tail: ...{safe_tail(adp_out, 120)}")

        out_md.append(f"## {i}. `{sid}`  -- lift={r['lift']:+.3f}  "
                      f"({'OK' if not loop else 'LOOP ARTIFACT - SKIP'})")
        out_md.append("")
        out_md.append(f"**Source (truncated):** {s['source'][:300]}...")
        out_md.append("")
        out_md.append(f"**Gold:** {s['reference']}")
        out_md.append("")
        out_md.append(f"**Base** ({base_t:.1f}s, R-1={rb['rouge1']:.3f}, PVR={sb['passive_voice_pct']:.2f}):")
        out_md.append(f"```\n{base_out}\n```")
        out_md.append("")
        out_md.append(f"**Phase-3** ({adp_t:.1f}s, R-1={ra['rouge1']:.3f}, PVR={sa['passive_voice_pct']:.2f}):")
        out_md.append(f"```\n{adp_out}\n```")
        out_md.append("")

        if not loop:
            clean_winners.append({
                "id": sid,
                "csv_lift_r1": r["lift"],
                "regen_base_r1": rb["rouge1"],
                "regen_adp_r1": ra["rouge1"],
                "regen_lift_r1": ra["rouge1"] - rb["rouge1"],
                "base_pvr": sb["passive_voice_pct"],
                "adp_pvr": sa["passive_voice_pct"],
                "preview": s["source"][:80],
            })

    OUT = REPO / "results" / "top_winners.md"
    OUT.write_text("\n".join(out_md), encoding="utf-8")
    print(f"\nfull dump -> {OUT}")

    # Output JSON for the curated_picks list
    SUM = REPO / "results" / "clean_winners.json"
    SUM.write_text(json.dumps(clean_winners, indent=2), encoding="utf-8")
    print(f"clean winners -> {SUM} ({len(clean_winners)} videos)")


if __name__ == "__main__":
    main()
