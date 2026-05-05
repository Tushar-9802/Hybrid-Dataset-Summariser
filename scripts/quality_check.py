"""Generation-quality probe.

Runs base + phase3 on a handful of test-split videos, dumps full outputs to
results/demo_quality.md and prints a compact table to stdout. Used to vet the
adapter's qualitative behavior before showing the demo.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import app as demo  # noqa: E402

OUT = REPO / "results" / "demo_quality.md"
N_SAMPLES = 4
MAX_TOKENS = 160


def fmt_metrics(s):
    return (f"PVR={s['passive_voice_pct']:.2f}  "
            f"Nom={s['nominalization_pct']:.3f}  "
            f"TTR={s['type_token_ratio']:.3f}  "
            f"ASL={s['avg_sent_len']:.1f}  "
            f"len={int(s['word_count'])}w")


def fmt_rouge(r):
    return f"R1={r['rouge1']:.3f}  R2={r['rouge2']:.3f}  RL={r['rougeL']:.3f}"


def main():
    samples = demo.load_video_samples(n=N_SAMPLES)
    if len(samples) < N_SAMPLES:
        print(f"[qc] only {len(samples)} samples available")

    available = tuple(k for k, p in demo.ADAPTERS.items() if p.exists())
    model, tok, loaded = demo.load_model_and_adapters(available)
    print(f"[qc] adapters loaded: {loaded}")
    print(f"[qc] generating on {len(samples)} samples (max_new_tokens={MAX_TOKENS})\n")

    md = ["# Generation-quality check — base vs phase3 adapter", ""]
    md.append(f"Generated on {len(samples)} test-split videos. "
              f"Decoding: greedy, max_new_tokens={MAX_TOKENS}, "
              "no_repeat_ngram_size=3.")
    md.append("")

    rows = []
    for i, s in enumerate(samples, 1):
        print(f"=== Sample {i}: {s['id']} ===")
        base, base_t = demo.generate(model, tok, s["source"],
                                     use_adapter=None, max_new_tokens=MAX_TOKENS)
        adp, adp_t = demo.generate(model, tok, s["source"],
                                   use_adapter="phase3", max_new_tokens=MAX_TOKENS)

        bs = demo.style_metrics(base)
        as_ = demo.style_metrics(adp)
        rb = demo.compute_rouge(base, s["reference"])
        ra = demo.compute_rouge(adp, s["reference"])

        rows.append({
            "id": s["id"], "base_t": base_t, "adp_t": adp_t,
            "rouge1": (rb["rouge1"], ra["rouge1"]),
            "rouge2": (rb["rouge2"], ra["rouge2"]),
            "pvr": (bs["passive_voice_pct"], as_["passive_voice_pct"]),
            "nom": (bs["nominalization_pct"], as_["nominalization_pct"]),
        })

        print(f"  REF      : {s['reference'][:200]}...")
        print(f"  BASE   ({base_t:5.1f}s | {fmt_metrics(bs)} | {fmt_rouge(rb)})")
        print(f"           {base[:300]}")
        print(f"  PHASE3 ({adp_t:5.1f}s | {fmt_metrics(as_)} | {fmt_rouge(ra)})")
        print(f"           {adp[:300]}")
        print()

        # Markdown dump (full outputs)
        md.append(f"## Sample {i} — `{s['id']}`")
        md.append("")
        md.append("**Source transcript (truncated):**")
        md.append(f"> {s['source'][:600]}{'...' if len(s['source'])>600 else ''}")
        md.append("")
        md.append("**Gold reference:**")
        md.append(f"> {s['reference']}")
        md.append("")
        md.append(f"**Base Mistral-7B** ({base_t:.1f}s · {fmt_metrics(bs)} · {fmt_rouge(rb)}):")
        md.append("")
        md.append(f"```\n{base}\n```")
        md.append("")
        md.append(f"**+ Phase-3 adapter** ({adp_t:.1f}s · {fmt_metrics(as_)} · {fmt_rouge(ra)}):")
        md.append("")
        md.append(f"```\n{adp}\n```")
        md.append("")

    # Write the markdown dump first so a print failure can't lose it
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[qc] full outputs written to: {OUT}\n")

    # Aggregate (ASCII-only, Windows cp1252-safe)
    print("=" * 78)
    print(f"{'sample':<14}  {'R1 base->adp':<18}  {'R2 base->adp':<18}  {'PVR base->adp':<18}")
    for r in rows:
        print(f"{r['id']:<14}  "
              f"{r['rouge1'][0]:.3f} -> {r['rouge1'][1]:.3f}    "
              f"{r['rouge2'][0]:.3f} -> {r['rouge2'][1]:.3f}    "
              f"{r['pvr'][0]:.2f} -> {r['pvr'][1]:.2f}")


if __name__ == "__main__":
    main()
