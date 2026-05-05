"""Headless smoke test for app.py model + adapter swap path.

Runs base + phase3 generation on a single test-split video transcript and
prints both summaries plus their style metrics. Exits non-zero on failure.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import app as demo  # noqa: E402  (must follow path setup)


def main():
    print(f"[smoke] HDF5 exists: {demo.HDF5_PATH.exists()}")
    print(f"[smoke] Adapters present: {[k for k, p in demo.ADAPTERS.items() if p.exists()]}")

    samples = demo.load_video_samples(n=4)
    print(f"[smoke] Loaded {len(samples)} test-split video samples")
    if not samples:
        print("[smoke] FAIL: no samples")
        sys.exit(1)

    sample = samples[0]
    print(f"[smoke] Using sample id={sample['id']}, source_len={len(sample['source'])} chars")

    available = tuple(k for k, p in demo.ADAPTERS.items() if p.exists())
    model, tok, loaded = demo.load_model_and_adapters(available)
    print(f"[smoke] Loaded adapters: {loaded}")

    base_summary, base_t = demo.generate(
        model, tok, sample["source"], use_adapter=None, max_new_tokens=120,
    )
    print(f"[smoke] BASE   ({base_t:.1f}s): {base_summary[:300]}")

    adp_summary, adp_t = demo.generate(
        model, tok, sample["source"], use_adapter="phase3", max_new_tokens=120,
    )
    print(f"[smoke] PHASE3 ({adp_t:.1f}s): {adp_summary[:300]}")

    base_style = demo.style_metrics(base_summary)
    adp_style = demo.style_metrics(adp_summary)
    print(f"[smoke] BASE   style: pvr={base_style['passive_voice_pct']:.3f} "
          f"nom={base_style['nominalization_pct']:.3f} "
          f"ttr={base_style['type_token_ratio']:.3f} "
          f"asl={base_style['avg_sent_len']:.1f}")
    print(f"[smoke] PHASE3 style: pvr={adp_style['passive_voice_pct']:.3f} "
          f"nom={adp_style['nominalization_pct']:.3f} "
          f"ttr={adp_style['type_token_ratio']:.3f} "
          f"asl={adp_style['avg_sent_len']:.1f}")

    rouge_b = demo.compute_rouge(base_summary, sample["reference"])
    rouge_a = demo.compute_rouge(adp_summary, sample["reference"])
    print(f"[smoke] BASE   rouge: r1={rouge_b['rouge1']:.3f} r2={rouge_b['rouge2']:.3f} rL={rouge_b['rougeL']:.3f}")
    print(f"[smoke] PHASE3 rouge: r1={rouge_a['rouge1']:.3f} r2={rouge_a['rouge2']:.3f} rL={rouge_a['rougeL']:.3f}")

    if base_summary == adp_summary:
        print("[smoke] WARN: base and adapter produced identical output — adapter may not be applied.")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
