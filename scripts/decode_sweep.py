"""Sweep decoding configs on the 4 quality-probe samples to find one that
suppresses the Phase-3 token-loop artifact without killing content (the way
repetition_penalty=1.15 did)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import app as demo  # noqa: E402

CONFIGS = [
    {"name": "paper_default",  "max_new_tokens": 160, "repetition_penalty": 1.00, "no_repeat_ngram_size": 3},
    {"name": "short_budget",   "max_new_tokens":  96, "repetition_penalty": 1.00, "no_repeat_ngram_size": 3},
    {"name": "gentle_rep_pen", "max_new_tokens": 160, "repetition_penalty": 1.05, "no_repeat_ngram_size": 3},
    {"name": "gentle+ngram4",  "max_new_tokens": 160, "repetition_penalty": 1.05, "no_repeat_ngram_size": 4},
    {"name": "short+gentle",   "max_new_tokens": 120, "repetition_penalty": 1.05, "no_repeat_ngram_size": 4},
]


def has_loop_artifact(text: str) -> bool:
    """Heuristic: detect token-suffix repetition loops."""
    import re
    if re.search(r"(\w{3,})\1{2,}", text):  # "tion" "tion" "tion" etc.
        return True
    if re.search(r"\b(\w{3,})\b(?:\s+\1\b){2,}", text):  # "word word word"
        return True
    if re.search(r"(?:\bto\b\s*){5,}|(?:\btt+t+\b)", text):  # totot collapse
        return True
    return False


def main():
    samples = demo.load_video_samples(n=4)
    available = tuple(k for k, p in demo.ADAPTERS.items() if p.exists())
    model, tok, _ = demo.load_model_and_adapters(available)

    print(f"\n{'='*100}")
    print(f"{'sample':<14} {'config':<18} {'mode':<7} {'r1':<7} {'r2':<7} {'pvr':<6} {'words':<6} {'loop':<5}")
    print("=" * 100)

    summary = {}
    for cfg in CONFIGS:
        summary[cfg["name"]] = {"adp_loops": 0, "adp_r1_sum": 0.0}

    for i, s in enumerate(samples, 1):
        for cfg in CONFIGS:
            kwargs = {k: v for k, v in cfg.items() if k != "name"}

            base, _ = demo.generate(model, tok, s["source"], use_adapter=None, **kwargs)
            adp, _ = demo.generate(model, tok, s["source"], use_adapter="phase3", **kwargs)

            rb = demo.compute_rouge(base, s["reference"])
            ra = demo.compute_rouge(adp, s["reference"])
            sb = demo.style_metrics(base)
            sa = demo.style_metrics(adp)

            base_loop = has_loop_artifact(base)
            adp_loop = has_loop_artifact(adp)

            print(f"{s['id']:<14} {cfg['name']:<18} {'BASE':<7} "
                  f"{rb['rouge1']:<7.3f} {rb['rouge2']:<7.3f} {sb['passive_voice_pct']:<6.2f} "
                  f"{int(sb['word_count']):<6} {'Y' if base_loop else '.':<5}")
            print(f"{s['id']:<14} {cfg['name']:<18} {'PHASE3':<7} "
                  f"{ra['rouge1']:<7.3f} {ra['rouge2']:<7.3f} {sa['passive_voice_pct']:<6.2f} "
                  f"{int(sa['word_count']):<6} {'Y' if adp_loop else '.':<5}")

            summary[cfg["name"]]["adp_loops"] += int(adp_loop)
            summary[cfg["name"]]["adp_r1_sum"] += ra["rouge1"]

            if adp_loop:
                # Last 80 chars usually shows the collapse
                tail = adp[-80:].replace("\n", " ")
                print(f"   {' '*30} loop tail: ...{tail}")

        print("-" * 100)

    print("\n=== Aggregate (lower loop-count + higher R-1 = better) ===")
    print(f"{'config':<18} {'loops_seen':<12} {'mean_adp_R1':<12}")
    for name, agg in summary.items():
        n = len(samples)
        print(f"{name:<18} {agg['adp_loops']:<12} {agg['adp_r1_sum']/n:<12.3f}")


if __name__ == "__main__":
    main()
