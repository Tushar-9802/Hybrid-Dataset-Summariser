"""
app.py — Streamlit demo: Base Mistral-7B vs Phase-3 "Science" Adapter

Two-way comparison on the same input. Loads base Mistral-7B-v0.1 (4-bit NF4)
once, attaches the Phase 3 LoRA adapter, and uses PEFT's disable_adapter()
context to switch between paths without reloading.

Run:
    streamlit run app.py

Hardware target: laptop RTX 4070 (8 GB VRAM) — Mistral-7B 4-bit ~5 GB +
LoRA r=32 ~150 MB + activations fits comfortably.
"""

import re
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REPO_ROOT = Path(__file__).parent
HDF5_PATH = REPO_ROOT / "data" / "hdf5" / "engineering.h5"
MODEL_ID = "mistralai/Mistral-7B-v0.1"

# Each adapter resolves to a local checkpoint if present, otherwise to the HF
# Hub repo (auto-downloaded by PeftModel.from_pretrained). Phase 1 has no
# published HF mirror; it loads only when local checkpoint exists.
ADAPTERS = {
    "phase1": {
        "local": REPO_ROOT / "checkpoints" / "phase1" / "final",
        "hub": None,
    },
    "phase2": {
        "local": REPO_ROOT / "checkpoints" / "phase2" / "final",
        "hub": "Tushar9802/hybrid-summariser-phase2-lora",
    },
    "phase3": {
        "local": REPO_ROOT / "checkpoints" / "phase3" / "final",
        "hub": "Tushar9802/hybrid-summariser-crossmodal-lora",
    },
}


def resolve_adapter(key: str) -> str | None:
    """Return the path or HF Hub ID for an adapter, or None if unavailable."""
    cfg = ADAPTERS.get(key)
    if not cfg:
        return None
    local = cfg["local"]
    if local and Path(local).exists():
        return str(local)
    if cfg["hub"]:
        return cfg["hub"]
    return None


def adapter_source_label(key: str) -> str:
    """Human-readable indicator of where this adapter comes from."""
    cfg = ADAPTERS.get(key, {})
    if cfg.get("local") and Path(cfg["local"]).exists():
        return "local"
    if cfg.get("hub"):
        return f"HF Hub: {cfg['hub']}"
    return "unavailable"

PROMPT_TEMPLATE = "Summarize the following text.\n\nText: {source}\n\nSummary:"

NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism",
)

DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_NUM_VIDEO_EXAMPLES = 128  # > 75 → loads full test split (no stride pruning)

# Test-split video examples shown in the app. Picked across the seven CS/Eng
# subdomains the dataset covers (cs.AI, cs.CL, cs.LG, cs.CV, cs.RO, cs.SE, cs.DS).
# Aggregate context: across all 75 test-split videos, 58 (77%) are adapter wins
# by R-1 lift, 14 marginal, 3 losses. Full per-video numbers in
# results/top_winners.md and results/phase3/per_sample.csv.
EXAMPLES = [
    {
        "id": "p_VxqEBiNiA",
        "label": "KV-charts / Boolean algebra (R-1: 0.12 → 0.53, +0.41)",
        "blurb": "Base outputs literal numerical gibberish "
                 "(\"12 13 14 15 16 17 18 19 20 21 22...\"); adapter recovers "
                 "into a topic-correct K-map walkthrough naming x AND y, "
                 "x OR y, NOT x. Most striking single contrast.",
    },
    {
        "id": "1GYv4KxL8JQ",
        "label": "Incidence matrices in graph theory (R-1: 0.18 → 0.57, +0.40)",
        "blurb": "Base reads like the lecturer's literal procedural narration "
                 "(\"plus one will come in row 1, row 2...\"); adapter shifts "
                 "to abstractive third-person and surfaces the structural "
                 "claim (\"each column will sum to zero\").",
    },
    {
        "id": "0Eix0yYVapw",
        "label": "Basis-path testing / cyclomatic complexity (R-1: 0.21 → 0.57, +0.37)",
        "blurb": "Base produces stream-of-consciousness number fragments "
                 "(\"5 plus 4 is 9, 9 plus 6 is 15...\"); adapter delivers "
                 "the full method outline including the E - n + 2P formula.",
    },
    {
        "id": "bxYrYicHtIg",
        "label": "PID controller theory (R-1: 0.28 → 0.55, +0.31)",
        "blurb": "Base is 1st-person walkthrough (\"we have a muscle, we "
                 "want to stretch it...\"); adapter produces a 3rd-person "
                 "lecture summary covering Kp, Ki, Kd roles and tuning "
                 "tradeoffs.",
    },
    {
        "id": "LZz3TuTDAoA",
        "label": "ROS gmapping for autonomous robot navigation (R-1: 0.25 → 0.60, +0.26)",
        "blurb": "Base = step-by-step config instructions; adapter cleanly "
                 "extracts the three-stage structure (mapping → localization "
                 "→ navigation) of the gmapping pipeline.",
    },
]
EXAMPLE_IDS = {p["id"] for p in EXAMPLES}

# ━━ MODEL LOADING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner="Loading Mistral-7B (4-bit NF4) — first run downloads ~5 GB...")
def load_model_and_adapters(adapter_keys: tuple[str, ...]):
    """
    Load Mistral-7B-v0.1 once with 4-bit NF4 quantization, attach all
    requested LoRA adapters under their respective names. Returns
    (model, tokenizer, loaded_adapter_names).

    Use model.disable_adapter() context for the base path; model.set_adapter(name)
    to switch which adapter is active.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    loaded = []
    model = None
    for key in adapter_keys:
        src = resolve_adapter(key)
        if src is None:
            continue
        if model is None:
            model = PeftModel.from_pretrained(base, src, adapter_name=key)
        else:
            model.load_adapter(src, adapter_name=key)
        loaded.append(key)

    if model is None:
        # No adapters available — return raw base for "base only" mode
        model = base

    model.eval()
    return model, tokenizer, loaded


# ━━ HDF5 SAMPLE LOADING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner="Loading test-split video samples...")
def load_video_samples(n: int = DEFAULT_NUM_VIDEO_EXAMPLES):
    """Load N test-split video samples from HDF5: source transcript + reference summary."""
    import h5py
    from transformers import AutoTokenizer

    if not HDF5_PATH.exists():
        return []

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    samples = []
    with h5py.File(str(HDF5_PATH), "r") as hf:
        grp = hf["videos"]
        splits = grp["split"][:]
        test_idx = np.where(splits == 2)[0]

        # Take a deterministic spread across the test split
        if len(test_idx) > n:
            stride = max(1, len(test_idx) // n)
            picks = test_idx[::stride][:n]
        else:
            picks = test_idx

        for idx in picks:
            sid = grp["sample_id"][idx]
            if isinstance(sid, bytes):
                sid = sid.decode("utf-8")

            ids = grp["input_ids"][idx]
            mask = grp["attention_mask"][idx]
            real_len = int(mask.sum())
            source = tok.decode(ids[:real_len].tolist(), skip_special_tokens=True).strip()

            label_ids = grp["labels"][idx]
            label_real = label_ids[label_ids != pad_id].tolist()
            reference = tok.decode(label_real, skip_special_tokens=True).strip()

            if source and reference:
                samples.append({
                    "id": sid,
                    "source": source,
                    "reference": reference,
                    "preview": (source[:80] + "...") if len(source) > 80 else source,
                })
    return samples


# ━━ GENERATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate(model, tokenizer, source_text: str, *,
             use_adapter: str | None,
             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
             repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 3,
             num_beams: int = 4):
    """
    Generate one summary. If use_adapter is None, runs base via disable_adapter().
    Otherwise, calls model.set_adapter(use_adapter) before generation.
    """
    import torch
    from contextlib import nullcontext
    from peft import PeftModel

    prompt = PROMPT_TEMPLATE.format(source=source_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    is_peft = isinstance(model, PeftModel)
    if is_peft and use_adapter is not None:
        model.set_adapter(use_adapter)

    ctx = model.disable_adapter() if (is_peft and use_adapter is None) else nullcontext()

    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=(num_beams > 1),
            do_sample=False,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    summary = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return summary, elapsed


# ━━ STYLE METRICS (regex/suffix only — no spaCy dep) ━━━━━━━━━━━━━━━━━━━━━━━━━

PASSIVE_RE = re.compile(
    r"\b(?:is|was|were|been|being|are|am|get|got|gets)\s+"
    r"(?:\w+\s+){0,3}"
    r"(?:\w+ed|written|shown|given|taken|made|done|seen|known|found)\b",
    re.IGNORECASE,
)


def passive_voice_pct(text: str) -> float:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s.split()) >= 2]
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if PASSIVE_RE.search(s))
    return hits / len(sentences)


def nominalization_pct(text: str) -> float:
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text) if len(w) > 3]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.endswith(NOMINALIZATION_SUFFIXES))
    return hits / len(words)


def type_token_ratio(text: str) -> float:
    toks = text.lower().split()
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def avg_sentence_length(text: str) -> float:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s.split()) >= 2]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


@st.cache_resource
def get_rouge_scorer():
    from rouge_score.rouge_scorer import RougeScorer
    return RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(generated: str, reference: str) -> dict[str, float]:
    if not reference or not generated:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = get_rouge_scorer().score(reference, generated)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }


def style_metrics(text: str) -> dict[str, float]:
    return {
        "passive_voice_pct": passive_voice_pct(text),
        "nominalization_pct": nominalization_pct(text),
        "type_token_ratio": type_token_ratio(text),
        "avg_sent_len": avg_sentence_length(text),
        "word_count": float(len(text.split())),
    }


# ━━ UI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Hybrid-Dataset Summariser — Demo",
    layout="wide",
)

st.title("Hybrid-Dataset Summariser — Adapter Comparison")
st.caption(
    "Base **Mistral-7B-v0.1** (4-bit NF4) vs the Phase-3 LoRA adapter "
    "(curriculum + LoRA+ + OPLoRA + EWC + CrossCLR). "
    "Same input, two outputs. Across all 75 test-split videos: "
    "**58 (77%) adapter wins** by R-1 lift, 14 marginal, 3 losses."
)

with st.expander("About this demo", expanded=False):
    st.markdown(
        """
**The problem.** Fine-tuning Mistral-7B on 25 K academic papers improved
paper summarization (+6–8% ROUGE-1) but degraded video summarization by
14–50%. Adapters entangled domain knowledge with academic style conventions
(passive voice +112%, nominalization +85%) — outputs like *"this paper
presents"* on conversational video content. (IMPACT 2025, Springer.)

**The fix.** A 3-phase curriculum that progressively activates protection
mechanisms:
- **Phase 1** — LoRA+ asymmetric LRs on papers only (domain knowledge).
- **Phase 2** — OPLoRA orthogonal projection + EWC Fisher penalty
  (preserve base subspace; no catastrophic forgetting).
- **Phase 3** — CrossCLR contrastive alignment + tighter EWC + full
  composite loss (video specialization).

**What you're looking at.**
- *Left:* base Mistral-7B zero-shot summary.
- *Right:* same input, with the chosen LoRA adapter active.
- *Style metrics:* PVR ↓ and Nominalization ↓ are *good* — they show the
  adapter isn't leaking academic register. TTR ↑ = lexical diversity ↑.
- *ROUGE:* per-sample lift over base measures content recovery against
  the gold summary.

**Caveats.**
- Decoding is paper-faithful (beam search, no_repeat_ngram_size=3,
  matches `scripts/evaluate.py`). The Phase-3 adapter sometimes produces
  visible token-suffix repetition (e.g. *"matrix matrix matrix"*) — a
  known artifact of small-rank LoRA on a 4-bit base at deterministic
  decode. Per-video numbers in `results/phase3/per_sample.csv`; full
  inspection of top-12 wins in `results/top_winners.md`.
- Test split is **CS/Engineering only** (cs.AI, cs.CL, cs.LG, cs.CV,
  cs.RO, cs.SE, cs.DS) — same domain the adapter trained on. Out-of-domain
  videos (cooking, politics, music) will not work.
"""
    )

# Sidebar — config
with st.sidebar:
    st.header("Config")
    available_adapters = [k for k in ADAPTERS if resolve_adapter(k) is not None]
    if not available_adapters:
        st.error(
            "No adapters available locally or on HF Hub. "
            "Either place checkpoints under `checkpoints/<phase>/final/` or "
            "ensure `huggingface-cli login` has access to the published "
            "adapter repos."
        )
        st.stop()

    adapter_choice = st.selectbox(
        "Adapter to compare against base",
        options=available_adapters,
        index=available_adapters.index("phase3") if "phase3" in available_adapters else 0,
        format_func=lambda k: f"{k}  ({adapter_source_label(k)})",
        help="phase3 = full framework (best video R-1/R-2 in paper). "
             "phase2 = +OPLoRA +EWC. phase1 = LoRA+ only.",
    )
    max_new = st.slider("Max new tokens", 64, 320, DEFAULT_MAX_NEW_TOKENS, step=16)
    num_beams = st.slider(
        "Beam size", 1, 6, 2, step=1,
        help="2 = laptop-friendly default (8 GB VRAM headroom). "
             "4 = paper-faithful (matches scripts/evaluate.py — bump if you "
             "have a desktop GPU). 1 = greedy (fastest but loop-prone on "
             "this adapter).",
    )
    rep_pen = st.slider(
        "Repetition penalty", 1.0, 1.5, 1.0, step=0.05,
        help="1.0 = paper-faithful (off). The adapter has a token-suffix loop "
             "artifact at greedy decode; small values (1.05–1.10) reduce it but "
             "1.15+ over-corrects and pushes the model into sequential gibberish.",
    )
    no_rep_n = st.slider(
        "no_repeat_ngram_size", 0, 6, 3, step=1,
        help="Blocks repeated n-grams. 3 matches paper eval; higher reduces "
             "loop artifacts but can hurt fluency.",
    )
    n_examples = st.slider("# test-split video examples to load", 16, 128, DEFAULT_NUM_VIDEO_EXAMPLES, step=8)
    st.divider()
    st.markdown(
        "**Tip:** first generation pair takes longer (warm-up). "
        "Subsequent ones run at ~5–15 s on a laptop 4070."
    )

# Load model + adapters once
model, tokenizer, loaded = load_model_and_adapters(tuple(available_adapters))
if adapter_choice not in loaded:
    st.error(f"Adapter '{adapter_choice}' failed to load.")
    st.stop()

# Load examples
examples = load_video_samples(n=n_examples)

# Lookup so we can find example IDs inside the loaded sample window.
example_by_id = {e["id"]: e for e in examples}

# Input source
tab_picks, tab_examples, tab_custom = st.tabs(
    ["Examples", "Full test-split", "Paste your own"]
)

source_text = ""
reference_text = ""
choice = None
pick_choice = None

with tab_picks:
    available_picks = [p for p in EXAMPLES if p["id"] in example_by_id]
    if not available_picks:
        st.warning(
            "Sample window doesn't include the example IDs. "
            "Increase '# test-split video examples to load' in the sidebar "
            "to 128 (full split), or use the Full test-split tab."
        )
    else:
        labels = [p["label"] for p in available_picks]
        pick_choice = st.radio(
            "Pick a video:",
            options=range(len(available_picks)),
            format_func=lambda i: labels[i],
        )
        chosen = available_picks[pick_choice]
        st.caption(chosen["blurb"])
        source_text = example_by_id[chosen["id"]]["source"]
        reference_text = example_by_id[chosen["id"]]["reference"]

        with st.expander("Show full transcript", expanded=False):
            st.write(source_text)
        with st.expander("Show gold reference summary", expanded=False):
            st.write(reference_text)

with tab_examples:
    if not examples:
        st.warning("No test-split video samples available (HDF5 missing).")
    else:
        st.caption(
            "All 75 test-split videos. Quality varies — most are adapter "
            "wins (58/75 by R-1 lift), but 3 are clear losses and ~14 are "
            "marginal. See `results/phase3/per_sample.csv` for per-video "
            "ROUGE numbers."
        )
        labels = [f"{i+1}. {e['preview']}" for i, e in enumerate(examples)]
        choice = st.selectbox(
            "Pick any video transcript from the test split:",
            options=range(len(examples)),
            format_func=lambda i: labels[i],
        )
        if not source_text:
            source_text = examples[choice]["source"]
            reference_text = examples[choice]["reference"]

        with st.expander("Show full transcript", expanded=False):
            st.write(examples[choice]["source"])
        with st.expander("Show gold reference summary", expanded=False):
            st.write(examples[choice]["reference"])

with tab_custom:
    custom_text = st.text_area(
        "Paste a video transcript (or any source text):",
        height=240,
        placeholder="Paste a video transcript here, or use one of the "
                    "test-split tabs.",
    )
    custom_reference = st.text_area(
        "Optional: gold reference summary (for ROUGE)",
        height=80,
    )

# Decide which input is active. Custom input wins if filled. Otherwise use
# whichever tab populated source_text (Examples tab first, then full list).
if custom_text.strip():
    source_text = custom_text.strip()
    reference_text = custom_reference.strip()
    input_label = "your input"
elif pick_choice is not None and pick_choice >= 0 and EXAMPLES:
    input_label = f"example: {EXAMPLES[pick_choice]['id']}"
elif choice is not None and examples:
    input_label = f"test-split example #{choice+1}"
else:
    input_label = ""

st.divider()

go = st.button("Generate two-way summary", type="primary",
               disabled=not source_text)

if go and source_text:
    col_base, col_adapter = st.columns(2)

    with st.status(f"Generating on {input_label}...", expanded=True) as status:
        st.write("Base Mistral-7B (no adapter)...")
        base_summary, base_t = generate(
            model, tokenizer, source_text,
            use_adapter=None, max_new_tokens=max_new,
            repetition_penalty=rep_pen, no_repeat_ngram_size=no_rep_n,
            num_beams=num_beams,
        )
        st.write(f"  done in {base_t:.1f} s")

        st.write(f"Adapter: {adapter_choice}...")
        adp_summary, adp_t = generate(
            model, tokenizer, source_text,
            use_adapter=adapter_choice, max_new_tokens=max_new,
            repetition_penalty=rep_pen, no_repeat_ngram_size=no_rep_n,
            num_beams=num_beams,
        )
        st.write(f"  done in {adp_t:.1f} s")
        status.update(label="Generation complete", state="complete")

    base_style = style_metrics(base_summary)
    adp_style = style_metrics(adp_summary)
    base_rouge = compute_rouge(base_summary, reference_text) if reference_text else None
    adp_rouge = compute_rouge(adp_summary, reference_text) if reference_text else None

    with col_base:
        st.subheader("Base Mistral-7B")
        st.caption(f"zero-shot · {base_t:.1f} s")
        st.text_area("output_base", base_summary, height=260, label_visibility="collapsed")

    with col_adapter:
        st.subheader(f"+ {adapter_choice} adapter")
        st.caption(f"LoRA r=32 · {adp_t:.1f} s")
        st.text_area("output_adp", adp_summary, height=260, label_visibility="collapsed")

    st.divider()
    st.subheader("Style metrics")
    st.caption(
        "PVR (passive voice) and Nominalization% measure academic-register "
        "contamination — IMPACT 2025 found these spike when LoRA-trained on papers "
        "is applied to video. Lower = more conversational / less academic. "
        "TTR = lexical diversity. ASL = avg sentence length (words)."
    )

    metric_rows = [
        ("Passive voice %",       base_style["passive_voice_pct"] * 100,  adp_style["passive_voice_pct"] * 100,  "lower-is-better"),
        ("Nominalization %",      base_style["nominalization_pct"] * 100, adp_style["nominalization_pct"] * 100, "lower-is-better"),
        ("Type-token ratio",      base_style["type_token_ratio"],         adp_style["type_token_ratio"],         "higher-is-better"),
        ("Avg sentence length",   base_style["avg_sent_len"],             adp_style["avg_sent_len"],             "neutral"),
        ("Word count",            base_style["word_count"],               adp_style["word_count"],               "neutral"),
    ]

    cols = st.columns(len(metric_rows))
    for col, (name, b_val, a_val, direction) in zip(cols, metric_rows):
        delta = a_val - b_val
        if direction == "lower-is-better":
            delta_color = "inverse"
        elif direction == "higher-is-better":
            delta_color = "normal"
        else:
            delta_color = "off"
        col.metric(
            label=name,
            value=f"{a_val:.2f}",
            delta=f"{delta:+.2f} vs base",
            delta_color=delta_color,
            help=f"base={b_val:.2f} · adapter={a_val:.2f} · direction={direction}",
        )

    if base_rouge and adp_rouge:
        st.subheader("ROUGE vs reference")
        rcols = st.columns(3)
        for col, key in zip(rcols, ["rouge1", "rouge2", "rougeL"]):
            delta = adp_rouge[key] - base_rouge[key]
            col.metric(
                label=key.upper(),
                value=f"{adp_rouge[key]:.3f}",
                delta=f"{delta:+.3f} vs base",
                delta_color="normal",
                help=f"base={base_rouge[key]:.3f} · adapter={adp_rouge[key]:.3f}",
            )
    elif source_text and not reference_text:
        st.info("No reference summary provided — ROUGE skipped. "
                "Pick a test-split example to see ROUGE numbers.")

elif not source_text:
    st.info("Pick an example from the test split or paste your own transcript, "
            "then click **Generate**.")

# Footer
st.divider()
with st.expander("How this works"):
    st.markdown(f"""
- **Single base load**: Mistral-7B-v0.1 in 4-bit NF4 (~5 GB VRAM).
- **Adapter swap, no reload**: PEFT `model.set_adapter('{adapter_choice}')` for the
  trained path, `model.disable_adapter()` context for the base path. Same weights,
  no re-init between calls.
- **Decoding**: greedy, `max_new_tokens={max_new}`, `no_repeat_ngram_size=3`,
  identical for both paths — only the adapter changes.
- **Style metrics** are regex/suffix-based for portability (no spaCy install
  needed); the published paper numbers use spaCy-based PVR/NR. Values shown
  here are an approximation but track the same trend.
- **Loaded adapters**: {", ".join(loaded) or "none"}.
""")
