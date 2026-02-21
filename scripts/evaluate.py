"""
evaluate.py — Unified Evaluation Harness for Hybrid-Dataset Summariser
======================================================================

Measures summarisation quality, style degradation, and cross-modal consistency
at every project checkpoint: baseline (zero-shot), post-Phase 1/2/3, ablations.

Run from repo root:
    python scripts/evaluate.py --mode base --run-name baseline
    python scripts/evaluate.py --mode adapter --checkpoint checkpoints/phase1_ep3 \
                               --run-name phase1 --compare baseline
    python scripts/evaluate.py --mode adapter --checkpoint checkpoints/phase2_full \
                               --run-name phase2_full --compare baseline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEORETICAL FORMULATIONS
========================

The metrics and composite scores below synthesise measurement approaches from
six source papers. Each formula is derived first, then coded.

1. ROUGE SCORES (Lin, 2004)
────────────────────────────
Standard n-gram overlap between generated summary G and reference R.

    ROUGE-N = Σ_{s∈R} Σ_{gram_n∈s} Count_match(gram_n)
              ──────────────────────────────────────────
              Σ_{s∈R} Σ_{gram_n∈s} Count(gram_n)

    ROUGE-L uses Longest Common Subsequence:
        P_lcs = LCS(G, R) / |G|
        R_lcs = LCS(G, R) / |R|
        ROUGE-L = (1 + β²) × P_lcs × R_lcs / (R_lcs + β² × P_lcs)

    where β = R_lcs / P_lcs (F-measure weighting).

    We report F1 for all ROUGE variants.


2. BERTScore (Zhang et al., 2020)
──────────────────────────────────
Soft semantic similarity via contextual embeddings.

    Let x = (x₁, ..., x_k) be token embeddings of G,
        x̂ = (x̂₁, ..., x̂_l) be token embeddings of R.

    Precision:  P_BERT = (1/|x|) Σ_{x_i∈x} max_{x̂_j∈x̂} x_i^T x̂_j
    Recall:     R_BERT = (1/|x̂|) Σ_{x̂_j∈x̂} max_{x_i∈x} x_i^T x̂_j
    F1:         F_BERT = 2 × P_BERT × R_BERT / (P_BERT + R_BERT)

    We use DeBERTa-large-MNLI with baseline rescaling for calibrated scores.


3. STYLE DEGRADATION METRICS (from IMPACT 2025 findings)
─────────────────────────────────────────────────────────
IMPACT 2025 identified that LoRA fine-tuning on academic papers causes
14-50% increase in passive voice and nominalization in video summaries,
constituting measurable "domain contamination" even when ROUGE is stable.

  3a. Passive Voice Ratio (PVR)
      Using spaCy dependency parsing, a clause is passive if it contains
      a passive nominal subject (dep='nsubjpass'):

        PVR(G) = |{s ∈ sentences(G) : ∃ token t ∈ s with dep(t) = nsubjpass}|
                 ──────────────────────────────────────────────────────────────
                 |sentences(G)|

      Range: [0, 1]. Lower is more conversational. Higher signals academic
      register bleeding into video summaries.

  3b. Nominalization Ratio (NR)
      Nominalizations are nouns derived from verbs/adjectives via suffixes
      {-tion, -sion, -ment, -ness, -ity, -ance, -ence, -ism}.
      Content words = tokens with POS ∈ {NOUN, VERB, ADJ, ADV} minus stopwords.

        NR(G) = |{w ∈ content_words(G) : POS(w) = NOUN ∧ w ends with suffix}|
                ────────────────────────────────────────────────────────────────
                |content_words(G)|

  3c. Type-Token Ratio (TTR)
      Lexical diversity measure. Degradation reduces vocabulary breadth.

        TTR(G) = |unique_tokens(G)| / |tokens(G)|

  3d. Average Sentence Length (ASL)
      Fluency proxy. Academic contamination tends to increase ASL.

        ASL(G) = |tokens(G)| / |sentences(G)|


4. CROSS-MODAL CONSISTENCY (inspired by CrossCLR, Zolfaghari et al., 2021)
───────────────────────────────────────────────────────────────────────────
For topic-matched paper-video pair (p, v), generate summaries G_p, G_v.
Encode both with SBERT and compute cosine similarity:

    CMC(p, v) = cos(SBERT(G_p), SBERT(G_v))
              = SBERT(G_p)^T · SBERT(G_v)
                ─────────────────────────────
                ‖SBERT(G_p)‖ · ‖SBERT(G_v)‖

    CMC_mean = (1/|P|) Σ_{(p,v)∈P} CMC(p, v)

    where P is the set of cross-modal test pairs.

    Interpretation: high CMC means the model produces semantically aligned
    summaries for the same topic regardless of input modality — evidence
    of successful cross-modal transfer without domain contamination.


5. DEGRADATION GAP (D_gap) — composite metric
──────────────────────────────────────────────
Synthesised from IMPACT 2025 degradation measurements. Combines quality
and style metrics into a single scalar for phase-gating decisions.

    For quality metrics Q = {ROUGE-1, ROUGE-2, ROUGE-L, BERTScore}:
        δ_q = (q_after - q_before) / max(q_before, ε)

        Positive δ_q = improvement; negative = degradation.

    For style metrics S = {PVR, NR}:
        δ_s = -(s_after - s_before) / max(s_before, ε)

        Negated because increase in PVR/NR is degradation.
        Positive δ_s = improvement (less academic register);
        negative = degradation.

    Composite:
        D_gap = (Σ_{q∈Q} w_q·δ_q + Σ_{s∈S} w_s·δ_s) / (Σ_{m∈Q∪S} w_m)

    Default weights (tunable):
        w_{ROUGE-1} = 1.0      w_{ROUGE-2} = 1.0
        w_{ROUGE-L} = 1.0      w_{BERTScore} = 1.5
        w_{PVR}     = 0.75     w_{NR}        = 0.75

    BERTScore weighted higher: captures semantic preservation beyond n-gram
    overlap. Style metrics weighted lower: diagnostic, not primary quality.

    Phase-gating thresholds (on VIDEO split only):
        Phase 1 → Phase 2:  D_gap(videos) > -0.10
        Phase 2 → Phase 3:  D_gap(videos) > -0.05


6. STATISTICAL COMPARISON
─────────────────────────
  6a. Paired t-test
      For per-sample metric vectors a = (a₁,...,a_n), b = (b₁,...,b_n):
        d_i = a_i - b_i
        t = d̄ / (s_d / √n),  df = n - 1
        p = 2 × P(T > |t|)

  6b. Cohen's d (effect size)
        d = d̄ / s_d
      |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large

  6c. Wilcoxon signed-rank (non-parametric backup)
      Ranks absolute differences, applies sign. Robust when n < 30 or
      distributions are non-normal (relevant for 75-sample video split).

  6d. Bootstrap 95% CI
      Resample n values with replacement B=1000 times:
        CI = [percentile(θ̂*, 2.5), percentile(θ̂*, 97.5)]
      Preferred over normal approximation for small n.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import h5py

# ━━ CONFIGURATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HDF5_PATH = Path("data/hdf5/engineering.h5")
RESULTS_DIR = Path("results")
MODEL_ID = "mistralai/Mistral-7B-v0.1"
SBERT_MODEL = "all-mpnet-base-v2"
BERTSCORE_MODEL = "microsoft/deberta-large-mnli"

# Test split index in HDF5
TEST_SPLIT = 2

# D_gap weights
DGAP_WEIGHTS = {
    "rouge1":            1.0,
    "rouge2":            1.0,
    "rougeL":            1.0,
    "bertscore_f1":      1.5,
    "passive_voice_pct": 0.75,
    "nominalization_pct": 0.75,
}
# Metrics where increase = degradation (style metrics)
INVERTED_METRICS = {"passive_voice_pct", "nominalization_pct"}

# Epsilon for safe division
EPS = 1e-8

# Nominalization suffixes
NOMINALIZATION_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism")

# Prompt template for base Mistral-7B (no instruction tuning)
PROMPT_TEMPLATE = "Summarize the following text.\n\nText: {source}\n\nSummary:"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ━━ DATA STRUCTURES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Sample:
    """Single evaluation sample — source text, reference summary, metadata."""
    sample_id: str
    modality: str          # "paper" or "video"
    source_text: str
    reference: str
    category: str
    split: int


@dataclass
class PairSample:
    """Cross-modal pair — both sides."""
    pair_id: str
    similarity: float
    paper_source: str
    paper_reference: str
    video_source: str
    video_reference: str


@dataclass
class SampleResult:
    """Per-sample evaluation metrics."""
    sample_id: str
    modality: str
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_f1: float = 0.0
    passive_voice_pct: float = 0.0
    nominalization_pct: float = 0.0
    type_token_ratio: float = 0.0
    avg_sent_len: float = 0.0
    generation_len: int = 0
    generated_text: str = ""


# ━━ HDF5 TEST LOADER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HDF5TestLoader:
    """Load and decode test-split samples from engineering.h5."""

    def __init__(self, hdf5_path: Path, tokenizer):
        self.path = hdf5_path
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.categories = []  # filled on load

    def _decode(self, token_ids: np.ndarray, mask: np.ndarray) -> str:
        """Decode token IDs to text, respecting attention mask."""
        real_len = int(mask.sum())
        ids = token_ids[:real_len].tolist()
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def _decode_labels(self, label_ids: np.ndarray) -> str:
        """Decode label IDs, stripping padding."""
        real = label_ids[label_ids != self.pad_id].tolist()
        return self.tok.decode(real, skip_special_tokens=True).strip()

    def load_samples(self) -> tuple[list[Sample], list[Sample], list[PairSample]]:
        """
        Load all test-split samples.
        Returns: (paper_samples, video_samples, pair_samples)
        """
        papers, videos, pairs = [], [], []

        with h5py.File(str(self.path), "r") as hf:
            # Load category names
            self.categories = [
                c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in hf["metadata"]["categories"][:]
            ]

            # ── Papers ──
            grp = hf["papers"]
            splits = grp["split"][:]
            test_mask = splits == TEST_SPLIT
            test_indices = np.where(test_mask)[0]

            log.info(f"Loading {len(test_indices)} test papers...")
            for idx in test_indices:
                sid = grp["sample_id"][idx]
                if isinstance(sid, bytes):
                    sid = sid.decode("utf-8")
                cat_idx = int(grp["category"][idx])
                cat = self.categories[cat_idx] if cat_idx < len(self.categories) else "unknown"

                source = self._decode(grp["input_ids"][idx], grp["attention_mask"][idx])
                reference = self._decode_labels(grp["labels"][idx])

                if source and reference:
                    papers.append(Sample(
                        sample_id=sid, modality="paper",
                        source_text=source, reference=reference,
                        category=cat, split=TEST_SPLIT,
                    ))

            # ── Videos ──
            grp = hf["videos"]
            splits = grp["split"][:]
            test_mask = splits == TEST_SPLIT
            test_indices = np.where(test_mask)[0]

            log.info(f"Loading {len(test_indices)} test videos...")
            for idx in test_indices:
                sid = grp["sample_id"][idx]
                if isinstance(sid, bytes):
                    sid = sid.decode("utf-8")
                cat_idx = int(grp["category"][idx])
                cat = self.categories[cat_idx] if cat_idx < len(self.categories) else "unknown"

                source = self._decode(grp["input_ids"][idx], grp["attention_mask"][idx])
                reference = self._decode_labels(grp["labels"][idx])

                if source and reference:
                    videos.append(Sample(
                        sample_id=sid, modality="video",
                        source_text=source, reference=reference,
                        category=cat, split=TEST_SPLIT,
                    ))

            # ── Cross-modal pairs ──
            cm = hf["cross_modal"]
            pair_splits = cm["split"][:]
            test_mask = pair_splits == TEST_SPLIT
            test_indices = np.where(test_mask)[0]

            log.info(f"Loading {len(test_indices)} test pairs...")
            for idx in test_indices:
                pid = cm["pair_id"][idx]
                if isinstance(pid, bytes):
                    pid = pid.decode("utf-8")
                sim = float(cm["similarity"][idx])

                p_source = self._decode(cm["paper_input_ids"][idx], cm["paper_attention_mask"][idx])
                p_ref = self._decode_labels(cm["paper_labels"][idx])
                v_source = self._decode(cm["video_input_ids"][idx], cm["video_attention_mask"][idx])
                v_ref = self._decode_labels(cm["video_labels"][idx])

                if p_source and v_source:
                    pairs.append(PairSample(
                        pair_id=pid, similarity=sim,
                        paper_source=p_source, paper_reference=p_ref,
                        video_source=v_source, video_reference=v_ref,
                    ))

        log.info(f"Loaded: {len(papers)} papers, {len(videos)} videos, {len(pairs)} pairs")
        return papers, videos, pairs


# ━━ MODEL LOADER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModelLoader:
    """Load Mistral-7B with optional LoRA adapter, 4-bit quantized."""

    def __init__(self, model_id: str, adapter_path: Optional[str] = None):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer. Returns (model, tokenizer)."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        log.info(f"Loading tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log.info(f"Loading model: {self.model_id} (4-bit quantized)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,  # some versions warn; none crash
        )

        if self.adapter_path:
            from peft import PeftModel
            log.info(f"Loading LoRA adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        device_info = getattr(self.model, "hf_device_map", None) or str(next(self.model.parameters()).device)
        log.info(f"Model loaded. Device: {device_info}")
        return self.model, self.tokenizer

    def unload(self):
        """Free model VRAM."""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Model unloaded, VRAM freed.")


# ━━ SUMMARY GENERATOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SummaryGenerator:
    """Generate summaries from source texts using loaded model."""

    def __init__(self, model, tokenizer, beam: int = 4, max_new_tokens: int = 256):
        self.model = model
        self.tok = tokenizer
        self.beam = beam
        self.max_new_tokens = max_new_tokens

    def generate(self, source_text: str) -> str:
        """Generate a summary for one source text."""
        import torch

        prompt = PROMPT_TEMPLATE.format(source=source_text)

        inputs = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,  # leave room for generation
        )
        # Safe device resolution: device_map="auto" may not set .device
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.beam,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,         # deterministic for reproducibility
                temperature=1.0,         # irrelevant with do_sample=False, but explicit
                pad_token_id=self.tok.eos_token_id,  # silence repeated warnings
            )

        # Decode only the newly generated tokens (strip the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        summary = self.tok.decode(generated_ids, skip_special_tokens=True).strip()

        return summary

    def generate_batch(self, samples: list[Sample], label: str = "") -> dict[str, str]:
        """
        Generate summaries for a list of samples.
        Returns: {sample_id: generated_summary}
        """
        results = {}
        total = len(samples)

        for i, sample in enumerate(samples):
            if (i + 1) % 25 == 0 or i == 0:
                log.info(f"  [{label}] Generating {i+1}/{total}...")

            summary = self.generate(sample.source_text)

            # Fallback for empty generation
            if not summary.strip():
                summary = "(empty generation)"

            results[sample.sample_id] = summary

        return results


# ━━ METRICS ENGINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MetricsEngine:
    """Compute ROUGE, BERTScore, and style metrics."""

    def __init__(self):
        self._rouge_scorer = None
        self._nlp = None
        self._spacy_available = None  # tri-state: None=untried, True/False

    @property
    def rouge_scorer(self):
        if self._rouge_scorer is None:
            from rouge_score.rouge_scorer import RougeScorer
            self._rouge_scorer = RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        return self._rouge_scorer

    @property
    def nlp(self):
        if self._nlp is None:
            if self._spacy_available is False:
                return None
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                self._spacy_available = True
            except (ImportError, ValueError, OSError) as e:
                log.warning(
                    f"spaCy unavailable ({type(e).__name__}: {e}). "
                    "Using regex fallbacks for style metrics. "
                    "Fix: pip install --upgrade spacy thinc --force-reinstall && "
                    "python -m spacy download en_core_web_sm"
                )
                self._spacy_available = False
                return None
        return self._nlp

    # ── ROUGE ──

    def compute_rouge(self, generated: str, reference: str) -> dict[str, float]:
        """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    # ── BERTScore ──

    def compute_bertscore_batch(
        self, generated_list: list[str], reference_list: list[str]
    ) -> list[float]:
        """
        BERTScore F1 for a batch of (generated, reference) pairs.
        Computed in batch on GPU for efficiency.

        Includes safetensors crash prevention: on Windows with degraded
        HF cache (no symlinks), transformers may attempt broken safetensors
        auto-conversion even for cached pytorch models. We prevent this by
        setting SAFETENSORS env var and falling back to roberta-large if needed.
        """
        import os
        import torch
        from bert_score import score as bert_score_fn

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prevent transformers from attempting safetensors auto-conversion
        # which can crash on Windows due to broken HF Spaces API
        _orig_env = os.environ.get("SAFETENSORS_FAST_GPU")
        os.environ["SAFETENSORS_FAST_GPU"] = "0"

        try:
            _, _, f1 = bert_score_fn(
                generated_list,
                reference_list,
                model_type=BERTSCORE_MODEL,
                lang="en",
                verbose=True,
                rescale_with_baseline=True,
                device=device,
            )
            return f1.tolist()
        except (OSError, json.JSONDecodeError, Exception) as e:
            log.warning(f"BERTScore with {BERTSCORE_MODEL} failed: {e}")
            log.warning("Falling back to roberta-large (safe alternative)...")
            _, _, f1 = bert_score_fn(
                generated_list,
                reference_list,
                model_type="roberta-large",
                lang="en",
                verbose=True,
                rescale_with_baseline=True,
                device=device,
            )
            return f1.tolist()
        finally:
            # Restore original env
            if _orig_env is None:
                os.environ.pop("SAFETENSORS_FAST_GPU", None)
            else:
                os.environ["SAFETENSORS_FAST_GPU"] = _orig_env

    # ── STYLE: Passive Voice ──

    def passive_voice_pct(self, text: str) -> float:
        """
        PVR = |passive sentences| / |total sentences|

        A sentence is passive if any token has dep='nsubjpass' (spaCy),
        or matches passive regex patterns (fallback).
        """
        import re

        # Sentence splitting (shared by both paths)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if len(s.split()) >= 2]
        if not sentences:
            return 0.0

        nlp = self.nlp
        if nlp is not None:
            # ── spaCy path: dependency parsing ──
            doc = nlp(text)
            total_sents = 0
            passive_sents = 0
            for sent in doc.sents:
                total_sents += 1
                for token in sent:
                    if token.dep_ == "nsubjpass":
                        passive_sents += 1
                        break
            return passive_sents / max(total_sents, 1)
        else:
            # ── Regex fallback ──
            # Matches: "is/was/were/been/being/are/get/got/gets + past participle"
            passive_re = re.compile(
                r'\b(?:is|was|were|been|being|are|am|get|got|gets)\s+'
                r'(?:\w+\s+){0,3}'  # up to 3 intervening words (adverbs)
                r'(?:\w+ed|written|shown|given|taken|made|done|seen|known|found)\b',
                re.IGNORECASE,
            )
            passive_count = sum(1 for s in sentences if passive_re.search(s))
            return passive_count / len(sentences)

    # ── STYLE: Nominalization ──

    def nominalization_pct(self, text: str) -> float:
        """
        NR = |nominalizations| / |content words|

        spaCy path: Content words = NOUN/VERB/ADJ/ADV minus stopwords.
        Regex fallback: Content words ≈ all words > 3 chars not in stoplist.
        """
        nlp = self.nlp
        if nlp is not None:
            # ── spaCy path ──
            doc = nlp(text)
            content_count = 0
            nom_count = 0
            for token in doc:
                if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and not token.is_stop:
                    content_count += 1
                    if token.pos_ == "NOUN" and token.text.lower().endswith(
                        NOMINALIZATION_SUFFIXES
                    ):
                        nom_count += 1
            return nom_count / max(content_count, 1)
        else:
            # ── Regex fallback ──
            # Approximate content words as words > 3 chars
            words = [w.lower() for w in text.split() if len(w) > 3]
            if not words:
                return 0.0
            nom_count = sum(1 for w in words if w.endswith(NOMINALIZATION_SUFFIXES))
            return nom_count / len(words)

    # ── STYLE: Type-Token Ratio ──

    @staticmethod
    def type_token_ratio(text: str) -> float:
        """TTR = |unique tokens| / |total tokens|"""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    # ── STYLE: Average Sentence Length ──

    def avg_sentence_length(self, text: str) -> float:
        """ASL = |tokens| / |sentences|"""
        import re

        nlp = self.nlp
        if nlp is not None:
            doc = nlp(text)
            sents = list(doc.sents)
            if not sents:
                return 0.0
            total_tokens = sum(len(s) for s in sents)
            return total_tokens / len(sents)
        else:
            # Regex fallback
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = [s for s in sentences if len(s.split()) >= 2]
            if not sentences:
                return 0.0
            total_words = sum(len(s.split()) for s in sentences)
            return total_words / len(sentences)

    # ── Combined per-sample ──

    def evaluate_sample(
        self, generated: str, reference: str, sample_id: str, modality: str
    ) -> SampleResult:
        """Compute all non-BERTScore metrics for one sample."""
        rouge = self.compute_rouge(generated, reference)

        return SampleResult(
            sample_id=sample_id,
            modality=modality,
            rouge1=rouge["rouge1"],
            rouge2=rouge["rouge2"],
            rougeL=rouge["rougeL"],
            bertscore_f1=0.0,  # filled later in batch
            passive_voice_pct=self.passive_voice_pct(generated),
            nominalization_pct=self.nominalization_pct(generated),
            type_token_ratio=self.type_token_ratio(generated),
            avg_sent_len=self.avg_sentence_length(generated),
            generation_len=len(generated.split()),
            generated_text=generated,
        )


# ━━ CROSS-MODAL EVALUATOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CrossModalEvaluator:
    """
    Evaluate cross-modal consistency using SBERT cosine similarity
    between paper and video summaries of the same topic.
    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(SBERT_MODEL)
        return self._model

    def compute_consistency(
        self, paper_summaries: list[str], video_summaries: list[str]
    ) -> list[float]:
        """
        CMC(p, v) = cos(SBERT(G_p), SBERT(G_v))

        Args: parallel lists of paper and video summaries for matched pairs.
        Returns: list of cosine similarities.
        """
        p_embs = self.model.encode(
            paper_summaries, normalize_embeddings=True, convert_to_numpy=True
        )
        v_embs = self.model.encode(
            video_summaries, normalize_embeddings=True, convert_to_numpy=True
        )

        # Row-wise cosine similarity (embeddings are already normalized)
        similarities = np.sum(p_embs * v_embs, axis=1)
        return similarities.tolist()


# ━━ STATISTICAL COMPARATOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StatisticalComparator:
    """Compare two evaluation runs with paired statistical tests."""

    @staticmethod
    def bootstrap_ci(
        values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42
    ) -> tuple[float, float]:
        """
        Bootstrap confidence interval.
        CI = [percentile(θ̂*, α/2×100), percentile(θ̂*, (1-α/2)×100)]
        """
        rng = np.random.RandomState(seed)
        boot_means = np.empty(n_boot)
        n = len(values)
        for i in range(n_boot):
            sample = rng.choice(values, size=n, replace=True)
            boot_means[i] = np.mean(sample)
        alpha = (1 - ci) / 2
        return (
            float(np.percentile(boot_means, alpha * 100)),
            float(np.percentile(boot_means, (1 - alpha) * 100)),
        )

    @staticmethod
    def paired_t_test(a: np.ndarray, b: np.ndarray) -> dict:
        """
        Paired t-test: t = d̄ / (s_d / √n)
        Returns: t-statistic, p-value, mean difference.
        """
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(a, b)
        d = a - b
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "mean_diff": float(np.mean(d)),
            "std_diff": float(np.std(d, ddof=1)),
        }

    @staticmethod
    def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cohen's d = d̄ / s_d
        Effect size: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
        """
        d = a - b
        sd = np.std(d, ddof=1)
        if sd < EPS:
            return 0.0
        return float(np.mean(d) / sd)

    @staticmethod
    def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
        """
        Wilcoxon signed-rank test (non-parametric backup).
        Robust when n < 30 or distributions are non-normal.
        """
        from scipy.stats import wilcoxon

        d = a - b
        # Wilcoxon requires at least one non-zero difference
        if np.all(np.abs(d) < EPS):
            return {"statistic": 0.0, "p_value": 1.0}

        stat, p_val = wilcoxon(d)
        return {"statistic": float(stat), "p_value": float(p_val)}

    def compare_runs(
        self,
        baseline_results: list[SampleResult],
        current_results: list[SampleResult],
    ) -> dict:
        """
        Compare two runs on matched samples.
        Returns statistical tests for each metric.
        """
        # Build lookup by sample_id
        base_map = {r.sample_id: r for r in baseline_results}
        curr_map = {r.sample_id: r for r in current_results}

        # Find common sample IDs
        common_ids = sorted(set(base_map.keys()) & set(curr_map.keys()))
        if not common_ids:
            log.warning("No common sample IDs between runs.")
            return {}

        metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1",
                    "passive_voice_pct", "nominalization_pct",
                    "type_token_ratio", "avg_sent_len"]

        comparison = {"n_common": len(common_ids)}

        for metric in metrics:
            a = np.array([getattr(base_map[sid], metric) for sid in common_ids])
            b = np.array([getattr(curr_map[sid], metric) for sid in common_ids])

            comparison[metric] = {
                "baseline_mean": float(np.mean(a)),
                "current_mean": float(np.mean(b)),
                "paired_t": self.paired_t_test(b, a),  # b - a = improvement
                "cohens_d": self.cohens_d(b, a),
                "wilcoxon": self.wilcoxon_test(b, a),
            }

        return comparison


# ━━ D_GAP CALCULATOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_dgap(
    before: dict[str, float], after: dict[str, float],
    weights: dict[str, float] = None,
) -> dict:
    """
    Compute composite degradation gap.

    D_gap = (Σ w_m·δ_m) / (Σ w_m)

    where δ_m = (after - before)/max(before, ε)       for quality metrics
          δ_m = -(after - before)/max(before, ε)      for style metrics (inverted)

    Returns: {"d_gap": float, "per_metric": {metric: delta}, "weights_used": {}}
    """
    if weights is None:
        weights = DGAP_WEIGHTS

    per_metric = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for metric, w in weights.items():
        b = before.get(metric)
        a = after.get(metric)

        if b is None or a is None:
            continue

        raw_delta = (a - b) / max(abs(b), EPS)

        # Invert for style metrics: increase is degradation
        if metric in INVERTED_METRICS:
            delta = -raw_delta
        else:
            delta = raw_delta

        per_metric[metric] = {
            "before": round(b, 6),
            "after": round(a, 6),
            "raw_delta": round(raw_delta, 6),
            "delta": round(delta, 6),
            "weight": w,
        }

        weighted_sum += w * delta
        total_weight += w

    d_gap = weighted_sum / max(total_weight, EPS)

    return {
        "d_gap": round(d_gap, 6),
        "per_metric": per_metric,
        "weights_used": weights,
        "interpretation": _interpret_dgap(d_gap),
    }


def _interpret_dgap(d: float) -> str:
    if d > 0.05:
        return "improvement"
    elif d > -0.02:
        return "stable"
    elif d > -0.05:
        return "mild_degradation"
    elif d > -0.10:
        return "moderate_degradation"
    else:
        return "severe_degradation"


# ━━ AGGREGATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def aggregate_metrics(
    results: list[SampleResult], comparator: StatisticalComparator
) -> dict:
    """
    Aggregate per-sample results into summary statistics.
    Returns: {"n": int, "rouge1": {"mean", "std", "ci95"}, ...}
    """
    if not results:
        return {"n": 0}

    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1",
               "passive_voice_pct", "nominalization_pct",
               "type_token_ratio", "avg_sent_len", "generation_len"]

    agg = {"n": len(results)}

    for metric in metrics:
        values = np.array([getattr(r, metric) for r in results], dtype=np.float64)
        ci = comparator.bootstrap_ci(values)
        agg[metric] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values, ddof=1)), 6),
            "median": round(float(np.median(values)), 6),
            "ci95": [round(ci[0], 6), round(ci[1], 6)],
        }

    return agg


# ━━ OUTPUT FORMATTER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_results(
    run_name: str,
    args: argparse.Namespace,
    paper_results: list[SampleResult],
    video_results: list[SampleResult],
    cross_modal: dict,
    comparison: Optional[dict],
    d_gap: Optional[dict],
    runtime_sec: float,
):
    """Save all results to results/{run_name}/"""
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    comparator = StatisticalComparator()

    # Aggregate
    all_results = paper_results + video_results
    paper_agg = aggregate_metrics(paper_results, comparator)
    video_agg = aggregate_metrics(video_results, comparator)
    combined_agg = aggregate_metrics(all_results, comparator)

    # ── metrics.json ──
    metrics_json = {
        "run_name": run_name,
        "model": MODEL_ID,
        "adapter": args.checkpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation": {
            "beam": args.beam,
            "max_new_tokens": 256,
            "prompt_template": PROMPT_TEMPLATE,
            "temperature": 1.0,
            "do_sample": False,
        },
        "metrics": {
            "papers": paper_agg,
            "videos": video_agg,
            "combined": combined_agg,
            "cross_modal": cross_modal,
        },
        "d_gap": d_gap,
        "comparison": comparison,
        "runtime_sec": round(runtime_sec, 1),
        "git_sha": get_git_sha(),
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False))
    log.info(f"Saved: {metrics_path}")

    # ── per_sample.csv ──
    csv_path = run_dir / "per_sample.csv"
    header = (
        "sample_id,modality,rouge1,rouge2,rougeL,bertscore_f1,"
        "passive_voice_pct,nominalization_pct,type_token_ratio,"
        "avg_sent_len,generation_len"
    )
    lines = [header]
    for r in all_results:
        lines.append(
            f"{r.sample_id},{r.modality},"
            f"{r.rouge1:.6f},{r.rouge2:.6f},{r.rougeL:.6f},{r.bertscore_f1:.6f},"
            f"{r.passive_voice_pct:.6f},{r.nominalization_pct:.6f},"
            f"{r.type_token_ratio:.6f},{r.avg_sent_len:.2f},{r.generation_len}"
        )
    csv_path.write_text("\n".join(lines))
    log.info(f"Saved: {csv_path}")

    # ── Console summary ──
    print_summary(run_name, paper_agg, video_agg, combined_agg, cross_modal,
                  d_gap, comparison, runtime_sec)

    return metrics_json


def print_summary(
    run_name, paper_agg, video_agg, combined_agg, cross_modal,
    d_gap, comparison, runtime_sec,
):
    """Print readable evaluation summary to console."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS: {run_name}")
    print(f"{'='*70}")

    for label, agg in [("Papers", paper_agg), ("Videos", video_agg), ("Combined", combined_agg)]:
        n = agg.get("n", 0)
        if n == 0:
            continue
        print(f"\n  ── {label} (n={n}) ──")
        for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
            m = agg.get(metric, {})
            mean = m.get("mean", 0)
            ci = m.get("ci95", [0, 0])
            print(f"    {metric:<18s}  {mean:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")

        print()
        for metric in ["passive_voice_pct", "nominalization_pct",
                        "type_token_ratio", "avg_sent_len"]:
            m = agg.get(metric, {})
            mean = m.get("mean", 0)
            std = m.get("std", 0)
            print(f"    {metric:<18s}  {mean:.4f}  (±{std:.4f})")

    # Cross-modal
    if cross_modal and cross_modal.get("n", 0) > 0:
        cmc = cross_modal.get("inter_modal_similarity", {})
        print(f"\n  ── Cross-Modal Consistency (n={cross_modal['n']}) ──")
        print(f"    CMC mean: {cmc.get('mean', 0):.4f}  (±{cmc.get('std', 0):.4f})")

    # D_gap
    if d_gap:
        dg = d_gap["d_gap"]
        interp = d_gap["interpretation"]
        print(f"\n  ── D_gap ──")
        print(f"    D_gap = {dg:+.4f}  ({interp})")

        # Per-metric breakdown
        for m, info in d_gap.get("per_metric", {}).items():
            delta = info["delta"]
            symbol = "↑" if delta >= 0 else "↓"
            print(f"      {m:<20s}  δ={delta:+.4f} {symbol}")

    # Comparison
    if comparison and comparison.get("n_common", 0) > 0:
        print(f"\n  ── Statistical Comparison (n={comparison['n_common']}) ──")
        for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
            info = comparison.get(metric, {})
            if not info:
                continue
            t_info = info.get("paired_t", {})
            p_val = t_info.get("p_value", 1.0)
            cd = info.get("cohens_d", 0.0)
            sig = "*" if p_val < 0.05 else " "
            sig2 = "**" if p_val < 0.01 else sig
            print(f"    {metric:<18s}  Δ={t_info.get('mean_diff', 0):+.4f}  "
                  f"p={p_val:.4f}{sig2}  d={cd:+.3f}")

    print(f"\n  Runtime: {runtime_sec:.0f}s ({runtime_sec/60:.1f} min)")
    print(f"{'='*70}\n")


# ━━ MAIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summarisation model on test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["base", "adapter"], default="base",
        help="'base' = Mistral-7B zero-shot; 'adapter' = base + LoRA checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to LoRA adapter directory (required if mode=adapter)"
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Name for this evaluation run (e.g., 'baseline', 'phase1', 'phase2_no_ewc')"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Run name to compare against (loads results/{name}/per_sample.csv)"
    )
    parser.add_argument(
        "--beam", type=int, default=4,
        help="Beam search width (default: 4; use 1 for greedy during training)"
    )
    parser.add_argument(
        "--hdf5", type=str, default=str(HDF5_PATH),
        help=f"Path to HDF5 dataset (default: {HDF5_PATH})"
    )
    parser.add_argument(
        "--skip-cross-modal", action="store_true",
        help="Skip cross-modal pair evaluation (saves time)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.mode == "adapter" and not args.checkpoint:
        parser.error("--checkpoint is required when --mode=adapter")

    np.random.seed(args.seed)
    start_time = time.time()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        log.error(f"HDF5 file not found: {hdf5_path}")
        return

    # ── Step 1: Load model + tokenizer ──
    log.info("=" * 60)
    log.info("STEP 1: Loading model")
    log.info("=" * 60)

    loader = ModelLoader(
        MODEL_ID, adapter_path=args.checkpoint if args.mode == "adapter" else None
    )
    model, tokenizer = loader.load()

    # ── Step 2: Load test data ──
    log.info("=" * 60)
    log.info("STEP 2: Loading test data from HDF5")
    log.info("=" * 60)

    data_loader = HDF5TestLoader(hdf5_path, tokenizer)
    paper_samples, video_samples, pair_samples = data_loader.load_samples()

    # ── Step 3: Generate summaries ──
    log.info("=" * 60)
    log.info("STEP 3: Generating summaries")
    log.info("=" * 60)

    generator = SummaryGenerator(model, tokenizer, beam=args.beam)

    paper_generations = generator.generate_batch(paper_samples, label="papers")
    video_generations = generator.generate_batch(video_samples, label="videos")

    # Cross-modal pair generations (generate from pair-specific input_ids)
    pair_paper_gens = {}
    pair_video_gens = {}
    if not args.skip_cross_modal and pair_samples:
        log.info(f"Generating for {len(pair_samples)} cross-modal pairs...")
        for i, pair in enumerate(pair_samples):
            if (i + 1) % 25 == 0:
                log.info(f"  [pairs] Generating {i+1}/{len(pair_samples)}...")

            pair_paper_gens[pair.pair_id] = generator.generate(pair.paper_source)
            pair_video_gens[pair.pair_id] = generator.generate(pair.video_source)

    # ── Step 4: Unload model, free VRAM ──
    log.info("=" * 60)
    log.info("STEP 4: Unloading model")
    log.info("=" * 60)
    loader.unload()

    # ── Step 5: Compute metrics ──
    log.info("=" * 60)
    log.info("STEP 5: Computing metrics")
    log.info("=" * 60)

    engine = MetricsEngine()

    # ROUGE + style for papers
    log.info("Computing ROUGE + style metrics for papers...")
    paper_results = []
    for sample in paper_samples:
        gen = paper_generations[sample.sample_id]
        result = engine.evaluate_sample(gen, sample.reference, sample.sample_id, "paper")
        paper_results.append(result)

    # ROUGE + style for videos
    log.info("Computing ROUGE + style metrics for videos...")
    video_results = []
    for sample in video_samples:
        gen = video_generations[sample.sample_id]
        result = engine.evaluate_sample(gen, sample.reference, sample.sample_id, "video")
        video_results.append(result)

    # BERTScore (batch, on GPU after model unloaded)
    log.info("Computing BERTScore (batch)...")
    all_results = paper_results + video_results
    all_generated = [r.generated_text for r in all_results]
    all_references = []
    for sample in paper_samples + video_samples:
        all_references.append(sample.reference)

    bert_scores = engine.compute_bertscore_batch(all_generated, all_references)

    for i, result in enumerate(all_results):
        result.bertscore_f1 = bert_scores[i]

    # Split back into paper/video
    paper_results = all_results[:len(paper_samples)]
    video_results = all_results[len(paper_samples):]

    # ── Step 6: Cross-modal consistency ──
    cross_modal = {"n": 0}
    if not args.skip_cross_modal and pair_samples and pair_paper_gens:
        log.info("=" * 60)
        log.info("STEP 6: Cross-modal consistency")
        log.info("=" * 60)

        cm_eval = CrossModalEvaluator()

        p_sums = [pair_paper_gens[p.pair_id] for p in pair_samples]
        v_sums = [pair_video_gens[p.pair_id] for p in pair_samples]

        cmc_scores = cm_eval.compute_consistency(p_sums, v_sums)

        comparator_cm = StatisticalComparator()
        cmc_arr = np.array(cmc_scores)
        ci = comparator_cm.bootstrap_ci(cmc_arr)
        cross_modal = {
            "n": len(pair_samples),
            "inter_modal_similarity": {
                "mean": round(float(np.mean(cmc_arr)), 6),
                "std": round(float(np.std(cmc_arr, ddof=1)), 6),
                "median": round(float(np.median(cmc_arr)), 6),
                "ci95": [round(ci[0], 6), round(ci[1], 6)],
            },
        }

    # ── Step 7: Comparison with baseline (if requested) ──
    comparison = None
    d_gap = None

    if args.compare:
        log.info("=" * 60)
        log.info(f"STEP 7: Comparing against '{args.compare}'")
        log.info("=" * 60)

        baseline_csv = RESULTS_DIR / args.compare / "per_sample.csv"
        baseline_json = RESULTS_DIR / args.compare / "metrics.json"

        if not baseline_csv.exists():
            log.warning(f"Baseline per_sample.csv not found: {baseline_csv}")
        elif not baseline_json.exists():
            log.warning(f"Baseline metrics.json not found: {baseline_json}")
        else:
            # Load baseline per-sample results
            baseline_results = _load_per_sample_csv(baseline_csv)

            comparator = StatisticalComparator()

            # Split baseline by modality for D_gap
            base_paper = [r for r in baseline_results if r.modality == "paper"]
            base_video = [r for r in baseline_results if r.modality == "video"]

            # Statistical comparison on all samples
            comparison = comparator.compare_runs(baseline_results, all_results)

            # D_gap on VIDEO split (the money metric — this is what we're protecting)
            baseline_metrics = json.loads(baseline_json.read_text())
            base_video_agg = baseline_metrics.get("metrics", {}).get("videos", {})

            comparator_dgap = StatisticalComparator()
            current_video_agg = aggregate_metrics(video_results, comparator_dgap)

            if base_video_agg.get("n", 0) > 0:
                before = {m: base_video_agg[m]["mean"]
                          for m in DGAP_WEIGHTS if m in base_video_agg}
                after = {m: current_video_agg[m]["mean"]
                         for m in DGAP_WEIGHTS if m in current_video_agg}

                d_gap = compute_dgap(before, after)

                # Phase-gating check
                dg_val = d_gap["d_gap"]
                if dg_val > -0.05:
                    d_gap["gate_status"] = "PASS (< 5% degradation)"
                elif dg_val > -0.10:
                    d_gap["gate_status"] = "CAUTION (5-10% degradation)"
                else:
                    d_gap["gate_status"] = "FAIL (>10% degradation)"

    # ── Step 8: Save everything ──
    log.info("=" * 60)
    log.info("STEP 8: Saving results")
    log.info("=" * 60)

    runtime = time.time() - start_time
    save_results(
        args.run_name, args,
        paper_results, video_results,
        cross_modal, comparison, d_gap,
        runtime,
    )

    log.info("Done.")


# ━━ UTILITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_per_sample_csv(path: Path) -> list[SampleResult]:
    """Load per_sample.csv from a previous run."""
    results = []
    lines = path.read_text().strip().split("\n")

    if len(lines) < 2:
        return results

    # Skip header
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 11:
            continue

        results.append(SampleResult(
            sample_id=parts[0],
            modality=parts[1],
            rouge1=float(parts[2]),
            rouge2=float(parts[3]),
            rougeL=float(parts[4]),
            bertscore_f1=float(parts[5]),
            passive_voice_pct=float(parts[6]),
            nominalization_pct=float(parts[7]),
            type_token_ratio=float(parts[8]),
            avg_sent_len=float(parts[9]),
            generation_len=int(parts[10]),
        ))

    return results


if __name__ == "__main__":
    main()