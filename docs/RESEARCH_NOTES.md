
# Research Paper Notes - Consolidated

## Paper 1: CrossCLR (ICCV 2021)

**Cross-modal Contrastive Learning For Multi-modal Video Representations**

### Core Contribution

Inter + intra-modality contrastive learning with false negative pruning.

### Key Formulas

```
L(xi) = -log[δ(xi,yi) / (δ(xi,yi) + Σ_inter + λΣ_intra)]
δ(xi,yj) = exp(fx(xi)^T·fy(yj) / τ)
C(xi) = (1/M)Σ(xi^T·xj)/(||xi||·||xj||)  [connectivity]
w(xi) = exp(C(xi)/κ)  [loss weighting]
```

### Hyperparameters

* τ = 0.03 (temperature)
* γ = 0.9 (connectivity threshold)
* λ = 0.7-0.8 (intra-modality weight)
* κ = 35e-4 (weighting scale)
* Queue size = 3000-5000

### Results

* Youcook2 R@1: 19.5% (text→video)
* CIDEr-D: 61.10 (captioning)

### Using for Project

* Contrastive loss with inter + intra-modality terms
* Connectivity-based false negative pruning
* Queue mechanism for reliable statistics
* Loss weighting by sample influence

---

## Paper 2: MoNA (ICML 2024)

**Learning Modality Knowledge Alignment for Cross-Modality Transfer**

### Core Contribution

Two-stage meta-learning measuring and minimizing modality gap.

### Key Formulas

```
D(Ms,Mt) = inf_{π,B} d(P(Y^s_{π,B}|X̂), P(Y^t|X̂))
L'_outer = λL_outer + L_inner
L_outer = L_align + L_uniform
Gradient alignment: min L_inner - λα(∇L_outer)·(∇L_inner)
```

### Hyperparameters

* λ = 0.3-0.5 (source/target balance)
* Stage 1: 5-10 epochs
* LR = 3e-5

### Results

* NAS-Bench-360: SOTA 9/10 tasks
* CIFAR-100: 6.48% error

### Using for Project

* Modality gap measurement D(M_paper, M_video)
* Gradient alignment principle validates EWC
* Davies-Bouldin index for knowledge preservation
* Linear probe accuracy tracking

---

## Paper 3: LoRA+ (ICML 2024)

**Efficient Low Rank Adaptation of Large Models**

### Core Contribution

Asymmetric learning rates ηB = λ·ηA for efficient feature learning.

### Key Theory

```
ΔZ_B = B·ΔZ_A + ΔB·Z_A + ΔB·ΔZ_A
       \__δ¹__/   \__δ²__/   \__δ³__/
Efficiency: δ¹ = δ² = Θ(1) requires η_A = Θ(n^{-1}), η_B = Θ(1)
Optimal: η_B/η_A = Θ(n), practical λ ∈ {4,8,16}
```

### Hyperparameters

* λ = 16 (RoBERTa), λ = 2-4 (LLaMA)
* r ∈ {4,8,16,64}, α ∈ {8,16}

### Results

* MMLU: +1.3% (44.0% vs 42.7%)
* 2× convergence speedup
* Harder tasks show larger gains

### Using for Project

* Asymmetric learning rates: ηA = 1e-4, ηB = 8e-4
* Monitor δ¹/δ² norms (target: both Θ(1))
* Lambda sweep: {4, 8, 16} in Week 5

---

## Paper 4: OPLoRA (AAAI 2026)

**Orthogonal Projection LoRA Prevents Catastrophic Forgetting**

### Core Contribution

Double-sided orthogonal projection preserves top-k singular directions.

### Key Theory

```
ΔW = PL·BA·PR
PL = I - Uk·Uk^T, PR = I - Vk·Vk^T
Guarantee: Uk^T·W'·Vk = Σk (top-k preserved)
ρk = ||Qk·ΔW||²F / ||ΔW||²F  [alignment metric]
```

### Hyperparameters

* k ∈ {16, 128}
* r = 32-64
* Overhead: 19% training time

### Results

* LLaMA-2 7B: +5-7% forgetting resistance
* ρk < 0.003 (minimal interference)
* SOTA on commonsense, math, code tasks

### Using for Project

* Start k=16, scale to k=128 if ρk > 0.5
* One-time SVD: 5.5 min, cached
* Monitor ρk every 100 steps in Phase 2-3
* Target: ρk < 0.3

---

## Paper 5: DoRA (arXiv 2024)

**Weight-Decomposed Low-Rank Adaptation**

### Core Contribution

Decompose weights into magnitude + direction, apply LoRA only to direction.

### Key Formula

```
W = m · (V/||V||c) = ||W||c · (W/||W||c)
DoRA: W' = m' · (V + ΔV)/||V + ΔV||c
```

### Results

* LLaMA-7B: +3.7% vs LoRA (commonsense)
* VL-BART: +0.9% (image-text)
* Shows FT-like learning patterns

### Using for Project

* **Optional enhancement** if LoRA+ insufficient
* Magnitude-only tuning for MLP layers
* Expected gain: +2-4%
* VRAM cost: +0.2GB

---

## Paper 6: EWC (PNAS 2017)

**Overcoming Catastrophic Forgetting in Neural Networks**

### Core Contribution

Fisher information weighted quadratic penalty on parameters.

### Key Formula

```
L_EWC = λ_ewc · Σ_i [F_i · (θ_i - θ*_i)²]
F_i = E[(∂log p(y|x,θ) / ∂θ_i)²]
```

### Using for Project

* λ_ewc = 200 (Phase 2), 400 (Phase 3)
* Compute Fisher after Phase 1
* Diagonal approximation (efficient)

---

## Paper 7: LfVS (CVPR 2024)

**Learning from Long-Form Video for Summarization**

### Core Contribution

LLM-based extractive summarization using pseudo-ground truth.

### Key Insights

* Text encoder improves video summarization (+2.3% F1)
* Cross-modal attention critical
* Pretraining on large-scale pseudo-labels transfers

### Using for Project

* Validates text+video multi-modal approach
* Cross-attention between paper/video embeddings
* Evaluation metrics: ROUGE, CIDEr-D, BERTScore

---

## Paper 8: VISTA (2024)

**Video Summarization Dataset for Academic Videos**

### Core Contribution

18.6K academic video-summary pairs with plan-based generation.

### Key Statistics

* Videos: 6.8 min avg, 16.36 shots
* Summaries: 192.6 tokens, 7.19 sentences
* Domains: multiple academic fields

### Using for Project

* Validates academic video summarization task
* Provides baseline metrics
* Plan-based approach: 2-stage (plan → summary)

---

## Paper 9: CoMM (2024)

**Cross-Modal Mutual Learning**

### Core Contribution

Bidirectional knowledge transfer between modalities.

### Key Formula

```
L_mutual = L_src→tgt + L_tgt→src
```

### Using for Project

* Bidirectional paper↔video alignment
* Validates curriculum mixing strategy

---

## Integrated Framework Summary

### Six Synergistic Methods

1. **OPLoRA** - Subspace preservation (k=16/128)
2. **LoRA+** - Asymmetric LR (λ=8, ηB=8·ηA)
3. **CrossCLR** - Contrastive learning (τ=0.03, λ=0.7)
4. **MoNA** - Gradient alignment (D_gap monitoring)
5. **EWC** - Fisher weighting (λ=200/400)
6. **DoRA** - Optional (magnitude decomposition)

### Curriculum Learning

* **Phase 1** (2 epochs): 100% papers, LoRA+ only
* **Phase 2** (1 epoch): 70/30 mix, +OPLoRA k=16, +EWC λ=200
* **Phase 3** (1 epoch): 50/50 mix, OPLoRA k=16/128, EWC λ=400

### Composite Loss

```
L_total = L_CE 
        + 0.3·L_crossclr 
        + 0.2·L_diversity 
        + 0.4·L_terminology
        + λ_ewc·L_EWC
```

### Key Monitoring Metrics

* **ρk** < 0.3 (subspace interference)
* **δ¹/δ²** ∈ [0.5, 2.0] (feature efficiency)
* **D_gap** < 0.7 (modality gap)
* **Probe accuracy** > 85% (knowledge retention)

### Expected Performance

* Paper R1: ≥0.35 (maintain baseline)
* Video R1: ≥0.37 (+42% vs degraded)
* Training: 8 weeks (2× speedup from LoRA+)
* Combined gain: +8-12%

---

## Critical Hyperparameters (Week 5 Tuning)

* **λ_ratio** ∈ {4, 8, 16} - LoRA+ asymmetry
* **k** ∈ {16, 128} - OPLoRA projection rank
* **λ_ewc** ∈ {200, 400} - EWC strength
* **τ** = 0.03 - CrossCLR temperature (fixed)
* **γ** = 0.9 - Connectivity threshold (fixed)

### Decision Criteria

* If **ρk > 0.5** in Phase 2 → scale k: 16→128
* If **D_gap > 0.7** after Phase 1 → extend to 3 epochs
* If **δ¹/δ² ∉ [0.5,2.0]** → adjust λ_ratio
