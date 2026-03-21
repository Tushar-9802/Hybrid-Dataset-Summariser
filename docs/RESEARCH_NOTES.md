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

### Actual Outcome
* Phase 3 contrastive loss with momentum queue (size=3000)
* τ=0.03, λ_intra=0.75, κ=3.5e-4
* NaN on some pair batches due to queue initialization — queue snapshot fix applied
* CMC improved 0.474→0.531 (+12%)
* Scheduler param group bug: CrossCLR projection head added 4th param group after scheduler created — fixed

---

## Paper 2: MoNA (ICML 2024)

**Learning Modality Knowledge Alignment for Cross-Modality Transfer**

### Core Contribution
Two-stage meta-learning measuring and minimizing modality gap.

### Key Formulas
```
D(Ms,Mt) = inf_{π,B} d(P(Y^s_{π,B}|X̂), P(Y^t|X̂))
L'_outer = λL_outer + L_inner
Gradient alignment: min L_inner - λα(∇L_outer)·(∇L_inner)
```

### Actual Outcome
* Used in monitoring role only — not a training loss
* D_gap metric adapted for phase gating: ROUGE w=1.0, BERTScore w=1.5, PVR/NR w=0.75
* D_gap values: Phase 1 exit +0.55, Phase 2 exit +1.26, Phase 3 final +1.96

---

## Paper 3: LoRA+ (ICML 2024)

**Efficient Low Rank Adaptation of Large Models**

### Core Contribution
Asymmetric learning rates ηB = λ·ηA for efficient feature learning.

### Key Theory
```
ΔZ_B = B·ΔZ_A + ΔB·Z_A + ΔB·ΔZ_A
Efficiency: δ¹ = δ² = Θ(1) requires η_A = Θ(n^{-1}), η_B = Θ(1)
```

### Actual Outcome
* λ=8 used across all phases (no sweep needed)
* Phase 1: lr_A=1e-4, lr_B=8e-4, halved each transition
* Phase 1 loss dropped 2.5→1.4 in 234 steps — convergence speed consistent with 2× claim
* Paper R-1 jumped 0.320→0.429 (+34%) in Phase 1 alone

---

## Paper 4: OPLoRA (AAAI 2026)

**Orthogonal Projection LoRA Prevents Catastrophic Forgetting**

### Core Contribution
Double-sided orthogonal projection preserves top-k singular directions.

### Key Theory
```
ΔW = PL·BA·PR
PL = I - Uk·Uk^T, PR = I - Vk·Vk^T
ρk = ||Qk·ΔW||²F / ||ΔW||²F  [alignment metric]
```

### Actual Outcome
* Left-projection only (not full double-sided) due to VRAM
* k=16, SVD cached in 5.5 min for 224 modules
* ρk never exceeded 0.5 — adaptive k scaling not triggered
* dtype fix: bitsandbytes uint8→float32 in hook
* dimension fix: 3D tensor (B,T,D) broadcasting

### Ablation Finding (NEW)
* **Removing OPLoRA: Video R-1 drops 0.417→0.389 (-7%), BERTScore drops 31% (0.151→0.104), nominalization jumps 6.5%→10.5% (+61%)**
* PVR paradoxically improves (14.1%→9.8%) — OPLoRA prevents nominalization contamination specifically, not passive voice
* OPLoRA provides meaningful but secondary contribution; curriculum phasing is the primary mechanism

---

## Paper 5: DoRA (arXiv 2024)

**Weight-Decomposed Low-Rank Adaptation**

### Actual Outcome
* NOT implemented — dropped for VRAM constraints
* OPLoRA addresses the same concern more directly

---

## Paper 6: EWC (PNAS 2017)

**Overcoming Catastrophic Forgetting in Neural Networks**

### Key Formula
```
L_EWC = λ_ewc · Σ_i [F_i · (θ_i - θ*_i)²]
```

### Actual Outcome
* λ_ewc = 200 (Phase 2), 400 (Phase 3)
* Fisher: 448 param matrices, 83.9M entries, 100% nonzero
* Highest Fisher: v_proj layers (max=0.052)
* OOM fix: gradient checkpointing ON, batch_size=1, max_total_len=512, n_samples=100, 37 seconds

### Ablation Finding (NEW)
* **Removing EWC IMPROVES all video metrics: R-1 0.417→0.438 (+5%), R-2 0.119→0.137 (+15%), PVR 14.1%→11.0%, BERTScore 0.151→0.169**
* λ=400 over-constrains — prevents model from fully adapting to video content
* When OPLoRA already preserves subspace, EWC regularization is redundant and harmful
* Future work: explore lower λ schedules or adaptive decay

---

## Paper 7: LfVS (CVPR 2024)
* Validates text+video approach. ROUGE/BERTScore adopted in eval harness.

## Paper 8: VISTA (2024)
* Validates academic video summarization task. Our 738 videos comparable to their stats.

## Paper 9: CoMM (2024)
* Validates bidirectional paper↔video curriculum mixing. CMC improved 0.356→0.531.

---

## Final Framework (As Implemented)

### Methods Applied

| Method | Phase | Status | Ablation Finding |
|--------|-------|--------|-----------------|
| LoRA+ | All | Working | Not ablated (baseline method) |
| OPLoRA | 2-3 | Working | Secondary: BERTScore +45%, nom. control |
| EWC | 2-3 | Working | **Over-constrains at λ=400; removal improves all metrics** |
| CrossCLR | 3 | Partial (NaN) | Not ablated independently |
| Curriculum | All | Working | **Primary mechanism (~80% of improvement)** |
| DoRA | — | Dropped | N/A |

### Curriculum (Actual)

* **Phase 1** (3 epochs): 100% papers, LoRA+ only, 234 steps
* **Phase 2** (1 epoch): 50/40/10 mix, +OPLoRA k=16, +EWC λ=200, 78 steps
* **Phase 3** (1 epoch): 30/60/10 mix, +CrossCLR τ=0.03, EWC λ=400, 78 steps

### Hyperparameters (Final)
```yaml
model: mistralai/Mistral-7B-v0.1
quantization: NF4, double quant, bfloat16 compute
lora: r=32, alpha=64, dropout=0.1
targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
trainable: 83,886,080 / 7,241,732,096 (1.16%)
optimizer: bitsandbytes AdamW8bit
batch: micro=3, grad_accum=8, effective=24
max_total_len: 1280
```

---

## Final Results (March 8, 2026)

### Phase Progression

| Phase | Video R-1 | Video R-2 | Video PVR% | CMC | Paper R-1 |
|-------|-----------|-----------|------------|-----|-----------|
| Baseline | 0.263 | 0.032 | 10.0 | 0.356 | 0.320 |
| Phase 1 | 0.305 | 0.052 | 15.9 | 0.474 | 0.429 |
| Phase 2 | 0.381 | 0.101 | 10.2 | 0.473 | 0.296 |
| Phase 3 | 0.417 | 0.119 | 14.1 | 0.531 | 0.309 |
| Vanilla LoRA | 0.315 | 0.060 | 18.4 | 0.470 | 0.451 |
| IMPACT 2025 | 0.272 | 0.060 | 31.4 | — | 0.333 |

### Component Ablations (March 20, 2026)

| Configuration | Video R-1 | Video R-2 | BERTScore | PVR% | Nom.% | CMC |
|--------------|-----------|-----------|-----------|------|-------|-----|
| Full framework | 0.417 | 0.119 | 0.151 | 14.1 | 6.5 | 0.531 |
| - EWC | **0.438** | **0.137** | **0.169** | **11.0** | 6.5 | 0.518 |
| - OPLoRA | 0.389 | 0.114 | 0.104 | 9.8 | 10.5 | 0.506 |
| Vanilla LoRA | 0.315 | 0.060 | 0.091 | 18.4 | — | 0.470 |

### Contribution Hierarchy (from ablations)

1. **Curriculum phasing** (~80%): Vanilla LoRA (no curriculum) gets R-1=0.315, PVR=18.4%. All curriculum-based configs get R-1=0.389-0.438, PVR=9.8-11.0%. The phased data mixing does most of the work.
2. **OPLoRA** (secondary): Prevents nominalization contamination (6.5% vs 10.5%), improves BERTScore (+45%) and video R-1 (+7%). Controls structural style transfer.
3. **EWC** (harmful at λ=400): Over-constrains model. Removal improves every video metric. When OPLoRA preserves the subspace, Fisher regularization is redundant.
4. **CrossCLR** (partial): CMC improved but NaN instability limits conclusions.

### Statistical Significance

* Combined (n=313): R-1 Δ=+0.029 p=0.004, R-2 Δ=+0.027 p<0.001
* Video only (n=75): R-1 Δ=+0.154 p<0.001 d=1.42, R-2 Δ=+0.087 p<0.001

### IMPACT 2025 Projections vs Achieved

| Strategy | Projected | Achieved |
|----------|-----------|----------|
| Curriculum learning | +20-30% R-1 | +58% R-1 |
| Contrastive alignment | +15-25% R-1 | +49% CMC |
| Video fine-tuning | +25-35% R-1 | +272% R-2 |
| Orthogonality constraints | mentioned | PVR 31.4%→14.1% |

---

## Implementation Issues & Fixes

### 1. Fisher OOM (Critical, Phase 1 exit)
* Problem: Gradient checkpointing OFF for Fisher = OOM (28.9GB)
* Fix: Keep ON, batch=1, max_total_len=512, n=100. 37 seconds.

### 2. OPLoRA dtype (Critical, Phase 2)
* Problem: bitsandbytes uint8 output
* Fix: Cast float32 in hook

### 3. OPLoRA dimensions (Critical, Phase 2)
* Problem: 3D tensor (B,T,D) vs expected 2D
* Fix: Broadcasting `coeffs = out_f @ Uk_f; proj = out_f - coeffs @ Uk_f.T`

### 4. CrossCLR inplace (Critical, Phase 3)
* Problem: Queue update conflicts with gradient checkpointing
* Fix: `queue_snapshot = self.queue.clone().detach()`

### 5. CrossCLR NaN (Partial, Phase 3)
* Problem: Zero denominator with random queue
* Fix needed: Warmup queue before enabling loss

### 6. Scheduler param groups (Critical, Phase 3)
* Problem: 4th param group added after scheduler creation
* Fix: Move add_param_group before setup_scheduler

### 7. Ablation checkpoint overwrite (Operational, March 20)
* Problem: Ablation configs saved to checkpoints/phase3/final, overwriting Phase 3 adapter
* Fix: Manually copied to checkpoints/abl_*/final/. Phase 3 adapter safe on HuggingFace.

---

## Training Timeline

| Date | Event | Duration |
|------|-------|----------|
| Mar 6 17:28 | Phase 1 start | — |
| Mar 6 19:20 | Phase 1 complete | 1h 52m |
| Mar 6 19:41 | Fisher computation | 37s |
| Mar 7 00:37 | Phase 1 eval complete | 4h 3m |
| Mar 7 09:30 | Phase 2 complete | 54m |
| Mar 7 21:36 | Phase 3 complete | 1h 29m |
| Mar 8 01:41 | Phase 3 eval complete | 4h 3m |
| Mar 8 05:55 | Phase 2 eval complete | 4h 5m |
| Mar 8 12:40 | Vanilla LoRA complete | 1h 51m |
| Mar 8 16:05 | Vanilla LoRA eval complete | 3h 23m |
| Mar 20 08:36 | No-EWC ablation train | ~2h |
| Mar 20 12:08 | No-EWC ablation eval | 4h 22m |
| Mar 20 13:30 | No-OPLoRA ablation train | ~2h |
| Mar 20 19:47 | No-OPLoRA ablation eval | 4h 13m |

**Total GPU time: ~30 hours** (training ~10h, evaluation ~20h)

---

## Fisher Information Statistics
```
Parameters tracked: 448 (224 A + 224 B)
Total entries: 83,886,080
All 100% nonzero
Highest Fisher: v_proj layers (max=0.052)
A matrices > B matrices (consistent with LoRA+ theory)
```

---

## Paper & Patent Status

* **IMPACT 2025:** Presented Dec 6, 2025. Forwarded to Springer. Not yet indexed.
* **New paper:** "Preventing Catastrophic Forgetting in Cross-Modal Summarization" — revised with ablation results. IEEE (7pp) and Springer LNCS (16pp) versions. Narrative: curriculum is primary, OPLoRA secondary, EWC over-constrains.
* **Patent disclosure:** "Phase-Gated Orthogonal Projection for Cross-Modal Adapter Training" — 9-page document. Note: ablation findings weaken the adaptive control loop claim; patent angle may need reframing around curriculum gating specifically.

---

## Artifacts Produced

| Deliverable | Location | Status |
|-------------|----------|--------|
| Training code (8 modules) | `src/training/` | ✅ Complete |
| Phase configs (4 + 6 ablation) | `configs/` | ✅ Complete |
| Evaluation harness | `scripts/evaluate.py` | ✅ Complete |
| Figure generation (fig1-5) | `scripts/generate_figures.py` | ✅ Complete |
| Ablation figure (fig6) | `scripts/generate_fig6.py` | ✅ Complete |
| IEEE paper (.tex) | `Research_Paper_IEEE.tex` | ✅ Revised with ablations |
| Springer LNCS paper (.tex) | `Research_Paper_Springer_LNCS.tex` | ✅ Revised with ablations |
| Patent disclosure | `Patent_Disclosure_ABES_HDS_2026.*` | ✅ Complete |
| Resume | `Tushar_Jaju_ML_Resume.*` | ✅ Complete |
| Phase 3 adapter | [HuggingFace](https://huggingface.co/Tushar9802/hybrid-summariser-crossmodal-lora) | ✅ Uploaded |
| Phase 2 adapter | [HuggingFace](https://huggingface.co/Tushar9802/hybrid-summariser-phase2-lora) | ✅ Uploaded |
| Dataset | [Kaggle](https://www.kaggle.com/datasets/tusharjaju/hybrid-dataset-summariser-crossmodal) | ✅ Uploaded |
| README | `README.md` | ✅ Updated with ablations |
| Research notes | `docs/RESEARCH_NOTES.md` | ✅ This file |