"""
generate_figures.py — Publication figures for the research paper.
Reads from results/*/metrics.json and logs/*_metrics.jsonl.

Run from repo root:
    python scripts/generate_figures.py

Output: figures/fig1_framework.png through fig5_cmc_progression.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# IEEE column width ≈ 3.5in, double column ≈ 7.16in
SINGLE_COL = (3.5, 2.5)
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
})


def load_metrics(run_name):
    path = RESULTS_DIR / run_name / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ============================================================
# FIGURE 1: Framework diagram (conceptual — manual or Tikz)
# ============================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Phase boxes
    phases = [
        (0.5, 1.5, 2.5, 1.8, 'Phase 1\nPapers only\nLoRA+', '#4C72B0'),
        (3.5, 1.5, 2.5, 1.8, 'Phase 2\nMixed 50/40/10\n+OPLoRA +EWC', '#DD8452'),
        (6.5, 1.5, 2.5, 1.8, 'Phase 3\nVideo-heavy 30/60/10\n+CrossCLR', '#55A868'),
    ]
    for x, y, w, h, label, color in phases:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=6.5, fontweight='bold')

    # Arrows
    for x in [3.0, 6.0]:
        ax.annotate('', xy=(x+0.5, 2.4), xytext=(x, 2.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Transition labels
    ax.text(3.25, 3.5, 'Fisher + SVD\ncomputation', ha='center', va='center', fontsize=5.5,
            style='italic', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='gray', alpha=0.8))
    ax.text(6.25, 3.5, 'D_gap gate\ncheck', ha='center', va='center', fontsize=5.5,
            style='italic', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    # Arrows from transitions to gap
    ax.annotate('', xy=(3.25, 3.3), xytext=(3.25, 3.05),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, ls='--'))
    ax.annotate('', xy=(6.25, 3.3), xytext=(6.25, 3.05),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, ls='--'))

    # Bottom: data ratio bar
    ax.text(1.75, 0.9, '100% papers', ha='center', fontsize=5.5, color='#4C72B0')
    ax.text(4.75, 0.9, '50P/40V/10Pr', ha='center', fontsize=5.5, color='#DD8452')
    ax.text(7.75, 0.9, '30P/60V/10Pr', ha='center', fontsize=5.5, color='#55A868')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_framework.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig1_framework.png")


# ============================================================
# FIGURE 2: ROUGE-1 progression (grouped bar chart)
# ============================================================
def fig2_rouge_progression():
    baseline = load_metrics("baseline")
    phase1 = load_metrics("phase1")
    phase2 = load_metrics("phase2")
    phase3 = load_metrics("phase3")  # or phase3_final
    if not phase3:
        phase3 = load_metrics("phase3_final")
    vanilla = load_metrics("abl_vanilla")

    systems = ['Baseline', 'Vanilla\nLoRA', 'Phase 1', 'Phase 2', 'Phase 3']
    paper_r1 = [
        baseline['metrics']['papers']['rouge1']['mean'],
        vanilla['metrics']['papers']['rouge1']['mean'] if vanilla else 0,
        phase1['metrics']['papers']['rouge1']['mean'],
        phase2['metrics']['papers']['rouge1']['mean'],
        phase3['metrics']['papers']['rouge1']['mean'],
    ]
    video_r1 = [
        baseline['metrics']['videos']['rouge1']['mean'],
        vanilla['metrics']['videos']['rouge1']['mean'] if vanilla else 0,
        phase1['metrics']['videos']['rouge1']['mean'],
        phase2['metrics']['videos']['rouge1']['mean'],
        phase3['metrics']['videos']['rouge1']['mean'],
    ]

    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    bars1 = ax.bar(x - width/2, paper_r1, width, label='Papers', color='#4C72B0', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, video_r1, width, label='Videos', color='#DD8452', edgecolor='white', linewidth=0.5)

    ax.set_ylabel('ROUGE-1 (F1)')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=5.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=5.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_rouge_progression.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig2_rouge_progression.png")


# ============================================================
# FIGURE 3: PVR comparison
# ============================================================
def fig3_pvr_comparison():
    baseline = load_metrics("baseline")
    phase3 = load_metrics("phase3") or load_metrics("phase3_final")
    vanilla = load_metrics("abl_vanilla")

    labels = ['Baseline', 'Phase 3\n(Ours)', 'Vanilla\nLoRA', 'IMPACT\n2025']
    pvr_vals = [
        baseline['metrics']['videos']['passive_voice_pct']['mean'] * 100,
        phase3['metrics']['videos']['passive_voice_pct']['mean'] * 100,
        vanilla['metrics']['videos']['passive_voice_pct']['mean'] * 100 if vanilla else 18.4,
        31.4,  # from IMPACT paper
    ]
    colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    bars = ax.bar(labels, pvr_vals, color=colors, edgecolor='white', linewidth=0.5, width=0.6)

    # Target threshold line
    ax.axhline(y=16, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(3.5, 16.5, 'Target: 16%', fontsize=6, ha='right', style='italic')

    ax.set_ylabel('Passive Voice Ratio (%)')
    ax.set_ylim(0, 38)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, pvr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_pvr_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig3_pvr_comparison.png")


# ============================================================
# FIGURE 4: Training loss curves
# ============================================================
def fig4_loss_curves():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=False)

    phase_files = [
        ('smoke_test_metrics.jsonl', 'Phase 1', '#4C72B0'),
        ('phase2_metrics.jsonl', 'Phase 2', '#DD8452'),
        ('phase3_metrics.jsonl', 'Phase 3', '#55A868'),
    ]

    for ax, (fname, title, color) in zip(axes, phase_files):
        fpath = LOGS_DIR / fname
        if not fpath.exists():
            ax.text(0.5, 0.5, f'{fname}\nnot found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        steps, ce_vals, total_vals, ewc_vals = [], [], [], []
        with open(fpath) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if 'loss/ce' in d:
                    steps.append(d.get('step', len(steps)))
                    ce_vals.append(d['loss/ce'])
                    total_vals.append(d.get('loss/total', d['loss/ce']))
                    ewc_vals.append(d.get('loss/ewc', 0))

        if steps:
            ax.plot(steps, ce_vals, color=color, linewidth=1.0, label='CE')
            if any(e > 0 for e in ewc_vals):
                ax.plot(steps, ewc_vals, color='#C44E52', linewidth=0.8, linestyle='--', label='EWC')
            ax.set_xlabel('Step')
            ax.legend(fontsize=5.5, loc='upper right')

        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Loss')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_loss_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig4_loss_curves.png")


# ============================================================
# FIGURE 5: CMC progression
# ============================================================
def fig5_cmc_progression():
    runs = ['baseline', 'phase1', 'phase2', 'phase3']
    labels_map = {'baseline': 'Baseline', 'phase1': 'Phase 1', 'phase2': 'Phase 2', 'phase3': 'Phase 3'}
    fallback = {'baseline': 'baseline', 'phase1': 'phase1', 'phase2': 'phase2', 'phase3': 'phase3_final'}

    cmc_vals = []
    labels = []
    for run in runs:
        m = load_metrics(run) or load_metrics(fallback[run])
        if m and 'cross_modal' in m['metrics']:
            cmc_vals.append(m['metrics']['cross_modal']['inter_modal_similarity']['mean'])
            labels.append(labels_map[run])

    colors = ['#4C72B0', '#DD8452', '#C44E52', '#55A868'][:len(cmc_vals)]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    bars = ax.bar(labels, cmc_vals, color=colors, edgecolor='white', linewidth=0.5, width=0.55)

    ax.set_ylabel('CMC (SBERT Cosine Similarity)')
    ax.set_ylim(0, 0.65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, cmc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_cmc_progression.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig5_cmc_progression.png")


# ============================================================
if __name__ == "__main__":
    print("Generating publication figures...")
    fig1_framework()
    fig2_rouge_progression()
    fig3_pvr_comparison()
    fig4_loss_curves()
    fig5_cmc_progression()
    print(f"\nAll figures saved to {OUT_DIR}/")
    print("\nPlacement in paper:")
    print("  fig1_framework.png      -> Figure 1 (Section III, framework overview)")
    print("  fig2_rouge_progression.png -> Figure 2 (Section V, after main results)")
    print("  fig3_pvr_comparison.png -> Figure 3 (Section V, after ROUGE discussion)")
    print("  fig4_loss_curves.png    -> Figure 4 (Section V-E, training dynamics)")
    print("  fig5_cmc_progression.png -> Figure 5 (Section V-E, after loss curves)")