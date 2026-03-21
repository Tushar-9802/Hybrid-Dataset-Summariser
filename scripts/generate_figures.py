"""
Generate fig6_ablation.png — Component ablation comparison.
Run from repo root: python scripts/generate_fig6.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'figures/fig6_ablation.png'

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
})

configs = ['Vanilla\nLoRA', 'Full\nFramework', '$-$ EWC', '$-$ OPLoRA']
video_r1 = [0.315, 0.417, 0.438, 0.389]
video_pvr = [18.4, 14.1, 11.0, 9.8]
video_bert = [0.091, 0.151, 0.169, 0.104]
video_nr = [None, 6.5, 6.5, 10.5]  # NR not available for vanilla

x = np.arange(len(configs))
width = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

# Left panel: Video R-1 and BERTScore
bars1 = ax1.bar(x - width/2, video_r1, width, label='Video R-1', color='#4C72B0',
                edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x + width/2, video_bert, width, label='BERTScore', color='#55A868',
                edgecolor='white', linewidth=0.5)

ax1.set_ylabel('Score')
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_ylim(0, 0.55)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Quality Metrics', fontsize=8, fontweight='bold')

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=5.5)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=5.5)

# Right panel: PVR and NR (style contamination)
colors_pvr = ['#C44E52', '#DD8452', '#55A868', '#4C72B0']
bars3 = ax2.bar(x - width/2, video_pvr, width, label='Passive Voice %',
                color=colors_pvr, edgecolor='white', linewidth=0.5)

# NR bars (skip vanilla which is None)
nr_vals = [0 if v is None else v for v in video_nr]
nr_colors = ['lightgray' if v is None else '#8172B2' for v in video_nr]
bars4 = ax2.bar(x + width/2, nr_vals, width, label='Nominalization %',
                color=nr_colors, edgecolor='white', linewidth=0.5)

ax2.axhline(y=16, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
ax2.text(3.6, 16.5, 'Target: 16%', fontsize=5.5, ha='right', style='italic', alpha=0.7)

ax2.set_ylabel('Contamination (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_ylim(0, 22)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Style Contamination', fontsize=8, fontweight='bold')

for bar, val in zip(bars3, video_pvr):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=5.5, fontweight='bold')
for bar, val in zip(bars4, video_nr):
    if val is not None:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=5.5)

fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved {OUT}")