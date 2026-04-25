"""Generate Slide 15 heatmap: Coda Type × Unit with transparent background."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──
data = [
    ('1+1+3', 160, 210, 116, 'all'),
    ('5R1',    78,  62,  96, 'all'),
    ('4D',     23,  65,  79, 'all'),
    ('7D1',    14,  52,  56, 'all'),
    ('3R1',    18,   8,  29, 'all'),
    ('4R1',     5,  12,  28, 'all'),
    ('3',       7,   5,  18, 'all'),
    ('5D1',     1,   3,  15, 'all'),
    ('5',       0,   2,  38, 'partial'),
    ('4',      12,   0,  14, 'partial'),
    ('7R1',     0,   9,  16, 'partial'),
    ('6R1',     3,   0,  11, 'partial'),
    ('6D',      0,   0,  35, 'exclusive'),
    ('9',       0,   0,  18, 'exclusive'),
    ('8',       0,   0,  15, 'exclusive'),
]

types = [d[0] for d in data]
counts = np.array([[d[1], d[2], d[3]] for d in data], dtype=float)
sharing = [d[4] for d in data]

# Row-normalize for color intensity
row_sums = counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
proportions = counts / row_sums

# ── Colors ──
UNIT_COLORS = ['#4c72b0', '#dd8452', '#55a868']
SHARING_COLORS = {'all': '#4f7088', 'partial': '#8e9bff', 'exclusive': '#e8e28b'}
BG_COLOR = '#D9D4CD'

# ── Figure ──
fig, ax = plt.subplots(figsize=(6, 8))
fig.patch.set_alpha(0)
ax.set_facecolor('none')

n_rows, n_cols = counts.shape

for i in range(n_rows):
    for j in range(n_cols):
        val = counts[i, j]
        prop = proportions[i, j]
        # Unit-tinted cell
        base_r = int(UNIT_COLORS[j][1:3], 16) / 255
        base_g = int(UNIT_COLORS[j][3:5], 16) / 255
        base_b = int(UNIT_COLORS[j][5:7], 16) / 255
        alpha = max(0.05, prop * 0.7) if val > 0 else 0.02
        cell_color = (base_r, base_g, base_b, alpha)
        rect = mpatches.FancyBboxPatch(
            (j + 0.02, n_rows - i - 1 + 0.02), 0.96, 0.96,
            boxstyle='round,pad=0.02',
            facecolor=cell_color,
            edgecolor='none',
        )
        ax.add_patch(rect)
        # Annotation
        txt = str(int(val)) if val > 0 else '—'
        ax.text(j + 0.5, n_rows - i - 0.5, txt,
                ha='center', va='center',
                fontsize=11, fontweight='bold' if val > 0 else 'normal',
                color='#111111' if val > 0 else '#999999')

# Sharing category strip on the left
for i, s in enumerate(sharing):
    y = n_rows - i - 0.5
    ax.plot(-0.3, y, 'o', markersize=7, color=SHARING_COLORS[s],
            clip_on=False, zorder=5)

# Section dividers
sections = []
current = sharing[0]
start = 0
for i, s in enumerate(sharing):
    if s != current:
        sections.append((current, start, i))
        current = s
        start = i
sections.append((current, start, len(sharing)))

for label, s_start, s_end in sections:
    y_top = n_rows - s_start
    y_bot = n_rows - s_end
    ax.axhline(y=y_top, color=SHARING_COLORS[label], alpha=0.3, linewidth=0.8, xmin=-0.05, xmax=1.02)

# Axes setup
ax.set_xlim(-0.5, n_cols)
ax.set_ylim(0, n_rows)
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_xticklabels(['Unit A', 'Unit D', 'Unit F'],
                    fontsize=12, fontweight='bold')
for idx, label in enumerate(ax.get_xticklabels()):
    label.set_color(UNIT_COLORS[idx])

ax.set_yticks([n_rows - i - 0.5 for i in range(n_rows)])
ax.set_yticklabels(types, fontsize=11, fontweight='bold')

ax.tick_params(axis='both', which='both', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Proportion bars on the right
bar_x_start = n_cols + 0.3
bar_width = 1.8
for i in range(n_rows):
    total = counts[i].sum()
    if total == 0:
        continue
    x_offset = bar_x_start
    for j in range(n_cols):
        w = (counts[i, j] / total) * bar_width
        if w > 0:
            rect = mpatches.FancyBboxPatch(
                (x_offset, n_rows - i - 1 + 0.3), w, 0.4,
                boxstyle='round,pad=0.01',
                facecolor=UNIT_COLORS[j],
                edgecolor='none',
            )
            ax.add_patch(rect)
            x_offset += w

ax.set_xlim(-0.6, bar_x_start + bar_width + 0.2)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=SHARING_COLORS['all'], label='All 3 units (9)'),
    mpatches.Patch(facecolor=SHARING_COLORS['partial'], label='2 units (6)'),
    mpatches.Patch(facecolor=SHARING_COLORS['exclusive'], label='Exclusive (5)'),
]
ax.legend(handles=legend_elements, loc='lower right',
          frameon=False, fontsize=9, ncol=3,
          bbox_to_anchor=(1.0, -0.04))

plt.tight_layout()
out = 'slide-app/public/figures/shared_vocabulary_heatmap.png'
fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0.1)
print(f'Saved → {out}')
plt.close()
