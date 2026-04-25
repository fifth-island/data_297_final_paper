"""
Generate real figures for Slide 11: The Rhythm Channel.
Produces from dswp_labels.csv:
  1. ici_violin.png        — Violin plot of ICI distributions by social unit
  2. tsne_by_unit.png      — t-SNE of standardised ICI vectors, coloured by unit
  3. tsne_by_codatype.png  — t-SNE of standardised ICI vectors, coloured by coda type
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE, "datasets", "dswp_labels.csv")
OUT_DIR = os.path.join(BASE, "slide-app", "public", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 200
BG_COLOR = "#D9D4CD"
TEXT_COLOR = "#111111"
TEXT_SEC = "#5A5A5A"

UNIT_COLORS = {"A": "#4c72b0cc", "D": "#dd8452cc", "F": "#55a868cc"}
UNIT_ORDER = ["A", "D", "F"]

# Top coda types for t-SNE coloring
TOP_TYPES = ["1+1+3", "5R1", "4D", "7D1", "3R1"]
TYPE_COLORS = {
    "1+1+3": "#e377c2",
    "5R1":   "#17becf",
    "4D":    "#bcbd22",
    "7D1":   "#9467bd",
    "3R1":   "#8c564b",
}
OTHER_COLOR = "#cccccc"

# ── Load & prepare data ─────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)
df = df[df["is_noise"] == 0].copy()  # 1,383 clean codas
print(f"Clean codas: {len(df)}")

# Parse ICI sequences into a padded matrix (max 9 ICIs)
MAX_ICI = 9

def parse_ici(seq_str):
    vals = [float(x) for x in seq_str.split("|")]
    padded = vals[:MAX_ICI] + [0.0] * (MAX_ICI - len(vals))
    return padded

ici_matrix = np.array([parse_ici(s) for s in df["ici_sequence"]])
units = df["unit"].values
coda_types = df["coda_type"].values

# Compute mean ICI per coda (for violin plot) — only non-zero values
mean_icis = []
for row in ici_matrix:
    nonzero = row[row > 0]
    mean_icis.append(np.mean(nonzero) * 1000 if len(nonzero) > 0 else 0)  # ms
mean_icis = np.array(mean_icis)

# ── Figure 1: Violin plot ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(4.5, 3.0), facecolor=BG_COLOR)
ax1.set_facecolor(BG_COLOR)

violin_data = [mean_icis[units == u] for u in UNIT_ORDER]
parts = ax1.violinplot(violin_data, positions=[1, 2, 3], showmeans=True,
                       showmedians=True, showextrema=False)

# Style the violins
for i, (body, u) in enumerate(zip(parts["bodies"], UNIT_ORDER)):
    body.set_facecolor(UNIT_COLORS[u])
    body.set_edgecolor(UNIT_COLORS[u])
    body.set_alpha(0.65)

parts["cmeans"].set_color(TEXT_COLOR)
parts["cmeans"].set_linewidth(1.5)
parts["cmedians"].set_color(TEXT_COLOR)
parts["cmedians"].set_linewidth(1.0)
parts["cmedians"].set_linestyle("--")

ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels([f"Unit {u}" for u in UNIT_ORDER], fontsize=10, fontweight="bold")
ax1.set_ylabel("Mean ICI (ms)", fontsize=10, color=TEXT_COLOR)
ax1.tick_params(axis="both", labelsize=8, colors=TEXT_COLOR)

# Add summary stats as annotations
for i, u in enumerate(UNIT_ORDER):
    d = violin_data[i]
    med = np.median(d)
    ax1.annotate(f"med={med:.0f}ms", (i + 1, med), xytext=(15, 5),
                 textcoords="offset points", fontsize=7, color=TEXT_SEC,
                 fontweight="600")

# Color x-tick labels
for i, label in enumerate(ax1.get_xticklabels()):
    label.set_color(UNIT_COLORS[UNIT_ORDER[i]])

for spine in ax1.spines.values():
    spine.set_color(TEXT_COLOR)
    spine.set_linewidth(0.5)

fig1.tight_layout(pad=0.5)
fig1.savefig(os.path.join(OUT_DIR, "ici_violin.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig1)
print("✓ ici_violin.png saved")

# ── t-SNE computation ───────────────────────────────────────────
scaler = StandardScaler()
ici_scaled = scaler.fit_transform(ici_matrix)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
coords = tsne.fit_transform(ici_scaled)
print(f"t-SNE computed: {coords.shape}")

# ── Figure 2: t-SNE coloured by unit ───────────────────────────
fig2, ax2 = plt.subplots(figsize=(4.0, 3.5), facecolor=BG_COLOR)
ax2.set_facecolor(BG_COLOR)

for u in UNIT_ORDER:
    mask = units == u
    ax2.scatter(coords[mask, 0], coords[mask, 1],
                c=UNIT_COLORS[u], s=8, alpha=0.5, label=f"Unit {u}",
                edgecolors="none")

ax2.set_xticks([])
ax2.set_yticks([])
ax2.legend(fontsize=8, loc="upper right", framealpha=0.8,
           edgecolor=BG_COLOR, facecolor=BG_COLOR)
for spine in ax2.spines.values():
    spine.set_visible(False)

fig2.tight_layout(pad=0.3)
fig2.savefig(os.path.join(OUT_DIR, "tsne_by_unit.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig2)
print("✓ tsne_by_unit.png saved")

# ── Figure 3: t-SNE coloured by coda type ──────────────────────
fig3, ax3 = plt.subplots(figsize=(4.0, 3.5), facecolor=BG_COLOR)
ax3.set_facecolor(BG_COLOR)

# Plot "other" types first (background)
other_mask = ~np.isin(coda_types, TOP_TYPES)
ax3.scatter(coords[other_mask, 0], coords[other_mask, 1],
            c=OTHER_COLOR, s=5, alpha=0.25, label="Other", edgecolors="none")

# Plot top types on top
for ct in TOP_TYPES:
    mask = coda_types == ct
    ax3.scatter(coords[mask, 0], coords[mask, 1],
                c=TYPE_COLORS[ct], s=12, alpha=0.7, label=ct,
                edgecolors="none")

ax3.set_xticks([])
ax3.set_yticks([])
ax3.legend(fontsize=7, loc="upper right", framealpha=0.8,
           edgecolor=BG_COLOR, facecolor=BG_COLOR, ncol=2)
for spine in ax3.spines.values():
    spine.set_visible(False)

fig3.tight_layout(pad=0.3)
fig3.savefig(os.path.join(OUT_DIR, "tsne_by_codatype.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig3)
print("✓ tsne_by_codatype.png saved")

print(f"\nAll figures saved to {OUT_DIR}")
