"""
Generate real figures for Slide 12: The Spectral Channel.
Produces from dswp_labels.csv + audio:
  1. mel_grid.png          — 3×2 grid of real mel-spectrograms (2 per unit)
  2. centroid_violin.png   — Violin plot of spectral centroids by unit
  3. ici_vs_centroid.png   — Scatter of mean ICI vs spectral centroid, coloured by unit
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Config ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE, "datasets", "dswp_audio")
LABELS_CSV = os.path.join(BASE, "datasets", "dswp_labels.csv")
OUT_DIR = os.path.join(BASE, "slide-app", "public", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 200
BG_COLOR = "#D9D4CD"
TEXT_COLOR = "#111111"
TEXT_SEC = "#5A5A5A"
SR = 22050
N_MELS = 64
FMAX = 8000

UNIT_COLORS = {"A": "#4c72b0cc", "D": "#dd8452cc", "F": "#55a868cc"}
UNIT_ORDER = ["A", "D", "F"]

# Representative codas: 2 per unit
SAMPLE_CODAS = [
    {"coda_id": 55,  "unit": "A", "type": "1+1+3"},
    {"coda_id": 42,  "unit": "A", "type": "4R2"},
    {"coda_id": 274, "unit": "D", "type": "5R1"},
    {"coda_id": 277, "unit": "D", "type": "1+1+3"},
    {"coda_id": 610, "unit": "F", "type": "1+1+3"},
    {"coda_id": 774, "unit": "F", "type": "4D"},
]

# ── Load data ───────────────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)
df = df[df["is_noise"] == 0].copy()
print(f"Clean codas: {len(df)}")

# ── Figure 1: 3×2 Mel-Spectrogram Grid ─────────────────────────
fig1 = plt.figure(figsize=(6.5, 6.75), facecolor=BG_COLOR)
gs = GridSpec(3, 2, figure=fig1, hspace=0.35, wspace=0.15)

for idx, spec in enumerate(SAMPLE_CODAS):
    row, col = idx // 2, idx % 2
    ax = fig1.add_subplot(gs[row, col])

    wav_path = os.path.join(AUDIO_DIR, f"{spec['coda_id']}.wav")
    y, sr = librosa.load(wav_path, sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_dB, sr=sr, fmax=FMAX, ax=ax, cmap="magma",
                             x_axis="time", y_axis="mel")
    ax.set_title(f"Unit {spec['unit']} · {spec['type']} · coda {spec['coda_id']}",
                 fontsize=8, color=UNIT_COLORS[spec["unit"]], fontweight="bold", pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("" if col == 1 else "Hz", fontsize=7)
    ax.tick_params(axis="both", labelsize=6, colors=TEXT_COLOR)

    # Only show x-label on bottom row
    if row < 2:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Time (s)", fontsize=7)

fig1.savefig(os.path.join(OUT_DIR, "mel_grid.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.08)
plt.close(fig1)
print("✓ mel_grid.png saved")

# ── Compute spectral centroids for all clean codas ──────────────
print("Computing spectral centroids for all clean codas...")
centroids = []
mean_icis = []

MAX_ICI = 9
for _, row in df.iterrows():
    cid = row["coda_id"]
    wav_path = os.path.join(AUDIO_DIR, f"{cid}.wav")
    if not os.path.exists(wav_path):
        continue
    try:
        y_audio, _ = librosa.load(wav_path, sr=SR)
        cent = librosa.feature.spectral_centroid(y=y_audio, sr=SR)
        centroids.append({"coda_id": cid, "unit": row["unit"],
                          "centroid": float(np.mean(cent))})
        # Also compute mean ICI
        ici_vals = [float(x) for x in row["ici_sequence"].split("|")]
        mean_icis.append({"coda_id": cid, "unit": row["unit"],
                          "mean_ici": np.mean(ici_vals) * 1000,
                          "centroid": float(np.mean(cent))})
    except Exception as e:
        print(f"  skip {cid}: {e}")

cent_df = pd.DataFrame(centroids)
ici_cent_df = pd.DataFrame(mean_icis)
print(f"  Computed centroids for {len(cent_df)} codas")

# ── Figure 2: Centroid Violin ───────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(4.5, 3.0), facecolor=BG_COLOR)
ax2.set_facecolor(BG_COLOR)

violin_data = [cent_df[cent_df["unit"] == u]["centroid"].values for u in UNIT_ORDER]
parts = ax2.violinplot(violin_data, positions=[1, 2, 3], showmeans=True,
                       showmedians=True, showextrema=False)

for i, (body, u) in enumerate(zip(parts["bodies"], UNIT_ORDER)):
    body.set_facecolor(UNIT_COLORS[u])
    body.set_edgecolor(UNIT_COLORS[u])
    body.set_alpha(0.65)

parts["cmeans"].set_color(TEXT_COLOR)
parts["cmeans"].set_linewidth(1.5)
parts["cmedians"].set_color(TEXT_COLOR)
parts["cmedians"].set_linewidth(1.0)
parts["cmedians"].set_linestyle("--")

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels([f"Unit {u}" for u in UNIT_ORDER], fontsize=10, fontweight="bold")
ax2.set_ylabel("Spectral Centroid (Hz)", fontsize=10, color=TEXT_COLOR)
ax2.tick_params(axis="both", labelsize=8, colors=TEXT_COLOR)

for i, u in enumerate(UNIT_ORDER):
    d = violin_data[i]
    med = np.median(d)
    ax2.annotate(f"med={med:.0f}Hz", (i + 1, med), xytext=(15, 5),
                 textcoords="offset points", fontsize=7, color=TEXT_SEC,
                 fontweight="600")

for i, label in enumerate(ax2.get_xticklabels()):
    label.set_color(UNIT_COLORS[UNIT_ORDER[i]])

for spine in ax2.spines.values():
    spine.set_color(TEXT_COLOR)
    spine.set_linewidth(0.5)

fig2.tight_layout(pad=0.5)
fig2.savefig(os.path.join(OUT_DIR, "centroid_violin.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig2)
print("✓ centroid_violin.png saved")

# ── Figure 3: ICI vs Centroid Scatter ───────────────────────────
fig3, ax3 = plt.subplots(figsize=(4.5, 3.5), facecolor=BG_COLOR)
ax3.set_facecolor(BG_COLOR)

for u in UNIT_ORDER:
    mask = ici_cent_df["unit"] == u
    sub = ici_cent_df[mask]
    ax3.scatter(sub["mean_ici"], sub["centroid"],
                c=UNIT_COLORS[u], s=8, alpha=0.45,
                label=f"Unit {u}", edgecolors="none")

# Compute and display Pearson r
from scipy.stats import pearsonr
r_val, p_val = pearsonr(ici_cent_df["mean_ici"], ici_cent_df["centroid"])
ax3.annotate(f"Pearson r = {r_val:.3f}", xy=(0.98, 0.95), xycoords="axes fraction",
             ha="right", va="top", fontsize=10, fontweight="bold", color=TEXT_COLOR,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=TEXT_COLOR, linewidth=0.5))

ax3.set_xlabel("Mean ICI (ms)", fontsize=10, color=TEXT_COLOR)
ax3.set_ylabel("Spectral Centroid (Hz)", fontsize=10, color=TEXT_COLOR)
ax3.tick_params(axis="both", labelsize=8, colors=TEXT_COLOR)
ax3.legend(fontsize=8, loc="lower right", framealpha=0.8,
           edgecolor=BG_COLOR, facecolor=BG_COLOR)

for spine in ax3.spines.values():
    spine.set_color(TEXT_COLOR)
    spine.set_linewidth(0.5)

fig3.tight_layout(pad=0.5)
fig3.savefig(os.path.join(OUT_DIR, "ici_vs_centroid.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig3)
print("✓ ici_vs_centroid.png saved")

print(f"\nAll figures saved to {OUT_DIR}")
print(f"\nStats: Pearson r(ICI, centroid) = {r_val:.4f}, p = {p_val:.4e}")
for u in UNIT_ORDER:
    sub = cent_df[cent_df["unit"] == u]["centroid"]
    print(f"  Unit {u}: median centroid = {sub.median():.0f} Hz, std = {sub.std():.0f} Hz")
