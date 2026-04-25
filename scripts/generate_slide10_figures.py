"""
Generate real figures for Slide 10: Anatomy of a Coda.
Produces 3 PNGs from coda 1077.wav (Unit F, type 1+1+3):
  1. waveform.png  — amplitude vs. time
  2. ici_timeline.png — click events with ICI brackets
  3. mel_spectrogram.png — 64-mel spectrogram, magma colormap
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Config ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE, "datasets", "dswp_audio")
LABELS_CSV = os.path.join(BASE, "datasets", "dswp_labels.csv")
OUT_DIR = os.path.join(BASE, "slide-app", "public", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

CODA_ID = 1077  # Unit F, 1+1+3, 5 clicks, 1.50s
SR = 22050      # librosa default resample rate
N_MELS = 64
FMAX = 8000
DPI = 200
BG_COLOR = "#D9D4CD"
TEXT_COLOR = "#111111"
UNIT_F_COLOR = "#4f7088"
LONG_GAP_COLOR = "#8e9bff"   # periwinkle for the "1+1" long gaps
SHORT_GAP_COLOR = "#e8e28b"  # soft yellow for the "3" short gaps

# ── Load data ───────────────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)
row = df[df["coda_id"] == CODA_ID].iloc[0]
ici_vals = [float(x) for x in row["ici_sequence"].split("|")]
n_clicks = int(row["n_clicks"])
duration = float(row["duration_sec"])
coda_type = row["coda_type"]
unit = row["unit"]

wav_path = os.path.join(AUDIO_DIR, f"{CODA_ID}.wav")
y, sr = librosa.load(wav_path, sr=SR)

# Compute click times from ICIs (cumulative sum starting at 0)
click_times = [0.0]
for ici in ici_vals:
    click_times.append(click_times[-1] + ici)

print(f"Coda {CODA_ID}: Unit {unit}, Type {coda_type}, {n_clicks} clicks, {duration:.3f}s")
print(f"ICI values (s): {ici_vals}")
print(f"Click times (s): {[f'{t:.4f}' for t in click_times]}")

# ── Figure 1: Waveform ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(6, 2.7), facecolor=BG_COLOR)
ax1.set_facecolor(BG_COLOR)

time_axis = np.linspace(0, len(y) / sr, num=len(y))
ax1.plot(time_axis, y, color=UNIT_F_COLOR, linewidth=0.4, alpha=0.85)
ax1.axhline(0, color=UNIT_F_COLOR, linewidth=0.3, alpha=0.2)

# Mark click times with subtle vertical lines
for ct in click_times:
    ax1.axvline(ct, color=UNIT_F_COLOR, linewidth=0.8, alpha=0.5, linestyle="--")

ax1.set_xlim(0, len(y) / sr)
ax1.set_xlabel("Time (s)", fontsize=9, color=TEXT_COLOR, fontfamily="sans-serif")
ax1.set_ylabel("Amplitude", fontsize=9, color=TEXT_COLOR, fontfamily="sans-serif")
ax1.tick_params(axis="both", labelsize=7, colors=TEXT_COLOR)
for spine in ax1.spines.values():
    spine.set_color(TEXT_COLOR)
    spine.set_linewidth(0.5)

fig1.tight_layout(pad=0.4)
fig1.savefig(os.path.join(OUT_DIR, "waveform.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig1)
print("✓ waveform.png saved")

# ── Figure 2: ICI Timeline ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 2.7), facecolor=BG_COLOR)
ax2.set_facecolor(BG_COLOR)

# Draw click dots on a horizontal line
y_line = 0.5
ax2.axhline(y_line, color=TEXT_COLOR, linewidth=0.8, alpha=0.3, zorder=1)

for i, ct in enumerate(click_times):
    ax2.plot(ct, y_line, "o", color=UNIT_F_COLOR, markersize=10, zorder=3,
             markeredgecolor=TEXT_COLOR, markeredgewidth=0.5)
    ax2.annotate(f"C{i+1}", (ct, y_line + 0.18), ha="center", va="bottom",
                 fontsize=7, color=TEXT_COLOR, fontweight="bold")

# Draw ICI brackets below
bracket_y = 0.22
for i, ici in enumerate(ici_vals):
    x_start = click_times[i]
    x_end = click_times[i + 1]
    x_mid = (x_start + x_end) / 2
    # Color: first 2 gaps are "long" (1+1), last 2 are "short" (3)
    color = LONG_GAP_COLOR if i < 2 else SHORT_GAP_COLOR
    # Bracket line
    ax2.plot([x_start, x_start], [bracket_y, bracket_y + 0.06], color=color, linewidth=1.2)
    ax2.plot([x_end, x_end], [bracket_y, bracket_y + 0.06], color=color, linewidth=1.2)
    ax2.plot([x_start, x_end], [bracket_y + 0.03, bracket_y + 0.03], color=color, linewidth=1.5)
    # ICI label
    ax2.annotate(f"{ici*1000:.0f} ms", (x_mid, bracket_y - 0.04), ha="center", va="top",
                 fontsize=8, color=color, fontweight="600")

ax2.set_xlim(-0.05, click_times[-1] + 0.05)
ax2.set_ylim(0, 0.85)
ax2.set_xlabel("Time (s)", fontsize=9, color=TEXT_COLOR, fontfamily="sans-serif")
ax2.set_yticks([])
ax2.tick_params(axis="x", labelsize=7, colors=TEXT_COLOR)
for spine in ["top", "right", "left"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["bottom"].set_color(TEXT_COLOR)
ax2.spines["bottom"].set_linewidth(0.5)

fig2.tight_layout(pad=0.4)
fig2.savefig(os.path.join(OUT_DIR, "ici_timeline.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig2)
print("✓ ici_timeline.png saved")

# ── Figure 3: Mel-Spectrogram ──────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 3.3), facecolor=BG_COLOR)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
S_dB = librosa.power_to_db(S, ref=np.max)

img = librosa.display.specshow(S_dB, sr=sr, fmax=FMAX, x_axis="time", y_axis="mel",
                                ax=ax3, cmap="magma")
ax3.set_xlabel("Time (s)", fontsize=9, color=TEXT_COLOR, fontfamily="sans-serif")
ax3.set_ylabel("Hz", fontsize=9, color=TEXT_COLOR, fontfamily="sans-serif")
ax3.tick_params(axis="both", labelsize=7, colors=TEXT_COLOR)

# Colorbar
cbar = fig3.colorbar(img, ax=ax3, format="%+.0f dB", pad=0.02)
cbar.ax.tick_params(labelsize=7, colors=TEXT_COLOR)
cbar.outline.set_edgecolor(TEXT_COLOR)
cbar.outline.set_linewidth(0.5)

for spine in ax3.spines.values():
    spine.set_color(TEXT_COLOR)
    spine.set_linewidth(0.5)

fig3.tight_layout(pad=0.4)
fig3.savefig(os.path.join(OUT_DIR, "mel_spectrogram.png"), dpi=DPI, facecolor=BG_COLOR,
             bbox_inches="tight", pad_inches=0.05)
plt.close(fig3)
print("✓ mel_spectrogram.png saved")

print(f"\nAll figures saved to {OUT_DIR}")
