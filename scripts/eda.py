"""
Phase 0 — Exploratory Data Analysis
Beyond WhAM: CS 297 Final Paper

Run this script to produce all EDA figures.
Figures are saved to: figures/eda/
"""

import csv
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "datasets")
AUDIO  = os.path.join(DATA, "dswp_audio")
LABELS = os.path.join(DATA, "dswp_labels.csv")
FIGS   = os.path.join(BASE, "figures", "eda")
os.makedirs(FIGS, exist_ok=True)

UNIT_COLORS = {"A": "#4C72B0", "D": "#DD8452", "F": "#55A868"}
UNIT_ORDER  = ["A", "D", "F"]

# ── Load labels ───────────────────────────────────────────────────────────────
df = pd.read_csv(LABELS)
df["ici_list"] = df["ici_sequence"].apply(
    lambda s: [float(x) for x in s.split("|")] if isinstance(s, str) and s else []
)
df["mean_ici"]  = df["ici_list"].apply(lambda x: np.mean(x) if x else np.nan)
df["mean_ici_ms"] = df["mean_ici"] * 1000          # convert to ms for readability
df["date_parsed"] = pd.to_datetime(df["date"], dayfirst=True)
df["year"] = df["date_parsed"].dt.year

df_clean = df[df["is_noise"] == 0].copy()
df_id    = df_clean[df_clean["individual_id"] != "0"].copy()

print(f"Total codas   : {len(df)}")
print(f"Clean codas   : {len(df_clean)}")
print(f"ID-labeled    : {len(df_id)}  (individual_id != 0)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Label distribution overview  (4 panels)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DSWP Label Distributions", fontsize=16, fontweight="bold")

# 1a — Social unit counts
ax = axes[0, 0]
unit_counts = df["unit"].value_counts()[UNIT_ORDER]
bars = ax.bar(UNIT_ORDER, unit_counts.values,
              color=[UNIT_COLORS[u] for u in UNIT_ORDER], edgecolor="black", linewidth=0.7)
for bar, val in zip(bars, unit_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xlabel("Social Unit"); ax.set_ylabel("Coda Count")
ax.set_title("(a) Social Unit Distribution")
ax.set_ylim(0, 1050)
noise_patch = plt.Rectangle((0,0),1,1, fc="none", ec="grey", ls="--")
ax.legend([noise_patch], [f"Includes {df['is_noise'].sum()} noise-tagged"], fontsize=9)

# 1b — Top 15 coda types (clean only)
ax = axes[0, 1]
ctype_counts = df_clean["coda_type"].value_counts().head(15)
colors_ct = [UNIT_COLORS["F"]] * len(ctype_counts)
ax.barh(ctype_counts.index[::-1], ctype_counts.values[::-1],
        color=colors_ct, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Count"); ax.set_title("(b) Top 15 Coda Types (clean only)")
ax.tick_params(axis="y", labelsize=8)

# 1c — Individual ID distribution (clean, identified)
ax = axes[1, 0]
idn_counts = df_id["individual_id"].value_counts().head(12)
ax.bar(range(len(idn_counts)), idn_counts.values,
       color="#8172B2", edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(idn_counts)))
ax.set_xticklabels(idn_counts.index, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Individual ID (IDN)"); ax.set_ylabel("Coda Count")
ax.set_title(f"(c) Individual ID Distribution  (n={len(df_id)}, unknown={df['individual_id'].eq('0').sum()})")

# 1d — Recording year distribution
ax = axes[1, 1]
year_unit = df.groupby(["year", "unit"]).size().unstack(fill_value=0)
year_unit = year_unit.reindex(columns=UNIT_ORDER, fill_value=0)
bottom = np.zeros(len(year_unit))
for unit in UNIT_ORDER:
    ax.bar(year_unit.index, year_unit[unit], bottom=bottom,
           color=UNIT_COLORS[unit], label=f"Unit {unit}", edgecolor="black", linewidth=0.4)
    bottom += year_unit[unit].values
ax.set_xlabel("Year"); ax.set_ylabel("Coda Count")
ax.set_title("(d) Recording Year × Social Unit")
ax.legend(title="Unit", fontsize=9)

plt.tight_layout()
path = os.path.join(FIGS, "fig1_label_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — ICI distribution per social unit and per coda type
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rhythm Channel: Inter-Click Interval (ICI) Distributions", fontsize=14, fontweight="bold")

# 2a — ICI distribution per unit (violin)
ax = axes[0]
unit_ici = [df_clean[df_clean["unit"] == u]["mean_ici_ms"].dropna().values for u in UNIT_ORDER]
parts = ax.violinplot(unit_ici, positions=range(len(UNIT_ORDER)), showmedians=True)
for i, (pc, u) in enumerate(zip(parts["bodies"], UNIT_ORDER)):
    pc.set_facecolor(UNIT_COLORS[u]); pc.set_alpha(0.7)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(UNIT_ORDER))); ax.set_xticklabels([f"Unit {u}" for u in UNIT_ORDER])
ax.set_ylabel("Mean ICI (ms)"); ax.set_title("(a) Mean ICI by Social Unit")
for i, (vals, u) in enumerate(zip(unit_ici, UNIT_ORDER)):
    ax.text(i, np.median(vals) + 2, f"{np.median(vals):.1f}ms", ha="center", fontsize=9)

# 2b — Mean ICI for top 10 coda types (boxplot)
ax = axes[1]
top_ctypes = df_clean["coda_type"].value_counts().head(10).index.tolist()
df_top = df_clean[df_clean["coda_type"].isin(top_ctypes)].copy()
order = df_top.groupby("coda_type")["mean_ici_ms"].median().sort_values().index.tolist()
data_by_type = [df_top[df_top["coda_type"] == ct]["mean_ici_ms"].dropna().values for ct in order]
bp = ax.boxplot(data_by_type, vert=True, patch_artist=True, labels=order,
                medianprops=dict(color="black", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#4C72B0"); patch.set_alpha(0.6)
ax.set_xlabel("Coda Type"); ax.set_ylabel("Mean ICI (ms)")
ax.set_title("(b) Mean ICI by Top 10 Coda Types")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
path = os.path.join(FIGS, "fig2_ici_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Duration and click count distributions
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Acoustic Properties of Clean Codas", fontsize=14, fontweight="bold")

# 3a — Duration distribution per unit
ax = axes[0]
for u in UNIT_ORDER:
    vals = df_clean[df_clean["unit"] == u]["duration_sec"]
    ax.hist(vals, bins=30, alpha=0.6, label=f"Unit {u}", color=UNIT_COLORS[u], density=True)
ax.set_xlabel("Duration (s)"); ax.set_ylabel("Density")
ax.set_title("(a) Coda Duration by Unit"); ax.legend()

# 3b — n_clicks distribution
ax = axes[1]
nc = df_clean["n_clicks"].astype(int)
click_counts = nc.value_counts().sort_index()
ax.bar(click_counts.index, click_counts.values, color="#DD8452", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Number of Clicks"); ax.set_ylabel("Count")
ax.set_title("(b) Clicks per Coda")
for x, v in click_counts.items():
    ax.text(x, v + 2, str(v), ha="center", fontsize=8)

# 3c — n_ici per unit
ax = axes[2]
for u in UNIT_ORDER:
    vals = df_clean[df_clean["unit"] == u]["n_ici"].astype(int)
    ax.hist(vals, bins=range(1, 15), alpha=0.6, label=f"Unit {u}",
            color=UNIT_COLORS[u], density=True)
ax.set_xlabel("Number of ICI values"); ax.set_ylabel("Density")
ax.set_title("(c) ICI Count by Unit"); ax.legend()

plt.tight_layout()
path = os.path.join(FIGS, "fig3_duration_clicks.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Coda type × Social unit heatmap  (do rhythm and unit overlap?)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

ct_unit = df_clean.groupby(["coda_type", "unit"]).size().unstack(fill_value=0)
ct_unit = ct_unit.reindex(columns=UNIT_ORDER, fill_value=0)
top20 = df_clean["coda_type"].value_counts().head(20).index
ct_unit_top = ct_unit.loc[ct_unit.index.isin(top20)]
ct_norm = ct_unit_top.div(ct_unit_top.sum(axis=1), axis=0)  # row-normalize

sns.heatmap(ct_norm, annot=ct_unit_top, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Proportion within coda type"},
            annot_kws={"size": 8})
ax.set_xlabel("Social Unit"); ax.set_ylabel("Coda Type")
ax.set_title("Coda Type × Social Unit  (counts shown, color = row proportion)\nAre coda types shared across units or unit-specific?")

plt.tight_layout()
path = os.path.join(FIGS, "fig4_codatype_unit_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — IDN=0 investigation
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Investigation of Unidentified Whales (IDN=0)", fontsize=13, fontweight="bold")

df_clean["id_known"] = df_clean["individual_id"].ne("0")

# 5a — IDN=0 by unit
ax = axes[0]
id_unit = df_clean.groupby(["unit", "id_known"]).size().unstack(fill_value=0)
id_unit.columns = ["Unknown (IDN=0)", "Identified"]
id_unit = id_unit.reindex(UNIT_ORDER)
id_unit.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"], edgecolor="black", linewidth=0.5)
ax.set_xlabel("Social Unit"); ax.set_ylabel("Count")
ax.set_title("(a) IDN=0 by Social Unit"); ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=0)

# 5b — IDN=0 by year
ax = axes[1]
id_year = df_clean.groupby(["year", "id_known"]).size().unstack(fill_value=0)
id_year.columns = ["Unknown", "Identified"]
id_year.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"], edgecolor="black", linewidth=0.5)
ax.set_xlabel("Year"); ax.set_ylabel("Count")
ax.set_title("(b) IDN=0 by Recording Year"); ax.legend(fontsize=9)

# 5c — IDN=0 by coda type (top 10)
ax = axes[2]
id_ct = df_clean[df_clean["coda_type"].isin(top20)].groupby(["coda_type", "id_known"]).size().unstack(fill_value=0)
id_ct.columns = ["Unknown", "Identified"]
pct_unknown = (id_ct["Unknown"] / id_ct.sum(axis=1) * 100).sort_values(ascending=False)
pct_unknown.plot(kind="bar", ax=ax, color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_ylabel("% Unknown (IDN=0)"); ax.set_xlabel("Coda Type")
ax.set_title("(c) % Unknown by Coda Type")
ax.tick_params(axis="x", rotation=45)
ax.axhline(df_clean["id_known"].eq(False).mean()*100, color="black", ls="--", lw=1.2, label=f"Overall {df_clean['id_known'].eq(False).mean()*100:.0f}%")
ax.legend(fontsize=9)

plt.tight_layout()
path = os.path.join(FIGS, "fig5_idn0_investigation.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Sample mel-spectrograms (one per unit, one per top coda type)
# ═══════════════════════════════════════════════════════════════════════════════
def pick_example(df_sub, n=6):
    """Pick n clean, non-noise examples from a subset, return coda_ids."""
    sub = df_sub[df_sub["is_noise"] == 0]
    return sub["coda_id"].tolist()[:n]

fig = plt.figure(figsize=(16, 8))
fig.suptitle("Sample Mel-Spectrograms by Social Unit (2 per unit)", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.3)

examples = {}
for u in UNIT_ORDER:
    ids = pick_example(df_clean[df_clean["unit"] == u], n=2)
    examples[u] = ids

plot_idx = 0
for u in UNIT_ORDER:
    for coda_id in examples[u]:
        ax = fig.add_subplot(gs[UNIT_ORDER.index(u), examples[u].index(coda_id)])
        wav_path = os.path.join(AUDIO, f"{coda_id}.wav")
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                 fmax=8000, ax=ax, cmap="magma")
        ctype = df_clean[df_clean["coda_id"] == coda_id]["coda_type"].values[0]
        dur   = df_clean[df_clean["coda_id"] == coda_id]["duration_sec"].values[0]
        ax.set_title(f"Unit {u}  |  ID {coda_id}  |  {ctype}  |  {dur:.2f}s", fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel("")
        plot_idx += 1

path = os.path.join(FIGS, "fig6_sample_spectrograms.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — t-SNE of raw ICI vectors colored by unit and coda type
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

MAX_ICI = 9
ici_matrix = np.zeros((len(df_clean), MAX_ICI))
for i, row in enumerate(df_clean.itertuples()):
    vals = row.ici_list
    for j, v in enumerate(vals[:MAX_ICI]):
        ici_matrix[i, j] = v

scaler = StandardScaler()
ici_scaled = scaler.fit_transform(ici_matrix)

print("Running t-SNE on ICI vectors...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb = tsne.fit_transform(ici_scaled)

df_clean = df_clean.copy()
df_clean["tsne_x"] = emb[:, 0]
df_clean["tsne_y"] = emb[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("t-SNE of Raw ICI Vectors", fontsize=14, fontweight="bold")

# 7a — colored by social unit
ax = axes[0]
for u in UNIT_ORDER:
    mask = df_clean["unit"] == u
    ax.scatter(df_clean.loc[mask, "tsne_x"], df_clean.loc[mask, "tsne_y"],
               c=UNIT_COLORS[u], label=f"Unit {u}", alpha=0.6, s=15, edgecolors="none")
ax.set_title("(a) Colored by Social Unit")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend()

# 7b — colored by coda type (top 8)
ax = axes[1]
top8 = df_clean["coda_type"].value_counts().head(8).index.tolist()
palette = sns.color_palette("tab10", len(top8))
for ct, color in zip(top8, palette):
    mask = df_clean["coda_type"] == ct
    ax.scatter(df_clean.loc[mask, "tsne_x"], df_clean.loc[mask, "tsne_y"],
               color=color, label=ct, alpha=0.6, s=15, edgecolors="none")
other_mask = ~df_clean["coda_type"].isin(top8)
ax.scatter(df_clean.loc[other_mask, "tsne_x"], df_clean.loc[other_mask, "tsne_y"],
           color="lightgrey", label="Other", alpha=0.3, s=8, edgecolors="none")
ax.set_title("(b) Colored by Coda Type (top 8)")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
path = os.path.join(FIGS, "fig7_tsne_ici.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Spectral centroid per unit  (from audio)
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing spectral centroids (this takes ~1-2 min)...")

SAMPLE = 200   # use a random sample to keep it fast
rng = np.random.default_rng(42)
sample_df = df_clean.groupby("unit").apply(
    lambda g: g.sample(min(len(g), SAMPLE//3), random_state=42)
).reset_index(drop=True)

centroids = []
for row in sample_df.itertuples():
    wav_path = os.path.join(AUDIO, f"{row.coda_id}.wav")
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroids.append(float(np.mean(cent)))

sample_df = sample_df.copy()
sample_df["spectral_centroid_hz"] = centroids

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Spectral Channel: Spectral Centroid by Social Unit", fontsize=13, fontweight="bold")

ax = axes[0]
unit_cents = [sample_df[sample_df["unit"] == u]["spectral_centroid_hz"].values for u in UNIT_ORDER]
parts = ax.violinplot(unit_cents, positions=range(len(UNIT_ORDER)), showmedians=True)
for pc, u in zip(parts["bodies"], UNIT_ORDER):
    pc.set_facecolor(UNIT_COLORS[u]); pc.set_alpha(0.7)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(UNIT_ORDER)))
ax.set_xticklabels([f"Unit {u}" for u in UNIT_ORDER])
ax.set_ylabel("Spectral Centroid (Hz)"); ax.set_title("(a) Centroid Distribution by Unit")
for i, (vals, u) in enumerate(zip(unit_cents, UNIT_ORDER)):
    ax.text(i, np.median(vals) + 30, f"{np.median(vals):.0f}Hz", ha="center", fontsize=9)

ax = axes[1]
ax.scatter(sample_df["mean_ici_ms"], sample_df["spectral_centroid_hz"],
           c=[UNIT_COLORS[u] for u in sample_df["unit"]], alpha=0.5, s=20, edgecolors="none")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=UNIT_COLORS[u], label=f"Unit {u}") for u in UNIT_ORDER]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel("Mean ICI (ms)"); ax.set_ylabel("Spectral Centroid (Hz)")
ax.set_title("(b) Rhythm vs. Spectral Channel  (are they independent?)")

plt.tight_layout()
path = os.path.join(FIGS, "fig8_spectral_centroid.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("EDA COMPLETE — Key numbers for the paper")
print("=" * 60)
print(f"Total codas                : {len(df)}")
print(f"Clean (non-noise) codas    : {len(df_clean)}")
print(f"Noise codas                : {df['is_noise'].sum()} ({df['is_noise'].mean()*100:.1f}%)")
print(f"Social units               : {df['unit'].nunique()}  (A={df['unit'].eq('A').sum()}, D={df['unit'].eq('D').sum()}, F={df['unit'].eq('F').sum()})")
print(f"Unit F imbalance           : {df['unit'].eq('F').mean()*100:.1f}% of all codas")
print(f"Coda types                 : {df_clean['coda_type'].nunique()}")
print(f"Top coda type (1+1+3)      : {df_clean['coda_type'].eq('1+1+3').sum()} codas ({df_clean['coda_type'].eq('1+1+3').mean()*100:.1f}%)")
print(f"Individual IDs (IDN!=0)    : {len(df_id)} codas, {df_id['individual_id'].nunique()} individuals")
print(f"IDN=0 (unidentified)       : {df['individual_id'].eq('0').sum()} ({df['individual_id'].eq('0').mean()*100:.1f}%)")
print(f"Date range                 : {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}")
print(f"Mean duration (clean)      : {df_clean['duration_sec'].mean():.3f}s  (std={df_clean['duration_sec'].std():.3f}s)")
print(f"Mean ICI (clean)           : {df_clean['mean_ici_ms'].mean():.1f}ms  (std={df_clean['mean_ici_ms'].std():.1f}ms)")
print()
print("Figures saved to:", FIGS)
