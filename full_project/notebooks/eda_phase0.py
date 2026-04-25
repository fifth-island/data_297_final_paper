"""Phase 0 EDA — executed as a script, outputs figures to figures/phase0/"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────────
BASE    = "/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project"
DATA    = f"{BASE}/datasets"
AUDIO   = f"{DATA}/dswp_audio"
FIGDIR  = f"{BASE}/figures/phase0"
os.makedirs(FIGDIR, exist_ok=True)

UNIT_COLORS = {"A": "#2196F3", "D": "#FF9800", "F": "#4CAF50"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── load labels ────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{DATA}/dswp_labels.csv")
clean = df[df.is_noise == 0].reset_index(drop=True)
print(f"Total: {len(df)}  Clean: {len(clean)}  Noise: {(df.is_noise==1).sum()}")

# ── parse ICI sequences ────────────────────────────────────────────────────────
def parse_ici(s):
    if pd.isna(s): return []
    return [float(x)*1000 for x in str(s).split("|")]  # ms

clean["ici_list"] = clean["ici_sequence"].apply(parse_ici)
clean["mean_ici_ms"] = clean["ici_list"].apply(lambda x: np.mean(x) if x else np.nan)
clean["year"] = pd.to_datetime(clean["date"], dayfirst=True).dt.year

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Label distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DSWP Label Distributions", fontsize=15, fontweight="bold")

# 1a unit distribution
unit_counts = clean["unit"].value_counts().sort_index()
colors = [UNIT_COLORS[u] for u in unit_counts.index]
bars = axes[0,0].bar(unit_counts.index, unit_counts.values, color=colors, edgecolor="black", linewidth=0.7)
axes[0,0].set_title("Social Unit Distribution (clean codas)")
axes[0,0].set_xlabel("Social Unit")
axes[0,0].set_ylabel("Count")
for bar, val in zip(bars, unit_counts.values):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f"{val}\n({val/len(clean)*100:.1f}%)",
                   ha="center", va="bottom", fontsize=9)

# 1b noise vs clean
noise_counts = df["is_noise"].value_counts().sort_index()
axes[0,1].bar(["Clean", "Noise"], noise_counts.values, color=["#4CAF50","#F44336"], edgecolor="black", linewidth=0.7)
axes[0,1].set_title("Clean vs. Noise Codas (all 1,501)")
axes[0,1].set_ylabel("Count")
for i, v in enumerate(noise_counts.values):
    axes[0,1].text(i, v+2, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

# 1c top 15 coda types
top_types = clean["coda_type"].value_counts().head(15)
axes[1,0].barh(top_types.index[::-1], top_types.values[::-1], color="#7986CB", edgecolor="black", linewidth=0.5)
axes[1,0].set_title("Top 15 Coda Types (clean codas)")
axes[1,0].set_xlabel("Count")

# 1d individual ID distribution (IDN-labeled only)
idn = clean[clean["individual_id"].astype(str) != "0"]
idn_counts = idn["individual_id"].value_counts().head(13)
axes[1,1].bar(idn_counts.index.astype(str), idn_counts.values, color="#EF5350", edgecolor="black", linewidth=0.5)
axes[1,1].set_title(f"Individual ID Distribution (N={len(idn)} labeled codas)")
axes[1,1].set_xlabel("Individual ID")
axes[1,1].set_ylabel("Count")
axes[1,1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig1_label_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig1 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — ICI distributions per coda type
# ══════════════════════════════════════════════════════════════════════════════
# Only top 10 coda types for readability
top10 = clean["coda_type"].value_counts().head(10).index.tolist()
ici_df = clean[clean["coda_type"].isin(top10)].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("ICI Distributions by Coda Type (top 10)", fontsize=14, fontweight="bold")

# violin
order = ici_df.groupby("coda_type")["mean_ici_ms"].median().sort_values().index.tolist()
sns.violinplot(data=ici_df, x="coda_type", y="mean_ici_ms", order=order,
               palette="Set2", inner="quartile", ax=axes[0])
axes[0].set_title("Mean ICI (ms) — Violin")
axes[0].set_xlabel("Coda Type")
axes[0].set_ylabel("Mean ICI (ms)")
axes[0].tick_params(axis="x", rotation=45)

# boxplot
sns.boxplot(data=ici_df, x="coda_type", y="mean_ici_ms", order=order,
            palette="Set2", showfliers=False, ax=axes[1])
axes[1].set_title("Mean ICI (ms) — Boxplot")
axes[1].set_xlabel("Coda Type")
axes[1].set_ylabel("Mean ICI (ms)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_ici_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig2 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Duration and click count distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Duration and Click Count Distributions", fontsize=14, fontweight="bold")

# duration histogram by unit
for unit, grp in clean.groupby("unit"):
    axes[0].hist(grp["duration_sec"], bins=40, alpha=0.6, label=f"Unit {unit}", color=UNIT_COLORS[unit])
axes[0].set_title("Duration by Social Unit")
axes[0].set_xlabel("Duration (s)")
axes[0].set_ylabel("Count")
axes[0].legend()

# n_clicks by unit
for unit, grp in clean.groupby("unit"):
    axes[1].hist(grp["n_clicks"], bins=range(1,16), alpha=0.6, label=f"Unit {unit}", color=UNIT_COLORS[unit], align="left")
axes[1].set_title("Click Count by Social Unit")
axes[1].set_xlabel("Number of Clicks")
axes[1].set_ylabel("Count")
axes[1].legend()

# duration vs n_clicks scatter
sc = axes[2].scatter(clean["n_clicks"], clean["duration_sec"],
                     c=[{"A":0,"D":1,"F":2}[u] for u in clean["unit"]],
                     cmap="Set1", alpha=0.3, s=10)
axes[2].set_title("Duration vs. Click Count")
axes[2].set_xlabel("Number of Clicks")
axes[2].set_ylabel("Duration (s)")
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor=UNIT_COLORS[u], markersize=8, label=f"Unit {u}") for u in ["A","D","F"]]
axes[2].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_duration_clicks.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig3 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Coda type × unit heatmap
# ══════════════════════════════════════════════════════════════════════════════
# top 15 coda types
top15 = clean["coda_type"].value_counts().head(15).index
ct_unit = pd.crosstab(clean[clean["coda_type"].isin(top15)]["coda_type"],
                       clean[clean["coda_type"].isin(top15)]["unit"])
ct_norm = ct_unit.div(ct_unit.sum(axis=1), axis=0)  # row-normalize

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Coda Type × Social Unit (top 15 types)", fontsize=14, fontweight="bold")

sns.heatmap(ct_unit, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Raw Counts")
axes[0].set_xlabel("Social Unit")
axes[0].set_ylabel("Coda Type")

sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
axes[1].set_title("Row-Normalized (fraction per coda type)")
axes[1].set_xlabel("Social Unit")
axes[1].set_ylabel("Coda Type")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_codatype_unit_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig4 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — IDN=0 investigation
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("IDN=0 (Unidentified Individual) Investigation", fontsize=14, fontweight="bold")

idn0 = clean[clean["individual_id"].astype(str) == "0"]
idn_known = clean[clean["individual_id"].astype(str) != "0"]

# by unit
idn0_unit = idn0["unit"].value_counts().sort_index()
idn_known_unit = idn_known["unit"].value_counts().sort_index()
x = np.arange(3)
w = 0.35
axes[0].bar(x - w/2, [idn0_unit.get(u,0) for u in ["A","D","F"]], w, label="IDN=0", color="#EF5350")
axes[0].bar(x + w/2, [idn_known_unit.get(u,0) for u in ["A","D","F"]], w, label="IDN known", color="#42A5F5")
axes[0].set_xticks(x); axes[0].set_xticklabels(["A","D","F"])
axes[0].set_title("IDN=0 by Social Unit")
axes[0].set_ylabel("Count"); axes[0].legend()

# by year for unit F
f_df = clean[clean["unit"] == "F"].copy()
f_df["id_status"] = f_df["individual_id"].apply(lambda x: "IDN=0" if x == 0 else "Known")
year_id = f_df.groupby(["year","id_status"]).size().unstack(fill_value=0)
year_id.plot(kind="bar", stacked=True, ax=axes[1], color=["#EF5350","#42A5F5"])
axes[1].set_title("Unit F: IDN=0 vs Known by Year")
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=45)

# overall IDN=0 fraction by unit
totals = clean.groupby("unit").size()
idn0_totals = idn0.groupby("unit").size().reindex(["A","D","F"], fill_value=0)
frac = (idn0_totals / totals * 100).fillna(0)
axes[2].bar(frac.index, frac.values, color=[UNIT_COLORS[u] for u in frac.index], edgecolor="black")
axes[2].set_title("IDN=0 Fraction by Social Unit (%)")
axes[2].set_xlabel("Social Unit"); axes[2].set_ylabel("% Unidentified")
for i, (idx, v) in enumerate(frac.items()):
    axes[2].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_idn0_investigation.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig5 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Sample mel-spectrograms by unit
# ══════════════════════════════════════════════════════════════════════════════
print("Loading mel spectrograms for fig6...")
X_mel_full = np.load(f"{DATA}/X_mel_full.npy")  # (1383, 64, 128)

# get indices per unit (clean codas only, 1383)
unit_idx = {u: np.where(clean["unit"].values == u)[0] for u in ["A","D","F"]}

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
fig.suptitle("Sample Mel-Spectrograms by Social Unit", fontsize=14, fontweight="bold")

np.random.seed(42)
for row, unit in enumerate(["A","D","F"]):
    idxs = unit_idx[unit]
    chosen = np.random.choice(idxs, size=4, replace=False)
    for col, idx in enumerate(chosen):
        mel = X_mel_full[idx]  # (64, 128)
        im = axes[row, col].imshow(mel, aspect="auto", origin="lower", cmap="magma")
        axes[row, col].set_title(f"Unit {unit} — coda #{clean.iloc[idx]['coda_id']}", fontsize=9)
        axes[row, col].set_xlabel("Time frames")
        axes[row, col].set_ylabel("Mel bin")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig6_sample_spectrograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig6 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — t-SNE of raw ICI vectors
# ══════════════════════════════════════════════════════════════════════════════
print("Computing t-SNE of ICI vectors...")

# build ICI matrix (zero-padded to length 9, in ms)
ici_cols = [f"ICI{i}" for i in range(1, 10)]
# reconstruct from ici_list
def build_ici_matrix(df_clean):
    mat = np.zeros((len(df_clean), 9), dtype=np.float32)
    for i, row in enumerate(df_clean["ici_list"]):
        for j, v in enumerate(row[:9]):
            mat[i, j] = v
    return mat

X_ici = build_ici_matrix(clean)
scaler = StandardScaler()
X_ici_scaled = scaler.fit_transform(X_ici)

tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
Z = tsne.fit_transform(X_ici_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("t-SNE of Zero-Padded ICI Vectors (1383 clean codas)", fontsize=14, fontweight="bold")

# colored by unit
for unit, grp_idx in [("A", clean["unit"]=="A"), ("D", clean["unit"]=="D"), ("F", clean["unit"]=="F")]:
    axes[0].scatter(Z[grp_idx, 0], Z[grp_idx, 1], c=UNIT_COLORS[unit], label=f"Unit {unit}",
                    alpha=0.5, s=8, rasterized=True)
axes[0].set_title("Colored by Social Unit")
axes[0].legend(markerscale=3); axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

# colored by coda type (top 5 + other)
top5 = clean["coda_type"].value_counts().head(5).index.tolist()
palette = sns.color_palette("tab10", len(top5)+1)
for i, ct in enumerate(top5):
    mask = clean["coda_type"] == ct
    axes[1].scatter(Z[mask, 0], Z[mask, 1], c=[palette[i]], label=ct, alpha=0.6, s=8, rasterized=True)
other = ~clean["coda_type"].isin(top5)
axes[1].scatter(Z[other, 0], Z[other, 1], c=[palette[-1]], label="Other", alpha=0.3, s=6, rasterized=True)
axes[1].set_title("Colored by Coda Type (top 5)")
axes[1].legend(markerscale=3, fontsize=8); axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig7_tsne_ici.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig7 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Spectral centroid distribution
# ══════════════════════════════════════════════════════════════════════════════
print("Computing spectral centroids for all 1383 clean codas...")

SR = 22050
N_SAMPLE = 200  # use subsample for speed if needed; we'll try full set

coda_ids = clean["coda_id"].values
centroids = []
for cid in coda_ids[:N_SAMPLE]:
    wav_path = f"{AUDIO}/{cid}.wav"
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        c = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroids.append(float(np.mean(c)))
    except Exception:
        centroids.append(np.nan)

units_sample = clean["unit"].values[:N_SAMPLE]
cent_df = pd.DataFrame({"centroid_hz": centroids, "unit": units_sample})
cent_df = cent_df.dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Spectral Centroid Distribution (200-coda sample)", fontsize=14, fontweight="bold")

for unit in ["A","D","F"]:
    sub = cent_df[cent_df["unit"]==unit]["centroid_hz"]
    axes[0].hist(sub, bins=25, alpha=0.6, label=f"Unit {unit}", color=UNIT_COLORS[unit])
axes[0].set_title("Spectral Centroid by Unit")
axes[0].set_xlabel("Spectral Centroid (Hz)"); axes[0].set_ylabel("Count"); axes[0].legend()

sns.violinplot(data=cent_df, x="unit", y="centroid_hz", palette=UNIT_COLORS, inner="box", ax=axes[1])
axes[1].set_title("Spectral Centroid Violin by Unit")
axes[1].set_xlabel("Social Unit"); axes[1].set_ylabel("Spectral Centroid (Hz)")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig8_spectral_centroid.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig8 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Summary stats
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== PHASE 0 SUMMARY STATS ===")
print(f"Total codas: {len(df)}")
print(f"Clean codas: {len(clean)} (noise: {len(df)-len(clean)})")
print(f"Social units: {dict(clean['unit'].value_counts().sort_index())}")
print(f"Unique coda types (clean): {clean['coda_type'].nunique()}")
idn0_mask = clean["individual_id"].astype(str) == "0"
print(f"IDN=0: {idn0_mask.sum()} ({idn0_mask.mean()*100:.1f}%)")
labeled = clean[~idn0_mask]
print(f"IDN!=0 (labeled): {len(labeled)}, {labeled['individual_id'].nunique()} individuals")
print(f"Mean duration: {clean['duration_sec'].mean():.3f}s (std={clean['duration_sec'].std():.3f}s)")
print(f"Mean ICI: {clean['mean_ici_ms'].mean():.1f}ms (std={clean['mean_ici_ms'].std():.1f}ms)")
if centroids:
    valid = [c for c in centroids if not np.isnan(c)]
    print(f"Spectral centroid (sample n={len(valid)}): mean={np.mean(valid):.0f}Hz (std={np.std(valid):.0f}Hz)")
print(f"Date range: {clean['year'].min()} – {clean['year'].max()}")
print("All Phase 0 figures saved.")
