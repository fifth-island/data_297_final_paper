"""Phase 2 — WhAM Probing
Layer-wise linear probes for 6 targets + UMAP + recording-year confound analysis.
Outputs: figures/phase2/, no new datasets.
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import spearmanr
import umap

BASE   = "/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project"
DATA   = f"{BASE}/datasets"
FIGDIR = f"{BASE}/figures/phase2"
os.makedirs(FIGDIR, exist_ok=True)

UNIT_COLORS = {"A": "#2196F3", "D": "#FF9800", "F": "#4CAF50"}
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── load data ──────────────────────────────────────────────────────────────────
df    = pd.read_csv(f"{DATA}/dswp_labels.csv")
clean = df[df.is_noise == 0].reset_index(drop=True)
clean["ici_list"]   = clean["ici_sequence"].apply(
    lambda s: [float(x)*1000 for x in str(s).split("|")] if not pd.isna(s) else [])
clean["mean_ici_ms"] = clean["ici_list"].apply(lambda x: np.mean(x) if x else np.nan)
clean["year"]        = pd.to_datetime(clean["date"], dayfirst=True).dt.year

train_idx    = np.load(f"{DATA}/train_idx.npy")
test_idx     = np.load(f"{DATA}/test_idx.npy")
train_id_idx = np.load(f"{DATA}/train_id_idx.npy")
test_id_idx  = np.load(f"{DATA}/test_id_idx.npy")

all_layers  = np.load(f"{DATA}/wham_embeddings_all_layers.npy")  # (1501, 20, 1280)
coda_ids_clean = clean["coda_id"].values
all_layers_clean = all_layers[coda_ids_clean - 1]  # (1383, 20, 1280)

# ── targets ────────────────────────────────────────────────────────────────────
y_unit  = clean["unit"].values
y_type  = clean["coda_type"].values
y_noise = df["is_noise"].values  # 1501 — need alignment below
y_clicks = clean["n_clicks"].values.astype(float)
y_ici    = clean["mean_ici_ms"].values
y_year   = clean["year"].values.astype(str)

idn_mask = clean["individual_id"].astype(str) != "0"
clean_idn = clean[idn_mask].reset_index(drop=True)
idn_positions = np.where(idn_mask.values)[0]
le_id = LabelEncoder()
y_id_all = le_id.fit_transform(clean_idn["individual_id"].astype(str))
all_layers_id = all_layers_clean[idn_positions]  # (763, 20, 1280)

# ── probe functions ────────────────────────────────────────────────────────────
def clf_probe(X, y, tr_idx, te_idx, C=1.0):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr_idx])
    Xte = sc.transform(X[te_idx])
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced",
                             random_state=42, solver="lbfgs")
    clf.fit(Xtr, y[tr_idx])
    pred = clf.predict(Xte)
    return f1_score(y[te_idx], pred, average="macro")

def reg_probe(X, y, tr_idx, te_idx):
    # exclude NaN
    valid = ~np.isnan(y)
    tr = tr_idx[valid[tr_idx]]
    te = te_idx[valid[te_idx]]
    if len(tr) < 10 or len(te) < 5: return 0.0
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    Xte = sc.transform(X[te])
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(Xtr, y[tr])
    pred = reg.predict(Xte)
    return r2_score(y[te], pred)

# ══════════════════════════════════════════════════════════════════════════════
# Layer-wise probe for all 6 targets
# ══════════════════════════════════════════════════════════════════════════════
print("Running layer-wise probes for 6 targets (20 layers × 6 targets)...")
n_layers = 20
probe_results = {t: [] for t in ["unit","coda_type","individual_id","n_clicks","mean_ici","year"]}

for layer in range(n_layers):
    Xl = all_layers_clean[:, layer, :]
    Xl_id = all_layers_id[:, layer, :]

    f_unit  = clf_probe(Xl, y_unit, train_idx, test_idx)
    f_type  = clf_probe(Xl, y_type, train_idx, test_idx)
    f_year  = clf_probe(Xl, y_year, train_idx, test_idx)
    r_click = reg_probe(Xl, y_clicks, train_idx, test_idx)
    r_ici   = reg_probe(Xl, y_ici,    train_idx, test_idx)
    f_id    = clf_probe(Xl_id, y_id_all, train_id_idx, test_id_idx)

    probe_results["unit"].append(f_unit)
    probe_results["coda_type"].append(f_type)
    probe_results["individual_id"].append(f_id)
    probe_results["n_clicks"].append(r_click)
    probe_results["mean_ici"].append(r_ici)
    probe_results["year"].append(f_year)

    print(f"  L{layer:02d}: unit={f_unit:.3f}  type={f_type:.3f}  indivID={f_id:.3f}  "
          f"n_clicks_R²={r_click:.3f}  ici_R²={r_ici:.3f}  year={f_year:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Probing profile (6-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("WhAM Layer-wise Linear Probe", fontsize=14, fontweight="bold")

probe_meta = [
    ("unit",          "Social Unit Macro-F1",     "#2196F3", "F1"),
    ("coda_type",     "Coda Type Macro-F1",        "#FF9800", "F1"),
    ("individual_id", "Individual ID Macro-F1",    "#E91E63", "F1"),
    ("n_clicks",      "Click Count (R²)",          "#9C27B0", "R²"),
    ("mean_ici",      "Mean ICI (R²)",             "#00BCD4", "R²"),
    ("year",          "Recording Year Macro-F1",   "#607D8B", "F1"),
]

layers = list(range(n_layers))
for ax, (key, title, color, metric) in zip(axes.flat, probe_meta):
    vals = probe_results[key]
    ax.plot(layers, vals, marker="o", color=color, linewidth=2, markersize=5)
    best_l = int(np.argmax(vals))
    ax.scatter([best_l], [vals[best_l]], s=100, color="red", zorder=5)
    ax.annotate(f"L{best_l}\n{vals[best_l]:.3f}", xy=(best_l, vals[best_l]),
                xytext=(best_l+0.5, vals[best_l]+0.01), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("WhAM Layer")
    ax.set_ylabel(metric)
    ax.set_xticks(range(0, 20, 2))

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_probe_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("Probing profile saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — UMAP of WhAM embeddings (best layer = L19)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing UMAP of WhAM L19...")
Xl19 = all_layers_clean[:, 19, :]
sc = StandardScaler()
Xl19_sc = sc.fit_transform(Xl19)
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
Z_umap = reducer.fit_transform(Xl19_sc)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("WhAM L19 Embeddings — UMAP", fontsize=14, fontweight="bold")

for unit in ["A","D","F"]:
    m = y_unit == unit
    axes[0].scatter(Z_umap[m,0], Z_umap[m,1], c=UNIT_COLORS[unit], label=f"Unit {unit}",
                    alpha=0.5, s=8, rasterized=True)
axes[0].set_title(f"Social Unit (F1={probe_results['unit'][19]:.3f})")
axes[0].legend(markerscale=3); axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

top5 = pd.Series(y_type).value_counts().head(5).index.tolist()
pal = sns.color_palette("tab10", 6)
for i, ct in enumerate(top5):
    m = y_type == ct
    axes[1].scatter(Z_umap[m,0], Z_umap[m,1], c=[pal[i]], label=ct, alpha=0.5, s=8, rasterized=True)
other = ~np.isin(y_type, top5)
axes[1].scatter(Z_umap[other,0], Z_umap[other,1], c=[pal[-1]], label="Other", alpha=0.3, s=5, rasterized=True)
axes[1].set_title(f"Coda Type (F1={probe_results['coda_type'][19]:.3f})")
axes[1].legend(fontsize=8, markerscale=3); axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

years = sorted(clean["year"].unique())
year_pal = sns.color_palette("viridis", len(years))
for i, yr in enumerate(years):
    m = y_year == str(yr)
    axes[2].scatter(Z_umap[m,0], Z_umap[m,1], c=[year_pal[i]], label=str(yr), alpha=0.5, s=8, rasterized=True)
axes[2].set_title(f"Recording Year (F1={probe_results['year'][19]:.3f})")
axes[2].legend(fontsize=8, markerscale=3); axes[2].set_xlabel("UMAP 1"); axes[2].set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print("WhAM UMAP saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Recording-year confound analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Recording-year confound analysis...")

# Cramér's V for unit × year
from scipy.stats import chi2_contingency
ct = pd.crosstab(clean["unit"], clean["year"])
chi2, p, dof, _ = chi2_contingency(ct.values)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"Cramér's V(unit, year) = {cramers_v:.3f} (χ²={chi2:.1f}, p={p:.2e})")

# Spearman correlation between unit-F1 and year-F1 across layers
rho, p_rho = spearmanr(probe_results["unit"], probe_results["year"])
print(f"Spearman ρ(unit-F1, year-F1) = {rho:.3f}, p={p_rho:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Recording Year Confound Analysis", fontsize=14, fontweight="bold")

# unit vs year F1 per layer
axes[0].plot(layers, probe_results["unit"], marker="o", color="#2196F3", label=f"Unit F1")
axes[0].plot(layers, probe_results["year"], marker="s", color="#607D8B", linestyle="--", label=f"Year F1")
axes[0].set_title(f"Unit vs Year F1 per Layer\nSpearman ρ={rho:.3f}, p={p_rho:.3f}")
axes[0].set_xlabel("WhAM Layer"); axes[0].set_ylabel("Macro-F1")
axes[0].legend(); axes[0].set_xticks(range(0,20,2))

# scatter: unit F1 vs year F1 per layer
axes[1].scatter(probe_results["year"], probe_results["unit"], c=layers, cmap="viridis", s=60)
for i in range(0, 20, 4):
    axes[1].annotate(f"L{i}", (probe_results["year"][i], probe_results["unit"][i]),
                     xytext=(3,3), textcoords="offset points", fontsize=8)
axes[1].set_xlabel("Year Macro-F1"); axes[1].set_ylabel("Unit Macro-F1")
axes[1].set_title(f"Unit F1 vs Year F1 (per layer)\nCramér's V = {cramers_v:.3f}")

# unit × year heatmap
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title("Unit × Recording Year (counts)")
axes[2].set_xlabel("Year"); axes[2].set_ylabel("Social Unit")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_year_confound.png", dpi=150, bbox_inches="tight")
plt.close()
print("Year confound figure saved")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
best_unit = int(np.argmax(probe_results["unit"]))
best_type = int(np.argmax(probe_results["coda_type"]))
best_id   = int(np.argmax(probe_results["individual_id"]))
best_yr   = int(np.argmax(probe_results["year"]))

print("\n=== PHASE 2 SUMMARY ===")
print(f"Best unit layer:   L{best_unit} → F1={probe_results['unit'][best_unit]:.3f}")
print(f"Best type layer:   L{best_type} → F1={probe_results['coda_type'][best_type]:.3f}")
print(f"Best indivID layer: L{best_id} → F1={probe_results['individual_id'][best_id]:.3f}")
print(f"Best year layer:   L{best_yr} → F1={probe_results['year'][best_yr]:.3f}")
print(f"Best n_clicks layer: L{int(np.argmax(probe_results['n_clicks']))} → R²={max(probe_results['n_clicks']):.3f}")
print(f"Best mean_ici layer: L{int(np.argmax(probe_results['mean_ici']))} → R²={max(probe_results['mean_ici']):.3f}")
print(f"Cramér's V(unit, year) = {cramers_v:.3f}")
print(f"Spearman ρ(unit-F1, year-F1) = {rho:.3f}, p={p_rho:.3f}")
print("All Phase 2 figures saved.")
