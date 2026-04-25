"""Phase 1 Baselines — executed as a script.
1A: Raw ICI → LogReg (unit, coda_type, individual_id)
1B: WhAM L10 embedding → LogReg
1C: Mean-pooled mel → LogReg
Outputs: figures to figures/phase1/, phase1_results.csv to datasets/
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import umap

BASE   = "/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project"
DATA   = f"{BASE}/datasets"
FIGDIR = f"{BASE}/figures/phase1"
os.makedirs(FIGDIR, exist_ok=True)

UNIT_COLORS = {"A": "#2196F3", "D": "#FF9800", "F": "#4CAF50"}
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── load labels ────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{DATA}/dswp_labels.csv")
clean = df[df.is_noise == 0].reset_index(drop=True)

# ── load split indices (index into clean 1383-coda array) ─────────────────────
train_idx    = np.load(f"{DATA}/train_idx.npy")
test_idx     = np.load(f"{DATA}/test_idx.npy")
train_id_idx = np.load(f"{DATA}/train_id_idx.npy")
test_id_idx  = np.load(f"{DATA}/test_id_idx.npy")

# ── labels ─────────────────────────────────────────────────────────────────────
y_unit = clean["unit"].values
y_type = clean["coda_type"].values

# individual ID: only labeled codas; encode
idn_mask = clean["individual_id"].astype(str) != "0"
clean_idn = clean[idn_mask].reset_index(drop=True)
le_id = LabelEncoder()
y_id_all = le_id.fit_transform(clean_idn["individual_id"].astype(str))

def run_probe(X, y, tr_idx, te_idx, label="", max_iter=2000, C=1.0):
    """Fit logistic regression probe and return (macro_f1, accuracy)."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr_idx])
    Xte = scaler.transform(X[te_idx])
    ytr, yte = y[tr_idx], y[te_idx]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced",
                             random_state=42, solver="lbfgs")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, average="macro")
    acc = accuracy_score(yte, pred)
    print(f"  {label}: F1={f1:.3f}  Acc={acc:.3f}")
    return f1, acc, clf, pred, yte, scaler

# ══════════════════════════════════════════════════════════════════════════════
# 1A — Raw ICI LogReg
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Baseline 1A: Raw ICI → LogReg ===")

def parse_ici_matrix(df_clean):
    mat = np.zeros((len(df_clean), 9), dtype=np.float32)
    for i, row in enumerate(df_clean["ici_sequence"]):
        if pd.isna(row): continue
        vals = [float(x)*1000 for x in str(row).split("|")]
        for j, v in enumerate(vals[:9]):
            mat[i, j] = v
    return mat

X_ici = parse_ici_matrix(clean)
scaler_ici = StandardScaler()
X_ici_sc = scaler_ici.fit_transform(X_ici)

r1a_unit = run_probe(X_ici_sc, y_unit, train_idx, test_idx, "unit")
r1a_type = run_probe(X_ici_sc, y_type, train_idx, test_idx, "coda_type")

# individual ID probe using id-split
X_ici_id = parse_ici_matrix(clean_idn)
scaler_ici_id = StandardScaler()
X_ici_id_sc = scaler_ici_id.fit_transform(X_ici_id)
r1a_id = run_probe(X_ici_id_sc, y_id_all, train_id_idx, test_id_idx, "individual_id")

# save confusion matrices for unit and type
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Baseline 1A: Raw ICI LogReg", fontsize=14, fontweight="bold")
for ax, (label, result) in zip(axes, [("Social Unit", r1a_unit), ("Coda Type (top)", r1a_type)]):
    _, _, clf, pred, yte, _ = result
    classes = np.unique(yte)
    cm = confusion_matrix(yte, pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{label}\nMacro-F1={result[0]:.3f}")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_1a_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("1A confusion matrix saved")

# ══════════════════════════════════════════════════════════════════════════════
# 1C — Mel-spectrogram LogReg
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Baseline 1C: Mean-pooled Mel → LogReg ===")
X_mel = np.load(f"{DATA}/X_mel_all.npy")  # (1383, 64)

r1c_unit = run_probe(X_mel, y_unit, train_idx, test_idx, "unit")
r1c_type = run_probe(X_mel, y_type, train_idx, test_idx, "coda_type")

# mel individual ID probe
idn_clean_idx = np.where(idn_mask.values)[0]  # positions in clean array
# need to map train_id_idx / test_id_idx (which index into clean_idn) to clean array
# Actually: clean_idn was reset_index(drop=True), so clean_idn.index maps to idn_mask cumsum
idn_positions = np.where(idn_mask.values)[0]  # absolute positions in clean (1383)
X_mel_id = X_mel[idn_positions]
r1c_id = run_probe(X_mel_id, y_id_all, train_id_idx, test_id_idx, "individual_id")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Baseline 1C: Mean-pooled Mel LogReg", fontsize=14, fontweight="bold")
for ax, (label, result) in zip(axes, [("Social Unit", r1c_unit), ("Coda Type (top)", r1c_type)]):
    _, _, clf, pred, yte, _ = result
    classes = np.unique(yte)
    cm = confusion_matrix(yte, pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{label}\nMacro-F1={result[0]:.3f}")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_1c_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("1C confusion matrix saved")

# ══════════════════════════════════════════════════════════════════════════════
# 1B — WhAM L10 LogReg
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Baseline 1B: WhAM L10 → LogReg ===")
wham_emb = np.load(f"{DATA}/wham_embeddings.npy")  # (1501, 1280) — all 1501 codas

# align wham to clean: use coda_id as key (coda_id = 1-based, index = 0-based)
coda_ids_clean = clean["coda_id"].values  # shape (1383,)
wham_clean = wham_emb[coda_ids_clean - 1]  # (1383, 1280)

r1b_unit = run_probe(wham_clean, y_unit, train_idx, test_idx, "unit")
r1b_type = run_probe(wham_clean, y_type, train_idx, test_idx, "coda_type")

# WhAM individual ID
wham_id = wham_clean[idn_positions]  # (763, 1280)
r1b_id = run_probe(wham_id, y_id_all, train_id_idx, test_id_idx, "individual_id")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Baseline 1B: WhAM L10 LogReg", fontsize=14, fontweight="bold")
for ax, (label, result) in zip(axes, [("Social Unit", r1b_unit), ("Coda Type", r1b_type)]):
    _, _, clf, pred, yte, _ = result
    classes = np.unique(yte)
    cm = confusion_matrix(yte, pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{label}\nMacro-F1={result[0]:.3f}")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_1b_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("1B confusion matrix saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG — Baseline comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════
tasks = ["Unit Macro-F1", "CodaType Macro-F1", "IndivID Macro-F1"]
ici_scores  = [r1a_unit[0], r1a_type[0], r1a_id[0]]
mel_scores  = [r1c_unit[0], r1c_type[0], r1c_id[0]]
wham_scores = [r1b_unit[0], r1b_type[0], r1b_id[0]]

x = np.arange(len(tasks))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w,   ici_scores,  w, label="ICI LogReg (1A)",  color="#7986CB")
ax.bar(x,       mel_scores,  w, label="Mel LogReg (1C)",   color="#4DB6AC")
ax.bar(x + w,   wham_scores, w, label="WhAM L10 (1B)",     color="#FF8A65")
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
ax.set_title("Phase 1 — Baseline Comparison (Macro-F1)", fontsize=13, fontweight="bold")
ax.legend()
for xi, scores in zip([x-w, x, x+w], [ici_scores, mel_scores, wham_scores]):
    for xj, v in zip(xi, scores):
        ax.text(xj, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Baseline comparison figure saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG — WhAM t-SNE and UMAP
# ══════════════════════════════════════════════════════════════════════════════
print("Computing WhAM t-SNE (test split)...")
Xte_wham = wham_clean[test_idx]  # (277, 1280)
sc = StandardScaler()
Xte_wham_sc = sc.fit_transform(Xte_wham)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
Z_tsne = tsne.fit_transform(Xte_wham_sc)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("WhAM L10 Embeddings — t-SNE (test set)", fontsize=13, fontweight="bold")

y_unit_te = y_unit[test_idx]
y_type_te = y_type[test_idx]

for unit in ["A","D","F"]:
    mask = y_unit_te == unit
    axes[0].scatter(Z_tsne[mask,0], Z_tsne[mask,1], c=UNIT_COLORS[unit], label=f"Unit {unit}",
                    alpha=0.6, s=15, rasterized=True)
axes[0].set_title("Colored by Social Unit")
axes[0].legend(markerscale=2); axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

top5 = pd.Series(y_type).value_counts().head(5).index.tolist()
palette = sns.color_palette("tab10", 6)
for i, ct in enumerate(top5):
    m = y_type_te == ct
    axes[1].scatter(Z_tsne[m,0], Z_tsne[m,1], c=[palette[i]], label=ct, alpha=0.6, s=15, rasterized=True)
other = ~np.isin(y_type_te, top5)
axes[1].scatter(Z_tsne[other,0], Z_tsne[other,1], c=[palette[-1]], label="Other", alpha=0.3, s=8, rasterized=True)
axes[1].set_title("Colored by Coda Type (top 5)")
axes[1].legend(fontsize=8, markerscale=2); axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("WhAM t-SNE saved")

# UMAP (all clean)
print("Computing WhAM UMAP (all clean)...")
sc2 = StandardScaler()
Xall_sc = sc2.fit_transform(wham_clean)
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
Z_umap = reducer.fit_transform(Xall_sc)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("WhAM L10 Embeddings — UMAP (all 1383 clean codas)", fontsize=13, fontweight="bold")
for unit in ["A","D","F"]:
    m = y_unit == unit
    axes[0].scatter(Z_umap[m,0], Z_umap[m,1], c=UNIT_COLORS[unit], label=f"Unit {unit}",
                    alpha=0.4, s=6, rasterized=True)
axes[0].set_title("Colored by Social Unit")
axes[0].legend(markerscale=3); axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
for i, ct in enumerate(top5):
    m = y_type == ct
    axes[1].scatter(Z_umap[m,0], Z_umap[m,1], c=[palette[i]], label=ct, alpha=0.5, s=6, rasterized=True)
other = ~np.isin(y_type, top5)
axes[1].scatter(Z_umap[other,0], Z_umap[other,1], c=[palette[-1]], label="Other", alpha=0.3, s=4, rasterized=True)
axes[1].set_title("Colored by Coda Type (top 5)")
axes[1].legend(fontsize=8, markerscale=3); axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print("WhAM UMAP saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG — ICI rhythm patterns (coda type examples)
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting ICI rhythm patterns...")
top6_types = clean["coda_type"].value_counts().head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("ICI Rhythm Patterns by Coda Type (mean ± std)", fontsize=13, fontweight="bold")
for ax, ct in zip(axes.flat, top6_types):
    sub = clean[clean["coda_type"] == ct]
    rows = []
    for row in sub["ici_sequence"]:
        if pd.isna(row): continue
        vals = [float(x)*1000 for x in str(row).split("|")]
        rows.append(vals[:9])
    # pad to 9
    padded = np.zeros((len(rows), 9))
    for i, r in enumerate(rows):
        padded[i, :len(r)] = r
    mean_ici = np.mean(padded, axis=0)
    std_ici  = np.std(padded, axis=0)
    n_click_mode = int(np.round(np.mean([len(r) for r in rows])))
    ax.plot(range(1, 10), mean_ici, marker="o", color="#3F51B5")
    ax.fill_between(range(1, 10), mean_ici-std_ici, mean_ici+std_ici, alpha=0.2, color="#3F51B5")
    ax.set_title(f"{ct} (n={len(rows)})")
    ax.set_xlabel("ICI index"); ax.set_ylabel("ICI (ms)")
    ax.set_xlim(0.5, 9.5)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_ici_rhythm_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("ICI rhythm patterns saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG — Mean mel profiles by unit
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting mean mel profiles by unit...")
X_mel_full = np.load(f"{DATA}/X_mel_full.npy")  # (1383, 64, 128)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Mean Mel-Spectrogram Profile by Social Unit", fontsize=13, fontweight="bold")
for ax, unit in zip(axes, ["A","D","F"]):
    idxs = np.where(y_unit == unit)[0]
    mean_mel = X_mel_full[idxs].mean(axis=0)  # (64, 128)
    im = ax.imshow(mean_mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Unit {unit} (n={len(idxs)})")
    ax.set_xlabel("Time frames"); ax.set_ylabel("Mel bin")
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_mean_mel_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Mean mel profiles saved")

# ══════════════════════════════════════════════════════════════════════════════
# WhAM layer-wise probe (all 20 layers, social unit and coda type)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing layer-wise WhAM probe...")
all_layers = np.load(f"{DATA}/wham_embeddings_all_layers.npy")  # (1501, 20, 1280)
all_layers_clean = all_layers[coda_ids_clean - 1]  # (1383, 20, 1280)

layer_unit_f1 = []
layer_type_f1 = []
for layer in range(20):
    Xl = all_layers_clean[:, layer, :]
    f1u, _, _, _, _, _ = run_probe(Xl, y_unit, train_idx, test_idx, f"L{layer:02d}-unit")
    f1t, _, _, _, _, _ = run_probe(Xl, y_type, train_idx, test_idx, f"L{layer:02d}-type")
    layer_unit_f1.append(f1u)
    layer_type_f1.append(f1t)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(20), layer_unit_f1, marker="o", label="Social Unit", color="#2196F3")
ax.plot(range(20), layer_type_f1, marker="s", label="Coda Type", color="#FF9800")
ax.set_xlabel("WhAM Layer"); ax.set_ylabel("Macro-F1")
ax.set_title("WhAM Layer-wise Linear Probe", fontsize=13, fontweight="bold")
ax.legend(); ax.set_xticks(range(20))
ax.axhline(r1a_unit[0], color="#2196F3", linestyle="--", alpha=0.5, label=f"ICI unit={r1a_unit[0]:.3f}")
ax.axhline(r1a_type[0], color="#FF9800", linestyle="--", alpha=0.5, label=f"ICI type={r1a_type[0]:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_layerwise_probe.png", dpi=150, bbox_inches="tight")
plt.close()
print("WhAM layer-wise probe saved")

# ══════════════════════════════════════════════════════════════════════════════
# Save phase1_results.csv
# ══════════════════════════════════════════════════════════════════════════════
results = pd.DataFrame([
    {"model": "ICI_LogReg_1A",  "unit_f1": r1a_unit[0], "codatype_f1": r1a_type[0], "individ_f1": r1a_id[0],
     "unit_acc": r1a_unit[1], "individ_acc": r1a_id[1]},
    {"model": "Mel_LogReg_1C",  "unit_f1": r1c_unit[0], "codatype_f1": r1c_type[0], "individ_f1": r1c_id[0],
     "unit_acc": r1c_unit[1], "individ_acc": r1c_id[1]},
    {"model": "WhAM_L10_1B",    "unit_f1": r1b_unit[0], "codatype_f1": r1b_type[0], "individ_f1": r1b_id[0],
     "unit_acc": r1b_unit[1], "individ_acc": r1b_id[1]},
])
results.to_csv(f"{DATA}/phase1_results.csv", index=False)
print("\nphase1_results.csv saved:")
print(results.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== PHASE 1 SUMMARY ===")
print(f"1A ICI:  unit={r1a_unit[0]:.3f}  type={r1a_type[0]:.3f}  indivID={r1a_id[0]:.3f}")
print(f"1C Mel:  unit={r1c_unit[0]:.3f}  type={r1c_type[0]:.3f}  indivID={r1c_id[0]:.3f}")
print(f"1B WhAM: unit={r1b_unit[0]:.3f}  type={r1b_type[0]:.3f}  indivID={r1b_id[0]:.3f}")
best_unit  = max(range(20), key=lambda i: layer_unit_f1[i])
best_type  = max(range(20), key=lambda i: layer_type_f1[i])
print(f"WhAM best layer for unit: L{best_unit} (F1={layer_unit_f1[best_unit]:.3f})")
print(f"WhAM best layer for type: L{best_type} (F1={layer_type_f1[best_type]:.3f})")
print("All Phase 1 figures saved.")
