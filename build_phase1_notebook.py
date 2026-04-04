"""
Generates phase1_baselines.ipynb
Run once: python3 build_phase1_notebook.py
"""
import json, os, random, string

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_baselines.ipynb")

def md(source): return {"cell_type":"markdown","id":None,"metadata":{},"source":source}
def code(source): return {"cell_type":"code","id":None,"execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Phase 1 — Baselines
## *Beyond WhAM* · CS 297 Final Paper · April 2026

---

This notebook establishes the three baselines against which the DCCE (Dual-Channel \
Contrastive Encoder) will be compared in Phase 3:

| Baseline | Input | Method | Goal |
|---|---|---|---|
| **1A — Raw ICI** | Zero-padded ICI vector (length 9) | Logistic Regression | Floor for the rhythm encoder |
| **1C — Raw Mel** | Mean-pooled mel-spectrogram | Logistic Regression | Floor for the spectral encoder |
| **1B — WhAM** | WhAM embeddings (512d) | Logistic Regression | Primary comparison target (current SOTA) |

All three share the same train/test split (80/20, stratified by social unit, seed=42) \
and the same evaluation protocol (**macro-F1** as primary metric, accuracy secondary).

**Why macro-F1?** Unit F comprises 59.4% of clean codas; the most common coda type \
(`1+1+3`) makes up 35.1%. A model predicting the majority class would achieve high \
accuracy but near-zero macro-F1. Macro-F1 weights every class equally and directly \
tests biological discriminability.
"""))

# ── SETUP ─────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup and Data Loading"))

cells.append(code("""\
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (classification_report, f1_score,
                             accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

BASE   = os.path.abspath(".")
DATA   = os.path.join(BASE, "datasets")
AUDIO  = os.path.join(DATA, "dswp_audio")
LABELS = os.path.join(DATA, "dswp_labels.csv")
FIGS   = os.path.join(BASE, "figures", "phase1")
os.makedirs(FIGS, exist_ok=True)

UNIT_COLORS = {"A": "#4C72B0", "D": "#DD8452", "F": "#55A868"}
SEED = 42
"""))

cells.append(code("""\
# Load and prepare labels
df = pd.read_csv(LABELS)
df["ici_list"] = df["ici_sequence"].apply(
    lambda s: [float(x) for x in s.split("|")] if isinstance(s, str) and s else [])
df["mean_ici_ms"] = df["ici_list"].apply(lambda x: np.mean(x)*1000 if x else np.nan)

# Working set: clean codas only
df_clean = df[df["is_noise"] == 0].copy().reset_index(drop=True)

# Individual-ID subset: only codas with known IDN
df_id = df_clean[df_clean["individual_id"] != "0"].copy().reset_index(drop=True)

print(f"Clean codas (social unit + coda type tasks) : {len(df_clean)}")
print(f"IDN-labeled codas (individual ID task)      : {len(df_id)}  |  {df_id['individual_id'].nunique()} individuals")
print(f"Social units                                : {sorted(df_clean['unit'].unique())}")
print(f"Coda types                                  : {df_clean['coda_type'].nunique()}")
"""))

# ── SHARED SPLIT ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Shared Train/Test Split

**Design decisions from EDA:**
- Stratified by social unit — Unit F = 59.4%, random split would skew test set
- 80/20 ratio — 1,106 train / 277 test on the clean set
- Random seed = 42 fixed for all experiments

This exact split will be reused in Phases 2, 3, and 4.
"""))

cells.append(code("""\
def make_split(data, stratify_col, test_size=0.2, seed=SEED):
    \"\"\"Return (train_idx, test_idx) with stratified split.\"\"\"
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(data))
    train_idx, test_idx = next(sss.split(idx, data[stratify_col]))
    return train_idx, test_idx

# Main split (stratified by unit)
train_idx, test_idx = make_split(df_clean, "unit")
df_train = df_clean.iloc[train_idx].reset_index(drop=True)
df_test  = df_clean.iloc[test_idx].reset_index(drop=True)

# ID-subset split (stratified by individual_id)
train_id_idx, test_id_idx = make_split(df_id, "individual_id")
df_id_train = df_id.iloc[train_id_idx].reset_index(drop=True)
df_id_test  = df_id.iloc[test_id_idx].reset_index(drop=True)

print(f"Main split  — train: {len(df_train)}  test: {len(df_test)}")
print(f"  Train unit distribution: {df_train['unit'].value_counts().to_dict()}")
print(f"  Test  unit distribution: {df_test['unit'].value_counts().to_dict()}")
print(f"ID split    — train: {len(df_id_train)}  test: {len(df_id_test)}")

# Save split indices for reuse in later phases
np.save(os.path.join(DATA, "train_idx.npy"), train_idx)
np.save(os.path.join(DATA, "test_idx.npy"),  test_idx)
np.save(os.path.join(DATA, "train_id_idx.npy"), train_id_idx)
np.save(os.path.join(DATA, "test_id_idx.npy"),  test_id_idx)
print("\\nSplit indices saved to datasets/")
"""))

# ── EVALUATION HELPER ─────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Evaluation Helper

We define a single `evaluate()` function used by all three baselines. It reports \
macro-F1, accuracy, and a per-class breakdown. Confusion matrices are produced for \
the social-unit task (3 classes) where visual inspection is meaningful.
"""))

cells.append(code("""\
def evaluate(model, X_train, X_test, y_train, y_test, task_name, label_names=None, plot_cm=True):
    \"\"\"Fit logistic regression and report metrics.\"\"\"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy  = accuracy_score(y_test, y_pred)

    print(f"{'='*55}")
    print(f"Task: {task_name}")
    print(f"  Macro-F1 : {macro_f1:.4f}   Accuracy: {accuracy:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=label_names,
                                 zero_division=0))

    if plot_cm and label_names is not None and len(label_names) <= 10:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{task_name}\\nMacro-F1={macro_f1:.3f}  Acc={accuracy:.3f}")
        plt.tight_layout()
        fname = task_name.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(FIGS, f"cm_{fname}.png"), dpi=130, bbox_inches="tight")
        plt.show()

    return {"macro_f1": macro_f1, "accuracy": accuracy}


def make_lr(class_weight="balanced"):
    \"\"\"Logistic regression with balanced class weights and generous max_iter.\"\"\"
    return LogisticRegression(
        max_iter=2000, random_state=SEED,
        class_weight=class_weight,   # compensates for unit F imbalance
        solver="lbfgs", multi_class="multinomial")
"""))

# ── BASELINE 1A ───────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 4. Baseline 1A — Raw ICI → Logistic Regression

### Motivation

**Leitão et al. (arXiv:2307.05304)** showed that ICI-based clustering closely aligns \
with social-unit and clan assignments, suggesting the raw ICI space carries biological \
signal even without any learned representation. **Gero et al. (2016)** used ICI sequences \
directly to define the 21-type taxonomy that underpins our labels.

The t-SNE analysis in Phase 0 confirmed that raw ICIs form tight, distinct clusters by \
coda type but that social units are intermixed within those clusters. We therefore \
expect:

- **Coda type** classification: high macro-F1 (ICI *is* the coda type, by definition)
- **Social unit** classification: moderate macro-F1 (micro-variation signal exists but is subtle)
- **Individual ID** classification: low macro-F1 (individual style is finer-grained than unit style)

This baseline sets the floor: if DCCE-rhythm-only cannot beat it, the GRU encoder adds \
no value over a simple linear model on raw features.

### Feature construction

- Extract `ICI1`–`ICI9` from labels (pre-computed, no audio needed)
- Zero-pad shorter sequences to length 9
- Apply `StandardScaler` (confirmed necessary in EDA: ICI range spans ~90ms–350ms+)
"""))

cells.append(code("""\
# Build ICI feature matrix
MAX_ICI = 9

def build_ici_matrix(data):
    X = np.zeros((len(data), MAX_ICI))
    for i, row in enumerate(data.itertuples()):
        for j, v in enumerate(row.ici_list[:MAX_ICI]):
            X[i, j] = v
    return X

X_ici_train = build_ici_matrix(df_train)
X_ici_test  = build_ici_matrix(df_test)

# Normalise
scaler_ici = StandardScaler()
X_ici_train = scaler_ici.fit_transform(X_ici_train)
X_ici_test  = scaler_ici.transform(X_ici_test)

print(f"ICI feature matrix shape  — train: {X_ici_train.shape}  test: {X_ici_test.shape}")
print(f"Feature range after scaling — min: {X_ici_train.min():.2f}  max: {X_ici_train.max():.2f}")
"""))

cells.append(code("""\
results_1a = {}

# ── Task 1: Social unit ────────────────────────────────────────────────────
results_1a["unit"] = evaluate(
    make_lr(),
    X_ici_train, X_ici_test,
    df_train["unit"], df_test["unit"],
    task_name="1A — ICI → Social Unit",
    label_names=["A", "D", "F"])
"""))

cells.append(code("""\
# ── Task 2: Coda type ─────────────────────────────────────────────────────
# Use top-10 most frequent types to keep the confusion matrix readable
top10_types = df_clean["coda_type"].value_counts().head(10).index.tolist()
mask_train = df_train["coda_type"].isin(top10_types)
mask_test  = df_test["coda_type"].isin(top10_types)

results_1a["coda_type_top10"] = evaluate(
    make_lr(),
    X_ici_train[mask_train], X_ici_test[mask_test],
    df_train.loc[mask_train, "coda_type"], df_test.loc[mask_test, "coda_type"],
    task_name="1A — ICI → Coda Type (top 10)",
    label_names=top10_types)

# Full 22-type evaluation (no confusion matrix — too many classes)
results_1a["coda_type_all"] = evaluate(
    make_lr(),
    X_ici_train, X_ici_test,
    df_train["coda_type"], df_test["coda_type"],
    task_name="1A — ICI → Coda Type (all 22)",
    label_names=None, plot_cm=False)
"""))

cells.append(code("""\
# ── Task 3: Individual ID ─────────────────────────────────────────────────
X_id_train = build_ici_matrix(df_id_train)
X_id_test  = build_ici_matrix(df_id_test)
scaler_id = StandardScaler()
X_id_train = scaler_id.fit_transform(X_id_train)
X_id_test  = scaler_id.transform(X_id_test)

results_1a["individual_id"] = evaluate(
    make_lr(),
    X_id_train, X_id_test,
    df_id_train["individual_id"], df_id_test["individual_id"],
    task_name="1A — ICI → Individual ID",
    label_names=None, plot_cm=False)

print("\\n1A Summary:")
for k, v in results_1a.items():
    print(f"  {k:25s}  Macro-F1={v['macro_f1']:.4f}  Acc={v['accuracy']:.4f}")
"""))

# ── BASELINE 1C ───────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 5. Baseline 1C — Raw Mel-Spectrogram → Logistic Regression

### Motivation

**Beguš et al. (2024)** showed that the spectral texture within coda clicks carries \
vowel-like formant variation correlated with individual and social-unit identity. The \
spectral centroid analysis in Phase 0 confirmed that spectral variance is high across \
the dataset (8,894 ± 2,913 Hz) and that rhythm and spectral channels are empirically \
uncorrelated (Pearson r ≈ 0).

This baseline tests whether a simple mean-pooled mel-spectrogram — the raw spectral \
representation without any learned encoder — carries social-unit or coda-type signal. \
It establishes the floor for the DCCE spectral encoder, analogous to what Baseline 1A \
does for the rhythm encoder.

### Feature construction

- Load each WAV with librosa (native sample rate, mono)
- Compute mel-spectrogram: 64 mel bins, `fmax=8000 Hz` (confirmed by EDA)
- **Mean-pool** across time → fixed 64-dimensional feature vector per coda
- Apply `StandardScaler`

Mean-pooling discards temporal structure but retains the average spectral shape. \
This is intentionally weak — a learned CNN will exploit the temporal structure that \
this baseline ignores.
"""))

cells.append(code("""\
print("Computing mel-spectrogram features (this takes ~3-5 min for 1,383 codas)...")

def compute_mel_features(data, n_mels=64, fmax=8000):
    \"\"\"Mean-pooled mel-spectrogram for each coda. Returns (N, n_mels) array.\"\"\"
    feats = []
    for row in data.itertuples():
        y, sr = librosa.load(os.path.join(AUDIO, f"{row.coda_id}.wav"),
                             sr=None, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feats.append(mel_db.mean(axis=1))   # mean across time → (n_mels,)
    return np.array(feats)

X_mel_all = compute_mel_features(df_clean)
np.save(os.path.join(DATA, "X_mel_all.npy"), X_mel_all)

X_mel_train = X_mel_all[train_idx]
X_mel_test  = X_mel_all[test_idx]

scaler_mel = StandardScaler()
X_mel_train = scaler_mel.fit_transform(X_mel_train)
X_mel_test  = scaler_mel.transform(X_mel_test)

print(f"Mel feature matrix — train: {X_mel_train.shape}  test: {X_mel_test.shape}")
"""))

cells.append(code("""\
results_1c = {}

# ── Task 1: Social unit ────────────────────────────────────────────────────
results_1c["unit"] = evaluate(
    make_lr(),
    X_mel_train, X_mel_test,
    df_train["unit"], df_test["unit"],
    task_name="1C — Mel → Social Unit",
    label_names=["A", "D", "F"])
"""))

cells.append(code("""\
# ── Task 2: Coda type (top 10 + all) ────────────────────────────────────
mask_train = df_train["coda_type"].isin(top10_types)
mask_test  = df_test["coda_type"].isin(top10_types)

results_1c["coda_type_top10"] = evaluate(
    make_lr(),
    X_mel_train[mask_train], X_mel_test[mask_test],
    df_train.loc[mask_train, "coda_type"], df_test.loc[mask_test, "coda_type"],
    task_name="1C — Mel → Coda Type (top 10)",
    label_names=top10_types)

results_1c["coda_type_all"] = evaluate(
    make_lr(),
    X_mel_train, X_mel_test,
    df_train["coda_type"], df_test["coda_type"],
    task_name="1C — Mel → Coda Type (all 22)",
    label_names=None, plot_cm=False)
"""))

cells.append(code("""\
# ── Task 3: Individual ID ─────────────────────────────────────────────────
# Reindex mel features to ID subset
df_clean_reset = df_clean.reset_index(drop=True)
id_mask = df_clean_reset["individual_id"] != "0"
X_mel_id = X_mel_all[id_mask.values]
scaler_mel_id = StandardScaler()
X_mel_id_train = scaler_mel_id.fit_transform(X_mel_id[train_id_idx])
X_mel_id_test  = scaler_mel_id.transform(X_mel_id[test_id_idx])

results_1c["individual_id"] = evaluate(
    make_lr(),
    X_mel_id_train, X_mel_id_test,
    df_id_train["individual_id"], df_id_test["individual_id"],
    task_name="1C — Mel → Individual ID",
    label_names=None, plot_cm=False)

print("\\n1C Summary:")
for k, v in results_1c.items():
    print(f"  {k:25s}  Macro-F1={v['macro_f1']:.4f}  Acc={v['accuracy']:.4f}")
"""))

# ── COMPARISON ────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 6. Baseline Comparison (1A vs 1C)

Before introducing WhAM, we compare the two raw-feature baselines to understand \
which channel carries more signal for each task.
"""))

cells.append(code("""\
tasks   = ["unit", "coda_type_all", "individual_id"]
labels  = ["Social Unit", "Coda Type (all 22)", "Individual ID"]
x       = np.arange(len(tasks))
width   = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 1 Baselines: Raw Feature Comparison (1A ICI vs 1C Mel-Spec)",
             fontsize=13, fontweight="bold")

for ax_idx, (metric, ylabel) in enumerate([("macro_f1", "Macro-F1"),
                                            ("accuracy", "Accuracy")]):
    ax = axes[ax_idx]
    vals_1a = [results_1a[t][metric] for t in tasks]
    vals_1c = [results_1c[t][metric] for t in tasks]
    bars1 = ax.bar(x - width/2, vals_1a, width, label="1A — Raw ICI",
                   color="#4C72B0", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width/2, vals_1c, width, label="1C — Raw Mel",
                   color="#DD8452", edgecolor="black", linewidth=0.6)
    for bar, val in zip(list(bars1)+list(bars2), vals_1a+vals_1c):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05)
    ax.set_title(f"({chr(97+ax_idx)}) {ylabel}")
    ax.legend(fontsize=9)
    ax.axhline(1/3, color="grey", ls="--", lw=0.8, label="random (3-class)")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_1a_vs_1c_comparison.png"), dpi=130, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""\
### Interpretation

The comparison above tells us the *raw signal strength* of each channel before any \
learned encoding:

- **If 1A (ICI) >> 1C (Mel) on coda type**: coda type is fundamentally a rhythm \
  phenomenon, consistent with decades of bioacoustics (Watkins & Schevill 1977; Gero 2016).
- **If 1C (Mel) >= 1A (ICI) on social unit**: the spectral channel carries social \
  identity signal that is *not* reducible to rhythm patterns — the central empirical \
  claim of Beguš et al. (2024).
- **The gap between 1A and 1C on social unit** is the motivation for DCCE: neither \
  channel alone captures the full social signal. The fusion model should outperform both.

These results set concrete numerical targets that DCCE-full must exceed to constitute \
a genuine contribution.
"""))

# ── BASELINE 1B SCAFFOLD ──────────────────────────────────────────────────────
cells.append(md("""\
---
## 7. Baseline 1B — WhAM Embeddings → Logistic Regression

### Motivation

**WhAM** (Paradise et al., NeurIPS 2025, arXiv:2512.02206) is the current state of \
the art for sperm whale coda representation. It is a transformer-based masked acoustic \
token model fine-tuned from VampNet (a music audio generative model). Its classification \
results — social unit, rhythm type, and vowel type — are emergent byproducts of a \
generative objective, not a purpose-built representation objective.

Running WhAM on our exact DSWP split and reporting its linear probe accuracy gives us \
a fair, reproducible comparison target. The numbers in the WhAM paper used a different, \
larger dataset; we need to regenerate them on our 80/20 split.

### Setup

Download weights from Zenodo (CC-BY-NC-ND 4.0, ~3.1 GB total):
```
DOI: 10.5281/zenodo.17633708
Files: coarse.pth (1.3 GB), c2f.pth (1.1 GB), codec.pth (601 MB), wavebeat.pth (33 MB)
```

Clone the WhAM repository:
```bash
git clone https://github.com/Project-CETI/wham.git
cd wham
pip install -e .
pip install -e ./vampnet
```
"""))

cells.append(code("""\
# ── Baseline 1B: WhAM embeddings ────────────────────────────────────────────
# This cell will run once WhAM weights are downloaded and the repo is installed.
# It extracts 512-dimensional embeddings for all 1,501 DSWP codas and saves them.
# Then runs the same logistic regression probes as 1A and 1C.

WHAM_EMBEDDINGS = os.path.join(DATA, "wham_embeddings.npy")

if os.path.exists(WHAM_EMBEDDINGS):
    print("Loading pre-computed WhAM embeddings...")
    X_wham_all = np.load(WHAM_EMBEDDINGS)
    print(f"  Shape: {X_wham_all.shape}")

    # Slice to clean codas only
    clean_ids = df_clean["coda_id"].values - 1   # 0-indexed
    X_wham_clean = X_wham_all[clean_ids]

    X_wham_train = X_wham_clean[train_idx]
    X_wham_test  = X_wham_clean[test_idx]

    scaler_wham = StandardScaler()
    X_wham_train = scaler_wham.fit_transform(X_wham_train)
    X_wham_test  = scaler_wham.transform(X_wham_test)

    results_1b = {}

    results_1b["unit"] = evaluate(
        make_lr(),
        X_wham_train, X_wham_test,
        df_train["unit"], df_test["unit"],
        task_name="1B — WhAM → Social Unit",
        label_names=["A", "D", "F"])

    results_1b["coda_type_all"] = evaluate(
        make_lr(),
        X_wham_train, X_wham_test,
        df_train["coda_type"], df_test["coda_type"],
        task_name="1B — WhAM → Coda Type (all 22)",
        label_names=None, plot_cm=False)

    # Individual ID subset
    id_mask = df_clean["individual_id"] != "0"
    X_wham_id = X_wham_clean[id_mask.values]
    scaler_wham_id = StandardScaler()
    X_wham_id_train = scaler_wham_id.fit_transform(X_wham_id[train_id_idx])
    X_wham_id_test  = scaler_wham_id.transform(X_wham_id[test_id_idx])

    results_1b["individual_id"] = evaluate(
        make_lr(),
        X_wham_id_train, X_wham_id_test,
        df_id_train["individual_id"], df_id_test["individual_id"],
        task_name="1B — WhAM → Individual ID",
        label_names=None, plot_cm=False)

    print("\\n1B Summary:")
    for k, v in results_1b.items():
        print(f"  {k:25s}  Macro-F1={v['macro_f1']:.4f}  Acc={v['accuracy']:.4f}")

else:
    print("WhAM embeddings not found.")
    print("Please download weights from https://zenodo.org/records/17633708")
    print("and install the WhAM repo: https://github.com/Project-CETI/wham")
    print("Then run the extraction script below to generate wham_embeddings.npy")
"""))

cells.append(code("""\
# ── WhAM embedding extraction script ─────────────────────────────────────────
# Run this cell after installing WhAM to produce wham_embeddings.npy
# (Only needed once — takes ~10-20 min on Apple MPS)

EXTRACT = False   # set to True once WhAM is installed

if EXTRACT:
    import sys
    sys.path.insert(0, os.path.expanduser("~/wham"))
    from wham.embedding import extract_embeddings

    wav_paths = [os.path.join(AUDIO, f"{i}.wav") for i in range(1, 1502)]
    print(f"Extracting embeddings for {len(wav_paths)} WAV files...")
    embeddings = extract_embeddings(wav_paths)   # returns (1501, dim) array
    np.save(WHAM_EMBEDDINGS, embeddings)
    print(f"Saved to {WHAM_EMBEDDINGS}  shape={embeddings.shape}")
"""))

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 8. Phase 1 Summary

The table below is the master results table for Phase 1. It will be extended with \
WhAM (1B) results once the embeddings are extracted, and referenced throughout \
Phases 2–3 as the comparison baseline.
"""))

cells.append(code("""\
# Build and display summary table
rows = []
for baseline, results, label in [
    ("1A — Raw ICI",  results_1a, "ICI → LogReg"),
    ("1C — Raw Mel",  results_1c, "Mel → LogReg"),
]:
    for task in ["unit", "coda_type_all", "individual_id"]:
        r = results[task]
        rows.append({
            "Baseline": baseline,
            "Task": task.replace("_all","").replace("_"," ").title(),
            "Macro-F1": f"{r['macro_f1']:.4f}",
            "Accuracy": f"{r['accuracy']:.4f}",
        })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

print("\\n(1B — WhAM embeddings: pending weight download)")
"""))

cells.append(md("""\
### Next steps

- **If 1A coda-type F1 is high and 1A unit F1 is moderate**: confirms EDA predictions; \
  proceed to Phase 2 (WhAM probing) with confidence that the ICI floor is established.
- **If 1C mel F1 on social unit > 1A**: spectral channel carries social signal independently \
  of rhythm — the Beguš et al. claim is validated empirically on DSWP. This strengthens \
  the motivation for DCCE's dual-encoder design.
- **Download WhAM weights** (Zenodo: 10.5281/zenodo.17633708) to complete Baseline 1B \
  and unlock Phases 2 and 3.
"""))

# ── Assign IDs ────────────────────────────────────────────────────────────────
for i, cell in enumerate(cells):
    cell["id"] = f"p1-{i:02d}-{''.join(random.choices(string.ascii_lowercase, k=6))}"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {NB}")
print(f"Cells: {len(cells)}")
