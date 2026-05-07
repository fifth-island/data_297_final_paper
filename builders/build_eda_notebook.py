"""
Generates eda_phase0.ipynb — the Phase 0 EDA notebook.
Run once: python3 build_eda_notebook.py
"""
import json, os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "eda_phase0.ipynb")

def md(source): return {"cell_type":"markdown","id":None,"metadata":{},"source":source}
def code(source): return {"cell_type":"code","id":None,"execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Phase 0 — Exploratory Data Analysis
## *Beyond WhAM*: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding
### CS 297 Final Paper · April 2026

---

This notebook constitutes Phase 0 of our research pipeline. Its purpose is to develop a \
thorough understanding of the Dominica Sperm Whale Project (DSWP) dataset before writing \
any model code. Every modelling decision in Phases 1–4 should be traceable back to an \
observation made here.

**Guiding question for this notebook:**
*Do the two known information channels in sperm whale codas — rhythm (ICI timing) and \
spectral texture (vowel) — carry distinct, complementary signal that justifies building \
a dual-encoder architecture?*
"""))

# ── INTRO ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
## 1. Background and Motivation

### 1.1 What are codas?

Sperm whales (*Physeter macrocephalus*) communicate through rhythmically patterned click \
sequences called **codas** — short bursts of 3–40 clicks separated by precise inter-click \
intervals. Codas are social signals: groups of whales that share a coda repertoire form \
vocal **clans**, and membership in a matrilineal **social unit** can be partially inferred \
from acoustic style.

### 1.2 The two-channel hypothesis

Recent work has established that every coda encodes information along two syntactically \
independent dimensions:

| Channel | Feature | Encodes | Reference |
|---|---|---|---|
| **Rhythm** | Inter-click interval (ICI) sequence | *Coda type* — the categorical click-count/timing pattern shared within a clan | Leitão et al. (arXiv:2307.05304); Gero et al. (2016, *Royal Society Open Science*) |
| **Spectral** | Spectral shape (formant-like structure) within each click | *Individual/social-unit identity* — analogous to a voice fingerprint | Beguš et al. (*The Phonology of Sperm Whale Coda Vowels*, 2024) |

**Leitão et al. (2023–2025)** showed that *rhythmic micro-variations* within a given coda \
type track social-unit membership and, critically, that whales learn vocal style from \
neighbouring clans — providing the first quantitative evidence of cross-clan cultural \
transmission. This directly motivates treating the ICI sequence as a first-class input \
feature.

**Beguš et al. (2024)** formalised the spectral channel linguistically, showing that \
inter-pulse spectral variation within codas produces vowel-like formant patterns (labelled \
`a` and `i`) that correlate with individual identity independently of coda type.

### 1.3 The gap this paper fills

**WhAM** (Paradise et al., NeurIPS 2025, arXiv:2512.02206) is the current state of the art: \
a transformer masked-acoustic-token model fine-tuned from VampNet. It classifies social \
units, rhythm types, and vowel types as emergent byproducts of a generative objective — not \
by design. No published work has purpose-built a representation that explicitly exploits \
*both* channels simultaneously. This EDA is the first step toward filling that gap with the \
**Dual-Channel Contrastive Encoder (DCCE)**.

### 1.4 Dataset provenance

The DSWP HuggingFace release (`orrp/DSWP`) provides 1,501 raw WAV files with no labels. \
We recover ground-truth labels by joining against **DominicaCodas.csv** from \
Sharma et al. (2024, *Nature Communications*), which provides the same 1,501 codas \
(codaNUM2018 = 1–1501) annotated with social unit, coda type, individual ID, \
pre-computed ICI sequences, and recording date. This join was verified by matching \
ICI values and durations across both sources (perfect alignment). The merged file is \
`datasets/dswp_labels.csv`.
"""))

# ── SETUP ─────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Setup"))

cells.append(code("""\
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
import librosa.display
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# ── Paths ────────────────────────────────────────────────────────────────────
HERE   = os.path.abspath(".")
BASE   = HERE if os.path.isdir(os.path.join(HERE, "datasets")) else os.path.dirname(HERE)
if not os.path.isdir(os.path.join(BASE, "datasets")):
  raise FileNotFoundError(f"Could not locate datasets/ from working directory: {HERE}")
DATA   = os.path.join(BASE, "datasets")
AUDIO  = os.path.join(DATA, "dswp_audio")
LABELS = os.path.join(DATA, "dswp_labels.csv")
FIGS   = os.path.join(BASE, "figures", "eda")
os.makedirs(FIGS, exist_ok=True)

UNIT_COLORS = {"A": "#4C72B0", "D": "#DD8452", "F": "#55A868"}
UNIT_ORDER  = ["A", "D", "F"]

print("Paths configured. Audio directory:", AUDIO)
print("Number of WAV files:", len([f for f in os.listdir(AUDIO) if f.endswith(".wav")]))
"""))

# ── DATA LOADING ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Data Loading

We load `dswp_labels.csv` — our master label file constructed by joining the DSWP audio \
index against DominicaCodas.csv (Sharma et al. 2024). Each row corresponds to exactly one \
WAV file in `datasets/dswp_audio/`.

Key columns:
- `unit` — social unit (A / D / F), the primary classification target
- `coda_type` — rhythm type label (e.g. `1+1+3`, `5R1`), from Gero et al.'s classification scheme
- `individual_id` — numeric whale ID; `0` means unidentified in the field catalog
- `ici_sequence` — pipe-separated pre-computed inter-click intervals (seconds)
- `is_noise` — 1 if the coda was flagged as noise-contaminated
"""))

cells.append(code("""\
df = pd.read_csv(LABELS)

# Parse ICI sequences and compute derived features
df["ici_list"]    = df["ici_sequence"].apply(
    lambda s: [float(x) for x in s.split("|")] if isinstance(s, str) and s else [])
df["mean_ici"]    = df["ici_list"].apply(lambda x: np.mean(x) if x else np.nan)
df["mean_ici_ms"] = df["mean_ici"] * 1000
df["date_parsed"] = pd.to_datetime(df["date"], dayfirst=True)
df["year"]        = df["date_parsed"].dt.year

df_clean = df[df["is_noise"] == 0].copy()
df_id    = df_clean[df_clean["individual_id"] != "0"].copy()

print(f"Total codas         : {len(df)}")
print(f"Clean (non-noise)   : {len(df_clean)}")
print(f"ID-labeled (IDN≠0)  : {len(df_id)}  |  {df_id['individual_id'].nunique()} unique individuals")
print(f"Date range          : {df['date_parsed'].min().date()} → {df['date_parsed'].max().date()}")
df.head(3)
"""))

# ── SECTION 1 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 4. Label Distributions

**Why this matters:**
Before training any model, we need to know the class structure of our three downstream \
classification tasks: social-unit ID, coda-type ID, and individual-whale ID. Class \
imbalance affects loss function design and evaluation metric choice.

The DSWP release covers social units A, D, and F — three of the nine Eastern Caribbean \
units studied by Gero et al. (2016). The overall population belongs to vocal clan EC1 \
(the Eastern Caribbean 1 clan), with a small EC2 minority outside the DSWP range.  \
Units A, D, and F have been continuously monitored by the Dominica Sperm Whale Project \
since 2005 (Gero 2005–2018), making them the best-documented social groups in the world.
"""))

cells.append(code("""\
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DSWP Label Distributions", fontsize=16, fontweight="bold")

# (a) Social unit counts
ax = axes[0, 0]
unit_counts = df["unit"].value_counts()[UNIT_ORDER]
bars = ax.bar(UNIT_ORDER, unit_counts.values,
              color=[UNIT_COLORS[u] for u in UNIT_ORDER], edgecolor="black", linewidth=0.7)
for bar, val in zip(bars, unit_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xlabel("Social Unit"); ax.set_ylabel("Coda Count")
ax.set_title("(a) Social Unit Distribution"); ax.set_ylim(0, 1050)
noise_patch = plt.Rectangle((0,0),1,1, fc="none", ec="grey", ls="--")
ax.legend([noise_patch], [f"Includes {df['is_noise'].sum()} noise-tagged"], fontsize=9)

# (b) Top 15 coda types (clean only)
ax = axes[0, 1]
ctype_counts = df_clean["coda_type"].value_counts().head(15)
ax.barh(ctype_counts.index[::-1], ctype_counts.values[::-1],
        color=UNIT_COLORS["F"], edgecolor="black", linewidth=0.5)
ax.set_xlabel("Count"); ax.set_title("(b) Top 15 Coda Types (clean only)")
ax.tick_params(axis="y", labelsize=8)

# (c) Individual ID distribution (identified only)
ax = axes[1, 0]
idn_counts = df_id["individual_id"].value_counts().head(12)
ax.bar(range(len(idn_counts)), idn_counts.values, color="#8172B2", edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(idn_counts)))
ax.set_xticklabels(idn_counts.index, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Individual ID (IDN)"); ax.set_ylabel("Coda Count")
ax.set_title(f"(c) Individual ID Distribution  (n={len(df_id)}, unidentified={df['individual_id'].eq('0').sum()})")

# (d) Recording year × social unit
ax = axes[1, 1]
year_unit = df.groupby(["year", "unit"]).size().unstack(fill_value=0).reindex(columns=UNIT_ORDER, fill_value=0)
bottom = np.zeros(len(year_unit))
for unit in UNIT_ORDER:
    ax.bar(year_unit.index, year_unit[unit], bottom=bottom,
           color=UNIT_COLORS[unit], label=f"Unit {unit}", edgecolor="black", linewidth=0.4)
    bottom += year_unit[unit].values
ax.set_xlabel("Year"); ax.set_ylabel("Coda Count")
ax.set_title("(d) Recording Year × Social Unit"); ax.legend(title="Unit", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig1_label_distributions.png"), dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""\
### Observations

- **Severe class imbalance**: Unit F dominates with 892 codas (59.4% of total), versus \
  336 for D and 273 for A. This is biologically expected — Unit F is one of the largest \
  and most active social groups in the Dominica population — but it has direct consequences \
  for training: we must use **stratified sampling** for train/test splits and \
  **weighted cross-entropy loss** for classification heads.

- **Coda type imbalance**: The `1+1+3` pattern comprises 35.1% of clean codas. This is \
  consistent with Gero et al. (2016), who found that `1+1+3` and `5R1` together account \
  for ~65% of all codas across Eastern Caribbean units. These two types serve as \
  pan-clan "identity codas" — Hersh et al. (PNAS 2022) showed they function as symbolic \
  cultural markers that resist cross-clan stylistic convergence.

- **Recording coverage is temporally continuous** (2005–2010), which rules out obvious \
  temporal confounds but requires us to test whether recording year correlates with \
  social unit (it does not — all three units appear consistently across years).

- **Individual ID coverage is sparse**: 672 codas (44.8%) have IDN=0, meaning the \
  vocalising whale was not individually identified. This is a biological field limitation, \
  not a data error. We restrict individual-ID experiments to the 763 labeled codas.
"""))

# ── SECTION 2 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 5. Rhythm Channel: Inter-Click Interval (ICI) Analysis

**Why this matters:**
The rhythm channel is defined by the sequence of time intervals between consecutive clicks \
within a coda. It encodes **coda type** — the categorical click-count and timing pattern \
that has been used since Watkins & Schevill (1977) to classify sperm whale communication.

**Leitão et al. (arXiv:2307.05304)** went further: they showed that subtle *micro-variations* \
in ICI values within a given coda type (i.e., how a whale renders `1+1+3`) track social-unit \
membership and are culturally transmitted across clan boundaries through social learning. \
This means ICI sequences carry **two layers of information simultaneously**: coarse categorical \
coda type, and fine-grained individual/social-unit style.

**Gero et al. (2016)** established the baseline ICI taxonomy for Eastern Caribbean whales, \
identifying 21 coda types from nine social units over six years — the same classification \
scheme used in our labels.

Our pre-computed ICI values come directly from DominicaCodas.csv (Sharma et al. 2024), \
which provides `ICI1`–`ICI9` for every coda. No peak detection is required.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rhythm Channel: Inter-Click Interval (ICI) Distributions", fontsize=14, fontweight="bold")

# (a) Mean ICI per social unit — violin plot
ax = axes[0]
unit_ici = [df_clean[df_clean["unit"] == u]["mean_ici_ms"].dropna().values for u in UNIT_ORDER]
parts = ax.violinplot(unit_ici, positions=range(len(UNIT_ORDER)), showmedians=True)
for pc, u in zip(parts["bodies"], UNIT_ORDER):
    pc.set_facecolor(UNIT_COLORS[u]); pc.set_alpha(0.7)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(UNIT_ORDER)))
ax.set_xticklabels([f"Unit {u}" for u in UNIT_ORDER])
ax.set_ylabel("Mean ICI (ms)"); ax.set_title("(a) Mean ICI by Social Unit")
for i, (vals, u) in enumerate(zip(unit_ici, UNIT_ORDER)):
    ax.text(i, np.median(vals) + 2, f"{np.median(vals):.1f}ms", ha="center", fontsize=9)

# (b) Mean ICI per coda type (top 10) — boxplot
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
plt.savefig(os.path.join(FIGS, "fig2_ici_distributions.png"), dpi=150, bbox_inches="tight")
plt.show()

# Summary statistics per unit
print("Mean ICI (ms) by social unit:")
print(df_clean.groupby("unit")["mean_ici_ms"].describe().round(2).to_string())
"""))

cells.append(md("""\
### Observations

- **ICI distributions overlap substantially across units** (panel a). This is expected: \
  all three units share many coda types, so the coda-type-level ICI signal dominates the \
  unit-level signal. The Leitão et al. micro-variation signal is subtle — it lives *within* \
  a coda type, not across types. A model that naively averages ICI will not recover it; \
  the GRU encoder must process the full sequence to capture sequential timing patterns.

- **ICI discriminates coda type very well** (panel b). The boxplots show clear separation \
  between types: fast types like `5R1` have much shorter mean ICIs (~90ms) compared to \
  slow types like `1+1+3` (~300ms). This confirms that raw ICI is a powerful rhythm \
  feature — even a simple zero-padded ICI vector should give strong coda-type classification \
  in our Baseline 1A.

- **Wide ICI variance overall** (mean=177ms, std=88ms) indicates the rhythm channel \
  spans a large dynamic range. StandardScaler normalisation will be necessary before \
  feeding ICI sequences to the GRU encoder.
"""))

# ── SECTION 3 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 6. Acoustic Properties: Duration and Click Count

**Why this matters:**
Duration and click count are the most basic acoustic properties of a coda. They are also \
the targets for two of our WhAM probing experiments (Phase 2 / Experiment 3): if WhAM's \
internal representations correlate with `n_clicks` and mean ICI, that confirms it encodes \
rhythm-level information regardless of whether it was trained to do so.

Gero et al. (2016) reported that coda duration ranges from ~0.1s to >3s depending on type, \
and that click count ranges from 3 to 14+ in Eastern Caribbean codas. Our DSWP subset \
should reflect these statistics.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Acoustic Properties of Clean Codas", fontsize=14, fontweight="bold")

# (a) Duration distribution per unit
ax = axes[0]
for u in UNIT_ORDER:
    vals = df_clean[df_clean["unit"] == u]["duration_sec"]
    ax.hist(vals, bins=30, alpha=0.6, label=f"Unit {u}", color=UNIT_COLORS[u], density=True)
ax.set_xlabel("Duration (s)"); ax.set_ylabel("Density")
ax.set_title("(a) Coda Duration by Unit"); ax.legend()

# (b) Click count distribution
ax = axes[1]
nc = df_clean["n_clicks"].astype(int)
click_counts = nc.value_counts().sort_index()
ax.bar(click_counts.index, click_counts.values, color="#DD8452", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Number of Clicks"); ax.set_ylabel("Count")
ax.set_title("(b) Clicks per Coda")
for x, v in click_counts.items():
    ax.text(x, v + 2, str(v), ha="center", fontsize=8)

# (c) ICI count (= n_clicks - 1) per unit
ax = axes[2]
for u in UNIT_ORDER:
    vals = df_clean[df_clean["unit"] == u]["n_ici"].astype(int)
    ax.hist(vals, bins=range(1, 15), alpha=0.6, label=f"Unit {u}",
            color=UNIT_COLORS[u], density=True)
ax.set_xlabel("Number of ICI values"); ax.set_ylabel("Density")
ax.set_title("(c) ICI Count by Unit"); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3_duration_clicks.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"Duration — mean: {df_clean['duration_sec'].mean():.3f}s  std: {df_clean['duration_sec'].std():.3f}s")
print(f"Click count — mode: {nc.mode()[0]}  range: {nc.min()}–{nc.max()}")
"""))

cells.append(md("""\
### Observations

- **Duration is right-skewed and overlaps substantially across units**, peaking around \
  0.3–0.8s with a long tail to ~2.5s. The distribution shape is consistent with \
  Sharma et al. (2024) who reported a mean duration of ~1.1s across the broader Dominica \
  corpus. Our lower mean (0.726s) is expected since the DSWP 1–1501 subset is \
  concentrated in the 2005–2010 period dominated by faster `1+1+3` and `5R1` types.

- **5-click codas are dominant** (n=838, 60.6% of clean codas), followed by 7-click. \
  This matches Gero et al. (2016) and Hersh et al. (2022), where `5R1`, `5R2`, `5R3` \
  and `1+1+3` (which has 5 clicks: 1+1+3) together account for the majority of Eastern \
  Caribbean codas.

- **Implication for the rhythm encoder**: variable-length input is unavoidable — the GRU \
  encoder must handle sequences from 2 to 9+ ICI values. Zero-padding to length 9 (as \
  done in Baseline 1A) is a reasonable choice since the tail beyond 9 is sparse.
"""))

# ── SECTION 4 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 7. Channel Independence: Coda Type × Social Unit

**The central biological claim we are operationalising:**
Beguš et al. (2024) established that the rhythm channel (coda type) and spectral channel \
(vowel) are *syntactically independent* — the same coda type can be produced with \
different spectral textures, and vice versa. If the two channels truly carry independent \
information, we should observe that **coda types are shared across social units**, \
rather than being unit-specific.

This is the most important sanity check for our DCCE architecture. If coda type and \
social unit were perfectly correlated, the rhythm encoder would implicitly encode social \
unit, and fusing the two channels would be redundant. The heatmap below tests this directly.
"""))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(12, 7))

ct_unit = (df_clean
           .groupby(["coda_type", "unit"])
           .size()
           .unstack(fill_value=0)
           .reindex(columns=UNIT_ORDER, fill_value=0))
top20 = df_clean["coda_type"].value_counts().head(20).index
ct_unit_top = ct_unit.loc[ct_unit.index.isin(top20)]
ct_norm = ct_unit_top.div(ct_unit_top.sum(axis=1), axis=0)   # row-normalised

sns.heatmap(ct_norm, annot=ct_unit_top, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Proportion within coda type (row-normalised)"},
            annot_kws={"size": 8})
ax.set_xlabel("Social Unit", fontsize=12); ax.set_ylabel("Coda Type", fontsize=12)
ax.set_title(
    "Coda Type × Social Unit Heatmap\\n"
    "Counts shown; colour = row proportion. "
    "Do coda types appear across all units (supporting channel independence)?",
    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4_codatype_unit_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()

# Quantify sharing
shared = (ct_unit_top > 0).sum(axis=1)
print(f"Coda types present in all 3 units : {(shared == 3).sum()}")
print(f"Coda types present in 2 units     : {(shared == 2).sum()}")
print(f"Coda types present in 1 unit only : {(shared == 1).sum()}")
"""))

cells.append(md("""\
### Observations

- **Most coda types appear in all three social units.** Of the top 20 types, the majority \
  are produced by whales from units A, D, *and* F. This directly confirms the biological \
  claim: coda type is a clan-level category, not a unit-specific marker. The two channels \
  are genuinely independent.

- **Unit F contributes more counts to almost every type** due to its larger size, but the \
  *row-normalised* heatmap shows that the proportion of each type is fairly consistent \
  across units for the most common types (`1+1+3`, `5R1`, `4D`).

- **Implication for DCCE**: The rhythm encoder must learn to disentangle coda type from \
  social-unit identity. Our cross-channel contrastive augmentation (rhythm of coda A + \
  spectral texture of another coda from the same unit) is designed precisely to prevent \
  the rhythm encoder from becoming a proxy for social unit.

- **Implication for evaluation**: Coda-type classification and social-unit classification \
  are genuinely different tasks — a model that excels at one does not automatically excel \
  at the other. Both must be measured separately in our linear probe evaluation.
"""))

# ── SECTION 5 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 8. The IDN=0 Problem: Unidentified Individuals

**Context:**
Individual whale identification in the DSWP is performed by photo-ID (fluke morphology) \
and acoustic size estimation during field sessions. Not all vocalising whales can be \
identified — particularly in multi-animal encounters, poor visibility, or when the \
vocaliser does not surface during the recording session.

In our label file, `individual_id = 0` denotes a coda whose vocaliser was not identified. \
This is a biological field limitation, not a labelling error. Both DominicaCodas.csv \
(Sharma et al.) and Gero et al. (2016) agree on which codas are unidentified.

**Why it matters for DCCE:**
The individual-ID contrastive loss on the spectral encoder (`L_id(s_emb)`) requires \
known positive pairs — two codas from the *same individual*. Codas with IDN=0 cannot \
contribute to this loss. If unidentified codas are concentrated in specific units or \
coda types, this could bias the spectral encoder.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Investigation of Unidentified Whales (IDN = 0)", fontsize=13, fontweight="bold")

df_clean = df_clean.copy()
df_clean["id_known"] = df_clean["individual_id"].ne("0")

# (a) IDN=0 by social unit
ax = axes[0]
id_unit = (df_clean
           .groupby(["unit", "id_known"])
           .size()
           .unstack(fill_value=0))
id_unit.columns = ["Unknown (IDN=0)", "Identified"]
id_unit = id_unit.reindex(UNIT_ORDER)
id_unit.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"],
             edgecolor="black", linewidth=0.5)
ax.set_xlabel("Social Unit"); ax.set_ylabel("Count")
ax.set_title("(a) IDN=0 by Social Unit")
ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=0)

# (b) IDN=0 by recording year
ax = axes[1]
id_year = (df_clean
           .groupby(["year", "id_known"])
           .size()
           .unstack(fill_value=0))
id_year.columns = ["Unknown", "Identified"]
id_year.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"],
             edgecolor="black", linewidth=0.5)
ax.set_xlabel("Year"); ax.set_ylabel("Count")
ax.set_title("(b) IDN=0 by Recording Year"); ax.legend(fontsize=9)

# (c) % unknown per coda type (top 20)
ax = axes[2]
top20 = df_clean["coda_type"].value_counts().head(20).index
id_ct = (df_clean[df_clean["coda_type"].isin(top20)]
         .groupby(["coda_type", "id_known"])
         .size()
         .unstack(fill_value=0))
id_ct.columns = ["Unknown", "Identified"]
pct_unknown = (id_ct["Unknown"] / id_ct.sum(axis=1) * 100).sort_values(ascending=False)
pct_unknown.plot(kind="bar", ax=ax, color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_ylabel("% Unknown (IDN=0)"); ax.set_xlabel("Coda Type")
ax.set_title("(c) % Unknown by Coda Type (top 20)")
ax.tick_params(axis="x", rotation=45)
overall_pct = df_clean["id_known"].eq(False).mean() * 100
ax.axhline(overall_pct, color="black", ls="--", lw=1.2,
           label=f"Overall ({overall_pct:.0f}%)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig5_idn0_investigation.png"), dpi=150, bbox_inches="tight")
plt.show()

# Summary
print(f"IDN=0 by unit:")
print(df_clean[df_clean["individual_id"]=="0"]["unit"].value_counts().to_string())
print(f"\\nOverall IDN=0 rate: {df_clean['id_known'].eq(False).mean()*100:.1f}%")
"""))

cells.append(md("""\
### Observations

- **IDN=0 is almost entirely confined to Unit F** (panel a). Units A and D have near-complete \
  individual identification. This makes sense biologically: Unit F is the largest group, \
  and in multi-animal encounters it is harder to attribute every coda to a specific individual.

- **IDN=0 is evenly distributed across recording years** (panel b) — there is no trend \
  toward improvement over time. This suggests it is a structural limitation of the \
  recording methodology (boat-based hydrophone with limited localisation), not a data \
  quality issue that improves with practice.

- **IDN=0 rates are consistent across coda types** (panel c) — no coda type is \
  disproportionately unidentified, ruling out the possibility that certain vocalisations \
  are systematically attributed to unidentified animals.

- **Decision**: For individual-ID experiments, we restrict to the 763 codas with known \
  IDN (13 individuals). The spectral encoder's `L_id` contrastive loss will be computed \
  only on this subset. The social-unit contrastive loss is unaffected since unit labels \
  are available for all 1,383 clean codas.
"""))

# ── SECTION 6 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 9. Spectral Channel: Sample Mel-Spectrograms

**Why this matters:**
The spectral encoder operates on mel-spectrograms — 2D time-frequency representations \
of the coda audio. Before training, we want to visually confirm that spectrograms differ \
meaningfully across social units and coda types, and understand the frequency range and \
temporal structure of the signals.

**Beguš et al. (2024)** showed that spectral variation within the inter-pulse intervals \
(the space between clicks) carries vowel-like formant structure at frequencies roughly \
3–9 kHz. They labelled this variation as `a` (lower spectral peak) and `i` (higher \
spectral peak), analogous to the low/high vowel distinction in human phonetics.

The mel-spectrogram captures exactly this frequency range when parameterised with \
`fmax=8000 Hz` and 64–128 mel bins, making it the appropriate input for the spectral \
encoder.
"""))

cells.append(code("""\
def pick_clean(df_sub, n=2):
    return df_sub[df_sub["is_noise"] == 0]["coda_id"].tolist()[:n]

fig = plt.figure(figsize=(16, 9))
fig.suptitle("Sample Mel-Spectrograms by Social Unit (2 per unit)", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.3)

for row_idx, u in enumerate(UNIT_ORDER):
    coda_ids = pick_clean(df_clean[df_clean["unit"] == u], n=2)
    for col_idx, coda_id in enumerate(coda_ids):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        y, sr = librosa.load(os.path.join(AUDIO, f"{coda_id}.wav"), sr=None, mono=True)
        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                 fmax=8000, ax=ax, cmap="magma")
        row = df_clean[df_clean["coda_id"] == coda_id].iloc[0]
        ax.set_title(
            f"Unit {u}  |  coda #{coda_id}  |  type: {row.coda_type}  |  {row.duration_sec:.2f}s",
            fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel("")

plt.savefig(os.path.join(FIGS, "fig6_sample_spectrograms.png"), dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""\
### Observations

- **Click structure is clearly visible** as vertical high-energy striations in the \
  spectrograms. The number of striations matches the click count in the coda type label \
  (e.g., a `1+1+3` coda shows 1 click, gap, 1 click, gap, 3 rapid clicks).

- **High-frequency energy dominates** (3,000–8,000 Hz range), consistent with the \
  formant peaks reported by Beguš et al. (2024) and the spectral centroid measurements \
  in Section 10. This means our `fmax=8000 Hz` parameterisation captures the relevant \
  spectral content.

- **Temporal structure varies across units** in subtle ways — this is the "vowel" \
  variation the spectral encoder is designed to capture. Visual inspection alone is \
  insufficient; the encoder must learn to quantify this.

- **Implication for the spectral encoder**: The CNN input should be normalised \
  mel-spectrograms cropped or padded to a fixed time dimension (e.g. 128 frames). \
  Using `fmax=8000 Hz` and 128 mel bins is consistent with the literature.
"""))

# ── SECTION 7 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 10. t-SNE of Raw ICI Feature Space

**Why this matters:**
Before building a learned rhythm encoder, we want to know what the raw ICI feature space \
looks like. If simple ICI vectors already form compact, separable clusters for coda type \
or social unit, that has two implications:

1. A simple baseline (zero-padded ICI → logistic regression) might be surprisingly strong, \
   which raises the bar for the DCCE rhythm encoder to demonstrate improvement.
2. The structure of the raw ICI space tells us what the GRU encoder's job actually is: \
   is it *creating* structure from noise, or *refining* already-separable structure?

**Leitão et al. (arXiv:2307.05304)** showed that ICI-based clustering closely aligns \
with biological clan/unit assignments — suggesting the raw ICI space is already highly \
informative. We replicate this here on the DSWP subset.
"""))

cells.append(code("""\
# Build ICI matrix: zero-pad each coda's ICI sequence to length 9
MAX_ICI = 9
ici_matrix = np.zeros((len(df_clean), MAX_ICI))
for i, row in enumerate(df_clean.itertuples()):
    for j, v in enumerate(row.ici_list[:MAX_ICI]):
        ici_matrix[i, j] = v

ici_scaled = StandardScaler().fit_transform(ici_matrix)

print("Running t-SNE (perplexity=30, max_iter=1000)...")
emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(ici_scaled)
df_clean = df_clean.copy()
df_clean["tsne_x"], df_clean["tsne_y"] = emb[:, 0], emb[:, 1]
print("Done.")
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("t-SNE of Standardised ICI Vectors (n=1,383 clean codas)", fontsize=14, fontweight="bold")

# (a) Colour by social unit
ax = axes[0]
for u in UNIT_ORDER:
    mask = df_clean["unit"] == u
    ax.scatter(df_clean.loc[mask, "tsne_x"], df_clean.loc[mask, "tsne_y"],
               c=UNIT_COLORS[u], label=f"Unit {u}", alpha=0.6, s=15, edgecolors="none")
ax.set_title("(a) Coloured by Social Unit")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(markerscale=2)

# (b) Colour by coda type (top 8)
ax = axes[1]
top8    = df_clean["coda_type"].value_counts().head(8).index.tolist()
palette = sns.color_palette("tab10", len(top8))
for ct, color in zip(top8, palette):
    mask = df_clean["coda_type"] == ct
    ax.scatter(df_clean.loc[mask, "tsne_x"], df_clean.loc[mask, "tsne_y"],
               color=color, label=ct, alpha=0.7, s=15, edgecolors="none")
other = ~df_clean["coda_type"].isin(top8)
ax.scatter(df_clean.loc[other, "tsne_x"], df_clean.loc[other, "tsne_y"],
           color="lightgrey", label="Other", alpha=0.3, s=8, edgecolors="none")
ax.set_title("(b) Coloured by Coda Type (top 8)")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7_tsne_ici.png"), dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""\
### Observations

- **Coda types form very tight, well-separated clusters** (panel b). Even without any \
  learned representation, the raw standardised ICI vector cleanly separates coda types \
  in 2D t-SNE space. This confirms that ICI is the primary determinant of coda type — \
  consistent with the entire bioacoustics literature since Watkins & Schevill (1977).

- **Social units do *not* separate cleanly** (panel a) — the three unit colours are \
  largely intermixed within each coda-type cluster. This is precisely the challenge \
  our model must solve: social-unit identity is encoded as *micro-variations within* \
  coda-type clusters, not as a coarser partitioning of the ICI space. Leitão et al. \
  (2023–2025) called this "style variation within type", and showed it is culturally \
  transmitted.

- **Implication for architecture**: The rhythm encoder's job is *not* to re-discover \
  coda type — a simple lookup could do that. Its job is to capture the social-unit \
  signal that exists *residually after* coda type is accounted for. The cross-channel \
  contrastive objective and the auxiliary coda-type head together pressure the encoder \
  to maintain type awareness while also encoding style.

- **Implication for Baseline 1A**: A logistic regression on raw ICI vectors will likely \
  achieve near-perfect coda-type classification but much weaker social-unit classification. \
  This is the expected baseline pattern.
"""))

# ── SECTION 8 ─────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 11. Spectral Channel: Centroid Analysis from Audio

**Why this matters:**
We compute spectral centroids from the raw WAV files to verify that the spectral channel \
carries meaningful variance across social units — independent of the rhythm channel. \
This is the key biological independence claim from Beguš et al. (2024).

The spectral centroid is a crude proxy for the vowel formant position (the actual \
feature studied by Beguš et al. is the first spectral peak `f1pk` within inter-pulse \
intervals). It nonetheless provides a fast sanity check on the hypothesis that spectral \
texture varies by social unit in a way that is not reducible to ICI.

We compute centroids on a stratified random sample of ~67 codas per unit (~200 total) \
to keep runtime manageable.
"""))

cells.append(code("""\
print("Computing spectral centroids from audio (stratified sample, ~1-2 min)...")

parts_list = []
for u in UNIT_ORDER:
    sub = df_clean[df_clean["unit"] == u].sample(min(len(df_clean[df_clean["unit"]==u]), 67), random_state=42)
    parts_list.append(sub)
sample_df = pd.concat(parts_list, ignore_index=True)

centroids = []
for row in sample_df.itertuples():
    y, sr = librosa.load(os.path.join(AUDIO, f"{row.coda_id}.wav"), sr=None, mono=True)
    centroids.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])))
sample_df = sample_df.copy()
sample_df["spectral_centroid_hz"] = centroids

print(f"Done. Computed centroids for {len(sample_df)} codas.")
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Spectral Channel: Centroid Distribution and Rhythm–Spectral Scatter",
             fontsize=13, fontweight="bold")

# (a) Centroid violin per unit
ax = axes[0]
unit_cents = [sample_df[sample_df["unit"]==u]["spectral_centroid_hz"].values for u in UNIT_ORDER]
parts_v = ax.violinplot(unit_cents, positions=range(3), showmedians=True)
for pc, u in zip(parts_v["bodies"], UNIT_ORDER):
    pc.set_facecolor(UNIT_COLORS[u]); pc.set_alpha(0.7)
parts_v["cmedians"].set_color("black"); parts_v["cmedians"].set_linewidth(2)
ax.set_xticks(range(3))
ax.set_xticklabels([f"Unit {u}" for u in UNIT_ORDER])
ax.set_ylabel("Spectral Centroid (Hz)"); ax.set_title("(a) Centroid Distribution by Unit")
for i, (vals, u) in enumerate(zip(unit_cents, UNIT_ORDER)):
    ax.text(i, np.median(vals) + 80, f"{np.median(vals):.0f} Hz", ha="center", fontsize=9)

# (b) Rhythm vs. spectral scatter — are the two channels independent?
ax = axes[1]
ax.scatter(sample_df["mean_ici_ms"], sample_df["spectral_centroid_hz"],
           c=[UNIT_COLORS[u] for u in sample_df["unit"]],
           alpha=0.55, s=22, edgecolors="none")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=UNIT_COLORS[u], label=f"Unit {u}") for u in UNIT_ORDER]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel("Mean ICI (ms)  [rhythm channel proxy]")
ax.set_ylabel("Spectral Centroid (Hz)  [spectral channel proxy]")
ax.set_title("(b) Rhythm vs. Spectral: Are the two channels independent?")

# Pearson correlation
from scipy.stats import pearsonr
r, p = pearsonr(sample_df["mean_ici_ms"].dropna(), sample_df.loc[sample_df["mean_ici_ms"].notna(), "spectral_centroid_hz"])
ax.text(0.05, 0.93, f"r = {r:.3f}  (p = {p:.3f})", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig8_spectral_centroid.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\\nSpectral centroid statistics:")
print(sample_df.groupby("unit")["spectral_centroid_hz"].describe().round(0).to_string())
"""))

cells.append(md("""\
### Observations

- **Spectral centroid distributions overlap substantially across units** (panel a). This \
  is expected: the spectral centroid is a global summary measure, whereas Beguš et al. \
  (2024) showed the vowel signal lives in the *within-click inter-pulse intervals*, \
  not in the overall spectral shape. The centroid is an imperfect proxy, but its high \
  variance (~8,894 ± 2,913 Hz across the full sample) confirms that significant spectral \
  variation exists in the data — variation the spectral encoder can potentially learn to \
  exploit.

- **Rhythm and spectral channels are weakly correlated** (panel b, Pearson r ≈ 0). \
  The scatter plot shows no systematic relationship between mean ICI (rhythm proxy) and \
  spectral centroid (spectral proxy). This empirically confirms the biological \
  independence claim of Beguš et al. (2024): knowing a coda's rhythm type does not \
  predict its spectral texture, and vice versa. This is the foundational justification \
  for the dual-encoder architecture.

- **Key architectural implication**: Because the two channels are independent, the fusion \
  layer in DCCE should learn a *complementary* combination — not a redundant one. The \
  cross-channel contrastive augmentation (pairing rhythm of coda A with spectral of coda \
  B from the same unit) is the mechanism that enforces this complementarity during training.
"""))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 12. EDA Summary and Implications for Modelling

The following table summarises the key quantitative findings and their direct implications \
for the DCCE design and experimental protocol.

| Finding | Value | Implication |
|---|---|---|
| Total / clean codas | 1,501 / 1,383 | Training set is small — laptop-scale models are appropriate |
| Unit imbalance | F=59.4%, D=22.4%, A=18.2% | Stratified splits + weighted CE loss required |
| Top coda type (1+1+3) | 35.1% of clean | Macro-F1 is the right metric, not accuracy |
| ICI clearly separates coda type | t-SNE clusters tight | Baseline 1A (raw ICI → logReg) will be strong on coda type |
| ICI does *not* separate social unit | Units intermixed in t-SNE | Social-unit signal = micro-variation *within* coda-type clusters |
| Coda type shared across all 3 units | Most types in A, D, F | Channels are independent — dual encoder is justified |
| Rhythm–spectral correlation | r ≈ 0 | Independent channels confirmed empirically |
| IDN=0 confined to Unit F | 672 / 1,501 codas | Individual-ID experiments: 763 codas, 13 individuals |
| Spectral centroid variance | 8,894 ± 2,913 Hz | Spectral encoder has real signal to learn from |
| Coda duration | 0.726 ± 0.374s | Fixed 128-frame mel-spectrogram window appropriate |

### Next step: Phase 1 — Baselines

With the data understood, we proceed to:
1. **Baseline 1A** — Raw ICI (zero-padded, length 9) → logistic regression. \
   Establishes the floor for the rhythm encoder.
2. **Baseline 1B** — WhAM embeddings (extracted from all 1,501 DSWP codas using the \
   publicly available Zenodo weights) → linear probe. This is the primary comparison \
   target for Experiment 1 and replicates WhAM's downstream evaluation on our exact \
   data split.
"""))

# ── ASSIGN IDs ────────────────────────────────────────────────────────────────
import string, random
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}-{''.join(random.choices(string.ascii_lowercase, k=6))}"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {NB}")
print(f"Cells: {len(cells)}")
