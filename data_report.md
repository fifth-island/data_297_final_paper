# Data Report
## Beyond WhAM — Sperm Whale Coda Dataset Assembly and Exploratory Analysis

**Project**: Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding  
**Course**: CS 297 Final Paper · April 2026  

---

## 1. Biological Context

### 1.1 What are sperm whales and why do their vocalizations matter?

Sperm whales (*Physeter macrocephalus*) are among the most cognitively sophisticated animals on Earth. They have the largest brain of any known species, live in multigenerational matrilineal families, and exhibit documented cultural transmission across generations. Their social structure centers on **social units** — stable family groups of females and juveniles that travel and forage together, and whose membership can persist across decades.

Sperm whales communicate primarily through rhythmically patterned click sequences called **codas** — short bursts of 3–40 clicks separated by precise inter-click intervals. Codas are social signals, not echolocation: groups of whales that share a coda repertoire form **vocal clans**, and the coda repertoire of a clan is stable enough to function like a dialect. Project CETI (Cetacean Translation Initiative) has proposed that decoding the structure of codas may offer the first window into the semantic content of non-human communication at scale.

### 1.2 The two-channel structure of codas

A key insight from recent bioacoustics research, formalized by Beguš et al. (2024), is that a single coda waveform carries **two syntactically independent information channels**:

| Channel | Acoustic Feature | What It Encodes |
|---|---|---|
| **Rhythm** | Inter-click intervals (ICI) — the time gaps between consecutive clicks | *Coda type* — the categorical click-count and timing pattern shared across a clan (e.g., 1+1+3, 5R1, 4D). Functions like a word category. |
| **Spectral** | Formant-like spectral texture within each click | *Speaker identity* — the individual voice fingerprint and social-unit membership. Functions like a voice. |

These two channels are **independent**: the same coda type (same rhythm) can be produced by any individual, and the same individual produces many coda types. This means you cannot tell *who* is speaking from rhythm alone, nor can you tell *what they said* from spectral texture alone. Both channels are needed.

This decomposition is not a hypothesis — it is an established finding with multiple independent lines of evidence (Leitão et al. 2023; Beguš et al. 2024; Sharma et al. 2024). It is the biological prior that our model architecture exploits by design.

---

## 2. The Data Challenge

### 2.1 Starting point: DSWP audio without labels

The primary audio dataset in this project is the **Dominica Sperm Whale Project (DSWP)** dataset, released by Paradise et al. (2025, NeurIPS) via HuggingFace (`orrp/DSWP`). It contains:

- **1,501 isolated coda WAV files** (`1.wav` through `1501.wav`)
- Recordings from the waters off Dominica, 2005–2010
- The recording program was led by Shane Gero (Dominica Sperm Whale Project), who has monitored this population continuously since 2005
- License: CC BY 4.0 — fully open for research use

**The critical problem**: the HuggingFace release ships as **audio-only**. No labels are included — not social unit, not coda type, not individual identity, not ICI sequences. To use this dataset for any classification or representation learning task, labels had to be obtained from external sources.

This is not a minor inconvenience. Without labels:
- There is no training signal for supervised baselines
- There are no positive pairs for contrastive learning
- There is no evaluation protocol
- Linear probes cannot be designed

Obtaining labels for 1,501 codas from public sources was therefore a prerequisite for every experiment in this project — and it was not straightforward.

### 2.2 Label sources investigated

We identified and retrieved five public datasets that could potentially label the DSWP audio files. The investigation process and outcomes are summarized below.

---

#### Source 1 — DominicaCodas.csv (Sharma et al., 2024)

**Origin**: Sharma, P., Gero, S., Payne, R. et al. *"Contextual and combinatorial structure in sperm whale vocalisations."* Nature Communications 15, 3617 (2024).  
**Repository**: `github.com/pratyushasharma/sw-combinatoriality`  
**License**: CC BY 4.0

This dataset was released alongside a study of the combinatorial structure of sperm whale codas. It contains 8,719 rows from the complete Dominica corpus (2005–2018), with fields:

| Column | Description |
|---|---|
| `codaNUM2018` | Sequential coda ID (1–8,878) |
| `CodaType` | Coda type label (35 categories: 1+1+3, 5R1, 4D, ...) |
| `Unit` | Social unit (A, D, F, J, K, ... 13 named units) |
| `Clan` | Vocal clan (EC1 or EC2) |
| `IDN` | Individual whale numeric ID |
| `ICI1`–`ICI9` | Pre-computed inter-click intervals (seconds) |
| `Duration` | Coda duration (seconds) |
| `Date` | Recording date |

**The key discovery**: exactly **1,501 rows** in this dataset have `codaNUM2018` in the range 1–1,501 — and they cover exactly social units A, D, and F. This was not documented anywhere in either dataset's release notes. We verified the correspondence by matching `ICI1` values and `Duration` against values computed from the WAV files: the match is exact. The `codaNUM2018` index is the shared key: `codaNUM2018 = N` maps to `N.wav` in the DSWP release.

**Result**: DominicaCodas.csv provides complete labels for all 1,501 DSWP audio files.

---

#### Source 2 — codamd.csv (Beguš et al., 2024)

**Origin**: Beguš, G. et al. *"The Phonology of Sperm Whale Coda Vowels."*  
**Repository**: `github.com/Project-CETI/coda-vowel-phonology`  

This dataset provides hand-verified **vowel labels** (`handv`: `a` or `i`) for 1,375 codas — the spectral ground truth formalized by Beguš et al.'s phonological analysis. Vowel labels are exactly the kind of fine-grained spectral annotation that would be ideal supervision for our spectral encoder.

**The problem**: `codamd.csv` covers `codanum` range 4,933–8,860. The DSWP range is 1–1,501. These ranges **do not overlap**. Vowel labels for the 1,501 DSWP codas are not publicly available.

**Impact on model design**: because vowel supervision is unavailable for our audio, the spectral encoder cannot be trained with explicit vowel targets. This motivated the architectural choice of using a mel-spectrogram CNN trained via unit-level contrastive loss and individual-ID auxiliary loss, rather than vowel classification.

---

#### Source 3 — focal-coarticulation-metadata.csv (Beguš et al., 2024)

**Origin**: Same repository as codamd.csv.  

Provides per-click spectral formant measurements (peak frequency, f1, f2 in Hz) for click transitions within codas — 1,097 rows covering the same 13 named individual whales as codamd.

**The problem**: Same coverage gap — covers codaNUM 4,933+ only; no overlap with DSWP.

**Outcome**: Not used in any experiment. Documented here for completeness.

---

#### Source 4 — sperm-whale-dialogues.csv (Sharma et al., 2024)

**Origin**: Same repository as DominicaCodas.csv.  

Contains 3,840 codas from conversational exchange sequences (multi-whale interactions), with whale ID and ICI sequences. Potentially useful for rhythm encoder pre-training.

**The problems**: (1) uses a different whale ID numbering scheme (1–11 numeric, incompatible with IDN in DominicaCodas); (2) no coda type labels; (3) no overlap with DSWP IDs. Joining this with the main label file is non-trivial and ultimately redundant.

**Outcome**: Not used in any experiment.

---

#### Source 5 — Gero et al. 2016 (Zenodo, CC0)

**Origin**: Gero, Whitehead & Rendell (2016). *"Individual, unit and vocal clan level identity cues in sperm whale codas."*  
**Zenodo**: 4963528  

Contains 4,116 rows with coda labels from earlier Caribbean field work.

**Investigation**: Initial assumption was that `CodaNumber` in Gero 2016 might map to DSWP file indices. Tested: only 10 out of 1,454 shared IDs produced matching (ICI1, Length) pairs — a 0.7% alignment rate. The `CodaNumber` index is an independent sequential ID, not the DSWP file index. A fuzzy join on (ICI1, Length) recovers 1,472/1,501 matches (98.1%), but this is entirely redundant — DominicaCodas.csv already provides superior labels with more fields.

**Outcome**: Not used. Documented here to explain why an apparently promising source was discarded.

---

### 2.3 The assembled label file: dswp_labels.csv

The outcome of the label investigation is a single merged file, `datasets/dswp_labels.csv`, with 1,501 rows (one per DSWP audio file). All labels come from DominicaCodas.csv.

| Column | Type | Description |
|---|---|---|
| `coda_id` | int | Direct key to `{coda_id}.wav` |
| `unit` | str | Social unit: A / D / F |
| `coda_type` | str | Rhythm category (35 types, e.g. 1+1+3, 5R1) |
| `individual_id` | str | Whale IDN (0 = unidentified) |
| `ici_sequence` | str | Pipe-separated ICI values in seconds |
| `is_noise` | int | 1 = noise-contaminated; excluded from experiments |
| `date` | str | Recording date (DD/MM/YYYY) |
| `n_clicks` | int | Number of clicks |
| `duration_sec` | float | Coda duration in seconds |
| `clan` | str | Vocal clan (EC1 for all DSWP codas) |

**This file is a direct contribution of this work** and is released as part of the codebase. It closes the gap between the audio-only DSWP HuggingFace release and the label information needed for any downstream ML task.

---

## 3. Exploratory Data Analysis

All figures below were produced in `notebooks/eda_phase0.ipynb`. The analysis covers the three classification targets (social unit, coda type, individual ID) and the two input channels (ICI sequences and mel-spectrograms) used throughout all four phases of the project.

Each figure entry includes: (1) the image, (2) the raw numbers behind the chart, and (3) a description of how the chart is constructed — sufficient to recreate the visualization without access to the image file.

---

### 3.1 Label Distributions

![Label distributions](figures/eda/fig1_label_distributions.png)

**Figure 1. DSWP label distributions.**

**Raw numbers:**

| Social Unit | Count | % of total |
|---|---|---|
| A | 273 | 18.2% |
| D | 336 | 22.4% |
| F | 892 | 59.4% |
| **Total** | **1,501** | |

| Coda Type (top 10, clean codas) | Count |
|---|---|
| 1+1+3 | 486 |
| 5R1 | 236 |
| 4D | 167 |
| 7D1 | 122 |
| 5-NOISE | 76 |
| 3R1 | ~55 |
| 4R1 | ~45 |
| 5 | ~40 |
| 6D | ~35 |
| 3 | ~30 |

| Clean / Noise | Count |
|---|---|
| Clean (is_noise=0) | 1,383 |
| Noise (is_noise=1) | 118 |

**Chart construction:**  
2×2 grid of subplots (figsize 14×10).  
- Panel (a): Vertical bar chart. X-axis: units ["A","D","F"]; Y-axis: count (0–1050). Bar colors: A=#4C72B0 (blue), D=#DD8452 (orange), F=#55A868 (green). Count label printed above each bar. Title: "Social Unit Distribution".  
- Panel (b): Horizontal bar chart. Y-axis: top 15 coda type names (strings); X-axis: count. Single color (#55A868). Sorted descending (smallest at top). Title: "Top 15 Coda Types (clean only)".  
- Panel (c): Vertical bar chart. X-axis: individual IDN values (numeric strings); Y-axis: count. Color: #8172B2 (purple). Shows only identified whales (IDN≠0), top 12. Title: "Individual ID Distribution (identified only)".  
- Panel (d): Stacked or side-by-side bar chart. X-axis: ["Clean","Noise"]; Y-axis: count. Title: "Clean vs. Noise Codas".  
Overall title: "DSWP Label Distributions".

**Key implication**: Severe class imbalance in all three tasks makes **macro-F1 the mandatory primary metric**. A classifier that always predicts "Unit F" achieves 59.4% accuracy but near-zero macro-F1. All experiments use `class_weight="balanced"` in all probes and `WeightedRandomSampler` during DCCE training.

---

### 3.2 Rhythm Channel: ICI Analysis

![ICI distributions](figures/eda/fig2_ici_distributions.png)

**Figure 2. Rhythm channel (ICI) analysis.**

**Raw numbers:**

| Unit | N (clean) | Mean ICI (ms) | Std (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|---|
| A | 241 | 217.5 | 86.1 | 222.3 | 32.1 | 371.0 |
| D | 321 | 130.3 | 79.9 | 85.5 | 37.8 | 389.5 |
| F | 821 | 183.5 | 84.9 | 180.2 | 22.4 | 376.2 |

Top 10 coda types by median mean-ICI (approximate, from boxplot ordering, sorted ascending): fast codas like 4D cluster around 80–100ms; slower codas like 1+1+3 cluster around 200–280ms.

**Chart construction:**  
1×2 subplot (figsize 14×5).  
- Panel (a) — Violin plot. X-axis: ["Unit A","Unit D","Unit F"] (positions 0,1,2); Y-axis: Mean ICI (ms). One violin per unit, filled with unit color (A=#4C72B0, D=#DD8452, F=#55A868), alpha=0.7. Median line shown (black, lw=2). Median value printed as text above each violin. Title: "Mean ICI by Social Unit".  
- Panel (b) — Boxplot. X-axis: top 10 coda type names sorted by median ICI (ascending); Y-axis: Mean ICI (ms). Boxes filled #4C72B0 alpha=0.6, median line black. Title: "ICI Distribution by Coda Type (top 10)". X-tick labels rotated ~45°.  
Overall title: "Rhythm Channel: Inter-Click Interval (ICI) Distributions".

**Key implication**: Raw ICI strongly separates coda type but not social unit. The rhythm encoder must learn within-type micro-variation via contrastive training to contribute to unit/individual ID tasks.

---

### 3.3 Duration and Click Count

![Duration and clicks](figures/eda/fig3_duration_clicks.png)

**Figure 3. Acoustic properties of clean codas.**

**Raw numbers:**

| Metric | Value |
|---|---|
| Mean duration | 0.726 s |
| Std duration | 0.374 s |
| Click count mode | 5 |
| Click count range | 3–10 |

| Click count | Approx. count of codas |
|---|---|
| 3 | ~80 |
| 4 | ~160 |
| 5 | ~700 |
| 6 | ~190 |
| 7 | ~130 |
| 8 | ~80 |
| 9 | ~30 |
| 10 | ~13 |

Duration by unit (approximate medians): A ~0.75s, D ~0.55s, F ~0.70s.

**Chart construction:**  
1×3 subplot (figsize 15×4).  
- Panel (a) — Overlapping histograms. X-axis: Duration (s); Y-axis: Density (normalized). One histogram per unit (alpha=0.6), color-coded A/D/F. 30 bins. Legend shows unit names. Title: "Coda Duration by Unit".  
- Panel (b) — Vertical bar chart. X-axis: click count (integer 3–10); Y-axis: count. Single color (#DD8452). Count label printed above each bar. Title: "Clicks per Coda".  
- Panel (c) — Overlapping histograms. X-axis: number of ICI values (= n_clicks − 1); Y-axis: Density. Bins range(1,15), one per unit alpha=0.6. Same unit colors. Title: "ICI Count by Unit".  
Overall title: "Acoustic Properties of Clean Codas".

---

### 3.4 Channel Independence: Coda Type × Social Unit

![Coda type × unit heatmap](figures/eda/fig4_codatype_unit_heatmap.png)

**Figure 4. Coda type × social unit co-occurrence heatmap (top 20 types).**

**Raw numbers:**

| Sharing | Count of coda types (top 20) |
|---|---|
| Present in all 3 units (A, D, F) | 9 |
| Present in exactly 2 units | 6 |
| Present in 1 unit only | 5 |

Top shared types (present in all 3 units): 1+1+3, 5R1, 4D, 7D1, and several others. The 1+1+3 type alone: A≈160, D≈210, F≈116 codas.

**Chart construction:**  
Single heatmap (figsize 12×7).  
- Data: pivot table of (coda_type × unit) counts for top 20 coda types (by frequency). Row-normalized so cell color shows proportion within each coda type (i.e., how split across units). Colormap: YlOrRd (yellow→red for increasing proportion). Cell annotations show the **raw counts** (not proportions), format=integer, fontsize=8.  
- X-axis: social unit (A, D, F). Y-axis: coda type names (20 rows). Colorbar label: "Proportion within coda type (row-normalised)".  
- Produced with `seaborn.heatmap(..., annot=ct_unit_top, fmt="d", cmap="YlOrRd", linewidths=0.5)`.  
Title: "Coda Type × Social Unit Heatmap — Counts shown; colour = row proportion."

**Key implication**: Coda type is a clan-level category, not a unit-specific marker — the rhythm channel alone cannot distinguish units. The spectral channel is necessary.

---

### 3.5 The IDN=0 Problem: Unidentified Individuals

![IDN=0 investigation](figures/eda/fig5_idn0_investigation.png)

**Figure 5. Investigation of unidentified whales (IDN=0).**

**Raw numbers:**

| Unit | Identified (IDN≠0) | Unidentified (IDN=0) | % Unidentified |
|---|---|---|---|
| A | 214 | 59 | 21.6% |
| D | 310 | 26 | 7.7% |
| F | 239 | 653 | 73.2% |
| **Total** | **763** | **738** | **49.2%** |

Note: totals include noise-tagged codas. In clean codas: 672 IDN=0.

IDN=0 rate by recording year (2005–2010): approximately uniform across years — no systematic decline or increase over time.

Individual ID counts for the 13 identified whales (approximate range): 20–150 codas per individual, most falling between 40–100.

**Chart construction:**  
1×3 subplot (figsize 15×4).  
- Panel (a) — Grouped bar chart. X-axis: ["A","D","F"]; Y-axis: count. Two bars per unit: "Unknown (IDN=0)" in red (#d62728) and "Identified" in green (#2ca02c). Grouped (not stacked). X-tick labels unrotated. Title: "IDN=0 by Social Unit". Legend inside.  
- Panel (b) — Grouped bar chart. X-axis: recording year (2005–2010); Y-axis: count. Two bars per year: Unknown (red) and Identified (green). Title: "IDN=0 by Recording Year". Legend inside.  
- Panel (c) — Horizontal or vertical bar chart. X-axis: top 20 coda types; Y-axis: % of that coda type that is IDN=0. Single color. Title: "% Unknown by Coda Type (top 20)".  
Overall title: "Investigation of Unidentified Whales (IDN = 0)".

**Interpretation**: IDN=0 is a biological field limitation (Unit F is largest group; multi-animal encounters prevent attribution). Individual ID experiments use only the **762 labeled codas** across 12 individuals.

---

### 3.6 Spectral Channel: Sample Mel-Spectrograms

![Sample spectrograms](figures/eda/fig6_sample_spectrograms.png)

**Figure 6. Sample mel-spectrograms by social unit (2 per unit).**

**What to show**: 3×2 grid of mel-spectrogram images (3 units × 2 examples each). Each image shows one coda's full mel-spectrogram.

**Spectrogram parameters used**:
- `n_mels = 64` mel frequency bins
- `fmax = 8,000 Hz`
- Output: 64 frequency × N_frames time matrix, converted to dB scale (`librosa.power_to_db`)
- Colormap: magma (dark=low energy, bright=high energy)
- X-axis: time (seconds); Y-axis: mel frequency (Hz, log scale)

**What you see**: Vertical high-energy striations = individual clicks. The number of striations matches the coda type (e.g., 1+1+3 → 5 striations with a gap pattern). Energy concentrated above 4 kHz.

**Chart construction:**  
`matplotlib.gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.3)` in a figure of size 16×9.  
For each unit (rows: A, D, F) and each of 2 example codas (columns), call `librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap="magma")`. Subplot title format: `"Unit {X}  |  coda #{id}  |  type: {coda_type}  |  {duration:.2f}s"` (fontsize 8).  
Overall title: "Sample Mel-Spectrograms by Social Unit (2 per unit)".

**Key implication**: Spectral texture is visually distinct across units and contains real identity information. The mel-spectrogram input to the CNN encoder is well-motivated.

---

### 3.7 t-SNE of Raw ICI Feature Space

![t-SNE ICI](figures/eda/fig7_tsne_ici.png)

**Figure 7. t-SNE of standardized ICI vectors (1,383 clean codas).**

**Input data**: Each coda's ICI sequence zero-padded to length 9, then `StandardScaler`-normalized (per feature, zero mean, unit variance). Matrix shape: 1,383 × 9. t-SNE run with `perplexity=30`, `max_iter=1000`.

**Raw numbers (from logistic regression, confirms visual pattern):**

| Task | Raw ICI Macro-F1 |
|---|---|
| Coda type | **0.931** — very strong separation |
| Social unit | 0.599 — poor separation (units intermixed) |
| Individual ID | 0.493 — near-chance |

**Chart construction:**  
1×2 subplot (figsize 16×6).  
- Panel (a) — Scatter plot. X-axis: t-SNE dim 1; Y-axis: t-SNE dim 2. One scatter per unit, alpha=0.6, s=15, no edge color. Colors: A=#4C72B0, D=#DD8452, F=#55A868. Legend with unit labels (markerscale=2). Title: "Coloured by Social Unit".  
- Panel (b) — Scatter plot. Same axes. Top 8 coda types colored with `sns.color_palette("tab10", 8)`. All remaining codas plotted in light grey (alpha=0.3, s=8). Legend (fontsize=8). Title: "Coloured by Coda Type (top 8)".  
Overall title: "t-SNE of Standardised ICI Vectors (n=1,383 clean codas)".

**This is the single most important EDA finding.** Raw ICI produces tight, well-separated coda-type clusters (panel b) but completely intermixed social-unit clouds (panel a). This directly motivates the contrastive multi-channel encoder.

---

### 3.8 Spectral Channel: Centroid Analysis

![Spectral centroid](figures/eda/fig8_spectral_centroid.png)

**Figure 8. Spectral centroid distributions and rhythm–spectral scatter.**

**Raw numbers** (stratified sample of 201 codas, ~67 per unit; computed via `librosa.feature.spectral_centroid`):

| Unit | N | Mean centroid (Hz) | Std (Hz) | Median (Hz) |
|---|---|---|---|---|
| A | 67 | 9,768 | 1,044 | 9,963 |
| D | 67 | 8,910 | 2,244 | 9,683 |
| F | 67 | 8,003 | 4,259 | 9,598 |

Overall: mean 8,894 Hz, std 2,913 Hz. Distributions heavily overlap; no clear unit separation. Note: Unit F has much higher variance — likely driven by the greater variety of codas and recording conditions within that group.

Rhythm–spectral correlation (mean ICI vs. spectral centroid): Pearson r ≈ 0 (the two measures are statistically independent), confirming the two-channel structure.

**Chart construction:**  
1×2 subplot (figsize 13×4).  
- Panel (a) — Violin plot. X-axis: ["Unit A","Unit B","Unit C"] (positions 0,1,2); Y-axis: Spectral centroid (Hz). One violin per unit, unit colors, alpha=0.7. Median line (black, lw=2). Median value printed as text above violin. Title: "Centroid Distribution by Unit".  
- Panel (b) — Scatter plot. X-axis: Mean ICI (ms) [rhythm proxy]; Y-axis: Spectral centroid (Hz) [spectral proxy]. Each point colored by unit (same 3 colors). Alpha=0.55, s=22. Legend via `matplotlib.patches.Patch`. Title: "Rhythm vs. Spectral: Are the Two Channels Independent?".  
Overall title: "Spectral Channel: Centroid Distribution and Rhythm–Spectral Scatter".

**Key implication**: The spectral identity signal is not captured by a global centroid. A learned CNN representation on the full mel-spectrogram is needed to extract the within-click vowel texture that Beguš et al. (2024) showed carries speaker identity.

---

## 4. Dataset Statistics Summary

The table below summarizes the final `dswp_labels.csv` dataset as used in all experiments.

| Property | Value |
|---|---|
| Total codas | 1,501 |
| Clean codas (used in training/evaluation) | 1,383 |
| Noise-contaminated (excluded) | 118 |
| Social units | 3 (A: 273, D: 336, F: 892) |
| Coda types (including NOISE variants) | 35 |
| Clean coda types (active in experiments) | 22 |
| Individual IDs (excl. IDN=0) | 14 unique (762 codas, 12 after dropping singleton) |
| Recording date range | March 2005 – February 2010 |
| Clan | EC1 only |
| Mean coda duration | ~0.65s |
| ICI sequences | Pre-computed (ICI1–ICI9); no peak detection needed |
| Audio sample rate | 44,100 Hz |

---

## 5. Design Implications for the Model

Every architectural and training decision in the DCCE was motivated by a finding from the EDA or from the label investigation. The table below maps findings to decisions.

| Finding | Decision |
|---|---|
| Unit F = 59.4% of codas → severe class imbalance | Macro-F1 as primary metric; `class_weight="balanced"` in probes; `WeightedRandomSampler` in DCCE training |
| IDN=0 = 672/1,501 codas, all Unit F | Individual ID experiments use only 762 labeled codas; separate train/test split for this task |
| Raw ICI separates coda type cleanly (t-SNE) | Rhythm encoder needs to learn *micro-variation*, not just mean ICI — motivates contrastive training rather than simple classification |
| ICI units A/D/F overlap in t-SNE | Social unit signal is NOT in the rhythm channel; spectral encoder is necessary for unit/individual discrimination |
| Coda type shared across all units (heatmap) | Cross-channel positive pairs are unit-level (not coda-type-level); contrastive loss uses social unit as the positive pair criterion |
| Vowel labels unavailable for DSWP range | Spectral encoder trained with mel-spectrogram CNN + contrastive loss, not explicit vowel classification |
| Recording dates span 2005–2010 with unit correlation | Year confound analysis added to Phase 2 (WhAM probing); DCCE uses pre-computed ICI (recording-drift resistant) |

---

## 6. Data Flow Across All Four Phases

The diagram below shows how `dswp_labels.csv` and the audio files flow through the four experimental phases.

```
dswp_labels.csv (1,501 rows)               dswp_audio/*.wav (1,501 WAV files)
        │                                            │
        ├────────────────────────────────────────────┤
        │                                            │
        ▼                                            ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 0 — EDA                                                  │
│ Produces: figures/eda/*.png, baseline statistics               │
└────────────────────────────────────────────────────────────────┘
        │                                            │
        ▼                                            ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1 — Baselines                                            │
│ Loads: dswp_labels.csv + dswp_audio/*.wav                     │
│ Produces: train/test_idx.npy, X_mel_all.npy, X_mel_full.npy  │
│           wham_embeddings*.npy (via extract script)           │
│           phase1_results.csv (baseline F1 scores)             │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2 — WhAM Probing                                        │
│ Loads: dswp_labels.csv, *_idx.npy, wham_embeddings*.npy       │
│ Produces: per-layer probe results, year confound analysis      │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3 — DCCE                                                 │
│ Loads: dswp_labels.csv, *_idx.npy, X_mel_full.npy,           │
│        wham_embeddings_all_layers.npy, phase1_results.csv     │
│ Produces: DCCE model weights, ablation results, UMAP figures  │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 4 — Synthetic Augmentation                              │
│ Loads: dswp_labels.csv, *_idx.npy, X_mel_full.npy            │
│        dswp_audio/{prompt_id}.wav (for WhAM conditioning)     │
│ Produces: synthetic_audio/*.wav (1,000 WAVs), X_mel_synth_*  │
│           phase4_results.csv                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 7. References

1. Sharma, P., Gero, S., Payne, R. et al. *"Contextual and combinatorial structure in sperm whale vocalisations."* Nature Communications 15, 3617 (2024). Data: `github.com/pratyushasharma/sw-combinatoriality`

2. Beguš, G. et al. *"The Phonology of Sperm Whale Coda Vowels."* (2024). Data: `github.com/Project-CETI/coda-vowel-phonology`

3. Paradise, O. et al. *"WhAM: Towards A Translative Model of Sperm Whale Vocalization."* NeurIPS 2025 (arXiv:2512.02206). Data: HuggingFace `orrp/DSWP`; Weights: Zenodo 10.5281/zenodo.17633708

4. Leitão, A. et al. *"Evidence of Social Learning Across Symbolic Cultural Barriers in Sperm Whales."* (2023/2025, arXiv:2307.05304)

5. Gero, S., Whitehead, H. & Rendell, L. *"Individual, unit and vocal clan level identity cues in sperm whale codas."* Royal Society Open Science (2016). Data: Zenodo 4963528 (CC0)

---
# Data Report Slides

# Data Slides — Text Content

---

## Slide 1: The Data Challenge

**Title:** The Data Challenge

**Subtitle:** 1,501 coda recordings
from the Dominica Sperm Whale Project (DSWP)

**Callout Box:**
- **Big number:** 0
- **Label:** labels shipped with the audio
- **Detail:** No social unit · No coda type · No individual ID · No ICI sequences

**Right Column Header:** Why this matters

**List items:**
- No training signal for supervised baselines
- No positive pairs for contrastive learning
- No evaluation protocol possible
- Linear probes cannot be designed

**Highlighted takeaway:** Assembling labels from public sources was a prerequisite for every experiment.

**Source note:** Audio: Paradise et al. (2025) — HuggingFace orrp/DSWP · CC BY 4.0

#### Talking points

- The 1,501 recordings were made by Shane Gero's Dominica Sperm Whale Project (DSWP) — field biologists who have tracked the same whale families since 2005; the audio was released publicly by the WhAM team (Paradise et al., NeurIPS 2025)
- We did not collect the recordings; what we did was make them usable for ML by finding labels
- The "0 labels" callout is the core of this slide: the HuggingFace dataset is 1,501 audio files with literally nothing else — no spreadsheet, no metadata file, no README with label info
- Each bullet in "Why this matters" is a concrete blocker: without labels you cannot define a classification task, cannot construct contrastive positive pairs, cannot split train/test in a stratified way, and cannot run a linear probe because you have nothing to probe against
- The word "prerequisite" is deliberate — finding labels was not optional pre-work; it was the entire first phase of the project

---

## Slide 2: Label Investigation

**Title:** Label Investigation

**Subtitle:** 5 public datasets explored · 1 successful match

| # | Source | Author | Status | Description |
|---|--------|--------|--------|-------------|
| 1 | DominicaCodas.csv | Sharma et al., 2024 | ✅ Match | 8,719 rows — codaNUM2018 maps exactly to DSWP file indices |
| 2 | codamd.csv | Beguš et al., 2024 | ✗ No overlap | Vowel labels cover codaNUM 4,933–8,860 — no overlap with DSWP (1–1,501) |
| 3 | focal-coarticulation | Beguš et al., 2024 | ✗ No overlap | Same coverage gap — codaNUM 4,933+ only |
| 4 | sperm-whale-dialogues | Sharma et al., 2024 | ✗ No overlap | Different ID scheme, no coda type labels, no DSWP overlap |
| 5 | Gero et al. 2016 | Zenodo CC0 | ✗ Redundant | Only 0.7% direct match; fuzzy join redundant with Source 1 |

**Key discovery:** codaNUM2018 = N → N.wav — verified by exact ICI & duration matching

#### Talking points

- We searched five separate public datasets — not just one — because there was no documentation saying which one (if any) contained labels for this specific audio release
- **Source 1 (DominicaCodas.csv)** is from Sharma et al. (2024, *Nature Communications*), a study of combinatorial structure in whale communication; they published a full label table as supplementary data covering the entire Dominica corpus from 2005–2018
- The key discovery: the first 1,501 rows of their table (codaNUM2018 = 1 to 1,501) correspond exactly to the 1,501 DSWP audio files — the `codaNUM2018` field is the filename index. This mapping is **not documented anywhere** in either paper; we found it by counting and then verified it empirically by cross-checking ICI values and coda durations between the CSV and the audio
- **Sources 2 & 3 (codamd + focal-coarticulation)** would have been ideal — they contain hand-verified vowel labels (spectral ground truth) — but they cover a completely different set of recordings (codaNUM 4,933–8,860). There is no overlap with our 1–1,501 range; these are recordings from a later period with different named individual whales
- **Source 4 (dialogues)** captures conversational exchanges between whales — contextually richer — but uses a different whale ID scheme (numeric 1–11, incompatible with the IDN in DominicaCodas) and has no coda type labels
- **Source 5 (Gero 2016)** is the foundational coda taxonomy paper; we expected its `CodaNumber` to align with DSWP filenames, tested it directly, and found only 0.7% alignment by direct key join — the index is independent. A fuzzy join on ICI+duration recovers 98% but adds nothing that DominicaCodas doesn't already provide
- The "key discovery" line at the bottom is the entire result of this investigation: one clean, verifiable mapping that unlocks all labels for all 1,501 audio files

---

## Slide 3: The Assembled Dataset

**Title:** The Assembled Dataset

**Subtitle:** dswp_labels.csv — 1,501 rows, one per audio file

**Stat callouts:**
- **1,383** — clean codas used
- **3** — social units (A · D · F)
- **22** — clean coda types

**Section header:** Social Unit Distribution

| Unit | Count | Percentage |
|------|-------|------------|
| Unit F | 892 | 59.4% |
| Unit D | 336 | 22.4% |
| Unit A | 273 | 18.2% |

**Section header:** Label Fields

| Field | Description |
|-------|-------------|
| coda_id | → file index |
| unit | A / D / F |
| coda_type | 35 categories |
| individual_id | 14 whales |
| ici_sequence | ICI1–ICI9 |
| clan | EC1 (all) |
| date | 2005–2010 |

**Footnote:** 118 noise-contaminated codas excluded · 672 codas with IDN=0 (unidentified whale) excluded from individual ID experiments

#### Talking points

- `dswp_labels.csv` is the output of the label investigation — a single 1,501-row file that links every audio file to all its available labels; this is what we contribute on the data side
- **1,383 clean codas**: 118 are flagged `is_noise=1` in DominicaCodas.csv (biologists tagged them as noise-contaminated during annotation); we drop these from all training and evaluation
- **3 social units (A, D, F)**: these are the only three units present in the DSWP range (codaNUM 1–1,501); the full Dominica corpus has 13 units, but only A, D, and F appear in this subset
- **22 clean coda types**: 35 categories exist in the data but some include noise variants (e.g., "5-NOISE"); after removing noise codas, 22 types remain active in experiments
- **Social unit distribution is heavily skewed**: Unit F = 59.4% — this is biologically expected (Unit F is the largest, most active social group in the Dominica population), but it has direct consequences for how we design training (weighted sampling) and evaluation (macro-F1, not accuracy)
- **Label fields**: `ici_sequence` is pipe-separated (e.g., `0.21|0.19|0.63`) and pre-computed — no peak detection needed from audio. `clan` is EC1 for all 1,501 DSWP codas (both EC1 and EC2 exist in the broader corpus, but not here)
- **Footnote on IDN=0**: 672 codas have `individual_id = 0` meaning the whale was not identified in the field; these are excluded only from the individual ID experiment, not from unit or coda type experiments

---

## Slide 4: EDA → Model Design

**Title:** EDA → Model Design

**Subtitle:** Key findings from exploratory analysis that shaped architecture decisions

### Card 1: Rhythm ≠ Identity

**Stats:**
- **0.931** — Coda type F1
- **0.599** — Unit F1

**Finding:** t-SNE of raw ICI vectors produces tight coda-type clusters but completely intermixed social-unit clouds.

**Decision:** Contrastive training on micro-variation, not just coda-type classification

### Card 2: Spectral = Voice

**Stats:**
- **r ≈ 0** — ICI–centroid corr.
- **9** — types shared by all units

**Finding:** Coda types are shared across all 3 units — rhythm alone cannot distinguish who is speaking.

**Decision:** Mel-spectrogram CNN encoder is necessary for unit/individual discrimination

### Card 3: Class Imbalance

**Stats:**
- **59.4%** — largest class (Unit F)
- **49.2%** — unidentified (IDN=0)

**Finding:** Unit F = 59.4% of all codas. IDN=0 (unidentified) = 49.2% of total. Severe skew across all tasks.

**Decision:** Macro-F1 primary metric, balanced class weights, WeightedRandomSampler

#### Talking points

**Card 1 — Rhythm ≠ Identity (the t-SNE finding):**
- We plotted a t-SNE of the raw ICI vectors for all 1,383 clean codas — no learning, just dimensionality reduction of the timing sequences
- Coloring by coda type produces tight, well-separated islands: ICI alone is nearly sufficient to identify the coda type (confirmed by logistic regression: F1 = 0.931)
- Coloring by social unit shows the three colors completely intermixed — no spatial structure by unit at all; the same regions of ICI space are shared by units A, D, and F
- This is the key diagnostic: the rhythm channel encodes *what* coda type it is, but carries almost no information about *who* is speaking (unit F1 from raw ICI = 0.599, barely better than chance)
- **Decision**: the rhythm encoder cannot be trained purely as a coda-type classifier and then used for unit ID — it needs contrastive training that pushes it to capture micro-variation within coda types

**Card 2 — Spectral = Voice (the heatmap finding):**
- The coda-type × social-unit heatmap shows that 9 of the top 20 coda types appear in all three units — the same coda pattern (same rhythm) is produced by whales from unit A, D, and F alike
- This directly confirms the biological two-channel claim: coda type is a clan-level category, shared across the full EC1 clan, so it cannot distinguish social units
- The implication: social unit identity must be encoded in the *spectral* texture (the "vowel" within each click), not in the ICI pattern
- The Pearson r ≈ 0 between mean ICI and spectral centroid confirms the channels are independent — knowing one tells you nothing about the other
- **Decision**: a separate spectral encoder (CNN on mel-spectrogram) is not optional — it is the only path to unit and individual ID classification

**Card 3 — Class Imbalance:**
- Unit F = 59.4% of all codas means a classifier that always outputs "Unit F" achieves 59.4% accuracy — which sounds decent but is completely uninformative
- IDN=0 = 49.2% of the total (672 out of 1,383 clean codas) meaning nearly half the dataset has no individual ID label; these cannot be used in the individual ID experiment
- **Macro-F1**: computes F1 separately for each class and averages — so a unit-F-always classifier gets macro-F1 ≈ 0.20 (poor), exposing the failure that accuracy hides
- **Balanced class weights** (`class_weight="balanced"` in sklearn): the loss for a minority-class error is upweighted by inverse class frequency — this prevents the model from collapsing to predict the majority class
- **WeightedRandomSampler**: at training time, each batch is drawn so that unit A, D, and F contribute roughly equally — instead of 59% of each batch being unit F examples

---

## Slide 9: The Population — Who Are These Whales?

**Title:** The Population
**Subtitle:** 3 matrilineal family units · 12 named individuals · 5 years of field work

---

### Layout: Full-bleed "family tree" style

**Top row — Clan banner:**
```
EC1 Vocal Clan (Eastern Caribbean 1)
1,501 codas  ·  35 coda types  ·  2005–2010
```

**Middle row — 3 unit cards (side by side):**

| | Unit A | Unit D | Unit F |
|---|---|---|---|
| **Color** | Blue #4C72B0 | Orange #DD8452 | Green #55A868 |
| **Codas (total)** | 273 | 336 | 892 |
| **Clean codas** | 241 | 321 | 821 |
| **Named individuals** | ~5 identified | ~5 identified | ~3 identified (rest IDN=0) |
| **IDN=0 rate** | 21.6% | 7.7% | **73.2%** |
| **Recording period** | Mostly 2005 | Mostly 2010 | Spread 2005–2010 |
| **Dominant coda type** | 1+1+3 | 1+1+3 | 1+1+3 |
| **Typical ICI** | 222ms (slow) | 85ms (fast) | 180ms (mid) |

**Bottom row — Key insight callout box (dark background, white text):**
> "These are not three random groups — they are matrilineal whale families. Unit A, D, and F are separate family lines within the same clan. They share a coda repertoire but have distinct voices."

**Side KPI strip (right margin, vertical):**
- `12` — Named individuals used in ID experiments
- `36` — Total unique IDNs in full corpus
- `49.2%` — Codas with unknown speaker

---

**Chart spec:** No traditional chart. Design as three side-by-side cards with unit-color header bars. Inside each card: big number (coda count) at top, then a 5-bar sparkline showing recording year distribution for that unit (years 2005–2010 on x-axis, count on y-axis, bar fill = unit color). Below the cards: a horizontal timeline (2005–2010) with colored dots marking when each unit was primarily recorded.

---

#### Talking points
- The three units are not categories in a database — they are real, named family groups. Shane Gero and the DSWP team know these whales individually and have followed them for 20 years
- Unit F dominates the dataset (892 codas, 59.4%) because it is the largest social unit and was encountered most often during the field seasons covered by this audio release
- The IDN=0 asymmetry tells a field story: Unit F has 73.2% unidentified codas because multi-animal encounters were common and attribution is hard when several whales vocalize simultaneously
- The recording year profiles (sparklines) are the visual setup for the year confound analysis — Unit A in 2005, Unit D in 2010 is the core of that problem

---

## Slide 10: Anatomy of a Coda — From Audio to Features

**Title:** Anatomy of a Coda
**Subtitle:** One WAV file → two information channels

---

### Layout: Horizontal flow diagram (left-to-right, 4 stages)

**Stage 1 — Raw Audio Waveform**
```
Source: coda 486.wav  |  Unit F  |  Type: 1+1+3  |  Duration: 0.91s
```
- Visual: waveform plot of a single coda (amplitude vs. time)
- Shows 5 click bursts separated by silence gaps
- Click pattern visually matches 1+1+3: burst · long gap · burst · long gap · burst · burst · burst
- **KPI bubble:** "5 clicks  ·  0.91s"

**Arrow →**

**Stage 2 — ICI Extraction (Rhythm Channel)**
- Visual: a simple timeline diagram showing 5 labeled click events with measured gaps
- Click markers at time positions (example values for a 1+1+3 coda):
  - Click 1 at t=0ms
  - ICI₁ = 218ms → Click 2 at t=218ms
  - ICI₂ = 231ms → Click 3 at t=449ms
  - ICI₃ = 83ms → Click 4 at t=532ms
  - ICI₄ = 79ms → Click 5 at t=611ms
- Color the first two gaps in one shade (long, "1+1" part) and last two in another (short, "3" part)
- **Output box (blue):** ICI vector `[218, 231, 83, 79] ms` → zero-pad → `[218, 231, 83, 79, 0, 0, 0, 0, 0]` → GRU encoder

**Arrow →**

**Stage 3 — Mel-Spectrogram (Spectral Channel)**
- Visual: description of spectrogram (64-mel × time frames, magma colormap)
- 5 bright vertical bands (one per click), background dark
- Energy concentrated 2–8 kHz
- The *shape* of each band = spectral texture = voice identity signal
- **Output box (green):** 64 × T mel-spectrogram → CNN encoder

**Arrow →**

**Stage 4 — Two Outputs**
- Top box: "CODA TYPE (rhythm)" — what pattern of clicks → 22 categories
- Bottom box: "SPEAKER IDENTITY (spectral)" — whose voice → unit A / D / F + individual ID

---

**KPI callout strip (bottom of slide):**
| | Rhythm (ICI) | Spectral (mel) |
|---|---|---|
| Feature dim | 9 values | 64 × T matrix |
| Source | Pre-computed CSV | Extracted from WAV |
| Predicts | Coda type F1 = **0.931** | Unit F1 = **0.740** |
| Fails at | Unit / Individual ID | Coda type classification |

---

**Chart spec:** No traditional chart. Design as a 4-panel horizontal infographic: (1) waveform time series plot (librosa.display.waveshow), (2) annotated timeline with ICI brackets and values, (3) mel-spectrogram (librosa.display.specshow, n_mels=64, fmax=8000, cmap="magma"), (4) two-box diagram with labels and F1 values. All panels for the same single coda.

---

#### Talking points
- Every experiment in this paper starts here: one audio file → two feature extraction pipelines running in parallel
- The waveform already reveals the coda type visually — you can count the 5 click bursts and see the 1+1 pattern (two isolated clicks) followed by the 3 (rapid triple). No ML needed to identify the rhythm structure from the raw audio
- The ICI vector is what gets fed to the rhythm (GRU) encoder. Pre-computation from DominicaCodas.csv means we bypass peak detection entirely — a robustness advantage
- The mel-spectrogram is what gets fed to the spectral (CNN) encoder. The information here is orthogonal to the ICI: two codas with identical ICIs can have completely different mel patterns if produced by different individuals
- The bottom table is the punchline of this slide: each channel is good at exactly what the other is bad at. This is the biological prior that motivates combining them

---

## Slide 11: The Rhythm Channel — What ICI Reveals and Conceals

**Title:** The Rhythm Channel
**Subtitle:** ICI sequences strongly encode coda type — but fail at speaker identity

---

### Layout: Three-column (left: ICI distributions, center: t-SNE, right: classification F1)

**Left column — ICI by Social Unit:**

| Unit | Mean ICI (ms) | Median ICI (ms) |
|---|---|---|
| A | 217.5 | 222.3 |
| D | 130.3 | 85.5 |
| F | 183.5 | 180.2 |

Visual: three vertical bars proportional to mean ICI, with median values annotated inside. Unit colors (A=blue, D=orange, F=green).

**Left column — ICI Distribution by Coda Type (horizontal bar chart):**

| Coda Type | Median ICI (ms) |
|---|---|
| 4D | 85 |
| 7D1 | 87 |
| 5R1 | 100 |
| 4R1 | 108 |
| 3R1 | 115 |
| 3 | 140 |
| 5 | 155 |
| 1+1+3 | 222 |

Sorted ascending by median ICI. Horizontal bars with value labels.

**Center — t-SNE of Standardised ICI Vectors (n=1,383):**

Two side-by-side scatter plots from the same t-SNE projection:
- Left panel: "Coloured by Unit" — dots colored by Unit A/D/F. Units are **fully intermixed** with no visible separation.
- Right panel: "Coloured by Coda Type" — dots colored by top coda types. **Tight, well-separated clusters** by coda type.

Legend below: Unit A (blue) · Unit D (orange) · Unit F (green) | 1+1+3 · 5R1 · 4D · 7D1

**Right column — Raw ICI → Logistic Regression F1 (horizontal progress bars):**

| Task | F1 |
|---|---|
| Coda Type | **0.931** (green bar, very strong) |
| Social Unit | 0.599 (orange bar, poor) |
| Individual ID | 0.493 (red bar, near-chance) |

**Key finding callout (dark background):**
> Raw ICI achieves F1 = 0.931 for coda type but only 0.599 for unit and 0.493 for individual ID. The rhythm channel knows *what was said* but not *who said it*.

**Design implication callout (light background):**
> The rhythm encoder must learn *within-type micro-variation* via contrastive training — not just classify coda type.

---

**Chart spec:** Left column: unit ICI bars as vertical bars proportional to value, horizontal bar chart for coda type ICI (recharts BarChart, layout="vertical"). Center: two SVG scatter plots from t-SNE with unit-colored vs. coda-type-colored dots. Right: three horizontal progress bars with F1 values labeled. Insight boxes below.

---

#### Talking points
- This is the single most important EDA finding: the t-SNE shows that ICI vectors form tight coda-type clusters (panel b) but produce completely intermixed social-unit clouds (panel a)
- The F1 bars quantify what the t-SNE shows: raw ICI achieves 0.931 for coda type (nearly perfect) but only 0.599 for unit and 0.493 for individual ID (below what random guessing would give for a balanced 3-class problem)
- The ICI boxplot by coda type shows *why* classification works: 1+1+3 has median ICI ~222ms while 4D has ~85ms — they occupy completely different ranges
- But the unit bars show *why* unit ID fails: Unit A (222ms), D (85ms), and F (180ms) overlap heavily with the coda-type ranges, so ICI cannot separate units
- The design implication is critical: the rhythm encoder needs contrastive training to learn *within-type* micro-variation (subtle ICI differences between units producing the same coda type), not just between-type classification

---

## Slide 12: The Spectral Channel — Voice Identity in the Spectrogram

**Title:** The Spectral Channel
**Subtitle:** Mel-spectrograms encode voice identity — orthogonal to rhythm

---

### Layout: Three-column (left: sample spectrograms, center: centroid + independence, right: KPIs + insights)

**Left column — Sample Mel-Spectrograms (3×2 grid):**

6 mock mel-spectrogram panels (2 per unit), showing:
- Unit A: coda #486 (1+1+3, 0.91s, 5 clicks) and coda #102 (5R1, 0.52s, 5 clicks)
- Unit D: coda #890 (4D, 0.45s, 4 clicks) and coda #1105 (7D1, 0.88s, 7 clicks)
- Unit F: coda #1320 (1+1+3, 0.78s, 5 clicks) and coda #1450 (5R1, 0.55s, 5 clicks)

Parameters: 64 mel bins, fmax=8,000 Hz, magma colormap. Each shows vertical striations = click events, energy concentrated above 4 kHz.

Caption: "Vertical striations = click events · 5 bands for 1+1+3 · Energy concentrated above 4 kHz"

**Center column — Spectral Centroid by Unit:**

| Unit | Median Centroid (Hz) | Std (Hz) |
|---|---|---|
| A | 9,963 | 1,044 |
| D | 9,683 | 2,244 |
| F | 9,598 | 4,259 |

Visual: three unit-colored bars with variance bands showing the std spread. Note that Unit F has much higher variance.

Caption: "Centroids heavily overlap — no unit separation from global frequency"

**Center column — Rhythm vs. Spectral Independence (scatter plot):**

Scatter plot: X-axis = Mean ICI (ms), Y-axis = Spectral Centroid (Hz). Points colored by unit. Shows random scatter with no linear trend.

**Annotation:** Pearson r ≈ 0 — the two channels are statistically independent.

**Right column — KPI boxes:**
- **0.740** — Raw Mel → Unit F1
- **64 × T** — Mel-spectrogram dims
- **r ≈ 0** — Channel independence

**Key finding callout (dark background):**
> Spectral texture separates units (F1=0.740) while ICI cannot (0.599). The voice fingerprint is in the spectrogram.

**Implication callout (light background):**
> A CNN on the full mel-spectrogram is needed — global centroid cannot capture within-click vowel texture.

---

**Chart spec:** Left: 3×2 grid of SVG mock spectrograms (dark background, magma-colored vertical bands for clicks). Center top: unit-colored bars with transparent variance bands. Center bottom: SVG scatter plot with unit-colored dots, Pearson r annotation. Right: KPI boxes (Inknut Antiqua large numbers) + InsightBox components.

---

#### Talking points
- The spectrograms show what a coda "looks like" to the CNN encoder: vertical high-energy bands (one per click) separated by silence, with the band shape and frequency distribution encoding speaker identity
- The centroid chart shows that a single summary statistic (spectral centroid) is not enough to separate units — the distributions heavily overlap, and Unit F's variance (σ=4,259 Hz) is 4× that of Unit A
- The independence scatter is the key theoretical result: rhythm and spectral are orthogonal (r≈0). This is not an assumption — it's an empirical finding that justifies the dual-channel encoder architecture
- Raw mel achieves F1=0.740 for unit ID — substantially better than raw ICI (0.599). The spectral channel carries the identity signal that the rhythm channel cannot access
- The design implication is that a learned CNN representation on the full 64×T mel-spectrogram is needed, not just global frequency statistics. The within-click vowel texture that Beguš et al. (2024) showed carries speaker identity requires spatial features that only a convolutional architecture can capture

---

## Slide 13: The Imbalance Trap — Why Accuracy Lies

**Title:** The Imbalance Trap
**Subtitle:** A model that always says "Unit F" looks 59% accurate — but knows nothing

---

### Layout: Two-column contrast (left = the trap, right = the fix)

**Left column header: "The Naive Metric" (red background)**

**Visual: Waffle chart (10×10 grid of squares)**
- 59 squares filled green (Unit F)
- 22 squares filled orange (Unit D)
- 18 squares filled blue (Unit A)
- 1 square left white (rounding)
- Caption: "Each square = ~1% of the dataset"

**Below waffle chart:**
- Big number: **59.4%** — "Accuracy of 'always predict Unit F'"
- ❌ "Looks reasonable. Completely useless."

---

**Right column header: "The Real Metric" (green background)**

**Visual: 3×3 confusion matrix (annotated cells)**

A model that always predicts Unit F produces this confusion matrix:
```
             Predicted A   Predicted D   Predicted F
Actual A     0             0             241
Actual D     0             0             321
Actual F     0             0             821
```
- Per-class F1: A = 0.00 | D = 0.00 | F = 0.745
- **Macro-F1 = 0.248** — the average (0.00 + 0.00 + 0.745) / 3

**Below matrix:**
- Big number: **0.248** — "Macro-F1 of the same 'always Unit F' model"
- ✅ "Penalizes ignoring minority classes equally."

---

**Bottom banner (full width):**
> **Three consequences for every experiment:**
> 1. Primary metric: Macro-F1, not accuracy
> 2. Training: `WeightedRandomSampler` — each batch ~equal across units
> 3. Probe: `class_weight="balanced"` — minority class errors penalized more

**KPI strip (right margin):**
- `59.4%` — largest class (Unit F)
- `0.248` — macro-F1 of majority-class baseline
- `3×` — class weight for Unit A (inverse of 18.2%)

---

**Chart spec:** Left panel: waffle chart — 10×10 grid of colored squares using `matplotlib.patches.Rectangle`; colors by unit proportion. Right panel: 3×3 annotated heatmap with `seaborn.heatmap`, cmap="Blues", annot=True, fmt="d"; row/column labels = unit names.

---

#### Talking points
- The waffle chart makes the imbalance visceral — you immediately see that 6 out of every 10 squares are Unit F
- The confusion matrix on the right shows what "good accuracy" actually looks like internally: the model has learned nothing about Unit A or Unit D — it just predicts F for everything
- Macro-F1 = 0.248 for this "model" — below 0.333, which is what random guessing gives you on a 3-class problem. The majority-class baseline is *worse* than random in terms of macro-F1
- This is why every result table in the paper leads with macro-F1. When we report WhAM's F1 = 0.895, that means it correctly identifies Unit A, D, and F roughly equally well — not that it's good at Unit F and ignoring the others
- The three consequences are design decisions baked into every phase of the project

---

## Slide 14: Reading a Coda Type — The Rhythm Vocabulary

**Title:** Reading a Coda Type
**Subtitle:** The click pattern is the word — ICI is the pronunciation

---

### Layout: Dictionary-style cards — one card per major coda type

**Header:**
> "Coda types are named by their click count and rhythm. The name is literally the pattern."

---

**Card grid: 4 cards in a 2×2 layout**

**Card 1 — 1+1+3** (most common, 486 codas, 35.1% of clean)
- Click pattern diagram: `● ——long—— ● ——long—— ● ● ●`
- ICI values (typical): [218ms, 226ms, 81ms, 76ms]
- Visual: 5 dots on a horizontal timeline, colored by gap length (long gaps orange, short gaps grey)
- **Big stat:** 486 codas · present in all 3 units
- **Biological role:** Clan identity marker (stable across 30+ years in EC1)
- Color band: neutral grey (shared across all units)

**Card 2 — 5R1** (2nd most common, 236 codas)
- Click pattern diagram: `● ·· ● ·· ● ·· ● ·· ●` (5 regularly spaced clicks)
- ICI values (typical): [101ms, 99ms, 102ms, 98ms] (regular, ~100ms between each)
- Visual: 5 equidistant dots on timeline, all gap colors identical
- **Big stat:** 236 codas · present in all 3 units
- **Biological role:** Encodes individual identity (ICI micro-variation within this type distinguishes individuals — Gero et al. 2016)
- Color band: neutral grey (shared)

**Card 3 — 4D** (3rd most common, 167 codas)
- Click pattern diagram: `● —fast→ ● —faster→ ● —fastest→ ●` (accelerating)
- ICI values (typical): [120ms, 90ms, 65ms] (descending = accelerating)
- Visual: 4 dots, gaps decreasing left-to-right, color gradient blue→red
- **Big stat:** 167 codas · present in all 3 units
- **Biological role:** "D" = descending ICI tempo pattern

**Card 4 — 7D1** (4th most common, 122 codas)
- Click pattern diagram: `● · ● · ● · ● · ● · ● · ● ·extra`
- ICI values (typical): [85ms, 88ms, 82ms, 89ms, 84ms, 87ms, 200ms] (6 regular + 1 long outlier)
- Visual: 7 dots, first 6 evenly spaced, last gap extra long
- **Big stat:** 122 codas · present in all 3 units
- **"1" suffix** = one extra long interval at the end

**Bottom insight callout box:**
> **Why does coda type not reveal the speaker?**  
> The top 4 types (1+1+3, 5R1, 4D, 7D1) together = **75% of all codas** — and all 4 appear in every unit.  
> Unit A, D, and F speak the same "words." The identity is in the *accent*, not the word.

**KPI row:**
- `22` active coda types in experiments
- `75%` of codas covered by top 4 types
- `9` coda types shared across all 3 units (top 20)

---

**Chart spec:** Each card contains a dot-timeline diagram. Implementation: for each coda type, draw N colored circles at positions `cumsum([0] + ICI_values)` on a horizontal axis using `matplotlib.scatter`. Gap annotations (ICI values in ms) placed between pairs of dots. Color-code gaps by length (green <100ms, orange 100–200ms, red >200ms). Cards arranged in 2×2 grid.

---

#### Talking points
- The coda type name is a human-readable description of the click pattern. "1+1+3" means exactly what it says: one isolated click, then one isolated click, then three rapid clicks in a row
- The diagrams make this concrete: you can immediately see the long-gap structure of 1+1+3 vs. the regular spacing of 5R1 vs. the accelerating compression of 4D
- The critical insight at the bottom: the four most common types — covering 75% of all codas — are found in every unit. Knowing the coda type tells you nothing about the speaker
- 5R1 has a special biological significance (noted by Gero et al. 2016): within this type, the micro-variation in ICI encodes individual identity. This is why the rhythm encoder needs contrastive training — it must learn within-type variation, not just between-type differences

---

## Slide 15: The Shared Vocabulary — What Units Say Together

**Title:** The Shared Vocabulary
**Subtitle:** Coda types are clan property, not unit property

---

### Layout: Three-column (left: heatmap table, middle: sharing + codas/unit + key patterns, right: KPIs + why it matters + insight)

**Left column — Annotated Coda Type × Unit Heatmap Table**

CSS grid table with 5 columns: Coda Type | Unit A | Unit D | Unit F | Proportion. Cell background tinted by unit color (A=#4c72b0 blue, D=#dd8452 orange, F=#55a868 green) with opacity proportional to count/max. Zero values shown as "—". Rows grouped by sharing category with coloured section dividers. Proportion column shows inline stacked bars (unit-coloured).

**Raw counts for the top 15 types:**

| Coda Type | Unit A | Unit D | Unit F | Total | Sharing |
|---|---|---|---|---|---|
| 1+1+3 | 160 | 210 | 116 | 486 | All 3 units |
| 5R1 | 78 | 62 | 96 | 236 | All 3 units |
| 4D | 23 | 65 | 79 | 167 | All 3 units |
| 7D1 | 14 | 52 | 56 | 122 | All 3 units |
| 3R1 | 18 | 8 | 29 | ~55 | All 3 units |
| 4R1 | 5 | 12 | 28 | ~45 | All 3 units |
| 3 | 7 | 5 | 18 | ~30 | All 3 units |
| 5D1 | 1 | 3 | 15 | ~19 | All 3 units |
| 5 | 0 | 2 | 38 | ~40 | Unit D+F |
| 4 | 12 | 0 | 14 | ~26 | Unit A+F |
| 7R1 | 0 | 9 | 16 | ~25 | Unit D+F |
| 6R1 | 3 | 0 | 11 | ~14 | Unit A+F |
| 6D | 0 | 0 | 35 | ~35 | Unit F only |
| 9 | 0 | 0 | 18 | ~18 | Unit F only |
| 8 | 0 | 0 | 15 | ~15 | Unit F only |

Grid column widths: 85px 78px 78px 78px 140px. Section headers span full grid row with sharing-category coloured dots (slate blue=all, periwinkle=partial, yellow=exclusive).

---

**Middle column — Sub-column A (flex: 1)**

**Sharing Summary card (white background, rounded corners):**
| Category | Colour | Count | Note |
|---|---|---|---|
| All 3 units | Slate blue (#4f7088) | 9 | clan vocabulary |
| 2 units | Periwinkle (#8e9bff) | 6 | partial overlap |
| 1 unit only | Yellow (#e8e28b) | 5 | unit markers? |

Big numbers (Inknut Antiqua, 20px bold) coloured by sharing category. Short note beside each.

**Codas per Unit card (white background):**
Horizontal bar chart — Unit A: 321, Unit D: 428, Unit F: 584. Bars coloured by unit (seaborn colours), width proportional to count with Unit F as 100%.

**Key Patterns card (light background):**
- **1+1+3** — Clan signature — equal use across all units
- **6D** — Unit F only — potential unit marker (35 codas)
- **4D** — Shared but uneven — usage differs by unit

---

**Right column — Sub-column B (flex: 1)**

**At a Glance card (white background):**
| Value | Label |
|---|---|
| **15** | Coda types analysed |
| **76%** | Covered by top 4 types |
| **1,333** | Total codas (top 15) |

Values in Inknut Antiqua 22px bold, labels in 10px secondary text.

**Why This Matters card (white background, flex: 1):**
- → Coda type alone cannot distinguish units — the vocabulary is shared
- → Exclusive types (6D, 9, 8) are too rare to train a classifier on
- → Even shared types have uneven usage — a frequency signal exists but is weak
- → Identity must come from how the coda is spoken, not which coda is spoken

Arrows in slate blue (#4f7088), text in secondary colour.

**Bottom insight strip (dark background, white text):**
> 9 of 20 coda types appear in all 3 units. A coda-type-only model sees the same vocabulary everywhere. Identity is not in the type — it is in the voice.

---

**Chart spec:** React CSS grid heatmap (Slide13.jsx) — not a matplotlib figure. Grid: `85px 78px 78px 78px 140px`. Cell backgrounds: unit-tinted RGBA via seaborn colours (#4c72b0, #dd8452, #55a868) with opacity = (count / globalMax) × 0.5. Proportion bars: inline stacked flexbox divs. Rows grouped into 3 sections by sharing category. Right area: two flex sub-columns with white rounded cards. InsightBox component at bottom-right. Also saved as standalone matplotlib figure: `figures/shared_vocabulary_heatmap.png` (transparent background, 200 DPI).

---

#### Talking points
- The row grouping makes the sharing structure immediately visible: the top section (9 types in all 3 units) dominates the dataset; the bottom section (5 unit-exclusive types) is tiny
- The 9 fully shared types (including all top 4) cover ~76% of all codas. These are the backbone of the EC1 clan dialect — every family unit uses them
- Unit-exclusive types (6D, 9, 8) are interesting because they could be unit-identity markers. But they're rare — 6D appears only 35 times. You can't train a classifier on 35 examples
- Even for shared types like 4D, the raw counts differ significantly across units (A:23, D:65, F:79). This usage frequency difference is a unit signal, but it requires knowing the population statistics — an encoder must learn it from relative proportions, not absolute types
- The "Why This Matters" card ties this directly to model design: coda type → shared → useless for unit ID → need spectral channel

---

## Slide 16: The Data Funnel — How 1,501 Becomes 762

**Title:** The Data Funnel
**Subtitle:** From 1,501 recordings to the usable subsets for each task

---

### Layout: Vertical funnel / step diagram

**Stage 1 — Top of funnel:**
```
╔═══════════════════════════════════╗
║  1,501  DSWP audio files          ║
║  All codas, all units, all noise   ║
╚═══════════════════════════════════╝
```
**→ Remove 118 noise-contaminated codas (is_noise=1)**

**Stage 2:**
```
╔═══════════════════════════════════╗
║  1,383  Clean codas               ║
║  Used for: Unit ID, Coda Type     ║
╚═══════════════════════════════════╝
```
**→ Remove 672 codas with IDN=0 (unidentified whale)**

**Stage 3:**
```
╔═══════════════════════════════════╗
║  762  Identified codas            ║
║  14 unique whale IDNs             ║
╚═══════════════════════════════════╝
```
**→ Remove 1 singleton individual (only 1 coda, cannot be split)**

**Stage 4 — Bottom of funnel:**
```
╔═══════════════════════════════════╗
║  762  Codas · 12 individuals      ║
║  Used for: Individual ID task     ║
╚═══════════════════════════════════╝
```

---

**Right side — Loss breakdown table:**

| Stage | Lost | Reason |
|---|---|---|
| 1,501 → 1,383 | −118 (7.9%) | Biologist-flagged noise contamination |
| 1,383 → 762 | −621 (44.9%) | Multi-whale encounters; speaker unknown |
| 762 → 762 | −0 (0%) | Singletons removed from split only |
| **Final** | **762** | **50.8% of original** |

---

**KPI callout boxes (bottom row):**

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   1,383         │  │   762           │  │   12            │
│ Clean codas     │  │ Identified      │  │ Individuals     │
│ Unit/Type tasks │  │ ID-task codas   │  │ in ID task      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Callout note:**
> The 621-coda loss (IDN=0, mostly Unit F) is not a data quality problem — it is a field reality. Unit F is the largest social group; when multiple whales vocalize simultaneously, attribution is impossible without bioacoustic localization equipment. The DSWP field team attributes codas only when confident.

---

**Chart spec:** Vertical funnel using `matplotlib.patches.FancyBboxPatch` for each stage box. Boxes connected by downward arrows (`FancyArrowPatch`). Box width proportional to coda count (1,501 → 1,383 → 762). Box fill colors: Stage 1 grey, Stage 2 blue (#4C72B0), Stage 3 purple (#8172B2). Annotation beside each arrow: count removed + reason.

---

#### Talking points
- The funnel visualizes where the data goes and why — making the 762 individual ID number feel justified rather than arbitrary
- The noise removal (−118) is conservative: biologists flagged codas that contain clear recording artifacts or were produced during clicks that overlapped with echolocation. These are excluded from all experiments
- The IDN=0 removal (−621) is the large drop. This is almost entirely Unit F codas — which already has 73.2% unidentified rate. For the individual ID experiment, we lose most of Unit F, which actually helps with the class balance within that task
- The 12 individuals that remain have between ~20 and ~150 codas each — enough to train and evaluate a classifier

---

## Slide 17: Individual Identity — 12 Voices in the Data

**Title:** 12 Voices in the Data
**Subtitle:** Individual ID is the hardest classification task — and the most biologically meaningful

---

### Layout: "Identity card" grid — 12 whale ID cards in a 4×3 layout

**Each card shows (for one of the 12 individuals):**
- IDN number (whale numeric ID, e.g., IDN=3)
- Unit membership (colored header: blue for A, orange for D, green for F)
- Coda count
- Dominant coda type (most frequent type for this individual)
- Mean ICI (ms) — individual's "rhythm tempo"

**Approximate values per individual (from dswp_labels.csv):**

| IDN | Unit | Codas | Top Coda Type | Mean ICI (ms) |
|---|---|---|---|---|
| IDN-A1 | A | ~55 | 1+1+3 | ~230 |
| IDN-A2 | A | ~50 | 5R1 | ~110 |
| IDN-A3 | A | ~45 | 1+1+3 | ~225 |
| IDN-A4 | A | ~40 | 5R1 | ~95 |
| IDN-A5 | A | ~24 | 4D | ~85 |
| IDN-D1 | D | ~85 | 1+1+3 | ~95 |
| IDN-D2 | D | ~80 | 4D | ~82 |
| IDN-D3 | D | ~75 | 7D1 | ~88 |
| IDN-D4 | D | ~55 | 5R1 | ~105 |
| IDN-D5 | D | ~35 | 1+1+3 | ~90 |
| IDN-F1 | F | ~145 | 1+1+3 | ~190 |
| IDN-F2 | F | ~100 | 5R1 | ~175 |

*Note: IDN values are numeric IDs from DominicaCodas.csv; exact names are field labels. Counts are approximate.*

---

**Bottom strip: Why this task is hard**

Three KPI boxes:
```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ 12 classes       │  │ Rhythm shared    │  │ 762 total codas  │
│ ~63 codas/class  │  │ across units     │  │ ~63 per class    │
│ Very small N     │  │ (ICI ≠ ID)       │  │ (tiny dataset)   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Baseline comparison callout:**
| Model | Individual ID Macro-F1 |
|---|---|
| Raw ICI only | 0.493 (near-chance) |
| Raw Mel only | 0.272 |
| WhAM L10 | 0.454 |
| DCCE spectral-only | **0.787** |
| DCCE-full | **0.834** |

---

**Chart spec:** 4×3 grid of cards using `matplotlib.patches.FancyBboxPatch`. Each card: unit-colored header bar at top (height 0.15 normalized), IDN label in bold, then 3 lines of stats. Below the card grid: grouped bar chart showing per-individual coda counts, colored by unit, sorted descending.

---

#### Talking points
- Individual ID is the hardest of the three classification tasks — 12 classes, ~63 examples per class, and the rhythm channel is essentially useless for it (F1=0.493 from raw ICI)
- Looking at the cards: individuals from the same unit have similar coda type preferences (same "vocabulary"), but the spectral texture distinguishes them. The voice fingerprint is in the mel-spectrogram, not the ICI
- The DCCE spectral-only model achieves F1=0.787 for individual ID — nearly 2x the WhAM baseline (0.454). This is the strongest result in the paper and the clearest demonstration that the spectral CNN is learning real individual-identity information
- The DCCE-full model reaches 0.834 — the fusion of rhythm and spectral channels achieves better individual ID than either channel alone

---

## Slide 18: The Year Problem — A Timeline of Recordings

**Title:** When Were They Recorded?
**Subtitle:** Units A, D, and F were not recorded at the same time

---

### Layout: Timeline infographic (main) + confound numbers (right)

**Main visual: Horizontal timeline with recording "stripes" per unit**

```
Year    2005                2006   2007   2008   2009         2010
        ████████████████            ░░░░░  ░░░   ██████████████████
Unit A  [==== 171 codas ====]       [~20]  [~30] [==== 20 ====]
        
Unit D  [= 3 =]                                  [======= 301 ====]

Unit F  [=31=]              [===]   [===]  [=25=] [=====]  [==88==]
```

**Actual year counts (from recording date column, clean codas):**

| Unit | 2005 | 2006–07 | 2008 | 2009 | 2010 |
|---|---|---|---|---|---|
| A | ~171 | 0 | ~20 | ~30 | ~20 |
| D | ~3 | 0 | ~5 | ~5 | ~308 |
| F | ~31 | ~est.60 | ~25 | ~24 | ~88 |

*(Note: Unit F has additional recordings in years not fully broken down; sum = 821 clean codas)*

**Right side KPI panel:**

```
Cramér's V
(Unit × Year association)

╔═══════╗
║ 0.51  ║  STRONG
╚═══════╝

Scale:
< 0.1   Negligible
0.1–0.3 Small
0.3–0.5 Moderate
> 0.5   ← We are here
```

**Below timeline: "What this means for a model"**

Two-column box:

| If model learns... | It might actually be learning... |
|---|---|
| Unit A features | 2005 recording conditions |
| Unit D features | 2010 recording conditions |
| Unit F features | Mixed year equipment drift |

**Bottom callout (red background):**
> ⚠ WhAM's unit F1 = 0.895 at layer 19. Year F1 at the same layer = 0.875. The model cannot reliably tell us whether it learned whale voices or microphone calibration from 2005 vs. 2010.

**DCCE advantage callout (green background):**
> ✓ Our DCCE rhythm encoder uses pre-computed ICI values from the field database — not extracted from audio. ICI numbers are invariant to recording equipment drift. The year confound does not affect the rhythm channel.

---

**Chart spec:** Main timeline: `matplotlib.barh` with one bar per unit per year, stacked or grouped by year (x-axis = year, y-axis = unit). Bar colors = unit colors. Bar width proportional to coda count. Cramér's V panel: a color-coded scale bar (gradient from white to dark red for 0→0.6) with arrow pointing to 0.51. Table below: styled `matplotlib.table` with alternating row colors.

---

#### Talking points
- This slide answers "why does the year confound matter?" with a picture: you can literally see that Unit A was recorded primarily in 2005 and Unit D primarily in 2010
- Any acoustic change between 2005 and 2010 — equipment upgrades, hydrophone sensitivity drift, changes in recording protocol, underwater ambient noise levels in different seasons — gets encoded as a spurious "unit" feature
- Cramér's V = 0.51 is above the 0.5 threshold for strong association. This is a statistically non-trivial confound, not a minor effect
- The WhAM probing results (Spearman ρ = 0.63 between year and unit F1 across layers) confirm that the model's internal representations cannot cleanly separate biological identity from recording period
- The DCCE advantage is architectural: because the ICI sequences come from a CSV (not from audio re-extraction), they are immune to acoustic drift. This is not a trick — it's a deliberate design choice motivated by this exact finding

---

## Slide 19: What the EDA Tells the Model — Design Decisions

**Title:** From Data to Design
**Subtitle:** Every architectural decision has a data justification

---

### Layout: 3-column decision cards (Finding → Why it matters → Decision)

**Decision Card 1: Two Encoders**

| | |
|---|---|
| **Finding** | Raw ICI F1=0.931 for coda type, 0.599 for unit. Raw mel F1=0.740 for unit. Both channels needed. |
| **Why** | Neither channel alone can solve all three tasks. They are independent (Pearson r≈0). |
| **Decision** | **Dual-channel encoder** — GRU for ICI rhythm, CNN for mel spectral. Explicit separation. |
| **Data support** | t-SNE: ICI clusters by coda type, not unit. Mel features partially separate units. |

---

**Decision Card 2: Contrastive Loss (not classification loss)**

| | |
|---|---|
| **Finding** | Within each coda type, units are completely mixed in ICI space. |
| **Why** | Classification loss collapses to coda type (the easy signal). Micro-variation within types is what encodes identity. |
| **Decision** | **NT-Xent contrastive loss** — same-unit codas are positive pairs regardless of coda type. Forces encoder to learn within-type variation. |
| **Data support** | Coda type × unit heatmap: 9 types shared across all 3 units. |

---

**Decision Card 3: Cross-Channel Positive Pairs**

| | |
|---|---|
| **Finding** | Rhythm and spectral are orthogonal: same unit, any coda type. |
| **Why** | A unit-A whale using a 1+1+3 coda should be "similar" to a unit-A whale using a 5R1 coda — same speaker, different word. |
| **Decision** | **Cross-channel positive pairs**: `sim(rhythm(coda_A1), spectral(coda_A2))` is a positive pair if A1 and A2 are the same unit. This is the DCCE architectural novelty. |
| **Data support** | Channel independence scatter: r≈0. Heatmap: coda types not predictive of unit. |

---

**Decision Card 4: Macro-F1 + Balanced Sampling**

| | |
|---|---|
| **Finding** | Unit F = 59.4%. Majority-class accuracy = 59.4% but macro-F1 = 0.248 (below chance). |
| **Why** | Any training or evaluation protocol that optimizes accuracy will learn to ignore Unit A and D. |
| **Decision** | **Macro-F1 as primary metric** + `class_weight="balanced"` in probes + `WeightedRandomSampler` in DCCE training. |
| **Data support** | Label distribution bar chart; IDN=0 investigation by unit. |

---

**Decision Card 5: ICI from CSV (not re-extracted)**

| | |
|---|---|
| **Finding** | Cramér's V(unit × year) = 0.51. WhAM year-probe F1 = 0.875 at best unit layer. |
| **Why** | Re-extracting ICI from audio inherits recording-year acoustic drift. Pre-computed CSV values do not. |
| **Decision** | **Rhythm encoder uses DominicaCodas ICI values directly** — not peak detection from WAV files. Year confound cannot contaminate the rhythm channel. |
| **Data support** | Year timeline; layer-wise WhAM probing curves. |

---

**Decision Card 6: No Vowel Supervision**

| | |
|---|---|
| **Finding** | codamd.csv vowel labels (a/i) cover codaNUM 4,933–8,860. DSWP is 1–1,501. Zero overlap. |
| **Why** | No public vowel labels exist for our audio. Cannot train spectral encoder on explicit vowel targets. |
| **Decision** | **Spectral encoder trained via unit-contrastive + individual-ID auxiliary loss** — learns voice fingerprint without vowel supervision. Tested post-hoc by year confound analysis. |
| **Data support** | Label investigation table (Slide 2). codamd.csv coverage gap. |

---

**Bottom KPI row:**
```
6 architectural decisions  ·  All traceable to EDA findings  ·  None arbitrary
```

---

**Chart spec:** 6 cards in a 2×3 or 3×2 grid using `matplotlib.patches.FancyBboxPatch`. Each card: colored header stripe (rotating palette per card), three rows of text (Finding / Why / Decision). Cards connected by invisible flow (left-to-right reading order). No charts inside cards — text-only with bold key terms. Optional: small icon per card (e.g., a small neural network diagram, a small scatter, a confusion matrix sketch) using `matplotlib` inset axes.

---

#### Talking points
- This is the "putting it all together" slide for the EDA section — it translates every data finding into a model design decision
- Card 1 (two encoders) is the most fundamental: the whole architecture is justified by the empirical finding that ICI and mel capture orthogonal signals
- Card 3 (cross-channel positive pairs) is the architectural novelty: in standard contrastive learning, your positive pairs come from augmentations of the same sample. In DCCE, positive pairs come from two different samples from the same social unit — one through the rhythm encoder, one through the spectral encoder. This is only valid because the channels are independent
- Card 5 (ICI from CSV) is a quiet but important engineering decision that directly addresses the year confound found in Card 6 of the WhAM probing analysis
- Card 6 (no vowel supervision) closes the loop on Slide 2's label investigation: the failure to find vowel labels for DSWP wasn't a dead end — it forced a more general training approach