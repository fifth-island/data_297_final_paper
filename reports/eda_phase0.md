# Phase 0 — Exploratory Data Analysis

## *Beyond WhAM*: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding

### CS 297 Final Paper · April 2026

---

This report constitutes Phase 0 of our research pipeline. Its purpose is to develop a thorough understanding of the Dominica Sperm Whale Project (DSWP) dataset before writing any model code. Every modelling decision in Phases 1–4 should be traceable back to an observation made here.

**Guiding question:**
*Do the two known information channels in sperm whale codas — rhythm (ICI timing) and spectral texture (vowel) — carry distinct, complementary signal that justifies building a dual-encoder architecture?*

---

## 1. Background and Motivation

### 1.1 What are codas?

Sperm whales (*Physeter macrocephalus*) communicate through rhythmically patterned click sequences called **codas** — short bursts of 3–40 clicks separated by precise inter-click intervals. Codas are social signals: groups of whales that share a coda repertoire form vocal **clans**, and membership in a matrilineal **social unit** can be partially inferred from acoustic style.

### 1.2 The two-channel hypothesis

Recent work has established that every coda encodes information along two syntactically independent dimensions:

| Channel | Feature | Encodes | Reference |
|---|---|---|---|
| **Rhythm** | Inter-click interval (ICI) sequence | *Coda type* — the categorical click-count/timing pattern shared within a clan | Leitão et al. (arXiv:2307.05304); Gero et al. (2016, *Royal Society Open Science*) |
| **Spectral** | Spectral shape (formant-like structure) within each click | *Individual/social-unit identity* — analogous to a voice fingerprint | Beguš et al. (*The Phonology of Sperm Whale Coda Vowels*, 2024) |

**Leitão et al. (2023–2025)** showed that *rhythmic micro-variations* within a given coda type track social-unit membership and, critically, that whales learn vocal style from neighbouring clans — providing the first quantitative evidence of cross-clan cultural transmission.

**Beguš et al. (2024)** formalised the spectral channel linguistically, showing that inter-pulse spectral variation within codas produces vowel-like formant patterns (labelled `a` and `i`) that correlate with individual identity independently of coda type.

### 1.3 The gap this paper fills

**WhAM** (Paradise et al., NeurIPS 2025, arXiv:2512.02206) is the current state of the art: a transformer masked-acoustic-token model fine-tuned from VampNet. It classifies social units, rhythm types, and vowel types as emergent byproducts of a generative objective — not by design. No published work has purpose-built a representation that explicitly exploits *both* channels simultaneously. This EDA is the first step toward filling that gap with the **Dual-Channel Contrastive Encoder (DCCE)**.

### 1.4 Dataset provenance

The DSWP HuggingFace release (`orrp/DSWP`) provides 1,501 raw WAV files with no labels. We recover ground-truth labels by joining against **DominicaCodas.csv** from Sharma et al. (2024, *Nature Communications*), which provides the same 1,501 codas annotated with social unit, coda type, individual ID, pre-computed ICI sequences, and recording date. The merged file is `datasets/dswp_labels.csv`.

---

## 2. Data Loading

We load `dswp_labels.csv` — our master label file constructed by joining the DSWP audio index against DominicaCodas.csv (Sharma et al. 2024). Each row corresponds to exactly one WAV file in `datasets/dswp_audio/`.

Key columns:
- `unit` — social unit (A / D / F), the primary classification target
- `coda_type` — rhythm type label (e.g. `1+1+3`, `5R1`), from Gero et al.'s classification scheme
- `individual_id` — numeric whale ID; `0` means unidentified in the field catalog
- `ici_sequence` — pipe-separated pre-computed inter-click intervals (seconds)
- `is_noise` — 1 if the coda was flagged as noise-contaminated

**Dataset summary:**
- Total codas: 1,501
- Clean (non-noise): 1,383
- ID-labeled (IDN≠0): 763 (13 unique individuals)
- Date range: 2005–2010

---

## 3. Label Distributions

Before training any model, we need to know the class structure of our three downstream classification tasks: social-unit ID, coda-type ID, and individual-whale ID.

The DSWP release covers social units A, D, and F — three of the nine Eastern Caribbean units studied by Gero et al. (2016). The overall population belongs to vocal clan EC1 (the Eastern Caribbean 1 clan).

![Label Distributions](../figures/eda/fig1_label_distributions.png)

### Observations

- **Severe class imbalance**: Unit F dominates with 892 codas (59.4% of total), versus 336 for D and 273 for A. This is biologically expected — Unit F is one of the largest and most active social groups. Consequence: we must use **stratified sampling** for train/test splits and **weighted cross-entropy loss** for classification heads.

- **Coda type imbalance**: The `1+1+3` pattern comprises 35.1% of clean codas. This is consistent with Gero et al. (2016), who found that `1+1+3` and `5R1` together account for ~65% of all codas. These two types serve as pan-clan "identity codas" — Hersh et al. (PNAS 2022) showed they function as symbolic cultural markers.

- **Recording coverage is temporally continuous** (2005–2010), ruling out obvious temporal confounds.

- **Individual ID coverage is sparse**: 672 codas (44.8%) have IDN=0. We restrict individual-ID experiments to the 763 labeled codas.

---

## 4. Rhythm Channel: Inter-Click Interval (ICI) Analysis

The rhythm channel is defined by the sequence of time intervals between consecutive clicks within a coda. It encodes **coda type** — the categorical click-count and timing pattern used since Watkins & Schevill (1977) to classify sperm whale communication.

**Leitão et al. (arXiv:2307.05304)** showed that subtle *micro-variations* in ICI values within a given coda type track social-unit membership and are culturally transmitted across clan boundaries. This means ICI sequences carry **two layers of information simultaneously**: coarse categorical coda type, and fine-grained individual/social-unit style.

![ICI Distributions](../figures/eda/fig2_ici_distributions.png)

### Observations

- **ICI distributions overlap substantially across units** (panel a). The Leitão et al. micro-variation signal is subtle — it lives *within* a coda type, not across types. A model that naively averages ICI will not recover it; the GRU encoder must process the full sequence.

- **ICI discriminates coda type very well** (panel b). The boxplots show clear separation between types: fast types like `5R1` have much shorter mean ICIs (~90ms) compared to slow types like `1+1+3` (~300ms). Confirms raw ICI is a powerful rhythm feature.

- **Wide ICI variance overall** (mean=177ms, std=88ms). StandardScaler normalisation will be necessary before feeding ICI sequences to the GRU encoder.

---

## 5. Acoustic Properties: Duration and Click Count

Duration and click count are the most basic acoustic properties of a coda and targets for WhAM probing experiments (Phase 2).

![Duration and Clicks](../figures/eda/fig3_duration_clicks.png)

### Observations

- **Duration is right-skewed and overlaps substantially across units**, peaking around 0.3–0.8s with a long tail to ~2.5s. Mean: 0.726s ± 0.374s.

- **5-click codas are dominant** (n=838, 60.6% of clean codas), followed by 7-click. Matches Gero et al. (2016) and Hersh et al. (2022).

- **Implication for the rhythm encoder**: variable-length input is unavoidable. Zero-padding to length 9 is a reasonable choice since the tail beyond 9 is sparse.

---

## 6. Channel Independence: Coda Type × Social Unit

**The central biological claim we are operationalising:** Beguš et al. (2024) established that the rhythm channel (coda type) and spectral channel (vowel) are *syntactically independent*. If the two channels truly carry independent information, we should observe that **coda types are shared across social units**.

![Coda Type × Unit Heatmap](../figures/eda/fig4_codatype_unit_heatmap.png)

### Observations

- **Most coda types appear in all three social units.** Of the top 20 types, the majority are produced by whales from units A, D, *and* F. This directly confirms the biological claim: coda type is a clan-level category, not a unit-specific marker. The two channels are genuinely independent.

- **Unit F contributes more counts** due to its larger size, but the *row-normalised* heatmap shows that the proportion of each type is fairly consistent across units.

- **Implication for DCCE**: The rhythm encoder must learn to disentangle coda type from social-unit identity. Cross-channel contrastive augmentation is designed precisely for this.

### Sharing statistics
- Coda types present in all 3 units: 9
- Coda types present in 2 units: 6
- Coda types present in 1 unit only: 5

---

## 7. The IDN=0 Problem: Unidentified Individuals

Individual whale identification in the DSWP is performed by photo-ID (fluke morphology) and acoustic size estimation during field sessions. `individual_id = 0` denotes a coda whose vocaliser was not identified.

![IDN=0 Investigation](../figures/eda/fig5_idn0_investigation.png)

### Observations

- **IDN=0 is almost entirely confined to Unit F** (panel a). Units A and D have near-complete individual identification. Makes sense biologically: Unit F is the largest group.

- **IDN=0 is evenly distributed across recording years** (panel b) — no improvement trend over time. A structural limitation of the recording methodology.

- **IDN=0 rates are consistent across coda types** (panel c) — no systematic bias.

- **Decision**: For individual-ID experiments, restrict to the 763 codas with known IDN (13 individuals). The social-unit contrastive loss is unaffected since unit labels are available for all 1,383 clean codas.

---

## 8. Spectral Channel: Sample Mel-Spectrograms

The spectral encoder operates on mel-spectrograms — 2D time-frequency representations. Before training, we visually confirm that spectrograms differ meaningfully across social units and coda types.

**Beguš et al. (2024)** showed that spectral variation within the inter-pulse intervals carries vowel-like formant structure at frequencies roughly 3–9 kHz.

![Sample Spectrograms](../figures/eda/fig6_sample_spectrograms.png)

### Observations

- **Click structure is clearly visible** as vertical high-energy striations matching the click count in the coda type label.

- **High-frequency energy dominates** (3,000–8,000 Hz range), consistent with Beguš et al. (2024). Our `fmax=8000 Hz` parameterisation captures the relevant spectral content.

- **Temporal structure varies across units** in subtle ways — this is the "vowel" variation the spectral encoder must capture.

- **Implication**: The CNN input should be normalised mel-spectrograms cropped or padded to 128 frames. Using `fmax=8000 Hz` and 128 mel bins is consistent with the literature.

---

## 9. t-SNE of Raw ICI Feature Space

Before building a learned rhythm encoder, we examine the raw ICI feature space — zero-padded ICI vectors standardised and projected via t-SNE.

![t-SNE of ICI Vectors](../figures/eda/fig7_tsne_ici.png)

### Observations

- **Coda types form very tight, well-separated clusters** (panel b). Even without any learned representation, the raw standardised ICI vector cleanly separates coda types in 2D. Confirms ICI is the primary determinant of coda type.

- **Social units do *not* separate cleanly** (panel a) — the three unit colours are largely intermixed within each coda-type cluster. This is precisely the challenge: social-unit identity is encoded as *micro-variations within* coda-type clusters (Leitão et al. 2023–2025: "style variation within type").

- **Implication for architecture**: The rhythm encoder's job is *not* to re-discover coda type. Its job is to capture the social-unit signal that exists *residually after* coda type is accounted for.

- **Implication for Baseline 1A**: A logistic regression on raw ICI vectors will likely achieve near-perfect coda-type classification but much weaker social-unit classification.

---

## 10. Spectral Channel: Centroid Analysis from Audio

Spectral centroids computed from raw WAV files verify that the spectral channel carries meaningful variance across social units — independent of the rhythm channel.

![Spectral Centroid Analysis](../figures/eda/fig8_spectral_centroid.png)

### Observations

- **Spectral centroid distributions overlap substantially across units** (panel a). The centroid is an imperfect proxy (Beguš et al.'s vowel signal lives in within-click inter-pulse intervals), but high variance (~8,894 ± 2,913 Hz) confirms significant spectral variation exists.

- **Rhythm and spectral channels are weakly correlated** (panel b, Pearson r ≈ 0). No systematic relationship between mean ICI and spectral centroid. This empirically confirms the biological independence claim of Beguš et al. (2024): knowing a coda's rhythm type does not predict its spectral texture.

- **Key architectural implication**: Because the two channels are independent, the fusion layer in DCCE should learn a *complementary* combination. The cross-channel contrastive augmentation enforces this complementarity during training.

---

## 11. EDA Summary and Implications for Modelling

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
1. **Baseline 1A** — Raw ICI (zero-padded, length 9) → logistic regression. Establishes the floor for the rhythm encoder.
2. **Baseline 1B** — WhAM embeddings (extracted from all 1,501 DSWP codas using the publicly available Zenodo weights) → linear probe. Primary comparison target for Experiment 1.
