# Implementation Plan — Beyond WhAM
## CS 297 Final Paper

**Last updated**: 2026-04-03
**Status**: Phase 0 complete — EDA done, plan updated with concrete design decisions

---

## 1. Project Overview

**Title**: Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding

**Core claim**: A model purpose-built around the known biological decomposition of sperm whale codas (rhythm channel + spectral channel) produces better representations than WhAM, which learned those features as an emergent byproduct of a generative music-audio objective.

**Proposed model**: Dual-Channel Contrastive Encoder (DCCE)

```
Coda waveform
    │
    ├── Rhythm Encoder (2-layer GRU on ICI sequence) ──► r_emb (64d)
    │                                                          │
    └── Spectral Encoder (CNN on mel-spectrogram) ──────► s_emb (64d)
                                                               │
                                        Fusion MLP ──► z (64d joint embedding)
```

**Training objective**:
```
L = L_contrastive(z) + λ1 · L_type(r_emb) + λ2 · L_id(s_emb)
```

Key novelty: cross-channel positive pairs — rhythm of coda A + spectral texture of a different coda from the **same unit** = positive pair.

> **EDA update**: Originally stated "same whale" but IDN=0 = 44.8% of data (all from Unit F). Positive pairs must be constructed at the **social unit level**, not individual level, to avoid excluding the majority of Unit F. The individual-level contrastive loss `L_id(s_emb)` remains restricted to the 763 IDN-labeled codas.

**Three experiments**:
1. Representation quality — DCCE vs. WhAM vs. single-channel ablations (linear probe)
2. Synthetic data augmentation — does adding WhAM-generated codas improve classification?
3. WhAM probing — interpretability analysis of WhAM's internal representations

**Compute target**: MacBook (Apple MPS / CPU). All experiments must run on a laptop.

---

## 2. Literature Foundation

| Paper | Role in Our Work | Code/Data Available? |
|---|---|---|
| Goldwasser et al., NeurIPS 2023 (arXiv:2211.11081) | Theoretical justification: UMT of animal comms is feasible | No code needed |
| Leitão et al., 2023–2025 (arXiv:2307.05304) | Motivates separating rhythm channel; ICI micro-variation encodes cultural identity | No public code |
| Gubnitsky et al., 2024 (arXiv:2407.17119) | Automated coda detector — provides preprocessing baseline | GitHub: Project-CETI/Coda-detector (Zenodo: 10.5281/zenodo.14902261) |
| Paradise et al. (WhAM), NeurIPS 2025 (arXiv:2512.02206) | **Primary baseline** — current SOTA | Weights: Zenodo 10.5281/zenodo.17633708 (3.1 GB); Code: github.com/Project-CETI/wham |
| Beguš et al., 2024 (coda-vowel-phonology) | Formalizes spectral/vowel channel — two syntactically independent channels | GitHub: Project-CETI/coda-vowel-phonology |
| Sharma et al., Nature Comms 2024 | Combinatorial structure; provides DominicaCodas.csv with labels | GitHub: pratyushasharma/sw-combinatoriality; Zenodo: 10.5281/zenodo.10817697 |
| Gero, Whitehead & Rendell, 2016 | Foundational coda classification — 9 Caribbean units, 21 types | Zenodo: 4963528 (CC0) |

---

## 3. Data Inventory

### 3.1 Audio Data

| Dataset | Source | Size | Contents | Status |
|---|---|---|---|---|
| DSWP | HuggingFace `orrp/DSWP` | ~585 MB | 1,501 WAV files (`1.wav`–`1501.wav`), unlabeled | Not yet downloaded — download before Phase 1 |

**Note**: DSWP is audio-only. No labels are included in the HuggingFace release. All labels come from the CSV sources below.

### 3.2 Label Files (downloaded to `datasets/`)

| File | Source | Rows | Key Columns | DSWP Coverage | Status |
|---|---|---|---|---|---|
| `DominicaCodas.csv` | Sharma et al. 2024, GitHub | 8,719 | codaNUM2018, Unit, CodaType, IDN, ICI1-9, Duration, Clan, Date | **1,501 exact** (codaNUM2018 1–1501) | Downloaded |
| `codamd.csv` | Beguš et al., Project-CETI GitHub | 1,375 | codanum, whale (named), codatype, Duration, handv | None (codanum 4933–8860) | Downloaded |
| `focal-coarticulation-metadata.csv` | Beguš et al., Project-CETI GitHub | 1,097 | codanum, whale, coart (aa/ai/ia/ii), pkfq, f1pk, f2pk | None | Downloaded |
| `sperm-whale-dialogues.csv` | Sharma et al. 2024, GitHub | 3,840 | REC, nClicks, ICI1-28, Whale, TsTo | None (different recording scheme) | Downloaded |
| `gero2016.xlsx` | Gero et al. 2016, Zenodo (CC0) | 4,116 | CodaNumber, CodaName, Unit, WhaleID, ICI1-9, Length, Date | 1,454 (via CodaNumber), 1,472 via fuzzy join | Downloaded |

### 3.3 Master Label File

| File | Description | Status |
|---|---|---|
| `dswp_labels.csv` | **Primary label file** — 1,501 rows, one per DSWP audio file | Produced |

#### dswp_labels.csv Schema

| Column | Type | Example | Notes |
|---|---|---|---|
| `coda_id` | int | 1 | Direct key to `{coda_id}.wav` |
| `audio_file` | str | `1.wav` | DSWP filename |
| `date` | str | `04/03/2005` | Recording date (DD/MM/YYYY) |
| `unit` | str | `A` | Social unit — **primary classification target** (A/D/F) |
| `unit_num` | int | 1 | Numeric unit ID |
| `clan` | str | `EC1` | Vocal clan (EC1/EC2) |
| `individual_id` | str | `5722` | Individual whale ID (IDN=0 means unidentified) |
| `coda_type` | str | `1+1+3` | Rhythm/coda type — 35 categories including NOISE variants |
| `is_noise` | int | 0 | 1 = noise-contaminated coda, filter for clean analysis |
| `n_clicks` | int | 5 | Number of clicks |
| `duration_sec` | float | 1.188 | Duration in seconds |
| `ici_sequence` | str | `0.293\|0.282\|0.298\|0.315` | Pre-computed ICIs, pipe-separated |
| `n_ici` | int | 4 | Number of ICI values |
| `handv` | str | *(empty)* | Vowel label (a/i) — not available for DSWP range |
| `whale_name` | str | *(empty)* | Named whale — not available for DSWP range |
| `f1pk_hz` | float | *(empty)* | Spectral formant — not available for DSWP range |

#### dswp_labels.csv Statistics

- **Total rows**: 1,501 (all DSWP audio files covered)
- **Clean codas** (is_noise=0): 1,383
- **Noise codas** (is_noise=1): 118
- **Social units**: A (273), D (336), F (892) — class imbalance, F dominates
- **Coda types**: 35 total; top 5: 1+1+3 (486), 5R1 (236), 4D (167), 7D1 (122), 5-NOISE (76)
- **Individual IDs**: 14 unique values; IDN=0 (unknown) = 672 codas
- **Clans**: EC1 only in DSWP range
- **Date range**: March 2005 – February 2010

### 3.4 Known Label Gaps

| Label | Available for DSWP range? | Workaround |
|---|---|---|
| Social unit (A/D/F) | **Yes** — from DominicaCodas.csv | None needed |
| Coda type (35 categories) | **Yes** — from DominicaCodas.csv | None needed |
| ICI sequence | **Yes** — pre-computed, all 1,501 rows | None needed |
| Individual whale ID | **Partial** — IDN=0 for 672/1,501 codas | Use only the 829 labeled codas for individual ID probe |
| Vowel (handv) | **No** — codamd.csv covers codaNUM 4933+ only | Email Beguš/WhAM team; or skip vowel label probe |
| Named whale | **No** — same coverage gap as handv | Same |
| Spectral formant (f1pk) | **No** | Compute from WAV using librosa as approximation |

### 3.5 WhAM Model Weights

| File | Size | Download URL |
|---|---|---|
| `coarse.pth` | 1.3 GB | Zenodo: 10.5281/zenodo.17633708 |
| `c2f.pth` | 1.1 GB | Same |
| `codec.pth` | 601 MB | Same |
| `wavebeat.pth` | 33 MB | Same |

**License**: CC-BY-NC-ND 4.0 — non-commercial, no derivatives. Fine for course project.
**Code**: `github.com/Project-CETI/wham`

---

## 4. Key Findings from Data Scouting

### F1 — DSWP audio is fully labeled via DominicaCodas.csv
The DSWP HuggingFace dataset ships as audio-only, but `codaNUM2018` in Sharma et al.'s DominicaCodas.csv maps exactly 1:1 to DSWP filenames (`codaNUM2018=N` → `N.wav`). This was verified by matching ICI sequences and durations. All 1,501 audio files are covered.

### F2 — Gero 2016 does NOT map by CodaNumber
Despite the claim that DSWP file indices map to Gero's CodaNumber, this is false. Only 10/1,454 shared CodaNumbers have matching ICI+Length values (0.7%). The CodaNumber index in Gero is an independent sequential ID, not the DSWP file index. A fuzzy join on (ICI1, Length) recovers 1,472/1,501 matches (98.1%), but this is redundant — DominicaCodas.csv already gives superior labels.

### F3 — Vowel labels (handv) are unavailable for the DSWP range
codamd.csv covers codaNUM 4,933–8,860. The DSWP release covers codaNUM 1–1,501. These ranges do not overlap. Vowel labels require either the Beguš team's data or approximation from audio.

### F4 — IDN=0 is a biological limitation, not a data gap
672 codas have individual_id=0 in both DominicaCodas and Gero 2016. These whales were genuinely unidentified in the field — no dataset resolves this. EDA confirmed IDN=0 is almost entirely confined to Unit F (the largest group), and is temporally distributed uniformly across years (not a data quality issue). Individual ID experiments use only the 763 labeled codas across 13 individuals.

### F5 — WhAM annotation CSV is not publicly released
The `allcodas.csv` file referenced in the WhAM codebase is not in the GitHub repo or Zenodo deposit. It contains the full DSWP annotation catalog used in the WhAM paper. An email to the WhAM team (Orr Paradise) has been drafted requesting access.

### F6 — ICI features are pre-computed
DominicaCodas.csv includes pre-computed ICI1–ICI9 for all codas. The rhythm encoder does not require any peak detection preprocessing — the features are ready to use directly.

---

## 5. Implementation Plan

### Phase 0 — Exploratory Data Analysis
**Goal**: Deeply understand the data before writing any model code. Produce figures for the paper.
**Dependencies**: dswp_labels.csv + DSWP audio (download first)
**Output**: EDA notebook + key figures

Tasks:
- [x] Download DSWP audio from HuggingFace (`orrp/DSWP`) — 1,501 WAV files
- [x] Label distribution analysis: unit, coda type, individual ID, is_noise — fig1
- [x] Class imbalance visualization (unit F = 59.4%) — fig1
- [x] ICI distribution per coda type (violin + boxplot) — fig2
- [x] Mel-spectrogram grids by social unit — fig6
- [x] Duration distribution by coda type — fig3
- [x] t-SNE of raw ICI vectors, colored by unit and coda type — fig7
- [x] Coda type × unit heatmap (independence check) — fig4
- [x] Investigate IDN=0 codas — all from unit F, temporally distributed — fig5
- [x] Spectral centroid distribution — fig8

---

### Phase 1 — Baselines
**Goal**: Establish comparison points before building DCCE. Replicate WhAM's classification results on our exact data and split.
**Dependencies**: DSWP audio, dswp_labels.csv, WhAM weights (3.1 GB download)

#### Baseline 1A — Raw ICI logistic regression
- Zero-pad ICI sequences to length 9
- Train logistic regression (sklearn) predicting: unit, coda_type, individual_id
- Evaluate: top-1 accuracy, **macro-F1** (primary metric — see EDA note on imbalance)
- **EDA-updated expectation**: coda type will be very strong (t-SNE showed tight clusters); social unit will be weak (units are intermixed in ICI space). This gap is what the full DCCE must close.

#### Baseline 1B — WhAM embedding linear probe
- Install WhAM (`github.com/Project-CETI/wham`)
- Extract embeddings for all 1,501 DSWP codas using `wham/embedding/generate_embeddings.py`
- Run same logistic regression probes: unit, coda_type, individual_id
- This replicates the WhAM paper's downstream evaluation on our exact split
- **This is the primary comparison target for Experiment 1**

**Split**: 80/20 train/test, **stratified by unit** (essential — Unit F = 59.4%, naive split would distort results). Fix random seed=42. Use same split for all experiments.

**Metric**: **Macro-F1 is the primary metric** for all classification tasks. Top-1 accuracy is reported as secondary. Rationale: class imbalance is severe (unit F, coda type 1+1+3) — a model that predicts the majority class achieves high accuracy but low macro-F1.

#### Baseline 1C — Mel-spectrogram logistic regression (NEW — added post-EDA)
- Compute a fixed-size summary of the mel-spectrogram for each coda (mean pooled across time)
- Train logistic regression predicting: unit, coda_type
- This establishes the floor for the spectral encoder, analogous to what 1A does for the rhythm encoder
- Cost: minimal (no neural network, uses pre-computed spectrograms)

---

### Phase 2 — Experiment 3: WhAM Probing
**Goal**: First interpretability analysis of WhAM. Understand what its internal layers encode. Produces a publishable result independent of DCCE.
**Dependencies**: WhAM embeddings (from Phase 1), dswp_labels.csv

Tasks:
- [ ] Extract per-layer intermediate embeddings from WhAM (VampNet coarse + c2f)
- [ ] Train linear probes per layer predicting:
  - `n_clicks` (click count) — tests rhythm length encoding
  - mean ICI (computed from ici_sequence) — tests tempo encoding
  - `unit` — tests social identity encoding
  - `coda_type` — tests categorical rhythm encoding
  - `is_noise` — tests audio quality sensitivity
  - spectral centroid (pre-computed in EDA for a 200-coda sample; compute for all here) — tests spectral encoding
- [ ] Plot probe accuracy vs. layer depth (probing profile)
- [ ] UMAP of final embedding space colored by: unit, coda_type, individual_id, year
- [ ] Test recording year as a confound (EDA showed consistent coverage across years — but WhAM may still encode recording drift)

**EDA-updated expected finding**: Raw ICI already cleanly separates coda type even without any neural encoding (t-SNE). If WhAM's early layers merely replicate this, that is evidence it has not learned anything beyond the trivial ICI signal. The interesting result would be if later layers show *better* social-unit separation than the raw ICI baseline — that would mean WhAM is capturing micro-variation that raw ICI misses.

**New probe to add** (motivated by EDA): recording year probe. EDA found no obvious temporal confound in the labels, but WhAM was trained on recordings from multiple years — its embeddings may cluster by recording date in ways that confound social unit. This probe directly tests the confound that was not tested in the original WhAM paper.

---

### Phase 3 — Experiment 1: DCCE
**Goal**: Build and evaluate the Dual-Channel Contrastive Encoder. Compare against baselines from Phase 1.
**Dependencies**: DSWP audio, dswp_labels.csv, Phases 0 and 1 complete

#### Architecture

```python
# Rhythm Encoder: 2-layer GRU on ICI sequence
# Input: zero-padded ICI vector (length 9)
# Output: r_emb (64d)

# Spectral Encoder: small CNN on mel-spectrogram
# Input: 128 mel bins × 128 time frames
# Output: s_emb (64d)

# Fusion MLP: concat → LayerNorm → Linear → LayerNorm → z (64d)
```

#### Training

- Contrastive loss (NT-Xent / SimCLR): same-unit pairs = positive, different-unit = negative
- Cross-channel augmentation: rhythm(coda_A) + spectral(coda_B from same **unit**) = positive
- Auxiliary head on r_emb: coda_type classification
- Auxiliary head on s_emb: individual_id contrastive loss (763 IDN-labeled codas only — corrected from 829 after EDA recount)
- Hyperparameters: λ1=λ2=0.5 (default), batch size 64, 100 epochs
- **Weighted sampling**: sample batches with equal unit representation to counter the F=59.4% imbalance
- **ICI normalisation**: StandardScaler on ICI matrix before GRU input (EDA showed mean=177ms, std=88ms — wide range requires normalisation)
- **Mel-spectrogram parameters** (confirmed by EDA): 128 mel bins, fmax=8000 Hz, fixed 128-frame time window with zero-padding

#### Ablations (all must be run for the comparison to be valid)

| Model | Description |
|---|---|
| DCCE-rhythm-only | z = r_emb only, no spectral encoder |
| DCCE-spectral-only | z = s_emb only, no rhythm encoder |
| DCCE-late-fusion | Both encoders but no cross-channel augmentation |
| DCCE-full | Full model with cross-channel augmentation |

#### Evaluation (linear probe on frozen embeddings)

| Task | Labels | n | Metric |
|---|---|---|---|
| Social unit classification | unit (A/D/F) | 1,383 clean | **Macro-F1** (primary), top-1 accuracy |
| Coda type classification | coda_type (22 clean types) | 1,383 clean | **Macro-F1** (primary), top-1 accuracy |
| Individual ID classification | individual_id | 763 IDN-labeled | **Macro-F1** (primary), top-1 accuracy |

**EDA-updated primary hypothesis**: 
- DCCE-rhythm-only will already be strong on coda type (raw ICI is discriminative per t-SNE)
- DCCE-spectral-only and DCCE-full will be stronger than rhythm-only on social unit (the social signal lives in the spectral/style channel, not the type channel)
- DCCE-full should outperform both single-channel ablations on social unit (complementary channels)
- WhAM may beat DCCE on coda type (larger training data) but DCCE-full should match or beat WhAM on social unit (purpose-built vs. emergent)

---

### Phase 4 — Experiment 2: Synthetic Data Augmentation
**Goal**: First controlled augmentation study for sperm whale bioacoustics.
**Dependencies**: WhAM generation working on local hardware, dswp_labels.csv

Tasks:
- [ ] Use WhAM to generate N_synth ∈ {0, 100, 500, 1000, 2000} synthetic codas conditioned on real DSWP codas (one prompt per unit)
- [ ] Assign pseudo-labels to synthetic codas based on prompt's unit label
- [ ] Train DCCE (or simple CNN classifier) on D_train ∪ D_synth
- [ ] Evaluate on real-only D_test
- [ ] Plot accuracy vs. N_synth

**Caveat**: WhAM generation requires ~2 GB GPU memory. May need to run on coarse model only. If generation is too slow on Apple MPS, this experiment may be dropped or reduced in scope.

---

## 6. Validation Protocol

All experiments must be validated against biological ground truth:

| Claim | Validation | Source |
|---|---|---|
| Rhythm channel encodes coda type | r_emb probe accuracy on coda_type | dswp_labels.csv |
| Spectral channel encodes social identity | s_emb probe accuracy on unit | dswp_labels.csv |
| Joint embedding improves over either alone | Linear probe comparison across ablations | — |
| DCCE matches/beats WhAM on identity tasks | Comparison with Phase 1 Baseline 1B | WhAM embeddings |
| WhAM late layers encode social structure | Layer-wise probe profile from Experiment 3 | dswp_labels.csv |

---

## 7. Open Items / Blockers

| Item | Status | Action needed |
|---|---|---|
| DSWP audio download | **Done** | 1,501 WAV files in `datasets/dswp_audio/` |
| WhAM weights download | Not started | Download 3.1 GB from Zenodo before Phase 1 |
| handv vowel labels for DSWP range | Missing | Email sent to WhAM/Beguš team — awaiting reply |
| WhAM `allcodas.csv` | Missing | Same email — WhAM team has this file |
| WhAM generation feasibility on MPS | Unknown | Test during Phase 4 setup |
| IDN=0 for 672 codas | Biological limitation — confirmed by EDA | Use only 763 IDN-labeled codas for individual ID experiments |
| Class imbalance handling | **Design decision made** | Stratified splits + weighted CE loss (Unit F = 59.4%) |
| Metric selection | **Design decision made** | Macro-F1 is primary metric for all tasks |
| Mel-spectrogram parameters | **Confirmed by EDA** | 128 mel bins, fmax=8000 Hz, 128 time frames |
| ICI normalisation | **Confirmed by EDA** | StandardScaler required (range 90–300ms+) |

---

## 8. File Directory

```
data_297_final_paper/
├── implementation_plan.md          ← This file (living document)
├── research_paper.md               ← Full paper proposal
├── team_update.md                  ← Initial team communication
├── eda.py                          ← EDA script (standalone)
├── eda_phase0.ipynb                ← EDA notebook (fully executed, with outputs)
├── build_eda_notebook.py           ← Script that generated the notebook
├── datasets/
│   ├── dataset_report.md           ← Source analysis report
│   ├── dswp_labels.csv             ← PRIMARY LABEL FILE (1,501 rows)
│   ├── dswp_audio/                 ← 1,501 WAV files (1.wav – 1501.wav)
│   ├── DominicaCodas.csv           ← Sharma et al. 2024 (8,719 rows)
│   ├── codamd.csv                  ← Beguš et al. (1,375 rows, vowel labels)
│   ├── focal-coarticulation-metadata.csv  ← Beguš et al. (spectral peaks)
│   ├── sperm-whale-dialogues.csv   ← Sharma et al. (3,840 rows, dialogues)
│   └── gero2016.xlsx               ← Gero et al. 2016 (4,116 rows, CC0)
└── figures/
    └── eda/
        ├── fig1_label_distributions.png
        ├── fig2_ici_distributions.png
        ├── fig3_duration_clicks.png
        ├── fig4_codatype_unit_heatmap.png
        ├── fig5_idn0_investigation.png
        ├── fig6_sample_spectrograms.png
        ├── fig7_tsne_ici.png
        └── fig8_spectral_centroid.png
```

---

## 9. Results Tracker

*To be filled in as experiments are completed.*

### Phase 0 — EDA
| Finding | Value | Notes |
|---|---|---|
| Total / clean codas | 1,501 / 1,383 | 118 noise-tagged (7.9%) |
| Social unit distribution | A=273, D=336, F=892 | F = 59.4% — severe imbalance, must weight loss or stratify |
| Unique coda types (clean) | 22 | Top type: 1+1+3 = 486 codas (35.1%) |
| IDN=0 (unidentified) | 672 (44.8%) | All from unit F; individual ID experiments use only 763 codas, 13 individuals |
| Mean coda duration | 0.726s (std=0.374s) | Range roughly 0.1–2.5s |
| Mean ICI | 177.1ms (std=88.6ms) | Wide spread — rhythm channel is informative |
| Spectral centroid (sample) | 8,894Hz (std=2,913Hz) | High variance — spectral channel is informative |
| t-SNE of raw ICI vectors | Units partially cluster; coda types cluster strongly | Raw ICI already separates coda types well — validates rhythm encoder utility |
| Coda type × unit heatmap | Most coda types are shared across all 3 units | Coda type ≠ social unit; the channels are genuinely independent |

### Phase 1 — Baselines
| Model | Unit Macro-F1 | CodaType Macro-F1 | IndivID Macro-F1 | Notes |
|---|---|---|---|---|
| Raw ICI → LogReg (1A) | — | — | — | — |
| Mel-spectrogram → LogReg (1C) | — | — | — | — |
| WhAM → LogReg (1B) | — | — | — | — |

### Phase 2 — WhAM Probing
| Probe Target | Best Layer | Macro-F1 / R² | Notes |
|---|---|---|---|
| n_clicks | — | — | — |
| mean ICI | — | — | — |
| unit | — | — | — |
| coda_type | — | — | — |
| spectral centroid | — | — | — |
| recording year | — | — | New probe added post-EDA |

### Phase 3 — DCCE Experiment 1
| Model | Unit Macro-F1 | CodaType Macro-F1 | IndivID Macro-F1 | Notes |
|---|---|---|---|---|
| DCCE-rhythm-only | — | — | — | — |
| DCCE-spectral-only | — | — | — | — |
| DCCE-late-fusion | — | — | — | — |
| DCCE-full | — | — | — | — |

### Phase 4 — Synthetic Augmentation
| N_synth | Unit Acc | CodaType Acc | Notes |
|---|---|---|---|
| 0 (baseline) | — | — | — |
| 100 | — | — | — |
| 500 | — | — | — |
| 1000 | — | — | — |
| 2000 | — | — | — |

---

## 10. Change Log

| Date | Change |
|---|---|
| 2026-04-03 | Initial plan created. All datasets downloaded and analyzed. dswp_labels.csv produced. |
| 2026-04-03 | Phase 0 complete. DSWP audio downloaded. 8 EDA figures produced. eda_phase0.ipynb fully executed. |
| 2026-04-03 | Plan updated post-EDA: (1) positive pairs changed from same-whale to same-unit level; (2) IDN count corrected to 763; (3) macro-F1 adopted as primary metric; (4) stratified splits + weighted CE loss added; (5) mel-spectrogram params confirmed; (6) ICI normalisation confirmed; (7) Baseline 1C (mel-spec logReg) added; (8) recording-year probe added to Phase 2; (9) hypotheses sharpened based on t-SNE and channel independence findings. |
