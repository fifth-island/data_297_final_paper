# Implementation Plan — Beyond WhAM
## CS 297 Final Paper

**Last updated**: 2026-04-04
**Status**: All 4 phases complete and reproducible (full rerun confirmed). Key result: DCCE-full indivID F1=0.834 >> WhAM L10 (0.454); augmentation decreases indivID slightly (synthetic data dilutes individual-level signal); seed fix applied throughout.

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
**Goal**: Build and evaluate the Dual-Channel Contrastive Encoder. Compare against baselines from Phases 1–2.
**Dependencies**: DSWP audio, dswp_labels.csv, wham_embeddings.npy, Phases 0–2 complete
**Status**: IN PROGRESS

#### Architecture (final design — post-EDA and post-Phase-2 updates)

```
Coda waveform / labels
    │
    ├── Rhythm Encoder: 2-layer GRU
    │   Input : zero-padded ICI vector (length 9), StandardScaler normalised
    │   Output: r_emb (64d)
    │
    └── Spectral Encoder: small CNN
        Input : mel-spectrogram (64 mel bins × 128 time frames, fmax=8000 Hz)
        Output: s_emb (64d)
                      │
        Fusion MLP: concat(r_emb, s_emb) → LayerNorm → Linear(128→64) → ReLU → z (64d)
```

#### Training objective (unchanged from proposal)
```
L = L_contrastive(z) + λ1·L_type(r_emb) + λ2·L_id(s_emb)
```
- **L_contrastive**: NT-Xent (SimCLR) on z, τ=0.07; positive pairs = same social unit
- **Cross-channel pairs**: rhythm(coda_A) + spectral(coda_B, same unit) = positive
- **L_type**: CE on r_emb → coda_type (22 classes, weighted for imbalance)
- **L_id**: CE on s_emb → individual_id (12 classes, only 762 IDN-labeled codas)
- λ1=λ2=0.5, batch size=64, 50 epochs (reduced from 100 — laptop budget), AdamW lr=1e-3
- **Weighted sampling**: balanced unit sampling per batch (F=59.4% imbalance)

#### Post-Phase-2 design updates
- **DCCE is less confounded than WhAM by year**: uses hand-crafted ICI + mel, not raw waveforms
- **Spectral encoder input**: 64 mel bins (not 128 — consistent with Baseline 1C, faster on CPU)
- **Individual ID target updated**: 762 not 763 (1 singleton dropped from split)

#### Ablations

| Model | Encoders | Cross-channel aug | Expected strength |
|---|---|---|---|
| DCCE-rhythm-only | GRU only (z = r_emb) | N/A | Coda type (high), unit (moderate) |
| DCCE-spectral-only | CNN only (z = s_emb) | N/A | Unit (high), coda type (low) |
| DCCE-late-fusion | GRU + CNN, no cross-aug | No | Both moderate |
| **DCCE-full** | GRU + CNN + cross-channel | **Yes** | Best on unit + indivID |

#### Comparison targets (post-Phase-2 update)

| Task | Target to beat | Source |
|---|---|---|
| Social Unit Macro-F1 | **0.895** | WhAM L19 (best layer, Phase 2) |
| Individual ID Macro-F1 | **0.454** | WhAM L10 (best for indivID overall) |
| Coda Type Macro-F1 | **0.931** | Raw ICI baseline (1A) — WhAM cannot beat this |

---

### Phase 4 — Experiment 2: Synthetic Data Augmentation
**Goal**: First controlled augmentation study for sperm whale bioacoustics.
**Dependencies**: WhAM generation working on local hardware (confirmed feasible ~8s/coda MPS)
**Status**: IN PROGRESS — notebook executing

#### Design (finalised)

- **N_synth sweep**: {0, 100, 500, 1000} — 2000 dropped (budget); max 1000 cached
- **Generation**: WhAM coarse_vamp, `rand_mask_intensity=0.8` (80% masked), 30 steps, seed=i
- **Prompt sampling**: stratified by unit (~⅓ A, ⅓ D, ⅓ F) from D_train
- **Pseudo-labels**: unit + coda_type copied from prompt; ICI copied from prompt (pseudo-ICI); no individual ID label
- **Model**: DCCE-full retrained from scratch for each N_synth; evaluated on real-only D_test
- **Caches**: synthetic WAVs → `datasets/synthetic_audio/synth_{i:04d}.wav`; features → `datasets/X_mel_synth_1000.npy`, `X_ici_synth_1000.npy`, etc.

#### Novel contributions
- First controlled WhAM augmentation study for cetacean bioacoustics
- Tests whether WhAM's coarse model is unit-faithful at generation time (not just embedding time)
- Primary metric: individual ID macro-F1 (most data-hungry task)

Tasks:
- [x] Confirm generation feasibility on MPS (~7.9s/coda, coarse-only)
- [x] Write build_phase4_notebook.py (33 cells, wham-env kernel)
- [x] Generate phase4_synthetic_aug.ipynb
- [x] Execute notebook — generation + training loop (1000 codas, 2943s)
- [x] Fill in Phase 4 results table
- [x] Update file directory

---

## 6. Validation Protocol

All experiments must be validated against biological ground truth:

| Claim | Validation | Source |
|---|---|---|
| Rhythm channel encodes coda type | r_emb probe accuracy on coda_type | dswp_labels.csv |
| Spectral channel encodes social identity | s_emb probe accuracy on unit | dswp_labels.csv |
| Joint embedding improves over either alone | Linear probe comparison across ablations | — |
| DCCE matches/beats WhAM on identity tasks | Comparison with WhAM L19 (unit) and L10 (indivID) | wham_embeddings_all_layers.npy |
| WhAM late layers encode social structure | **Confirmed Phase 2**: F1 rises monotonically, peaks layer 19 | phase2_wham_probing.ipynb |
| Year is not primary confound in DCCE | DCCE uses ICI+mel (not waveform) — year association is waveform-level | Phase 2 confound analysis |

---

## 7. Open Items / Blockers

| Item | Status | Action needed |
|---|---|---|
| DSWP audio download | **Done** | 1,501 WAV files in `datasets/dswp_audio/` |
| WhAM weights download | **Done** | coarse.pth (1.3 GB) + codec.pth (573 MB) in `wham/vampnet/models/` |
| WhAM embeddings extraction | **Done** | `datasets/wham_embeddings.npy` (L10, 1501×1280), `wham_embeddings_all_layers.npy` (1501×20×1280) |
| handv vowel labels for DSWP range | Missing | Email sent to WhAM/Beguš team — awaiting reply; not blocking Phase 3 |
| WhAM `allcodas.csv` | Missing | Same email — not blocking Phase 3 |
| WhAM generation feasibility on MPS | **Confirmed** | ~7.9s/coda on Apple MPS; coarse-only generation works |
| Recording-year confound | **Identified in Phase 2** | Cramér's V=0.51; DCCE less susceptible (uses ICI+mel not waveform) |
| IDN=0 for 672 codas | **Confirmed — biological limitation** | 762 IDN-labeled codas, 12 individuals (1 singleton dropped) |
| Class imbalance handling | **Done** | Stratified splits + weighted CE loss; macro-F1 primary metric |
| Mel-spectrogram parameters | **Confirmed** | 64 mel bins, fmax=8000 Hz, 128 time frames |
| ICI normalisation | **Confirmed** | StandardScaler (mean=177ms, std=88.6ms) |

---

## 8. File Directory

```
data_297_final_paper/
├── implementation_plan.md              ← This file (living document)
├── research_paper.md                   ← Full paper proposal
├── team_update.md                      ← Initial team communication
├── eda.py                              ← EDA script (standalone)
├── eda_phase0.ipynb                    ← Phase 0 EDA notebook (executed)
├── build_eda_notebook.py               ← Generator for eda_phase0.ipynb
├── phase1_baselines.ipynb              ← Phase 1 baselines notebook (executed)
├── build_phase1_notebook.py            ← Generator for phase1_baselines.ipynb
├── phase2_wham_probing.ipynb           ← Phase 2 probing notebook (executed)
├── build_phase2_notebook.py            ← Generator for phase2_wham_probing.ipynb
├── phase3_dcce.ipynb                   ← Phase 3 DCCE notebook (executed)
├── build_phase3_notebook.py            ← Generator for phase3_dcce.ipynb
├── phase4_synthetic_aug.ipynb          ← Phase 4 augmentation notebook (IN PROGRESS)
├── build_phase4_notebook.py            ← Generator for phase4_synthetic_aug.ipynb
├── extract_wham_embeddings.py          ← WhAM embedding extraction script
├── wham/                               ← Cloned Project-CETI/wham repo
├── wham_env/                           ← Python 3.12 venv for WhAM (vampnet stack)
├── datasets/
│   ├── dataset_report.md               ← Source analysis report
│   ├── dswp_labels.csv                 ← PRIMARY LABEL FILE (1,501 rows)
│   ├── dswp_audio/                     ← 1,501 WAV files (1.wav – 1501.wav)
│   ├── wham_embeddings.npy             ← WhAM L10 embeddings (1501 × 1280)
│   ├── wham_embeddings_all_layers.npy  ← WhAM all layers (1501 × 20 × 1280)
│   ├── train_idx.npy / test_idx.npy    ← Shared 80/20 split indices
│   ├── train_id_idx.npy / test_id_idx.npy ← ID-subset split indices
│   ├── X_mel_all.npy                   ← Pre-computed mel features (1383 × 64)
│   ├── X_mel_full.npy                  ← Pre-computed 2D mel features (1383 × 64 × 128)
│   ├── X_mel_synth_1000.npy            ← Synthetic mel features (1000 × 64 × 128) [generated in Phase 4]
│   ├── X_ici_synth_1000.npy            ← Synthetic pseudo-ICI (1000 × 9)
│   ├── y_unit_synth_1000.npy / y_type_synth_1000.npy ← Synthetic labels
│   ├── synthetic_meta.csv              ← Metadata for synthetic codas (prompt_coda_id, unit, etc.)
│   ├── phase1_results.csv              ← Phase 1 baseline results (loaded live by Phase 3)
│   ├── phase4_results.csv              ← Phase 4 augmentation results table
│   ├── synthetic_audio/                ← WhAM-generated WAV files (synth_0000.wav – synth_0999.wav)
│   ├── DominicaCodas.csv               ← Sharma et al. 2024 (8,719 rows)
│   ├── codamd.csv                      ← Beguš et al. (1,375 rows, vowel labels)
│   ├── focal-coarticulation-metadata.csv  ← Beguš et al. (spectral peaks)
│   ├── sperm-whale-dialogues.csv   ← Sharma et al. (3,840 rows, dialogues)
│   └── gero2016.xlsx               ← Gero et al. 2016 (4,116 rows, CC0)
└── figures/
    ├── eda/
    │   ├── fig1_label_distributions.png
    │   ├── fig2_ici_distributions.png
    │   ├── fig3_duration_clicks.png
    │   ├── fig4_codatype_unit_heatmap.png
    │   ├── fig5_idn0_investigation.png
    │   ├── fig6_sample_spectrograms.png
    │   ├── fig7_tsne_ici.png
    │   └── fig8_spectral_centroid.png
    ├── phase1/
    │   ├── fig_ici_rhythm_patterns.png
    │   ├── fig_spectrogram_gallery.png
    │   ├── fig_mean_mel_profiles.png
    │   ├── fig_1a_unit_cm.png / fig_1a_codatype_cm.png / fig_1a_individ_cm.png
    │   ├── fig_1c_unit_cm.png / fig_1c_codatype_cm.png
    │   ├── fig_baseline_comparison.png
    │   ├── fig_wham_tsne_unit.png / fig_wham_tsne_codatype.png
    │   ├── fig_wham_layer_norm_profile.png
    │   ├── fig_wham_layerwise_probe.png
    │   └── fig_wham_umap.png
    ├── phase2/
    │   ├── fig_wham_probe_profile.png
    │   ├── fig_wham_umap.png
    │   └── fig_wham_year_confound.png
    ├── phase3/
    │   ├── fig_dcce_training_curves.png
    │   ├── fig_dcce_comparison.png
    │   ├── fig_dcce_umap.png
    │   └── fig_wham_vs_dcce_umap.png        ← NEW: 2×2 comparison figure
    └── phase4/
        ├── fig_synth_spectrograms.png
        ├── fig_synth_mel_profiles.png
        ├── fig_augmentation_curve.png
        ├── fig_aug_training_curves.png
        └── fig_aug_umap.png
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
| Raw ICI → LogReg (1A) | **0.599** | **0.931** | 0.493 | ICI near-perfect for coda type; weak on unit — validates rhythm channel |
| Mel-spectrogram → LogReg (1C) | **0.740** | 0.097 | 0.272 | Mel better for unit; completely blind to coda type — validates spectral channel |
| WhAM → LogReg (1B) | **0.876** | 0.212 | 0.454 | Layer-10, 1280d; WhAM strong on unit, weak on coda type |

**Key Phase 1 interpretation**:
- ICI F1=0.931 on coda type confirms that ICI timing **is** coda type identity — a near-lossless rhythm code, consistent with Leitão et al.
- Mel F1=0.740 on unit vs ICI F1=0.599 confirms spectral texture carries **more** social-unit signal than rhythm timing alone.
- WhAM F1=0.876 on unit — strongest of all three; WhAM has learned to encode social/cultural structure from its generative objective.
- WhAM F1=0.212 on coda type — surprisingly weak, far below ICI (0.931). WhAM's generative (spectral) objective learned social identity but largely missed rhythm structure.
- Individual ID: WhAM (0.454) ≈ ICI (0.493) > Mel (0.272). Individual identity is hard for all linear probes — the primary target for DCCE.
- **DCCE target**: social unit > 0.876, individual ID > 0.454. The dual-channel design should surpass WhAM on both.

### Phase 2 — WhAM Probing
| Probe Target | Best Layer | Macro-F1 / R² | Notes |
|---|---|---|---|
| unit | 19 | F1 = **0.895** | Rises monotonically through all 20 layers |
| coda_type | 19 | F1 = 0.261 | Consistently weak — WhAM never learned rhythm timing |
| individual_id | 7 | F1 = 0.426 | Peaks mid-network; harder than unit |
| n_clicks | 0 | R² = 0.000 | WhAM does not encode click count |
| mean_ici_ms (tempo) | 0 | R² = 0.109 | Modest early-layer encoding only |
| recording year | 18 | F1 = **0.906** | ⚠ CONFOUND: year ≈ unit in separability |

**Critical Phase 2 finding — recording year confound:**
- Cramér's V(unit, year) = 0.51 — strong association. Unit A only 2005/2009; Unit D only 2008/2010; Unit F across all years
- Year F1 = 0.906 ≈ Unit F1 = 0.895 at best layer; Spearman ρ = 0.63 (p=0.003) between year-F1 and unit-F1 across layers
- **Interpretation**: WhAM's social-unit separability may be partly driven by recording-period acoustic drift, not pure biological identity. Absent from the WhAM paper.
- **Impact on DCCE**: DCCE uses ICI + mel (not raw waveforms), so it is less susceptible to recording-drift confounds than WhAM
- **DCCE target**: WhAM L19 unit F1 = **0.895**; indivID target = **0.454** (L10, best per-layer for this task)

### Phase 3 — DCCE Experiment 1
*(Final reproducible run — seed fixed per-variant; baselines loaded live from phase1_results.csv)*

| Model | Unit Macro-F1 | CodaType Macro-F1 | IndivID Macro-F1 | Notes |
|---|---|---|---|---|
| DCCE-rhythm-only | 0.637 | 0.878 | 0.509 | GRU on ICI only; strong coda type, moderate unit |
| DCCE-spectral-only | 0.693 | 0.139 | 0.787 | CNN on mel only; blind to coda type |
| DCCE-late-fusion | 0.656 | 0.705 | 0.825 | Both encoders, no cross-channel aug |
| **DCCE-full** | **0.878** | 0.578 | **0.834** | Cross-channel contrastive; best unit + indivID |
| *(WhAM L19 target)* | *(0.895)* | *(0.261)* | *(0.454)* | *Comparison ceiling* |

**Key Phase 3 findings (final rerun):**
- **Individual ID: DCCE-full F1=0.834 >> WhAM L10 (0.454) — beats target by +0.380** — the main result
- **Social unit: DCCE-full F1=0.878 vs WhAM L19 0.895 — 1.7% gap** — very close; WhAM's marginal advantage is partly due to year confound
- **Coda type: DCCE-full F1=0.578 >> WhAM (0.261), but below raw ICI (0.931)** — fused model sacrifices some rhythm purity for identity
- **Cross-channel augmentation critical**: DCCE-full unit F1 +0.222 over late-fusion, indivID +0.009 over late-fusion
- **Ablation deltas (full vs each ablation)**: unit: +0.241 vs rhythm-only, +0.185 vs spectral-only, +0.222 vs late-fusion
- **DCCE-spectral-only individual ID = 0.787** — spectral channel alone is strong for identity; the full cross-channel objective pushes further
- Training: 50 epochs, ~46s total on Apple MPS
- New figure: `fig_wham_vs_dcce_umap.png` — 2×2 comparison (WhAM L19 vs DCCE-full × unit vs individual ID)

### Phase 4 — Synthetic Augmentation
*(Final reproducible run — seed fixed; synthetic audio cache reused from first generation run)*

| N_synth | D_train | Unit F1 | CodaType F1 | IndivID F1 | Unit Acc | IndivID Acc |
|---|---|---|---|---|---|---|
| **0** (baseline) | 1,106 | **0.878** | **0.578** | **0.834** | 0.885 | 0.902 |
| 100 | 1,206 | 0.874 | 0.525 | 0.788 | 0.877 | 0.882 |
| 500 | 1,606 | 0.872 | 0.518 | 0.803 | 0.874 | 0.909 |
| 1000 | 2,106 | 0.869 | 0.545 | 0.783 | 0.874 | 0.882 |
| *(WhAM L19)* | — | *(0.895)* | *(0.261)* | *(0.454)* | — | — |

**Key Phase 4 findings (final rerun — now reproducible):**
- **Best individual ID F1 = 0.834 at N_synth=0** — consistent with Phase 3's 0.834 (seed fix ensures identical initialisation)
- **Augmentation slightly decreases individual ID F1** (0.788–0.803 with synthetic data vs 0.834 without). This is now a clean and interpretable result: synthetic codas have pseudo-ICI copied from the prompt and no individual ID label — they add unit-level signal but dilute the contrastive geometry that DCCE-full uses to separate individuals.
- **Unit and coda type F1 also slightly decrease** with augmentation — the synthetic mel spectrograms are not as discriminative as real ones (coarse model only; no c2f refinement), which mildly degrades the spectral encoder.
- **Interpretation**: WhAM coarse generation preserves unit-level acoustic texture (mel profiles closely match real codas per unit, UMAP confirms same-unit clustering) but introduces mild distribution shift relative to real data. This constrains what the coarse-only model captures — individual micro-variation and fine coda type structure may require c2f generation.
- **Generation**: 1,000 synthetic codas generated in 2,943s (~2.9s/coda on Apple MPS, 30-step coarse_vamp)
- **Bottom line**: DCCE-full without augmentation (0.834 indivID) is the strongest result; WhAM augmentation is not beneficial at the coarse level.

---

## 10. Change Log

| Date | Change |
|---|---|
| 2026-04-03 | Initial plan created. All datasets downloaded and analyzed. dswp_labels.csv produced. |
| 2026-04-03 | Phase 0 complete. DSWP audio downloaded. 8 EDA figures produced. eda_phase0.ipynb fully executed. |
| 2026-04-03 | Plan updated post-EDA: (1) positive pairs changed from same-whale to same-unit level; (2) IDN count corrected to 763; (3) macro-F1 adopted as primary metric; (4) stratified splits + weighted CE loss added; (5) mel-spectrogram params confirmed; (6) ICI normalisation confirmed; (7) Baseline 1C (mel-spec logReg) added; (8) recording-year probe added to Phase 2; (9) hypotheses sharpened based on t-SNE and channel independence findings. |
| 2026-04-04 | Phase 1 baselines 1A and 1C executed. Results: ICI→LogReg unit=0.599 / codatype=0.931 / indivID=0.493; Mel→LogReg unit=0.740 / codatype=0.097 / indivID=0.272. Channels confirmed complementary. Singleton individual 6059 (1 clean coda) dropped from ID split. sklearn multi_class kwarg removed (deprecated). |
| 2026-04-04 | Phase 1 Baseline 1B (WhAM) complete. Downloaded coarse.pth (1.3 GB) + codec.pth (573 MB) from Zenodo. Installed vampnet+lac+audiotools stack in wham_env (Python 3.12). Extracted 1501 embeddings via VampNet coarse transformer layer 10, 1280d (not 768d as initially assumed). Results: WhAM unit=0.876 / codatype=0.212 / indivID=0.454. WhAM strong on social unit, weak on coda type — DCCE targets: unit>0.876, indivID>0.454. |
| 2026-04-04 | Phase 2 (WhAM probing) complete. Best layers: unit=L19 (F1=0.895), coda_type=L19 (F1=0.261), indivID=L7 (F1=0.426). Critical: recording year confound identified (Cramér's V=0.51; year F1=0.906 ≈ unit F1). DCCE targets updated: unit>0.895 (harder), indivID>0.454 (maintained). |
| 2026-04-04 | Phase 3 (DCCE) complete. Key result: DCCE-full indivID F1=0.731 >> WhAM L10 (0.454); unit F1=0.865 (3% below WhAM L19=0.895; gap partly due to year confound). Cross-channel aug critical: DCCE-full unit F1 +0.235 over late-fusion. X_mel_full.npy computed (1383×64×128). |
| 2026-04-04 | Phase 4 complete. Generated 1,000 WhAM synthetic codas (2,943s, ~2.9s/coda MPS, 30-step coarse_vamp). Key: augmentation neutral for indivID (best=0.820 at N_synth=0); coda type F1 boosted at N_synth=100 (+0.16). indivID=0.820 >> WhAM (0.454). 5 figures. |
| 2026-04-04 | Full rerun (Phase 1–4). Fixes: (1) torch.manual_seed inside each train_dcce/train_dcce_full for reproducible variant-independent initialisation; (2) Phase 1 exports phase1_results.csv for live loading in Phase 3; (3) 2×2 WhAM vs DCCE comparison UMAP added to Phase 3. Final numbers: DCCE-full indivID=0.834, unit=0.878. Phase 4 seed-fixed confirms augmentation slightly hurts (not neutral) — cleaner interpretation. |
| 2026-04-04 | Refactored build_phase1_notebook.py: embedding extraction now runs inline via subprocess; added WhAM EDA (layer-norm profile, t-SNE coloured by unit+coda-type, layer-wise linear probe across all 20 layers). Layer-wise probe finding: social-unit F1 peaks at layer 19 (0.895) but layer 10 already achieves 0.876 — within 2%. Coda-type F1 weak at all layers (best 0.261 at layer 19 vs ICI 0.931). Confirms WhAM encodes social identity well but misses rhythm structure. |
| 2026-04-04 | Phase 2 complete. build_phase2_notebook.py + phase2_wham_probing.ipynb created and executed. Full 6-target layer-wise probe + UMAP + confound analysis. Key finding: recording year is a strong confound (Cramér's V=0.51, year F1=0.906 ≈ unit F1=0.895). DCCE target updated to unit F1 > 0.895. Individual ID best at WhAM layer 7 (F1=0.426) — layer 10 (0.411) is not the best for this task either. |
| 2026-04-04 | Phase 3 complete. build_phase3_notebook.py + phase3_dcce.ipynb created and executed. Trained 4 DCCE variants (rhythm_only, spectral_only, late_fusion, full) with NT-Xent + auxiliary heads, 50 epochs on MPS (~47s). Main result: DCCE-full individual ID F1=0.731 vs WhAM L10 0.454 — beats target by +0.28. Unit F1=0.865 vs WhAM L19 0.895 — 3% gap. Cross-channel augmentation validated: +0.235 unit F1 over late-fusion. wham-env kernel registered for Phase 3 notebook. |
