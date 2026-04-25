# Dataset Report: Additional Public Sources for Sperm Whale Coda Analysis

**Project**: Beyond WhAM — Dual-Channel Contrastive Coda Representation (CS 297 Final Paper)
**Date**: April 2026
**Author**: Data scouting analysis

---

## Overview

This report documents four publicly available datasets retrieved from GitHub repositories associated with Project CETI and the Sharma et al. (2024) Nature Communications paper. All were retrieved without access restrictions. Together they substantially augment the raw DSWP HuggingFace dataset (`orrp/DSWP`), which contains only unlabeled audio.

---

## Source 1: DominicaCodas.csv

**Origin**: Sharma, P., Gero, S., Payne, R. et al. "Contextual and combinatorial structure in sperm whale vocalisations." *Nature Communications* 15, 3617 (2024).
**Repository**: `github.com/pratyushasharma/sw-combinatoriality/data/DominicaCodas.csv`
**License**: Open (Nature Communications data availability policy, CC BY 4.0 implied)

### Contents

| Field | Description |
|---|---|
| `codaNUM2018` | Sequential coda ID, 1–8878 |
| `Date` | Recording date (2005–2010) |
| `nClicks` | Number of clicks in coda |
| `Duration` | Total coda duration in seconds |
| `ICI1`–`ICI9` | Pre-computed inter-click intervals (seconds) |
| `CodaType` | Coda type label (35 types: 1+1+3, 5R1, 4D, etc.) |
| `Clan` | Vocal clan (EC1 or EC2) |
| `Unit` | Social unit (A–V, ZZZ) |
| `UnitNum` | Numeric unit identifier |
| `IDN` | Individual whale numeric ID |

### Statistics

- **Total rows**: 8,719 codas across the full Dominica corpus (2005–2018)
- **codaNUM2018 range**: 1–8,878
- **Clans**: EC1 (7,770 codas), EC2 (949 codas)
- **Social units**: 13 named units (A, D, F, J, K, N, P, R, S, T, U, V, ZZZ)
- **Individual whales**: 36 unique IDN values
- **Coda types**: 35 (including NOISE variants)
- **NOISE-tagged codas**: 600 total; **clean codas**: 8,119
- **ICI coverage**: All rows have ICI1 > 0 (pre-computed, no peak detection needed)

### Critical Finding: Exact Alignment with DSWP

The DSWP HuggingFace dataset contains exactly 1,501 audio files (`1.wav`–`1501.wav`). DominicaCodas.csv contains **exactly 1,501 rows with codaNUM2018 in range 1–1501**, covering only units A, D, and F. This is not a coincidence — the DSWP release is a subset of the full Dominica corpus, and the codaNUM2018 index is the shared key.

**DSWP subset (codaNUM2018 1–1501)**:
- Unit A: 273 codas
- Unit D: 336 codas
- Unit F: 892 codas
- NOISE codas: 118 | Clean codas: 1,383

This means every DSWP audio file can now be labeled with: social unit, coda type, individual whale ID, ICI sequence, duration, and date.

---

## Source 2: codamd.csv

**Origin**: Beguš, G. et al. "The Phonology of Sperm Whale Coda Vowels." Repository: `github.com/Project-CETI/coda-vowel-phonology`
**License**: Public (Project CETI, CC BY 4.0)

### Contents

| Field | Description |
|---|---|
| `codanum` | Coda ID (matches DominicaCodas codaNUM2018) |
| `focal` | Whether this was a focal recording (True/False) |
| `whale` | Named individual whale (e.g., ATWOOD, FORK) |
| `codatype` | Coda type label |
| `Duration` | Coda duration in seconds |
| `handv` | Hand-verified vowel category: `a` or `i` |

### Statistics

- **Total rows**: 1,375 codas
- **codanum range**: 4,933–8,860 (the later portion of the full corpus — does NOT overlap with DSWP 1–1501)
- **Named whales**: 13 (ATWOOD, FORK, FRUIT, JOCASTA, LADYO, LAIUS, NALGENE, PINCHY, SAM, SOPH, SOURSOP, TBB, TWEAK)
- **Vowel distribution**: `a` = 745, `i` = 397, unlabeled = 233
- **NOISE codas**: 79 | **Clean**: 1,296
- **Overlap with DominicaCodas**: 1,375 rows (perfect join on codanum)

### Key Point

The `handv` vowel label (`a` or `i`) is the spectral channel ground truth formalized in the Beguš et al. phonology paper. This is the label for the **spectral encoder** in the DCCE architecture. However, codamd covers codaNUM 4,933–8,860 only — it does not cover the DSWP 1–1501 range. Vowel labels for the DSWP subset remain unavailable publicly.

---

## Source 3: focal-coarticulation-metadata.csv

**Origin**: Beguš et al., same repository as codamd.csv
**License**: Public (Project CETI, CC BY 4.0)

### Contents

Per-click spectral transition metadata for consecutive click pairs within codas. Each row represents a transition from one click to the next within the same coda, capturing how spectral patterns change across a coda sequence.

| Field | Description |
|---|---|
| `codanum` | Coda ID |
| `whale` | Named individual |
| `coart` | Coarticulation type: `aa`, `ai`, `ia`, or `ii` |
| `prevhandvcat` / `handvcat` | Vowel category of previous/current click |
| `pkfq` | Spectral peak frequency of current click (Hz) |
| `f1pk`, `f2pk` | First and second spectral formant peaks (Hz) |
| `deltasec` | Time gap between clicks (seconds) |
| `codadt` / `codaenddt` | Timestamps for coda start/end |

### Statistics

- **Total rows**: 1,097 click-pair transitions
- **Unique codas covered**: 1,097
- **Coarticulation type distribution**: aa=670, ii=336, ai=45, ia=46
- **Whales**: Same 13 named individuals as codamd.csv
- **Spectral peak frequencies**: ~3,000–9,000 Hz range

### Key Point

This is the most granular spectral dataset available. It provides formant-level information (f1, f2) analogous to vowel formants in human phonetics. Useful primarily for Experiment 3 probing (validating that WhAM encodes spectral/vowel information) and as ground truth for the spectral encoder.

---

## Source 4: sperm-whale-dialogues.csv

**Origin**: Sharma et al. (2024), same repository as DominicaCodas.csv
**License**: Open (CC BY 4.0)

### Contents

| Field | Description |
|---|---|
| `REC` | Recording session identifier (e.g., `sw061b001_124`) |
| `nClicks` | Number of clicks |
| `Duration` | Coda duration |
| `ICI1`–`ICI28` | Up to 28 inter-click intervals |
| `Whale` | Numeric whale ID (1–11) |
| `TsTo` | Timestamp offset within recording |

### Statistics

- **Total rows**: 3,840 codas
- **Whale IDs**: 11 numeric IDs (1–11), different scheme from DominicaCodas IDN
- **Recording sessions**: 48 unique sessions
- **nClicks**: Heavily dominated by 5-click codas (2,949/3,840 = 77%)
- **ICI columns**: Up to ICI28 (for very long codas)
- **No CodaType column**: Coda type labels not present

### Key Point

This dataset represents conversational exchange sequences ("dialogues") between whales. It captures which whale vocalized when during multi-whale interactions, providing temporal context not in DominicaCodas. However, it lacks coda type labels and uses a different ID scheme, making direct joins with other sources non-trivial. Most useful as additional training data for the ICI/rhythm encoder if more volume is needed.

---

## Comparative Summary

| Dataset | Rows | DSWP overlap | Social unit | Coda type | Individual ID | ICI | Vowel (handv) |
|---|---|---|---|---|---|---|---|
| DominicaCodas.csv | 8,719 | **Yes (1,501 exact)** | Yes (13 units) | Yes (35 types) | Yes (36 IDs) | Yes (pre-computed) | No |
| codamd.csv | 1,375 | No (codaNUM 4933+) | No | Yes | Yes (named) | No | **Yes (a/i)** |
| focal-coarticulation | 1,097 | No | No | No | Yes (named) | No | Yes (per-click) |
| sperm-whale-dialogues | 3,840 | No | No | No | Partial (numeric) | Yes (up to ICI28) | No |

---

## Utility Assessment for Each Experiment

### Experiment 1 — DCCE Representation Quality

**DominicaCodas.csv is the primary label source.** The 1,501 DSWP-aligned rows provide:
- Social unit (A/D/F) for contrastive loss positive pair construction
- CodaType for the rhythm classification auxiliary head
- IDN for individual-level contrastive loss
- ICI1–ICI9 as direct input to the rhythm encoder (no preprocessing needed)

**codamd.csv** adds vowel (handv) labels but only for codaNUM 4,933+. After a join with DominicaCodas on codanum, these can label an additional ~1,296 clean codas outside the DSWP range — useful for pre-training the spectral encoder on a larger set.

**Status**: Fully unblocked for all three downstream tasks (social unit, coda type, individual ID).

### Experiment 2 — Synthetic Data Augmentation

**DominicaCodas.csv** provides evaluation labels for the held-out test set. WhAM-generated synthetic codas can be conditioned on real DSWP codas and evaluated against these labels. Fully unblocked.

### Experiment 3 — WhAM Probing

**DominicaCodas.csv** provides: `nClicks` (click count probe), `ICI1`–`ICI9` (mean ICI/tempo probe), `CodaType`, `Unit`, `IDN`, and `Date`.

**focal-coarticulation-metadata.csv** provides per-click spectral peak frequencies (`pkfq`, `f1pk`) — directly usable as the ground-truth spectral centroid target for the vowel encoding probe.

**Status**: Fully unblocked. Richer than originally planned — spectral probes can use actual measured formant frequencies rather than librosa-computed centroids.

---

## Files Produced

| File | Description |
|---|---|
| `DominicaCodas.csv` | Raw download, Sharma et al. |
| `codamd.csv` | Raw download, Beguš et al. |
| `focal-coarticulation-metadata.csv` | Raw download, Beguš et al. |
| `sperm-whale-dialogues.csv` | Raw download, Sharma et al. |
| `dswp_labels.csv` | **Merged label table — 1,501 DSWP codas with all available labels** |
