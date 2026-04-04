# Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding

**CS 297 Final Paper · April 2026**

This repository contains the full experimental pipeline for the paper. The work investigates whether a model purpose-built around the known biological decomposition of sperm whale codas into a **rhythm channel** (ICI timing) and a **spectral channel** (acoustic texture) produces better representations than WhAM — the current state-of-the-art generative model for cetacean bioacoustics (Paradise et al., NeurIPS 2025).

---

## Key Results

| Model | Social Unit F1 | Coda Type F1 | Individual ID F1 |
|---|---|---|---|
| Raw ICI → LogReg (baseline 1A) | 0.599 | **0.931** | 0.493 |
| Raw Mel → LogReg (baseline 1C) | 0.740 | 0.097 | 0.272 |
| WhAM L10 → LogReg (baseline 1B) | 0.876 | 0.212 | 0.454 |
| WhAM L19 → LogReg (best layer) | **0.895** | 0.261 | 0.426 |
| DCCE-rhythm-only | 0.637 | 0.878 | 0.509 |
| DCCE-spectral-only | 0.693 | 0.139 | 0.787 |
| DCCE-late-fusion | 0.656 | 0.705 | 0.825 |
| **DCCE-full (ours)** | **0.878** | 0.578 | **0.834** |

**Main finding**: DCCE-full individual ID F1 = **0.834**, versus WhAM's best of **0.454** — a +0.380 improvement. WhAM's marginal social-unit advantage (0.895 vs 0.878) is partially confounded by recording-year drift (Cramér's V = 0.51).

**Synthetic augmentation** (Phase 4): Adding WhAM-generated codas slightly *decreases* all metrics. The coarse model is unit-faithful but introduces mild distribution shift.

---

## Repository Structure

```
data_297_final_paper/
│
├── README.md                        ← This file
├── implementation_plan.md           ← Living document: design decisions, all results, change log
├── research_paper.md                ← Full paper write-up (draft)
├── team_update.md                   ← Initial project communication
│
├── notebooks/                       ← Executed Jupyter notebooks (one per phase)
│   ├── eda_phase0.ipynb             ← Phase 0: Exploratory Data Analysis
│   ├── phase1_baselines.ipynb       ← Phase 1: Raw ICI / Mel / WhAM baselines
│   ├── phase2_wham_probing.ipynb    ← Phase 2: WhAM layer-wise interpretability
│   ├── phase3_dcce.ipynb            ← Phase 3: DCCE training and evaluation
│   └── phase4_synthetic_aug.ipynb   ← Phase 4: WhAM synthetic data augmentation
│
├── builders/                        ← Python scripts that generate the notebooks above
│   ├── build_eda_notebook.py
│   ├── build_phase1_notebook.py
│   ├── build_phase2_notebook.py
│   ├── build_phase3_notebook.py
│   └── build_phase4_notebook.py
│
├── scripts/                         ← Standalone utility scripts
│   ├── eda.py                       ← Standalone EDA script (exploratory, pre-notebook)
│   └── extract_wham_embeddings.py   ← WhAM embedding extraction (called by Phase 1)
│
├── datasets/                        ← All data files (audio not tracked in git)
│   ├── dswp_labels.csv              ← PRIMARY label file (1,501 rows, one per audio file)
│   ├── phase1_results.csv           ← Phase 1 baseline results (auto-loaded by Phase 3)
│   ├── phase4_results.csv           ← Phase 4 augmentation sweep results
│   ├── wham_embeddings.npy          ← WhAM Layer-10 embeddings (1501 × 1280)
│   ├── wham_embeddings_all_layers.npy ← WhAM all layers (1501 × 20 × 1280)
│   ├── X_mel_full.npy               ← Pre-computed 2D mel spectrograms (1383 × 64 × 128)
│   ├── X_mel_all.npy                ← Pre-computed mean-pooled mel (1383 × 64)
│   ├── train_idx.npy / test_idx.npy ← Shared 80/20 split (stratified by unit, seed=42)
│   ├── train_id_idx.npy / test_id_idx.npy ← Split indices for IDN-labeled subset
│   ├── X_mel_synth_1000.npy         ← Synthetic mel spectrograms (1000 × 64 × 128)
│   ├── X_ici_synth_1000.npy         ← Synthetic pseudo-ICI (1000 × 9)
│   ├── y_unit_synth_1000.npy        ← Synthetic unit labels
│   ├── y_type_synth_1000.npy        ← Synthetic coda type labels
│   ├── synthetic_meta.csv           ← Metadata for synthetic codas (prompt IDs, units)
│   ├── synthetic_audio/             ← 1,000 WhAM-generated WAV files (synth_0000–0999)
│   ├── dswp_audio/                  ← 1,501 real DSWP WAV files (1.wav – 1501.wav)
│   ├── DominicaCodas.csv            ← Sharma et al. 2024 label source (8,719 rows)
│   ├── codamd.csv                   ← Beguš et al. vowel labels (1,375 rows)
│   ├── focal-coarticulation-metadata.csv ← Beguš et al. spectral peaks
│   ├── sperm-whale-dialogues.csv    ← Sharma et al. dialogue sequences
│   ├── gero2016.xlsx                ← Gero et al. 2016 (CC0)
│   └── dataset_report.md            ← Source analysis and join validation notes
│
├── figures/                         ← All generated figures, organised by phase
│   ├── eda/                         ← 8 EDA figures (label distributions, t-SNE, etc.)
│   ├── phase1/                      ← Baseline figures (confusion matrices, comparison bars, WhAM UMAP)
│   ├── phase2/                      ← WhAM probing (layer profile, UMAP, year confound)
│   ├── phase3/                      ← DCCE (training curves, comparison bar, UMAP, 2×2 comparison)
│   └── phase4/                      ← Augmentation (F1 curve, synthetic spectrograms, UMAP)
│
├── wham/                            ← Cloned Project-CETI/wham repository
│   └── vampnet/models/              ← WhAM model weights (coarse.pth, codec.pth)
│
└── wham_env/                        ← Python 3.12 virtual environment (PyTorch + WhAM stack)
```

> All files are already organised into the subdirectories shown above.

---

## Environments

There are **two Python environments** in this project:

| Environment | Purpose | Packages |
|---|---|---|
| System Python (registered as `python3-local` kernel) | Phases 0 and 1 — no PyTorch needed | numpy, pandas, librosa, sklearn, matplotlib, seaborn, umap-learn |
| `wham_env/` (registered as `wham-env` kernel) | Phases 2–4 — needs WhAM / PyTorch | all of the above + torch, vampnet, lac, audiotools, soundfile |

### Register kernels (one-time setup)

```bash
# python3-local kernel (for Phase 0 and Phase 1)
python3 -m ipykernel install --user --name python3-local --display-name "Python 3 (local)"

# wham-env kernel (for Phases 2–4)
source wham_env/bin/activate
python -m ipykernel install --user --name wham-env --display-name "Python (wham-env)"
```

---

## How to Reproduce

Run the phases **in order**. Each phase depends on outputs from the previous one.

### Prerequisites

1. Clone the WhAM repository and download weights:
   ```bash
   git clone https://github.com/Project-CETI/wham
   pip install -e wham/
   # Download coarse.pth and codec.pth from Zenodo 10.5281/zenodo.17633708
   # Place in: wham/vampnet/models/
   ```

2. Download DSWP audio from HuggingFace:
   ```bash
   # ~585 MB — 1,501 WAV files
   python3 -c "
   from huggingface_hub import snapshot_download
   snapshot_download('orrp/DSWP', local_dir='datasets/dswp_audio', repo_type='dataset')
   "
   ```

3. `datasets/dswp_labels.csv` is already in the repository (produced from DominicaCodas.csv + Gero 2016).

---

### Phase 0 — Exploratory Data Analysis

**Notebook**: `eda_phase0.ipynb`  
**Kernel**: `python3-local`  
**Outputs**: `figures/eda/` (8 figures)  
**Rebuild**:
```bash
# Run from project root
python3 builders/build_eda_notebook.py
jupyter nbconvert --to notebook --execute notebooks/eda_phase0.ipynb \
  --output notebooks/eda_phase0.ipynb \
  --ExecutePreprocessor.kernel_name=python3-local
```

---

### Phase 1 — Baselines

**Notebook**: `notebooks/phase1_baselines.ipynb`  
**Kernel**: `python3-local`  
**Runtime**: ~15 min (WhAM embedding extraction is the long step; cached after first run)  
**Outputs**:
- `datasets/train_idx.npy`, `test_idx.npy`, `train_id_idx.npy`, `test_id_idx.npy` — shared splits used by all subsequent phases
- `datasets/wham_embeddings.npy` (1501 × 1280) and `wham_embeddings_all_layers.npy` (1501 × 20 × 1280)
- `datasets/X_mel_all.npy` (1383 × 64) — mean-pooled mel features
- `datasets/phase1_results.csv` — baseline F1 values loaded by Phase 3
- `figures/phase1/` (13 figures)

**Rebuild**:
```bash
python3 builders/build_phase1_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase1_baselines.ipynb \
  --output notebooks/phase1_baselines.ipynb \
  --ExecutePreprocessor.kernel_name=python3-local \
  --ExecutePreprocessor.timeout=1800
```

---

### Phase 2 — WhAM Probing

**Notebook**: `notebooks/phase2_wham_probing.ipynb`  
**Kernel**: `wham-env`  
**Runtime**: ~2 min  
**Outputs**: `figures/phase2/` (3 figures)  
**Key findings**: layer-wise probe profile; year confound (Cramér's V = 0.51); best unit layer = 19 (F1 = 0.895)

**Rebuild**:
```bash
source wham_env/bin/activate
python3 builders/build_phase2_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase2_wham_probing.ipynb \
  --output notebooks/phase2_wham_probing.ipynb \
  --ExecutePreprocessor.kernel_name=wham-env
```

---

### Phase 3 — DCCE Training and Evaluation

**Notebook**: `notebooks/phase3_dcce.ipynb`  
**Kernel**: `wham-env`  
**Runtime**: ~5 min (4 variants × 50 epochs, ~46s each on Apple MPS)  
**Outputs**:
- `datasets/X_mel_full.npy` (1383 × 64 × 128) — full 2D mel spectrograms; cached after first run
- `figures/phase3/` (4 figures including `fig_wham_vs_dcce_umap.png`)

**Rebuild**:
```bash
source wham_env/bin/activate
python3 builders/build_phase3_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase3_dcce.ipynb \
  --output notebooks/phase3_dcce.ipynb \
  --ExecutePreprocessor.kernel_name=wham-env \
  --ExecutePreprocessor.timeout=3600
```

---

### Phase 4 — Synthetic Data Augmentation

**Notebook**: `notebooks/phase4_synthetic_aug.ipynb`  
**Kernel**: `wham-env`  
**Runtime**: ~1 hour total (~2,943s for 1,000 codas at 2.9s/coda on Apple MPS; cached after first run)  
**Outputs**:
- `datasets/synthetic_audio/` — 1,000 WAV files
- `datasets/X_mel_synth_1000.npy`, `X_ici_synth_1000.npy`, `y_unit_synth_1000.npy`, `y_type_synth_1000.npy`
- `datasets/synthetic_meta.csv`, `datasets/phase4_results.csv`
- `figures/phase4/` (5 figures)

**Note**: generation uses a skip-if-cached check — if `datasets/synthetic_audio/synth_0000.wav` already exists, it will be reused and only the mel extraction + training runs again (~10 min).

**Rebuild**:
```bash
source wham_env/bin/activate
python3 builders/build_phase4_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase4_synthetic_aug.ipynb \
  --output notebooks/phase4_synthetic_aug.ipynb \
  --ExecutePreprocessor.kernel_name=wham-env \
  --ExecutePreprocessor.timeout=86400
```

---

### Full Pipeline (all phases in order)

Run from the project root directory:

```bash
# Phase 0 and 1 — system Python
python3 builders/build_eda_notebook.py
jupyter nbconvert --to notebook --execute notebooks/eda_phase0.ipynb \
  --output notebooks/eda_phase0.ipynb \
  --ExecutePreprocessor.kernel_name=python3-local

python3 builders/build_phase1_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase1_baselines.ipynb \
  --output notebooks/phase1_baselines.ipynb \
  --ExecutePreprocessor.kernel_name=python3-local \
  --ExecutePreprocessor.timeout=1800

# Phases 2–4 — wham-env (run sequentially; each depends on the previous)
source wham_env/bin/activate

python3 builders/build_phase2_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase2_wham_probing.ipynb \
  --output notebooks/phase2_wham_probing.ipynb \
  --ExecutePreprocessor.kernel_name=wham-env

python3 builders/build_phase3_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase3_dcce.ipynb \
  --output notebooks/phase3_dcce.ipynb \
  --ExecutePreprocessor.kernel_name=wham-env \
  --ExecutePreprocessor.timeout=3600

python3 builders/build_phase4_notebook.py
jupyter nbconvert --to notebook --execute notebooks/phase4_synthetic_aug.ipynb \
  --output notebooks/phase4_synthetic_aug.ipynb \
  --ExecutePreprocessor.kernel_name=wham-env \
  --ExecutePreprocessor.timeout=86400
```

---

## Reproducibility Notes

- **Random seed**: `SEED = 42` throughout. `torch.manual_seed(SEED)` is set **inside** each `train_dcce()` / `train_dcce_full()` call, so each variant or N_synth condition is initialised independently and identically regardless of execution order.
- **Train/test split**: produced once in Phase 1 and saved to `datasets/train_idx.npy` etc. Loaded — never recomputed — in all subsequent phases.
- **Baseline values**: Phase 1 saves `datasets/phase1_results.csv`; Phase 3 loads this file rather than using hardcoded numbers.
- **Mel spectrograms**: `datasets/X_mel_full.npy` is computed once in Phase 3 and reused in Phase 4.
- **Synthetic audio**: `datasets/synthetic_audio/synth_XXXX.wav` files are cached; re-running Phase 4 reuses them.
- **Hardware**: all experiments run on a MacBook with Apple MPS. Timings are MPS-based; CPU will be 3–5× slower.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Positive pairs at **social unit** level (not individual) | IDN=0 accounts for 44.8% of data (all Unit F) — individual-level pairing would exclude the majority of Unit F codas |
| **Cross-channel** pairs: rhythm(A) + spectral(B) from same unit | Forces the joint embedding to capture unit identity from orthogonal signal axes; validated by +0.222 F1 over late-fusion |
| **Macro-F1** as primary metric | Unit F = 59.4% of data — accuracy would flatter models predicting the majority class |
| **WeightedRandomSampler** | Balances unit distribution in each training batch without oversampling |
| **WhAM Layer 19** as baseline ceiling | Layer-wise probe (Phase 2) shows F1 peaks at L19 = 0.895, not the JukeMIR-convention L10 = 0.876 |
| **ICI pseudo-labels** for synthetic codas | Click timing cannot be reliably extracted from coarse-only generation; prompt ICI is used as a proxy |

---

## Literature

| Paper | Role |
|---|---|
| Paradise et al., NeurIPS 2025 (WhAM) | Primary baseline — SOTA cetacean audio model |
| Sharma et al., Nature Comms 2024 | DominicaCodas.csv labels; combinatorial coda structure |
| Leitão et al., 2023–2025 | Theoretical basis for separating rhythm channel |
| Beguš et al., 2024 | Formalises spectral/vowel channel; two independent syntax axes |
| Chen et al., NeurIPS 2020 (SimCLR) | NT-Xent loss; τ=0.07 convention |
| Gubnitsky et al., 2024 | Automated coda detector — preprocessing baseline |
| Gero, Whitehead & Rendell, 2016 | Foundational unit/type classification |
| Goldwasser et al., NeurIPS 2023 | Theoretical justification for unsupervised animal communication modelling |
