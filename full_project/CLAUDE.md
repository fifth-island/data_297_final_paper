# Beyond WhAM — Claude Code Project Brief

## What this project is

A CS 297 final paper reproducing and extending sperm whale coda classification research across 4 experimental phases. The full specification — architecture, data schema, expected results, and task-by-task breakdown — is in `implementation_plan.md`. **Read that file first.**

## Your task

Implement all 4 phases by building Jupyter notebooks from scratch:

| Notebook to create | Phase | Goal |
|---|---|---|
| `notebooks/eda_phase0.ipynb` | Phase 0 | Exploratory data analysis + figures |
| `notebooks/phase1_baselines.ipynb` | Phase 1 | ICI logistic regression + WhAM embedding probes |
| `notebooks/phase2_wham_probing.ipynb` | Phase 2 | Layer-wise WhAM probing + UMAP |
| `notebooks/phase3_dcce.ipynb` | Phase 3 | Dual-Channel Contrastive Encoder (DCCE) |
| `notebooks/phase4_synthetic_aug.ipynb` | Phase 4 | Synthetic data augmentation sweep |

Also create a `reports/` folder and write a `.md` summary report after each phase.

## Environment

Two Python environments are available:

### 1. Standard `.venv` (for Phases 0, 1, 2, 3)
```bash
source /Users/joaoquintanilha/Downloads/data_297_final_paper/.venv/bin/activate
```
Packages available: numpy, pandas, scikit-learn, matplotlib, seaborn, librosa, torch, umap-learn, jupyter.

### 2. `wham_env` (for WhAM embedding extraction only)
```bash
source ./wham_env/bin/activate
```
Use this only when running WhAM model inference (embedding extraction). The WhAM code is in `./wham/`.

**For notebooks**: Use the `.venv` kernel unless you need to re-extract WhAM embeddings (pre-computed embeddings are in `datasets/` — you should not need to re-extract).

## Key paths

```
full_project/
├── CLAUDE.md                         ← this file
├── implementation_plan.md            ← full specification
├── datasets/
│   ├── dswp_labels.csv               ← PRIMARY label file (1,501 rows)
│   ├── dswp_audio/                   ← 1,501 WAV files (1.wav – 1501.wav)
│   ├── wham_embeddings.npy           ← WhAM L10 embeddings (1501 × 1280)
│   ├── wham_embeddings_all_layers.npy ← All layers (1501 × 20 × 1280)
│   ├── X_mel_all.npy                 ← Mean-pooled mel features (1383 × 64)
│   ├── X_mel_full.npy                ← 2D mel features (1383 × 64 × 128)
│   ├── X_mel_synth_1000.npy          ← Synthetic mel (1000 × 64 × 128)
│   ├── X_ici_synth_1000.npy          ← Synthetic ICI (1000 × 9)
│   ├── y_unit_synth_1000.npy         ← Synthetic unit labels
│   ├── y_type_synth_1000.npy         ← Synthetic coda type labels
│   ├── train_idx.npy / test_idx.npy  ← 80/20 split on clean codas (1383)
│   ├── train_id_idx.npy / test_id_idx.npy ← split on IDN-labeled subset (762)
│   ├── phase1_results.csv            ← Baseline results (load in Phase 3)
│   └── synthetic_meta.csv            ← Metadata for synthetic codas
├── wham/                             ← Project-CETI/wham repo
└── wham_env/                         ← WhAM Python virtualenv
```

## Critical data facts

- **1,501 DSWP audio files** in `datasets/dswp_audio/`, named `1.wav` through `1501.wav`
- **1,383 clean codas** after filtering `is_noise=0` from `dswp_labels.csv`
- **Split indices** (`train_idx.npy` / `test_idx.npy`) index into the **clean 1383-coda array**, not the full 1501. Always filter noise first, then apply these indices.
- **Unit distribution**: A=273, D=336, F=892 — severe imbalance. Use **macro-F1** as primary metric everywhere, not accuracy.
- **Individual ID experiments**: only 762 codas have a known individual ID (IDN≠0). Use `train_id_idx.npy` / `test_id_idx.npy` for these.
- **Mel parameters**: 64 bins, fmax=8000 Hz, 128 time frames, sr=22050
- **ICI normalisation**: StandardScaler (mean≈177ms, std≈88.6ms); zero-pad sequences to length 9

## WhAM embedding structure

`wham_embeddings_all_layers.npy` shape: `(1501, 20, 1280)` — all 1501 codas × 20 transformer layers × 1280-dim embedding. Layer indexing is 0-based. L10 = index 10 (best for individual ID), L19 = index 19 (best for social unit).

## Key results to reproduce (comparison targets)

| Task | Baseline to beat | Source |
|---|---|---|
| Social Unit Macro-F1 | 0.895 | WhAM L19 (Phase 2) |
| Individual ID Macro-F1 | 0.454 | WhAM L10 (Phase 2) |
| Coda Type Macro-F1 | 0.931 | Raw ICI logistic regression (Phase 1) |
| DCCE-full IndivID F1 | 0.834 | Phase 3 confirmed result |

## Output conventions

- Save figures to `figures/phase{N}/` (create dirs as needed)
- Save reports to `reports/phase{N}.md`
- Use `random_state=42` everywhere
- Stratify all train/test splits by `unit`
- Use `class_weight='balanced'` for all logistic regression probes
