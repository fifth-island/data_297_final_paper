"""
Generates phase4_synthetic_aug.ipynb
Run once: python3 build_phase4_notebook.py
Kernel: wham-env (has PyTorch, vampnet, librosa, pandas, soundfile)
"""
import json, os, random, string

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "phase4_synthetic_aug.ipynb")

def md(source): return {"cell_type":"markdown","id":None,"metadata":{},"source":source}
def code(source): return {"cell_type":"code","id":None,"execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Phase 4 — Experiment 2: Synthetic Data Augmentation
## *Beyond WhAM* · CS 297 Final Paper · April 2026

---

This notebook runs the second experiment: **can WhAM-generated synthetic codas improve \
DCCE classification, particularly for individual ID — the data-scarce task?**

We test the hypothesis that WhAM, as a generative model of sperm whale vocalisations, \
can act as a *domain-specific data augmentor* to supplement the limited DSWP training set.

### Experimental design

| N_synth | Synthetic codas added | Total D_train |
|---|---|---|
| 0 | None (Phase 3 baseline) | 1,106 |
| 100 | 100 new codas | 1,206 |
| 500 | 500 new codas | 1,606 |
| 1,000 | 1,000 new codas | 2,106 |

For each N_synth:
1. Sample N_synth prompt codas from D_train (stratified by unit: ~⅓ A, ⅓ D, ⅓ F)
2. Generate synthetic coda via WhAM coarse_vamp (80% random mask — unit-conditional)
3. Assign pseudo-labels: unit and coda type from the prompt coda
4. Retrain DCCE-full on D_train ∪ D_synth
5. Evaluate on **real-only** D_test (same split indices as all prior phases)

**Key metric**: Individual ID macro-F1 — most sensitive to additional training data \
(762 IDN-labeled codas; 12 classes; Phase 3 best: 0.731).

### Why this matters (Sharma et al. 2024; Goldwasser et al. 2023)

Sperm whale field data is expensive to collect: each unit has only 273–892 recorded \
codas. If WhAM faithfully captures unit-level acoustic identity, its generated codas \
could serve as a cheap expansion of training sets for downstream classification.

> **Novel contribution**: This is the first controlled study of WhAM as a data \
> augmentor for cetacean bioacoustics.

### Pseudo-label strategy (Tarvainen & Valpola, 2017)

For synthetic coda *i* generated from prompt coda *p*:
- **Unit label**: copied from unit(*p*) — preserved by unit-conditional generation
- **Coda type label**: copied from type(*p*) — used for the rhythm auxiliary head
- **ICI sequence (pseudo)**: copied from ICI(*p*) — the prompt's timing pattern; \
  WhAM generates mel-level structure but click timing structure is influenced by \
  the conditioning tokens
- **Individual ID**: **not assigned** — id_label head is masked for synthetic codas \
  (generation cannot preserve within-unit individual identity)
"""))

# ── SECTION 1: SETUP ──────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""\
import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import librosa
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

HERE      = os.path.abspath(".")
BASE      = HERE if os.path.isdir(os.path.join(HERE, "datasets")) else os.path.dirname(HERE)
if not os.path.isdir(os.path.join(BASE, "datasets")):
    raise FileNotFoundError(f"Could not locate datasets/ from working directory: {HERE}")
DATA      = os.path.join(BASE, "datasets")
AUDIO     = os.path.join(DATA, "dswp_audio")
SYNTH_DIR = os.path.join(DATA, "synthetic_audio")
FIGS      = os.path.join(BASE, "figures", "phase4")
WHAM_DIR  = os.path.join(BASE, "wham")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)

UNIT_COLORS = {"A": "#4C72B0", "D": "#DD8452", "F": "#55A868"}
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DEVICE = ("mps"  if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {DEVICE}")
print(f"PyTorch     : {torch.__version__}")
"""))

# ── SECTION 2: DATA LOADING ───────────────────────────────────────────────────
cells.append(md("""\
---
## 2. Data Loading

Identical loading pipeline to Phase 3 — same splits, same feature arrays, same \
label encoders. This ensures all N_synth conditions are evaluated on a common test set.
"""))

cells.append(code("""\
# ── Labels ────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "dswp_labels.csv"))
df["ici_list"] = df["ici_sequence"].apply(
    lambda s: [float(x) for x in s.split("|")] if isinstance(s, str) and s else [])

df_clean = df[df["is_noise"] == 0].copy().reset_index(drop=True)

df_id_all = df_clean[df_clean["individual_id"] != "0"].copy()
id_counts  = df_id_all["individual_id"].value_counts()
df_id = df_id_all[df_id_all["individual_id"].isin(
    id_counts[id_counts > 1].index)].copy().reset_index(drop=True)

# ── Shared splits ──────────────────────────────────────────────────────────────
train_idx    = np.load(os.path.join(DATA, "train_idx.npy"))
test_idx     = np.load(os.path.join(DATA, "test_idx.npy"))
train_id_idx = np.load(os.path.join(DATA, "train_id_idx.npy"))
test_id_idx  = np.load(os.path.join(DATA, "test_id_idx.npy"))

# ── ICI features (zero-padded, StandardScaler normalised) ─────────────────────
MAX_ICI = 9

def build_ici_matrix(data):
    X = np.zeros((len(data), MAX_ICI), dtype=np.float32)
    for i, row in enumerate(data.itertuples()):
        for j, v in enumerate(row.ici_list[:MAX_ICI]):
            X[i, j] = v
    return X

X_ici_all = build_ici_matrix(df_clean)
scaler_ici = StandardScaler()
X_ici_all  = scaler_ici.fit_transform(X_ici_all).astype(np.float32)

# ── Mel spectrograms (full 2D, pre-computed in Phase 3) ───────────────────────
X_mel_full = np.load(os.path.join(DATA, "X_mel_full.npy"))   # (1383, 64, 128)
N_MELS, N_FRAMES = 64, 128

# ── Label encoders ────────────────────────────────────────────────────────────
le_unit = LabelEncoder().fit(df_clean["unit"])
le_type = LabelEncoder().fit(df_clean["coda_type"])
le_id   = LabelEncoder().fit(df_id["individual_id"])

n_units = len(le_unit.classes_)
n_types = len(le_type.classes_)
n_ids   = len(le_id.classes_)

y_unit  = le_unit.transform(df_clean["unit"])
y_type  = le_type.transform(df_clean["coda_type"])
y_id    = le_id.transform(df_id["individual_id"])

id_in_clean = df_id["coda_id"].apply(
    lambda c: df_clean[df_clean["coda_id"] == c].index[0]).values

# Unit train indices lookup
unit_indices_train = {u: np.where(y_unit[train_idx] == le_unit.transform([u])[0])[0]
                      for u in le_unit.classes_}

print(f"Clean codas     : {len(df_clean)}")
print(f"IDN-labeled     : {len(df_id)}  ({n_ids} individuals)")
print(f"Mel shape       : {X_mel_full.shape}")
print(f"Train / Test    : {len(train_idx)} / {len(test_idx)}")
print(f"\\nUnit distribution in training set:")
for u in le_unit.classes_:
    n = unit_indices_train[u].shape[0]
    print(f"  Unit {u}: {n:4d} ({100*n/len(train_idx):.1f}%)")
"""))

# ── SECTION 3: DCCE ARCHITECTURE ─────────────────────────────────────────────
cells.append(md("""\
---
## 3. DCCE Architecture

Identical to Phase 3 — redefined here for notebook self-containment. \
See Phase 3 (§3–§4) for architecture rationale and loss derivation \
(Leitão et al., 2023; Beguš et al., 2024; Chen et al., 2020).
"""))

cells.append(code("""\
class RhythmEncoder(nn.Module):
    \"\"\"2-layer GRU on ICI sequence → 64-d embedding (Leitão et al., 2023).\"\"\"
    def __init__(self, hidden_dim=64, out_dim=64):
        super().__init__()
        self.gru  = nn.GRU(1, hidden_dim, num_layers=2, batch_first=True,
                           dropout=0.2, bidirectional=False)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        _, h = self.gru(x.unsqueeze(-1))   # (B,9) → (B,9,1) → h: (2,B,64)
        return self.proj(h[-1])            # last-layer hidden → (B,64)


class SpectralEncoder(nn.Module):
    \"\"\"3-block CNN on mel-spectrogram (64×128) → 64-d embedding (Beguš et al., 2024).\"\"\"
    def __init__(self, n_mels=64, n_frames=128, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout(0.3)
        flat_dim  = 64 * (n_mels // 8) * (n_frames // 8)
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))       # (B,64,128) → (B,64,8,16)
        return self.proj(self.drop(x.flatten(1)))


class DCCE(nn.Module):
    \"\"\"Dual-Channel Contrastive Encoder (full cross-channel variant).\"\"\"
    def __init__(self, n_types=22, n_ids=12, emb_dim=64):
        super().__init__()
        self.rhythm_enc   = RhythmEncoder(out_dim=emb_dim)
        self.spectral_enc = SpectralEncoder(out_dim=emb_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.head_type = nn.Linear(emb_dim, n_types)
        self.head_id   = nn.Linear(emb_dim, n_ids)

    def forward(self, ici, mel):
        r = self.rhythm_enc(ici)
        s = self.spectral_enc(mel)
        z = self.fusion(torch.cat([r, s], dim=1))
        return z, r, s


def nt_xent_loss(z1, z2, temperature=0.07):
    \"\"\"NT-Xent contrastive loss (Chen et al., SimCLR, NeurIPS 2020). τ=0.07.\"\"\"
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), float('-inf'))
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


_m = DCCE(n_types=n_types, n_ids=n_ids).to(DEVICE)
n_params = sum(p.numel() for p in _m.parameters())
print(f"DCCE architecture OK  |  {n_params:,} parameters"); del _m
"""))

# ── SECTION 4: DATASETS AND LOADER ───────────────────────────────────────────
cells.append(md("""\
---
## 4. Datasets and DataLoader

`CodaDataset` — wraps real DSWP codas (real ICI + mel + ground-truth labels).

`SyntheticCodaDataset` — wraps WhAM-generated codas. Pseudo-ICI and pseudo-type \
are copied from the conditioning prompt coda. Individual ID is **not** set \
(id_label = −1), so the auxiliary ID head is not trained on synthetic examples.

`make_loader` builds a `WeightedRandomSampler` for unit balance across the \
combined real + synthetic training pool (compensates for Unit F = 59.4% of real data).
"""))

cells.append(code("""\
class CodaDataset(Dataset):
    \"\"\"Real DSWP codas: (ici, mel, unit, type, id_label).\"\"\"
    def __init__(self, indices, X_ici, X_mel, y_unit, y_type,
                 y_id=None, id_clean_idx=None):
        self.idx = indices; self.X_ici = X_ici; self.X_mel = X_mel
        self.y_unit = y_unit; self.y_type = y_type
        self.y_id = y_id; self.id_clean_idx = id_clean_idx

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ci  = self.idx[i]
        ici = torch.tensor(self.X_ici[ci])
        mel = torch.tensor(self.X_mel[ci])
        u   = torch.tensor(self.y_unit[ci], dtype=torch.long)
        t   = torch.tensor(self.y_type[ci], dtype=torch.long)
        id_lbl = torch.tensor(-1, dtype=torch.long)
        if self.y_id is not None and self.id_clean_idx is not None:
            m = np.where(self.id_clean_idx == ci)[0]
            if len(m) > 0:
                id_lbl = torch.tensor(self.y_id[m[0]], dtype=torch.long)
        return ici, mel, u, t, id_lbl


class SyntheticCodaDataset(Dataset):
    \"\"\"
    WhAM-generated synthetic codas. Pseudo-labels from the prompt coda.
    Individual ID is not assigned (id_label = -1).
    \"\"\"
    def __init__(self, X_ici_synth, X_mel_synth, y_unit_synth, y_type_synth):
        self.X_ici  = X_ici_synth    # (N, 9)   — pseudo-ICI (from prompt)
        self.X_mel  = X_mel_synth    # (N, 64, 128)
        self.y_unit = y_unit_synth   # (N,)
        self.y_type = y_type_synth   # (N,)

    def __len__(self): return len(self.y_unit)

    def __getitem__(self, i):
        return (torch.tensor(self.X_ici[i]),
                torch.tensor(self.X_mel[i]),
                torch.tensor(self.y_unit[i], dtype=torch.long),
                torch.tensor(self.y_type[i], dtype=torch.long),
                torch.tensor(-1, dtype=torch.long))


BATCH_SIZE = 64

def make_loader(real_ds, synth_ds=None):
    \"\"\"Combined DataLoader with WeightedRandomSampler for unit balance.\"\"\"
    combined = ConcatDataset([real_ds, synth_ds]) if synth_ds else real_ds
    n_total  = len(combined)

    # Collect unit labels for the entire combined pool
    real_units  = [real_ds[i][2].item() for i in range(len(real_ds))]
    synth_units = ([synth_ds[i][2].item() for i in range(len(synth_ds))]
                   if synth_ds else [])
    all_units   = np.array(real_units + synth_units)

    counts  = np.bincount(all_units, minlength=n_units).astype(float)
    weights = (1.0 / counts)[all_units]
    sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
    return DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                      num_workers=0, drop_last=True)


# ── Real datasets ──────────────────────────────────────────────────────────────
ds_real_train = CodaDataset(train_idx, X_ici_all, X_mel_full, y_unit, y_type,
                            y_id, id_in_clean)
ds_test       = CodaDataset(test_idx,  X_ici_all, X_mel_full, y_unit, y_type)
loader_real   = make_loader(ds_real_train)

print(f"Real training dataset   : {len(ds_real_train)} codas  |  {len(loader_real)} batches")
print(f"Test  dataset           : {len(ds_test)} codas")
"""))

# ── SECTION 5: TRAINING AND EVALUATION HELPERS ───────────────────────────────
cells.append(md("""\
---
## 5. Training and Evaluation Helpers
"""))

cells.append(code("""\
def build_unit_train_map():
    \"\"\"Unit encoded-int → positions-within-train_idx lookup for partner sampling.\"\"\"
    return {u_enc: np.where(y_unit[train_idx] == u_enc)[0]
            for u_enc in range(n_units)}


def sample_partners(u_batch, unit_map, rng):
    \"\"\"For each unit label in the batch, sample a different real training coda index.\"\"\"
    partners = []
    for u in u_batch.cpu().numpy():
        candidates = unit_map[int(u)]
        partners.append(train_idx[rng.choice(candidates)])
    return np.array(partners)


def train_dcce_full(train_loader, n_epochs=50, lr=1e-3, lambda1=0.5, lambda2=0.5,
                    label=""):
    \"\"\"
    Train DCCE-full (cross-channel NT-Xent + auxiliary heads).
    Cross-channel partners are always sampled from the *real* training pool,
    regardless of whether the loader contains synthetic codas.

    Returns: (model, loss_history)
    \"\"\"
    # Fix seed before model construction so each N_synth condition
    # starts from the same initialisation — ensures comparability.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model    = DCCE(n_types=n_types, n_ids=n_ids).to(DEVICE)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    rng      = np.random.default_rng(SEED)
    unit_map = build_unit_train_map()
    history  = []

    model.train()
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0; nb = 0
        for ici, mel, u_lbl, t_lbl, id_lbl in train_loader:
            ici   = ici.to(DEVICE);   mel = mel.to(DEVICE)
            u_lbl = u_lbl.to(DEVICE); t_lbl = t_lbl.to(DEVICE)

            z, r_emb, s_emb = model(ici, mel)

            # Cross-channel partners from the real training pool (unit-matched)
            p_idx = sample_partners(u_lbl, unit_map, rng)
            z_p, _, _ = model(torch.tensor(X_ici_all[p_idx]).to(DEVICE),
                              torch.tensor(X_mel_full[p_idx]).to(DEVICE))

            loss = nt_xent_loss(z, z_p)
            loss = loss + lambda1 * F.cross_entropy(model.head_type(r_emb), t_lbl)

            valid_id = id_lbl >= 0
            if valid_id.any():
                loss = loss + lambda2 * F.cross_entropy(
                    model.head_id(s_emb[valid_id.to(DEVICE)]),
                    id_lbl[valid_id].to(DEVICE))

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); nb += 1

        sched.step()
        history.append(epoch_loss / nb)
        if (epoch + 1) % 10 == 0:
            print(f"  {label}  epoch {epoch+1:3d}/{n_epochs}  "
                  f"loss={history[-1]:.4f}  ({time.time()-t0:.0f}s)")

    print(f"  {label} done in {time.time()-t0:.0f}s")
    return model, history


def extract_z(model, indices):
    \"\"\"Extract fused embeddings z for given df_clean indices.\"\"\"
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(indices), BATCH_SIZE):
            bi = indices[i:i + BATCH_SIZE]
            z, _, _ = model(torch.tensor(X_ici_all[bi]).to(DEVICE),
                            torch.tensor(X_mel_full[bi]).to(DEVICE))
            out.append(z.cpu().numpy())
    return np.vstack(out)


def linear_probe(model):
    \"\"\"
    Frozen linear probe evaluation: LogisticRegression on z.
    Tasks: unit, coda_type, individual_id.
    Primary metric: macro-F1 (class-imbalance robust).
    \"\"\"
    Z_tr = extract_z(model, train_idx)
    Z_te = extract_z(model, test_idx)

    id_tr_ci = id_in_clean[train_id_idx]
    id_te_ci = id_in_clean[test_id_idx]
    Z_id_tr  = extract_z(model, id_tr_ci)
    Z_id_te  = extract_z(model, id_te_ci)

    sc    = StandardScaler(); sc_id = StandardScaler()
    Xtr   = sc.fit_transform(Z_tr);   Xte   = sc.transform(Z_te)
    Xid_tr= sc_id.fit_transform(Z_id_tr); Xid_te= sc_id.transform(Z_id_te)

    make_lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=SEED, solver="lbfgs")
    res = {}

    clf = make_lr().fit(Xtr, y_unit[train_idx])
    p   = clf.predict(Xte)
    res["unit"] = {"macro_f1": f1_score(y_unit[test_idx], p, average="macro", zero_division=0),
                   "accuracy": accuracy_score(y_unit[test_idx], p)}

    clf = make_lr().fit(Xtr, y_type[train_idx])
    p   = clf.predict(Xte)
    res["coda_type"] = {"macro_f1": f1_score(y_type[test_idx], p, average="macro", zero_division=0),
                        "accuracy": accuracy_score(y_type[test_idx], p)}

    clf = make_lr().fit(Xid_tr, y_id[train_id_idx])
    p   = clf.predict(Xid_te)
    res["individual_id"] = {"macro_f1": f1_score(y_id[test_id_idx], p, average="macro", zero_division=0),
                            "accuracy": accuracy_score(y_id[test_id_idx], p)}
    return res
"""))

# ── SECTION 6: BASELINE TRAINING (N_synth = 0) ───────────────────────────────
cells.append(md("""\
---
## 6. Baseline DCCE-full Training (N_synth = 0)

We re-run DCCE-full on the real training set to obtain a clean baseline \
within this notebook's code path (same hyperparameters, same random seed as Phase 3).
"""))

cells.append(code("""\
print("Training DCCE-full on real data only (N_synth=0) ...")
model_baseline, hist_baseline = train_dcce_full(loader_real, label="[N_synth=0]")
"""))

cells.append(code("""\
baseline_res = linear_probe(model_baseline)
print("Baseline (N_synth=0) results:")
for task, r in baseline_res.items():
    print(f"  {task:25s}  F1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")
"""))

# ── SECTION 7: WHAM GENERATION SETUP ─────────────────────────────────────────
cells.append(md("""\
---
## 7. WhAM Generation Setup

We load the WhAM coarse model and codec from the local clone of the \
Project-CETI/wham repository (Paradise et al., NeurIPS 2025). \
Only the coarse model + codec are needed for audio generation and decoding; \
the coarse-to-fine (c2f) model is not required.

### Masking strategy

We use `rand_mask_intensity=0.8` — 80% of discrete acoustic tokens are randomly \
masked and regenerated by the model, while 20% are preserved from the conditioning \
prompt. This provides a weak unit-level conditioning signal: the preserved tokens \
carry spectral information from the prompt coda's unit, guiding the generation \
toward the same unit's acoustic texture.

With 100% masking (unconditional generation), outputs would sample from WhAM's \
overall prior with no unit specificity. With lower masking (e.g. 30%), outputs \
would be near-reconstructions of the prompt, adding little new variation. \
80% is a reasonable compromise for *diverse but unit-conditional* synthesis.
"""))

cells.append(code("""\
# ── Add WhAM repo to Python path ──────────────────────────────────────────────
sys.path.insert(0, WHAM_DIR)

from vampnet.interface import Interface
from vampnet import mask as pmask
from audiotools import AudioSignal

COARSE_CKPT = os.path.join(WHAM_DIR, "vampnet", "models", "coarse.pth")
CODEC_CKPT  = os.path.join(WHAM_DIR, "vampnet", "models", "codec.pth")

print("Loading WhAM Interface (coarse + codec)...")
t0 = time.time()
interface = Interface(
    coarse_ckpt=COARSE_CKPT,
    codec_ckpt=CODEC_CKPT,
    device=DEVICE,
)
print(f"  Interface loaded in {time.time()-t0:.1f}s  |  device={DEVICE}")
"""))

cells.append(code("""\
# ── Generation function (uses vampnet.mask API — Paradise et al., 2025) ───────
def generate_synthetic_coda(prompt_coda_id, rand_mask_intensity=0.8,
                             n_steps=30, seed=None):
    \"\"\"
    Generate one synthetic coda conditioned on a real prompt coda.

    Encodes the prompt into discrete acoustic tokens, then applies the WhAM
    'codas' mask recipe (periodic_p=12, onset_mask_width=21, n_mask_codebooks=9)
    with rand_mask_intensity=0.8 (80% of tokens regenerated, 20% preserved
    for weak unit-level conditioning).

    Masking pipeline (adapted from Paradise et al., 2025 codas config):
      1. pmask.linear_random(z, 0.8)       — random 80% mask
      2. pmask.mask_and(periodic_mask)      — periodic structure (period=12)
      3. pmask.codebook_unmask(mask, 0)     — ncc=0: no protected codebooks
      4. pmask.codebook_mask(mask, 9)       — mask fine codebooks (9+)
    Note: onset_mask omitted (requires madmom/Cython build not available here).

    Args:
        prompt_coda_id    : integer coda ID (file: {id}.wav)
        rand_mask_intensity: fraction of tokens masked (0=no mask, 1=fully random)
        n_steps           : VampNet sampling steps (30 fast / 50 higher quality)
        seed              : RNG seed for reproducibility

    Returns:
        (audio_np: float32 ndarray, sample_rate: int)
    \"\"\"
    audio_path = os.path.join(AUDIO, f"{prompt_coda_id}.wav")
    sig = AudioSignal(audio_path)
    sig = interface.preprocess(sig)     # resample to model SR

    with torch.no_grad():
        z = interface.encode(sig)

        # ── Build mask (codas config, rand_mask_intensity override) ───────
        # API: periodic_mask(x, period, width, random_roll)
        #      codebook_mask(mask, start) — masks codebooks from index start onwards
        # Note: onset_mask (codas config: width=21) requires madmom which needs
        # Cython build infrastructure; omitted here. periodic_mask preserves
        # periodic temporal structure at every 12th token (period=12).
        mask = pmask.linear_random(z, rand_mask_intensity)
        mask = pmask.mask_and(mask, pmask.periodic_mask(
            z, 12, 1, random_roll=True))
        mask = pmask.codebook_unmask(mask, n_conditioning_codebooks=0)
        mask = pmask.codebook_mask(mask, start=9)

        zv = interface.coarse_vamp(
            z,
            mask=mask,
            sampling_steps=n_steps,
            mask_temperature=15.0,        # masktemp=1.5 * 10 (WhAM convention)
            sampling_temperature=1.0,
            gen_fn=interface.coarse.generate,
            seed=seed,
            sample_cutoff=0.17,
            typical_filtering=False,
            typical_mass=0.102,
            typical_min_tokens=47,
        )
        out_sig = interface.to_signal(zv)

    audio_np = out_sig.samples.squeeze().cpu().numpy().astype(np.float32)
    return audio_np, out_sig.sample_rate
"""))

cells.append(code("""\
# ── Generation sanity check (~8-10s on MPS) ───────────────────────────────────
test_coda_id = int(df_clean.iloc[train_idx[0]]["coda_id"])
print(f"Generating test coda from prompt coda {test_coda_id} ...")
t0 = time.time()
test_audio, test_sr = generate_synthetic_coda(test_coda_id, seed=0)
elapsed = time.time() - t0
print(f"  Generated {len(test_audio)/test_sr:.2f}s audio in {elapsed:.1f}s")
print(f"  Sample rate: {test_sr} Hz  |  Samples: {len(test_audio)}")
print(f"  Audio range: [{test_audio.min():.3f}, {test_audio.max():.3f}]")
print(f"\\n  Estimated time for 1000 codas: {1000*elapsed/60:.1f} min")
"""))

# ── SECTION 8: BULK GENERATION ───────────────────────────────────────────────
cells.append(md("""\
---
## 8. Bulk Synthetic Coda Generation

We generate N_SYNTH_MAX=1000 synthetic codas and cache them to disk. \
This generation step only runs once; subsequent notebook runs load from cache.

**Prompt sampling strategy**: prompts are stratified by unit (⌊N/3⌋ per unit) \
to ensure the synthetic pool is balanced across A, D, and F — unlike the real \
training set which is dominated by Unit F (59.4%).
"""))

cells.append(code("""\
N_SYNTH_MAX   = 1000
SYNTH_META_PATH = os.path.join(DATA, "synthetic_meta.csv")


def load_or_generate_synthetics(n_target, force_regen=False):
    \"\"\"
    Generate n_target synthetic codas from WhAM, or load from disk cache.

    Prompts are sampled from D_train, stratified by unit.
    For each synthetic coda we save:
      - WAV file  → datasets/synthetic_audio/synth_{i:04d}.wav
      - Mel       → datasets/X_mel_synth_{n}.npy  (n_target, 64, 128)
      - Pseudo-ICI→ datasets/X_ici_synth_{n}.npy  (n_target, 9)   from prompt
      - Unit/type → datasets/y_unit_synth_{n}.npy, y_type_synth_{n}.npy
      - Metadata  → datasets/synthetic_meta.csv

    Returns:
        meta_df, X_ici_s, X_mel_s, y_unit_s, y_type_s
    \"\"\"
    mel_cache  = os.path.join(DATA, f"X_mel_synth_{n_target}.npy")
    ici_cache  = os.path.join(DATA, f"X_ici_synth_{n_target}.npy")
    unit_cache = os.path.join(DATA, f"y_unit_synth_{n_target}.npy")
    type_cache = os.path.join(DATA, f"y_type_synth_{n_target}.npy")

    all_cached = all(os.path.exists(p) for p in
                     [SYNTH_META_PATH, mel_cache, ici_cache, unit_cache, type_cache])

    if all_cached and not force_regen:
        print(f"Loading cached synthetic data ({n_target} codas)...")
        meta_df  = pd.read_csv(SYNTH_META_PATH).head(n_target)
        X_ici_s  = np.load(ici_cache)[:n_target]
        X_mel_s  = np.load(mel_cache)[:n_target]
        y_unit_s = np.load(unit_cache)[:n_target]
        y_type_s = np.load(type_cache)[:n_target]
        print(f"  Loaded: {len(meta_df)} codas, mel {X_mel_s.shape}")
        return meta_df, X_ici_s, X_mel_s, y_unit_s, y_type_s

    print(f"Generating {n_target} synthetic codas ...")
    est_min = n_target * 8 // 60
    print(f"  Estimated time: ~{est_min} min on MPS  (can leave running)")

    # ── Stratified prompt sampling ─────────────────────────────────────────
    rng_p = np.random.default_rng(SEED + 1)
    n_per_unit = n_target // n_units
    prompt_clean_idxs = []
    for u in le_unit.classes_:
        u_enc     = le_unit.transform([u])[0]
        u_pos     = unit_indices_train[u]        # positions within train_idx
        u_ci      = train_idx[u_pos]              # positions within df_clean
        chosen    = rng_p.choice(u_ci, size=n_per_unit, replace=True)
        prompt_clean_idxs.extend(chosen.tolist())
    # Remainder (handle n_target not divisible by n_units)
    remainder = n_target - len(prompt_clean_idxs)
    if remainder > 0:
        extra = rng_p.choice(train_idx, size=remainder, replace=True)
        prompt_clean_idxs.extend(extra.tolist())
    rng_p.shuffle(prompt_clean_idxs)

    # ── Allocate output arrays ─────────────────────────────────────────────
    X_ici_s  = np.zeros((n_target, MAX_ICI),           dtype=np.float32)
    X_mel_s  = np.zeros((n_target, N_MELS, N_FRAMES),  dtype=np.float32)
    y_unit_s = np.zeros(n_target,                      dtype=np.int64)
    y_type_s = np.zeros(n_target,                      dtype=np.int64)
    meta_records = []

    t0_gen = time.time()
    for i, prompt_ci in enumerate(prompt_clean_idxs):
        prompt_row = df_clean.iloc[prompt_ci]
        coda_id    = int(prompt_row["coda_id"])
        unit_str   = prompt_row["unit"]
        type_str   = prompt_row["coda_type"]
        out_path   = os.path.join(SYNTH_DIR, f"synth_{i:04d}.wav")

        # Generate or load from partial cache
        if not os.path.exists(out_path):
            audio_np, sr = generate_synthetic_coda(coda_id,
                                                   rand_mask_intensity=0.8,
                                                   n_steps=30, seed=i)
            sf.write(out_path, audio_np, sr)

        # Extract mel spectrogram from synthetic WAV
        y_wav, sr = librosa.load(out_path, sr=None, mono=True)
        mel = librosa.feature.melspectrogram(y=y_wav, sr=sr,
                                              n_mels=N_MELS, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        T = mel_db.shape[1]
        if T >= N_FRAMES:
            X_mel_s[i] = mel_db[:, :N_FRAMES]
        else:
            X_mel_s[i, :, :T] = mel_db

        # Pseudo-ICI: copy from prompt coda (already StandardScaler-normalised)
        X_ici_s[i]  = X_ici_all[prompt_ci]
        y_unit_s[i] = le_unit.transform([unit_str])[0]
        y_type_s[i] = le_type.transform([type_str])[0]

        meta_records.append({
            "synth_id": i, "prompt_coda_id": coda_id,
            "prompt_clean_idx": prompt_ci, "unit": unit_str,
            "coda_type": type_str, "audio_path": out_path,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0_gen
            rate    = elapsed / (i + 1)
            print(f"  Generated {i+1:4d}/{n_target}  "
                  f"({elapsed:.0f}s elapsed, ~{rate:.1f}s/coda, "
                  f"~{(n_target - i - 1) * rate / 60:.1f} min remaining)")

    # ── Save all caches ────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(SYNTH_META_PATH, index=False)
    np.save(mel_cache,  X_mel_s);  np.save(ici_cache,  X_ici_s)
    np.save(unit_cache, y_unit_s); np.save(type_cache, y_type_s)
    print(f"\\nGeneration complete in {time.time()-t0_gen:.0f}s. Caches saved.")
    return meta_df, X_ici_s, X_mel_s, y_unit_s, y_type_s


t0_total = time.time()
meta_df, X_ici_synth, X_mel_synth, y_unit_synth, y_type_synth = \\
    load_or_generate_synthetics(N_SYNTH_MAX)

print(f"\\nSynthetic data ready: {len(meta_df)} codas  "
      f"({time.time()-t0_total:.0f}s total)")
print(f"\\nSynthetic unit distribution:")
for u in le_unit.classes_:
    u_enc = le_unit.transform([u])[0]
    n_u   = (y_unit_synth == u_enc).sum()
    print(f"  Unit {u}: {n_u} ({100*n_u/len(y_unit_synth):.1f}%)")
"""))

# ── SECTION 9: EDA OF SYNTHETIC CODAS ────────────────────────────────────────
cells.append(md("""\
---
## 9. Exploratory Analysis of Synthetic Codas

Before training with synthetic data we visualise samples to verify that WhAM is \
generating coherent whale vocalisations, not noise or silence. We also compare \
mean mel profiles of real vs. synthetic codas per unit — a mismatch would indicate \
that generation is not unit-faithful.
"""))

cells.append(code("""\
# ── Sample mel-spectrograms: 3 synthetic codas per unit ───────────────────────
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle("WhAM-Generated Synthetic Coda Mel-Spectrograms (3 per unit)",
             fontsize=13, fontweight="bold")

rng_vis = np.random.default_rng(42)
for row_i, unit in enumerate(le_unit.classes_):
    u_enc  = le_unit.transform([unit])[0]
    u_idx  = np.where(y_unit_synth == u_enc)[0]
    chosen = rng_vis.choice(u_idx, 3, replace=False)
    for col_i, si in enumerate(chosen):
        ax = axes[row_i, col_i]
        im = ax.pcolormesh(X_mel_synth[si], cmap="viridis", shading="auto")
        ax.set_title(f"Unit {unit} — synth {si}", fontsize=9)
        ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_synth_spectrograms.png"),
            dpi=130, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# ── Mean mel profile: real vs synthetic per unit ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.suptitle("Mean Mel Profile (±1 SD): Real vs WhAM-Synthetic — by Social Unit",
             fontsize=12, fontweight="bold")

for i, unit in enumerate(le_unit.classes_):
    ax    = axes[i]
    u_enc = le_unit.transform([unit])[0]

    # Real training codas for this unit
    real_ci    = train_idx[y_unit[train_idx] == u_enc]
    real_mel   = X_mel_full[real_ci].mean(axis=2)   # (N, 64) — mean over time
    real_mean  = real_mel.mean(axis=0)
    real_std   = real_mel.std(axis=0)

    # Synthetic codas for this unit
    synth_ci   = np.where(y_unit_synth == u_enc)[0]
    synth_mel  = X_mel_synth[synth_ci].mean(axis=2) # (N, 64)
    synth_mean = synth_mel.mean(axis=0)
    synth_std  = synth_mel.std(axis=0)

    bins = np.arange(N_MELS)
    ax.fill_between(bins, real_mean - real_std,  real_mean + real_std,
                    alpha=0.25, color=UNIT_COLORS[unit])
    ax.plot(bins, real_mean,  color=UNIT_COLORS[unit], lw=2,   label="Real")
    ax.fill_between(bins, synth_mean - synth_std, synth_mean + synth_std,
                    alpha=0.20, color="gray")
    ax.plot(bins, synth_mean, color="gray",            lw=2, ls="--", label="Synthetic")

    ax.set_title(f"Unit {unit}")
    ax.set_xlabel("Mel bin (0 = low freq, 63 = high freq)")
    if i == 0: ax.set_ylabel("Mean power (dB, time-averaged)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_synth_mel_profiles.png"),
            dpi=130, bbox_inches="tight")
plt.show()
print("Note: close profile overlap = unit-faithful generation (supports augmentation).")
print("Large shape mismatch = distribution shift (would explain null augmentation result).")
"""))

# ── SECTION 10: AUGMENTED TRAINING ───────────────────────────────────────────
cells.append(md("""\
---
## 10. Augmented DCCE Training

We sweep N_synth ∈ {100, 500, 1000}. For each:
1. Subset the first N_synth synthetic codas
2. Concatenate with the real training dataset
3. Re-train DCCE-full from scratch (same 50 epochs, same seed)
4. Evaluate on real-only D_test

The N_synth=0 baseline was trained in §6.
"""))

cells.append(code("""\
N_SYNTH_VALUES = [100, 500, 1000]

augmented_models    = {}
augmented_histories = {}
augmented_results   = {}

for N_synth in N_SYNTH_VALUES:
    print(f"\\n{'='*60}")
    print(f"N_synth = {N_synth}  |  D_train = {len(train_idx) + N_synth}")
    print(f"{'='*60}")

    synth_ds   = SyntheticCodaDataset(
        X_ici_synth[:N_synth], X_mel_synth[:N_synth],
        y_unit_synth[:N_synth], y_type_synth[:N_synth])
    loader_aug = make_loader(ds_real_train, synth_ds)

    print(f"  DataLoader: {len(loader_aug)} batches of {BATCH_SIZE}")

    model, hist = train_dcce_full(loader_aug, label=f"[N_synth={N_synth}]")
    augmented_models[N_synth]    = model
    augmented_histories[N_synth] = hist

    res = linear_probe(model)
    augmented_results[N_synth] = res
    print(f"\\n  Evaluation (N_synth={N_synth}):")
    for task, r in res.items():
        print(f"    {task:25s}  F1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")

print("\\nAll augmented experiments complete.")
"""))

# ── SECTION 11: RESULTS TABLE ─────────────────────────────────────────────────
cells.append(md("""\
---
## 11. Results Summary
"""))

cells.append(code("""\
# ── Collect all results ────────────────────────────────────────────────────────
all_n   = [0] + N_SYNTH_VALUES
all_res = {0: baseline_res}
all_res.update(augmented_results)

rows = []
for n in all_n:
    r = all_res[n]
    rows.append({
        "N_synth"     : n,
        "D_train"     : len(train_idx) + n,
        "Unit F1"     : f"{r['unit']['macro_f1']:.4f}",
        "CodaType F1" : f"{r['coda_type']['macro_f1']:.4f}",
        "IndivID F1"  : f"{r['individual_id']['macro_f1']:.4f}",
        "Unit Acc"    : f"{r['unit']['accuracy']:.4f}",
        "IndivID Acc" : f"{r['individual_id']['accuracy']:.4f}",
    })

results_df = pd.DataFrame(rows)
print("Phase 4 results:")
print(results_df.to_string(index=False))
print()
print("── Phase 2-3 references ─────────────────────────────────────")
print("  WhAM L19     Unit F1    = 0.895")
print("  WhAM L10     IndivID F1 = 0.454")
print("  DCCE-full    Unit F1    (Phase 3 best — compare row N_synth=0)")
print("  DCCE-full    IndivID F1 (Phase 3 best — target to beat with augmentation)")
"""))

# ── SECTION 12: VISUALIZATIONS ───────────────────────────────────────────────
cells.append(md("""\
---
## 12. Visualizations
"""))

cells.append(code("""\
# ── Augmentation curves: macro-F1 vs N_synth ──────────────────────────────────
tasks        = ["unit", "coda_type", "individual_id"]
task_labels  = ["Social Unit", "Coda Type (22)", "Individual ID"]
task_colors  = ["#4C72B0", "#DD8452", "#55A868"]
ref_lines    = [("WhAM L19", 0.895, "#888888", "--"),
                None,
                ("WhAM L10", 0.4535, "#CC6600", ":")]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("DCCE-full: Macro-F1 vs. N_synth (WhAM-generated synthetic codas)",
             fontsize=12, fontweight="bold")

for ax, task, label, color, ref in zip(axes, tasks, task_labels,
                                        task_colors, ref_lines):
    f1_vals = [all_res[n][task]["macro_f1"] for n in all_n]
    ax.plot(all_n, f1_vals, "o-", color=color, lw=2.5, ms=8,
            label="DCCE-full + D_synth")
    for x, y in zip(all_n, f1_vals):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    if ref is not None:
        ref_label, ref_val, ref_color, ref_ls = ref
        ax.axhline(ref_val, ls=ref_ls, color=ref_color, lw=1.5,
                   alpha=0.8, label=f"{ref_label} ({ref_val:.3f})")

    ax.set_xlabel("N_synth (synthetic codas added to D_train)")
    ax.set_ylabel("Macro-F1 (real D_test)")
    ax.set_title(f"({'abc'[tasks.index(task)]}) {label}")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(all_n)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_augmentation_curve.png"),
            dpi=130, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# ── Training loss curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist_baseline, lw=2, color="#AAAAAA", ls="--", label="N_synth=0 (baseline)")
aug_colors = {100: "#4C72B0", 500: "#DD8452", 1000: "#55A868"}
for n, hist in augmented_histories.items():
    ax.plot(hist, lw=2, color=aug_colors[n], label=f"N_synth={n}")

ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("DCCE-full Training Loss — Real vs Augmented (50 epochs, CosineAnnealingLR)")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_aug_training_curves.png"),
            dpi=130, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# ── UMAP / t-SNE: real vs synthetic embeddings in DCCE space ──────────────────
best_n = max(all_n, key=lambda n: all_res[n]["individual_id"]["macro_f1"])
best_model = augmented_models.get(best_n, model_baseline)
print(f"Best N_synth for IndivID F1: {best_n}  "
      f"(F1={all_res[best_n]['individual_id']['macro_f1']:.4f})")

# Real test embeddings
Z_real  = extract_z(best_model, test_idx)
u_real  = y_unit[test_idx]

# Synthetic embeddings
if best_n > 0:
    best_model.eval()
    Z_synth_parts = []
    X_ici_s = X_ici_synth[:best_n]; X_mel_s = X_mel_synth[:best_n]
    with torch.no_grad():
        for i in range(0, best_n, BATCH_SIZE):
            z, _, _ = best_model(
                torch.tensor(X_ici_s[i:i+BATCH_SIZE]).to(DEVICE),
                torch.tensor(X_mel_s[i:i+BATCH_SIZE]).to(DEVICE))
            Z_synth_parts.append(z.cpu().numpy())
    Z_synth  = np.vstack(Z_synth_parts)
    u_synth  = y_unit_synth[:best_n]
    Z_all    = np.vstack([Z_real, Z_synth])
    u_all    = np.concatenate([u_real, u_synth])
    sources  = np.array(["real"] * len(Z_real) + ["synthetic"] * len(Z_synth))
else:
    Z_all = Z_real; u_all = u_real
    sources = np.array(["real"] * len(Z_real))

sc_dim = StandardScaler()
Z_sc   = sc_dim.fit_transform(Z_all)

try:
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30,
                            min_dist=0.1, random_state=SEED)
    method  = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=SEED, perplexity=30)
    method  = "t-SNE"

coords = reducer.fit_transform(Z_sc)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"{method} of DCCE Embeddings: Real vs Synthetic "
             f"(N_synth={best_n}, D_test + D_synth)", fontsize=12, fontweight="bold")

# Left: by unit
ax = axes[0]
for u_enc, u_str in enumerate(le_unit.classes_):
    is_unit  = u_all == u_enc
    is_real  = sources == "real"
    is_synth = sources == "synthetic"
    if (is_unit & is_real).any():
        ax.scatter(coords[is_unit & is_real, 0], coords[is_unit & is_real, 1],
                   c=UNIT_COLORS[u_str], s=20, alpha=0.7, label=f"Unit {u_str} (real)")
    if (is_unit & is_synth).any():
        ax.scatter(coords[is_unit & is_synth, 0], coords[is_unit & is_synth, 1],
                   c=UNIT_COLORS[u_str], s=25, alpha=0.45, marker="x",
                   label=f"Unit {u_str} (synth)")
ax.set_title("(a) Coloured by social unit"); ax.legend(fontsize=7, ncol=2); ax.axis("off")

# Right: by source
ax = axes[1]
ax.scatter(coords[sources == "real", 0],       coords[sources == "real", 1],
           c="#4C72B0", s=18, alpha=0.65, label="Real (D_test)")
if best_n > 0:
    ax.scatter(coords[sources == "synthetic", 0], coords[sources == "synthetic", 1],
               c="#DD8452", s=18, alpha=0.50, marker="x", label="Synthetic (WhAM)")
ax.set_title("(b) Coloured by source (real / synthetic)")
ax.legend(fontsize=9); ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_aug_umap.png"), dpi=130, bbox_inches="tight")
plt.show()
print(f"\\nInterpretation: if synthetic codas (×) cluster with real codas (•) of the")
print(f"same unit, WhAM generation is unit-faithful — supporting the augmentation claim.")
"""))

# ── SECTION 13: DISCUSSION ────────────────────────────────────────────────────
cells.append(md("""\
---
## 13. Discussion and Paper Interpretation

### What we found

The augmentation curve (§12) tells us whether WhAM's synthetic data carries \
useful unit-level structure for DCCE training:

| Outcome | Interpretation |
|---|---|
| F1 **increases** with N_synth | WhAM generates unit-faithful codas; augmentation improves individual ID classification |
| F1 **flat** with N_synth | Synthetic codas replicate existing patterns — no new information for DCCE |
| F1 **decreases** with N_synth | Synthetic codas introduce distribution shift; pseudo-labels are noisy |

### Connections to the paper's core claim

This experiment directly tests Experiment 2 of the paper:

> *"Is WhAM useful not just as a feature extractor but as a domain-specific \
> data augmentor for cetacean bioacoustics?"*

If augmentation **works** → WhAM's generative distribution is biologically \
structured at the unit level, providing independent evidence for the representations \
found in Phase 2 (WhAM probing) and Phase 3 (DCCE).

If augmentation **fails** → This is also publishable: it constrains what WhAM's \
coarse model has learned. Fine-grained individual identity may require the c2f \
model or direct conditioning on spectral formant features (Beguš et al., 2024).

### Cross-experiment consistency check

Compare the embedding UMAP (§12, right panel) against the Phase 3 UMAP \
(Fig. fig_dcce_umap.png). If synthetic codas occupy similar embedding-space \
regions as real codas of the same unit, then the DCCE representation is stable \
under augmentation — a sign that the model is learning generalisable features, \
not overfitting to the specific audio distribution of the real training set.

### Limitations

1. **Pseudo-ICI**: The ICI sequence assigned to synthetic codas is copied from \
   the prompt, not extracted from the generated audio. A proper click detector \
   (Gubnitsky et al., 2024) could provide ground-truth ICI for the synthetic WAVs.
2. **80% masking is a heuristic**: The optimal conditioning strength is unknown. \
   An ablation over mask intensity (e.g., 30%, 50%, 80%, 100%) would clarify \
   this design choice.
3. **Coarse model only**: WhAM's c2f (coarse-to-fine) model adds fine-grained \
   spectral detail. We used coarse-only generation for computational budget reasons; \
   c2f generation may produce more unit-faithful outputs.
"""))

cells.append(code("""\
# ── Save results ──────────────────────────────────────────────────────────────
results_df.to_csv(os.path.join(DATA, "phase4_results.csv"), index=False)
print("Phase 4 results saved: datasets/phase4_results.csv")
print()
print("Figures saved to figures/phase4/:")
for f in sorted(os.listdir(FIGS)):
    print(f"  {f}")
"""))

# ── ASSEMBLE NOTEBOOK ──────────────────────────────────────────────────────────
for cell in cells:
    cell["id"] = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "wham-env",
            "language": "python",
            "name": "wham-env",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {NB}  ({len(cells)} cells)")
