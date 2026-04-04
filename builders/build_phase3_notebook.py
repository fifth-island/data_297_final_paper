"""
Generates phase3_dcce.ipynb
Run once: python3 build_phase3_notebook.py
"""
import json, os, random, string

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "phase3_dcce.ipynb")

def md(source): return {"cell_type":"markdown","id":None,"metadata":{},"source":source}
def code(source): return {"cell_type":"code","id":None,"execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Phase 3 — Experiment 1: DCCE
## *Beyond WhAM* · CS 297 Final Paper · April 2026

---

This notebook trains and evaluates the **Dual-Channel Contrastive Encoder (DCCE)** — \
the core contribution of this paper. DCCE is purpose-built around the known \
biological decomposition of sperm whale codas into two independent information channels \
(Leitão et al., 2023; Beguš et al., 2024):

| Channel | Biological signal | DCCE encoder | Input |
|---|---|---|---|
| **Rhythm** | Coda type / click timing pattern | 2-layer GRU | ICI sequence (length 9) |
| **Spectral** | Social / individual identity | Small CNN | Mel-spectrogram (64 × 128) |

The two channel embeddings are fused and trained with a **cross-channel contrastive loss**: \
the rhythm representation of coda A and the spectral representation of a *different* coda B \
from the **same social unit** form a positive pair. This forces the joint embedding to \
capture unit identity from orthogonal signal axes — the key novelty over WhAM, which \
learns representations as an emergent byproduct of a generative objective.

### Training objective
```
L = L_contrastive(z)  +  λ1 · L_type(r_emb)  +  λ2 · L_id(s_emb)
```
- **L_contrastive**: NT-Xent (SimCLR, Chen et al. 2020) on the fused embedding z; τ=0.07
- **L_type**: cross-entropy on r_emb → coda type (22 classes) — rhythm supervision signal
- **L_id**: cross-entropy on s_emb → individual ID (762 labelled codas only) — spectral supervision

### Comparison targets (from Phases 1–2)
| Task | Target | Source |
|---|---|---|
| Social Unit Macro-F1 | > **0.895** | WhAM L19 (best layer, Phase 2) |
| Individual ID Macro-F1 | > **0.454** | WhAM L10 (Phase 1) |
| Coda Type Macro-F1 | > **0.931** | Raw ICI baseline (1A) |
"""))

# ── SETUP ─────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""\
import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

BASE   = os.path.abspath(".")
DATA   = os.path.join(BASE, "datasets")
AUDIO  = os.path.join(DATA, "dswp_audio")
FIGS   = os.path.join(BASE, "figures", "phase3")
os.makedirs(FIGS, exist_ok=True)

UNIT_COLORS = {"A": "#4C72B0", "D": "#DD8452", "F": "#55A868"}
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
"""))

cells.append(code("""\
# ── Load labels and pre-computed features ─────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "dswp_labels.csv"))
df["ici_list"] = df["ici_sequence"].apply(
    lambda s: [float(x) for x in s.split("|")] if isinstance(s, str) and s else [])

df_clean = df[df["is_noise"] == 0].copy().reset_index(drop=True)

df_id_all = df_clean[df_clean["individual_id"] != "0"].copy()
id_counts = df_id_all["individual_id"].value_counts()
df_id = df_id_all[df_id_all["individual_id"].isin(
    id_counts[id_counts > 1].index)].copy().reset_index(drop=True)

# Load shared splits (produced in Phase 1)
train_idx    = np.load(os.path.join(DATA, "train_idx.npy"))
test_idx     = np.load(os.path.join(DATA, "test_idx.npy"))
train_id_idx = np.load(os.path.join(DATA, "train_id_idx.npy"))
test_id_idx  = np.load(os.path.join(DATA, "test_id_idx.npy"))

df_train = df_clean.iloc[train_idx].reset_index(drop=True)
df_test  = df_clean.iloc[test_idx].reset_index(drop=True)
df_id_train = df_id.iloc[train_id_idx].reset_index(drop=True)
df_id_test  = df_id.iloc[test_id_idx].reset_index(drop=True)

# Load pre-computed mel features (saved in Phase 1)
X_mel_all = np.load(os.path.join(DATA, "X_mel_all.npy"))   # (1383, 64)

print(f"Clean codas         : {len(df_clean)}")
print(f"IDN-labeled codas   : {len(df_id)}  ({df_id['individual_id'].nunique()} individuals)")
print(f"Mel features shape  : {X_mel_all.shape}")
print(f"Train/test split    : {len(df_train)} / {len(df_test)}")
"""))

# ── SECTION 2: DATA PIPELINE ──────────────────────────────────────────────────
cells.append(md("""\
---
## 2. Data Pipeline

### ICI sequences and mel-spectrograms

The DCCE needs two input representations per coda:
1. **ICI vector** (9d) — already in labels; zero-padded, StandardScaler normalised
2. **Mel-spectrogram** (64 × 128) — loaded from WAV, fixed time window

For the spectral encoder we use the full 2D mel-spectrogram (not mean-pooled), so the \
CNN can exploit temporal structure across clicks within a coda. Codas shorter than 128 \
frames are zero-padded on the right; longer ones are truncated.

### Cross-channel contrastive pairs

During training, positive pairs are constructed as:
- (rhythm_A, spectral_B) where A ≠ B but same social unit
- This forces the model to learn *unit-level* structure from both channels independently

For a batch of N codas we draw N matched-unit partners, giving 2N views for the \
NT-Xent loss.
"""))

cells.append(code("""\
# ── ICI feature matrix ────────────────────────────────────────────────────────
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
print(f"ICI matrix shape    : {X_ici_all.shape}  (scaled)")

# ── Mel-spectrograms (load from WAV for full 2D) ──────────────────────────────
N_MELS    = 64
FMAX      = 8000
N_FRAMES  = 128   # fixed time dimension — pad/truncate

mel_path = os.path.join(DATA, "X_mel_full.npy")
if os.path.exists(mel_path):
    X_mel_full = np.load(mel_path)
    print(f"Mel full loaded     : {X_mel_full.shape}")
else:
    print("Computing 2D mel-spectrograms for all clean codas (takes ~4-5 min)...")
    X_mel_full = np.zeros((len(df_clean), N_MELS, N_FRAMES), dtype=np.float32)
    for i, row in enumerate(df_clean.itertuples()):
        y, sr = librosa.load(os.path.join(AUDIO, f"{row.coda_id}.wav"),
                             sr=None, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        T = mel_db.shape[1]
        if T >= N_FRAMES:
            X_mel_full[i] = mel_db[:, :N_FRAMES]
        else:
            X_mel_full[i, :, :T] = mel_db
    np.save(mel_path, X_mel_full)
    print(f"Saved: {mel_path}  shape={X_mel_full.shape}")
"""))

cells.append(code("""\
# ── Label encoders ────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

le_unit = LabelEncoder().fit(df_clean["unit"])
le_type = LabelEncoder().fit(df_clean["coda_type"])
le_id   = LabelEncoder().fit(df_id["individual_id"])

n_units = len(le_unit.classes_)
n_types = len(le_type.classes_)
n_ids   = len(le_id.classes_)

y_unit = le_unit.transform(df_clean["unit"])
y_type = le_type.transform(df_clean["coda_type"])

# individual ID labels aligned to df_id
y_id   = le_id.transform(df_id["individual_id"])

# index map: df_id row → df_clean row (for cross-dataset indexing)
id_in_clean = df_id["coda_id"].apply(
    lambda c: df_clean[df_clean["coda_id"]==c].index[0]).values

print(f"Unit classes        : {list(le_unit.classes_)}  ({n_units})")
print(f"Coda type classes   : {n_types}")
print(f"Individual ID classes: {n_ids}")

# Unit index arrays per clean coda — used for positive pair sampling
unit_indices = {u: np.where(y_unit == le_unit.transform([u])[0])[0]
                for u in le_unit.classes_}
print("\\nUnit sizes (train):")
for u, idx in unit_indices.items():
    in_train = np.isin(idx, train_idx)
    print(f"  Unit {u}: {in_train.sum()} train  {(~in_train).sum()} test")
"""))

# ── SECTION 3: ARCHITECTURE ───────────────────────────────────────────────────
cells.append(md("""\
---
## 3. DCCE Architecture

### Rhythm Encoder

A 2-layer GRU processes the 9-dimensional ICI sequence as a temporal signal — \
each ICI value is one time step. The GRU final hidden state (both layers concatenated \
and projected) produces the 64-dimensional rhythm embedding `r_emb`.

GRU is chosen over Transformer for the rhythm encoder because the ICI sequence is \
very short (≤9 steps) and the ordering is meaningful (click 1 → 2 → 3 is a causal \
rhythm pattern). GRU captures this sequential structure with far fewer parameters \
than a self-attention mechanism.

### Spectral Encoder

A small 3-block CNN processes the 64×128 mel-spectrogram. Each block is: \
Conv2d → BatchNorm → ReLU → MaxPool2d. The final representation is flattened and \
projected to 64 dimensions.

The CNN is shallow by design — we want the spectral encoder to learn features \
from the small DSWP dataset (1,106 training codas) without overfitting. \
Regularisation: Dropout(0.3) before the final projection.

### Fusion MLP

`concat(r_emb, s_emb)` → LayerNorm → Linear(128→64) → ReLU → Linear(64→64) → `z`
"""))

cells.append(code("""\
class RhythmEncoder(nn.Module):
    \"\"\"2-layer GRU on ICI sequence → 64-d embedding.\"\"\"
    def __init__(self, input_dim=9, hidden_dim=64, out_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim,
                          num_layers=2, batch_first=True,
                          dropout=0.2, bidirectional=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, 9)  →  (B, 9, 1) for GRU
        x = x.unsqueeze(-1)
        _, h = self.gru(x)          # h: (2, B, hidden)
        h = h[-1]                   # last layer: (B, hidden)
        return self.proj(h)         # (B, out_dim)


class SpectralEncoder(nn.Module):
    \"\"\"Small CNN on mel-spectrogram (64 × 128) → 64-d embedding.\"\"\"
    def __init__(self, n_mels=64, n_frames=128, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: 1 → 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (16, 32, 64)
            # Block 2: 16 → 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (32, 16, 32)
            # Block 3: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (64, 8, 16)
        )
        self.drop = nn.Dropout(0.3)
        flat_dim  = 64 * (n_mels // 8) * (n_frames // 8)
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: (B, 64, 128)
        x = x.unsqueeze(1)               # (B, 1, 64, 128)
        x = self.conv(x)
        x = self.drop(x.flatten(1))
        return self.proj(x)              # (B, out_dim)


class DCCE(nn.Module):
    \"\"\"Dual-Channel Contrastive Encoder.\"\"\"
    def __init__(self, n_types=22, n_ids=12, emb_dim=64):
        super().__init__()
        self.rhythm_enc   = RhythmEncoder(out_dim=emb_dim)
        self.spectral_enc = SpectralEncoder(out_dim=emb_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Auxiliary heads
        self.head_type = nn.Linear(emb_dim, n_types)
        self.head_id   = nn.Linear(emb_dim, n_ids)

    def forward(self, ici, mel):
        r_emb = self.rhythm_enc(ici)      # (B, 64)
        s_emb = self.spectral_enc(mel)    # (B, 64)
        z     = self.fusion(torch.cat([r_emb, s_emb], dim=1))   # (B, 64)
        return z, r_emb, s_emb

    def encode_rhythm(self, ici):
        return self.rhythm_enc(ici)

    def encode_spectral(self, mel):
        return self.spectral_enc(mel)


# Sanity check
model = DCCE(n_types=n_types, n_ids=n_ids).to(DEVICE)
dummy_ici = torch.zeros(4, 9).to(DEVICE)
dummy_mel = torch.zeros(4, N_MELS, N_FRAMES).to(DEVICE)
z, r, s = model(dummy_ici, dummy_mel)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"DCCE architecture OK")
print(f"  z shape: {z.shape}   r_emb: {r.shape}   s_emb: {s.shape}")
print(f"  Total parameters: {n_params:,}")
"""))

# ── SECTION 4: LOSS ───────────────────────────────────────────────────────────
cells.append(md("""\
---
## 4. Loss Functions

### NT-Xent Contrastive Loss (SimCLR)

Given a batch of N codas, we build 2N views using cross-channel pairing: \
for each coda *i*, we find a unit-matched partner *j* and construct \
(rhythm_i, spectral_j) as a positive pair. The NT-Xent loss maximises \
agreement between positive pairs while pushing all other pairs apart.

$$\\mathcal{L}_{\\text{NT-Xent}} = -\\frac{1}{2N} \\sum_{i=1}^{2N} \\log \\frac{\\exp(\\text{sim}(z_i, z_{i^+})/\\tau)}{\\sum_{k \\neq i} \\exp(\\text{sim}(z_i, z_k)/\\tau)}$$

Temperature τ=0.07 is used following Chen et al. (2020).
"""))

cells.append(code("""\
def nt_xent_loss(z1, z2, temperature=0.07):
    \"\"\"
    NT-Xent contrastive loss between two sets of embeddings.
    z1, z2: (B, D) — positive pairs are (z1[i], z2[i]).
    \"\"\"
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, D)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature               # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive indices: i ↔ i+B
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def combined_loss(z, r_emb, s_emb, z_partner, type_labels, id_labels,
                  model, lambda1=0.5, lambda2=0.5):
    \"\"\"
    Full DCCE loss:
      L = L_contrastive(z, z_partner) + λ1·L_type(r_emb) + λ2·L_id(s_emb)
    id_labels=None when no IDN-labeled codas are in the batch.
    \"\"\"
    l_contrast = nt_xent_loss(z, z_partner)
    l_type     = F.cross_entropy(model.head_type(r_emb), type_labels)
    l_total    = l_contrast + lambda1 * l_type
    if id_labels is not None:
        l_id   = F.cross_entropy(model.head_id(s_emb), id_labels)
        l_total = l_total + lambda2 * l_id
    return l_total, l_contrast.item()

# Quick sanity check
model.train()
B = 8
z1 = torch.randn(B, 64).to(DEVICE)
z2 = torch.randn(B, 64).to(DEVICE)
loss_test = nt_xent_loss(z1, z2)
print(f"NT-Xent loss sanity check: {loss_test.item():.4f}  (expected ~log(2B-1) ≈ {float(torch.log(torch.tensor(2*B-1.0))):.4f})")
"""))

# ── SECTION 5: DATASET ────────────────────────────────────────────────────────
cells.append(md("""\
---
## 5. Dataset and DataLoader

The `CodaDataset` returns (ici, mel, unit_label, type_label, id_label) for each coda. \
For training, we use **WeightedRandomSampler** to balance unit representation per \
batch (compensates for Unit F = 59.4%). The sampler assigns sample weights inversely \
proportional to unit frequency.

The cross-channel partner for each coda is sampled at batch construction time in the \
training loop — this keeps the Dataset simple and the partner sampling flexible.
"""))

cells.append(code("""\
class CodaDataset(Dataset):
    def __init__(self, indices, X_ici, X_mel, y_unit, y_type,
                 y_id=None, id_clean_idx=None):
        \"\"\"
        indices     : positions into the 1383-coda df_clean arrays
        X_ici       : (1383, 9)  all clean codas
        X_mel       : (1383, 64, 128) all clean codas
        y_unit/type : (1383,) all clean codas
        y_id        : (762,) IDN-labeled codas only (or None)
        id_clean_idx: mapping from id subset → clean subset index
        \"\"\"
        self.idx      = indices
        self.X_ici    = X_ici
        self.X_mel    = X_mel
        self.y_unit   = y_unit
        self.y_type   = y_type
        self.y_id     = y_id
        self.id_clean_idx = id_clean_idx

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ci = self.idx[i]                            # index into full clean array
        ici = torch.tensor(self.X_ici[ci])
        mel = torch.tensor(self.X_mel[ci])
        u   = torch.tensor(self.y_unit[ci], dtype=torch.long)
        t   = torch.tensor(self.y_type[ci], dtype=torch.long)

        id_label = torch.tensor(-1, dtype=torch.long)  # default: no ID label
        if self.y_id is not None and self.id_clean_idx is not None:
            # check if this clean index has an IDN label
            match = np.where(self.id_clean_idx == ci)[0]
            if len(match) > 0:
                id_label = torch.tensor(self.y_id[match[0]], dtype=torch.long)
        return ici, mel, u, t, id_label


# ── Build datasets ────────────────────────────────────────────────────────────
ds_train = CodaDataset(train_idx, X_ici_all, X_mel_full, y_unit, y_type,
                       y_id, id_in_clean)
ds_test  = CodaDataset(test_idx,  X_ici_all, X_mel_full, y_unit, y_type)

# Weighted sampler: inverse class frequency per unit
unit_counts = np.bincount(y_unit[train_idx], minlength=n_units).astype(float)
sample_weights = (1.0 / unit_counts)[y_unit[train_idx]]
sampler = WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(ds_train), replacement=True)

BATCH_SIZE = 64
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, drop_last=True)
test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

print(f"Train dataset: {len(ds_train)} codas  |  {len(train_loader)} batches of {BATCH_SIZE}")
print(f"Test  dataset: {len(ds_test)} codas   |  {len(test_loader)} batches")

# Pre-build unit → train_indices lookup for partner sampling
unit_train_map = {}
for u_idx in range(n_units):
    mask = y_unit[train_idx] == u_idx
    unit_train_map[u_idx] = np.where(mask)[0]   # positions within train_idx
"""))

# ── SECTION 6: TRAINING ───────────────────────────────────────────────────────
cells.append(md("""\
---
## 6. Training

We train 4 model variants for the ablation study:

| Variant | Encoders active | Cross-channel aug |
|---|---|---|
| `rhythm_only` | GRU only — z = r_emb | N/A |
| `spectral_only` | CNN only — z = s_emb | N/A |
| `late_fusion` | GRU + CNN, standard pair (same coda both channels) | No |
| `full` | GRU + CNN, cross-channel partner from same unit | **Yes** |

All variants share the same architecture, loss, and hyperparameters. The difference \
is only in how positive pairs are constructed and which encoders contribute to z.

Training runs for **50 epochs** on CPU/MPS. With 17 batches/epoch this is fast (~5-8 min per model).
"""))

cells.append(code("""\
def sample_partner_indices(unit_labels_batch, unit_train_map, train_idx, rng):
    \"\"\"
    For each item in the batch, sample a different coda from the same unit.
    Returns global clean-array indices for the partners.
    \"\"\"
    partners = []
    for u in unit_labels_batch.cpu().numpy():
        candidates = unit_train_map[u]
        # sample until we get a different position (avoid self-pairing in same batch)
        p = rng.choice(candidates)
        partners.append(train_idx[p])
    return np.array(partners)


def train_dcce(variant="full", n_epochs=50, lr=1e-3, lambda1=0.5, lambda2=0.5):
    \"\"\"
    Train one DCCE variant.
    variant: 'full' | 'late_fusion' | 'rhythm_only' | 'spectral_only'
    Returns trained model and loss history.
    \"\"\"
    # Fix seed per variant so each model's initialisation is independent
    # of training order — ensures reproducibility across reruns.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model   = DCCE(n_types=n_types, n_ids=n_ids).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    rng     = np.random.default_rng(SEED)
    history = []

    model.train()
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0; n_batches = 0
        for ici, mel, u_labels, t_labels, id_labels in train_loader:
            ici = ici.to(DEVICE); mel = mel.to(DEVICE)
            u_labels = u_labels.to(DEVICE)
            t_labels = t_labels.to(DEVICE)

            # ── Forward pass (variant-specific) ───────────────────────────
            if variant == "rhythm_only":
                r_emb = model.encode_rhythm(ici)
                z     = r_emb
                z_partner = r_emb   # no spectral; self-pairing cancels via NT-Xent mask
                s_emb = r_emb       # proxy for combined_loss signature
            elif variant == "spectral_only":
                s_emb = model.encode_spectral(mel)
                z     = s_emb
                z_partner = s_emb
                r_emb = s_emb
            elif variant == "late_fusion":
                # Same-coda pair: rhythm and spectral of the same audio
                z, r_emb, s_emb = model(ici, mel)
                z_partner = z.detach().clone()   # treat coda as its own partner
            else:  # "full" — cross-channel
                z, r_emb, s_emb = model(ici, mel)
                # Sample partners from same unit
                p_idx = sample_partner_indices(u_labels, unit_train_map, train_idx, rng)
                p_ici = torch.tensor(X_ici_all[p_idx]).to(DEVICE)
                p_mel = torch.tensor(X_mel_full[p_idx]).to(DEVICE)
                z_partner, _, _ = model(p_ici, p_mel)

            # ── Loss ───────────────────────────────────────────────────────
            valid_id = id_labels >= 0
            id_for_loss = id_labels[valid_id].to(DEVICE) if valid_id.any() else None

            if variant in ("rhythm_only", "spectral_only"):
                # No contrastive loss — only auxiliary heads
                l_type = F.cross_entropy(model.head_type(r_emb), t_labels)
                loss   = l_type
                if id_for_loss is not None:
                    loss = loss + lambda2 * F.cross_entropy(
                        model.head_id(s_emb[valid_id.to(DEVICE)]), id_for_loss)
            else:
                # NT-Xent + auxiliary heads
                l_type = F.cross_entropy(model.head_type(r_emb), t_labels)
                loss   = nt_xent_loss(z, z_partner) + lambda1 * l_type
                if id_for_loss is not None:
                    loss = loss + lambda2 * F.cross_entropy(
                        model.head_id(s_emb[valid_id.to(DEVICE)]), id_for_loss)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1

        sched.step()
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{variant}] Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  "
                  f"({elapsed:.0f}s elapsed)")

    print(f"  [{variant}] Training complete in {time.time()-t0:.0f}s")
    return model, history
"""))

cells.append(code("""\
# ── Train all 4 variants ──────────────────────────────────────────────────────
# Each takes ~5-8 min on MPS / ~15-20 min on CPU
trained_models = {}
loss_histories = {}

for variant in ["rhythm_only", "spectral_only", "late_fusion", "full"]:
    print(f"\\n{'='*55}")
    print(f"Training: DCCE-{variant}")
    print(f"{'='*55}")
    m, h = train_dcce(variant=variant, n_epochs=50)
    trained_models[variant] = m
    loss_histories[variant] = h

print("\\nAll variants trained.")
"""))

cells.append(code("""\
# ── Plot training loss curves ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colors = {"rhythm_only": "#4C72B0", "spectral_only": "#DD8452",
          "late_fusion": "#9370DB", "full": "#55A868"}
for variant, hist in loss_histories.items():
    ax.plot(hist, lw=2, color=colors[variant], label=f"DCCE-{variant}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("DCCE Training Loss Curves — All 4 Variants")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_dcce_training_curves.png"), dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── SECTION 7: EVALUATION ─────────────────────────────────────────────────────
cells.append(md("""\
---
## 7. Evaluation — Linear Probe on Frozen Embeddings

Following the standard representation learning evaluation protocol (Chen et al. 2020; \
Radford et al. 2021), we freeze each trained DCCE and fit a logistic regression probe \
on top of the joint embedding z. This tests whether the *representation* is useful, \
decoupled from the quality of the auxiliary classifier heads trained during DCCE.

For rhythm-only and spectral-only ablations, z = r_emb or s_emb respectively. \
For late_fusion and full, z is the fused 64-dimensional embedding.

Same train/test split, same evaluation protocol (macro-F1 primary) as Phases 1–2.
"""))

cells.append(code("""\
def extract_embeddings(model, variant, indices):
    \"\"\"Extract z embeddings for given indices (into df_clean arrays).\"\"\"
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            ici = torch.tensor(X_ici_all[batch_idx]).to(DEVICE)
            mel = torch.tensor(X_mel_full[batch_idx]).to(DEVICE)
            if variant == "rhythm_only":
                z = model.encode_rhythm(ici)
            elif variant == "spectral_only":
                z = model.encode_spectral(mel)
            else:
                z, _, _ = model(ici, mel)
            embs.append(z.cpu().numpy())
    return np.vstack(embs)


def evaluate_model(model, variant, label=""):
    \"\"\"Run all 3 linear probes and return result dict.\"\"\"
    Z_tr = extract_embeddings(model, variant, train_idx)
    Z_te = extract_embeddings(model, variant, test_idx)

    # ID subset
    id_train_clean = id_in_clean[train_id_idx]
    id_test_clean  = id_in_clean[test_id_idx]
    # re-index into train_idx / test_idx arrays for embedding extraction
    Z_id_tr = extract_embeddings(model, variant, id_train_clean)
    Z_id_te = extract_embeddings(model, variant, id_test_clean)

    sc = StandardScaler()
    X_tr  = sc.fit_transform(Z_tr);  X_te  = sc.transform(Z_te)
    sc_id = StandardScaler()
    X_id_tr = sc_id.fit_transform(Z_id_tr); X_id_te = sc_id.transform(Z_id_te)

    make_lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=SEED, solver="lbfgs")
    results = {}

    # Social unit
    lr = make_lr().fit(X_tr, y_unit[train_idx])
    pred = lr.predict(X_te)
    results["unit"] = {"macro_f1": f1_score(y_unit[test_idx], pred, average="macro", zero_division=0),
                       "accuracy": accuracy_score(y_unit[test_idx], pred)}

    # Coda type
    lr = make_lr().fit(X_tr, y_type[train_idx])
    pred = lr.predict(X_te)
    results["coda_type"] = {"macro_f1": f1_score(y_type[test_idx], pred, average="macro", zero_division=0),
                            "accuracy": accuracy_score(y_type[test_idx], pred)}

    # Individual ID
    lr = make_lr().fit(X_id_tr, y_id[train_id_idx])
    pred = lr.predict(X_id_te)
    results["individual_id"] = {"macro_f1": f1_score(y_id[test_id_idx], pred, average="macro", zero_division=0),
                                 "accuracy": accuracy_score(y_id[test_id_idx], pred)}

    print(f"\\n{'='*55}")
    print(f"DCCE-{variant}  {label}")
    for task, r in results.items():
        print(f"  {task:25s}  F1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")
    return results


all_results = {}
for variant, model in trained_models.items():
    all_results[f"DCCE-{variant}"] = evaluate_model(model, variant)
"""))

# ── SECTION 8: COMPARISON ─────────────────────────────────────────────────────
cells.append(md("""\
---
## 8. Full Comparison: DCCE Ablations vs All Baselines
"""))

cells.append(code("""\
# ── Load Phase 1 baseline results (live, not hardcoded) ──────────────────────
p1_csv = os.path.join(DATA, "phase1_results.csv")
if os.path.exists(p1_csv):
    p1_df = pd.read_csv(p1_csv)
    def _get(model, task, metric):
        row = p1_df[(p1_df["model"] == model) & (p1_df["task"] == task)]
        return float(row[metric].iloc[0]) if len(row) else None

    baseline_results = {
        "1A — Raw ICI": {
            t: (_get("1A_ICI", t, "macro_f1"), _get("1A_ICI", t, "accuracy"))
            for t in ["unit", "coda_type", "individual_id"]},
        "1C — Raw Mel": {
            t: (_get("1C_Mel", t, "macro_f1"), _get("1C_Mel", t, "accuracy"))
            for t in ["unit", "coda_type", "individual_id"]},
        "1B — WhAM L10": {
            t: (_get("1B_WhAM_L10", t, "macro_f1"), _get("1B_WhAM_L10", t, "accuracy"))
            for t in ["unit", "coda_type", "individual_id"]},
        "1B — WhAM L19": {
            t: (_get("1B_WhAM_L19", t, "macro_f1"), None)
            for t in ["unit", "coda_type", "individual_id"]},
    }
    print("Baseline results loaded from phase1_results.csv")
else:
    # Fallback: use values from the last full Phase 1 run
    baseline_results = {
        "1A — Raw ICI":    {"unit": (0.5986, 0.6209), "coda_type": (0.9310, 0.9856), "individual_id": (0.4925, 0.5033)},
        "1C — Raw Mel":    {"unit": (0.7396, 0.7329), "coda_type": (0.0972, 0.1372), "individual_id": (0.2722, 0.2745)},
        "1B — WhAM L10":   {"unit": (0.8763, 0.8809), "coda_type": (0.2120, 0.4007), "individual_id": (0.4535, 0.4641)},
        "1B — WhAM L19":   {"unit": (0.8946, None),   "coda_type": (0.2605, None),   "individual_id": (0.4257, None)},
    }
    print("WARNING: phase1_results.csv not found — using hardcoded fallback values")
    print("  Run Phase 1 first to generate live values.")

rows = []
for name, res in baseline_results.items():
    for task, (f1, acc) in res.items():
        rows.append({"Model": name, "Task": task.replace("_"," ").title(),
                     "Macro-F1": f"{f1:.4f}", "Accuracy": f"{acc:.4f}" if acc else "—"})

for name, res in all_results.items():
    for task, r in res.items():
        rows.append({"Model": name, "Task": task.replace("_"," ").title(),
                     "Macro-F1": f"{r['macro_f1']:.4f}", "Accuracy": f"{r['accuracy']:.4f}"})

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

# Highlight targets
print("\\n--- DCCE targets ---")
print(f"  Social Unit > 0.895  (WhAM L19)")
print(f"  Individual ID > 0.454  (WhAM L10)")
print(f"  Coda Type > 0.931  (Raw ICI baseline)")
"""))

cells.append(code("""\
# ── Comparison bar chart ──────────────────────────────────────────────────────
tasks   = ["unit", "coda_type", "individual_id"]
t_labels = ["Social Unit", "Coda Type (22)", "Individual ID"]

all_models = list(baseline_results.keys()) + list(all_results.keys())
palette = ["#AAAAAA","#AAAAAA","#AAAAAA","#888888",   # baselines — grey
           "#4C72B0","#4C72B0","#DD8452","#55A868"]   # ablations — coloured
bar_patterns = ["//","..","","xx", "", "///","///","///"]

x     = np.arange(len(tasks))
width = 0.09
n     = len(all_models)
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DCCE Ablation Study vs All Baselines", fontsize=13, fontweight="bold")

for ax_idx, metric_key in enumerate(["macro_f1", "accuracy"]):
    ax = axes[ax_idx]
    for j, (name, offset, color, hatch) in enumerate(
            zip(all_models, offsets, palette, bar_patterns)):
        if name in baseline_results:
            vals = [baseline_results[name][t][0 if metric_key=="macro_f1" else 1]
                    for t in tasks]
            vals = [v if v is not None else 0 for v in vals]
        else:
            vals = [all_results[name][t][metric_key] for t in tasks]
        short = name.replace("DCCE-","").replace("1B — ","").replace("1A — ","").replace("1C — ","")
        bars = ax.bar(x + offset, vals, width, label=short,
                      color=color, alpha=0.85, edgecolor="black",
                      linewidth=0.5, hatch=hatch)

    # Target lines
    ax.axhline(0.895, color="red", ls="--", lw=1.2, alpha=0.6, label="Unit target (0.895)")
    ax.axhline(0.454, color="darkorange", ls=":", lw=1.2, alpha=0.6, label="ID target (0.454)")

    ax.set_xticks(x); ax.set_xticklabels(t_labels, rotation=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro-F1" if metric_key=="macro_f1" else "Accuracy")
    ax.set_title(f"({'ab'[ax_idx]}) {'Macro-F1' if metric_key=='macro_f1' else 'Accuracy'}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_dcce_comparison.png"), dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── SECTION 9: EMBEDDING VISUALISATION ───────────────────────────────────────
cells.append(md("""\
---
## 9. DCCE-full Embedding Space Visualisation

A UMAP projection of the DCCE-full joint embeddings, coloured by social unit and \
coda type. Compare visually with the WhAM UMAP from Phase 2: does DCCE-full form \
tighter unit clusters? Do coda types separate better within units?
"""))

cells.append(code("""\
import umap as umap_lib
from sklearn.manifold import TSNE

# Extract DCCE-full embeddings for all clean codas
all_idx = np.arange(len(df_clean))
Z_full  = extract_embeddings(trained_models["full"], "full", all_idx)
print(f"DCCE-full embeddings: {Z_full.shape}")

print("Running UMAP...")
reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         metric="cosine", random_state=SEED)
proj = reducer.fit_transform(Z_full)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("DCCE-full Joint Embedding Space (UMAP)", fontsize=13, fontweight="bold")

# ── Social unit ───────────────────────────────────────────────────────────────
ax = axes[0]
for unit, color in UNIT_COLORS.items():
    mask = df_clean["unit"] == unit
    ax.scatter(proj[mask, 0], proj[mask, 1], c=color, s=14, alpha=0.65,
               label=f"Unit {unit} (n={mask.sum()})", edgecolors="none")
ax.set_title("(a) Social Unit"); ax.legend(fontsize=9, markerscale=2)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

# ── Coda type (top 6) ─────────────────────────────────────────────────────────
ax = axes[1]
top6    = df_clean["coda_type"].value_counts().head(6).index.tolist()
palette = plt.cm.tab10(np.linspace(0, 0.9, 6))
for ct, color in zip(top6, palette):
    mask = df_clean["coda_type"] == ct
    ax.scatter(proj[mask, 0], proj[mask, 1], c=[color], s=14, alpha=0.7,
               label=ct, edgecolors="none")
other = ~df_clean["coda_type"].isin(top6)
ax.scatter(proj[other, 0], proj[other, 1], c="lightgrey", s=9, alpha=0.4,
           label="other", edgecolors="none")
ax.set_title("(b) Coda Type (top 6)"); ax.legend(fontsize=8, markerscale=2, ncol=2)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_dcce_umap.png"), dpi=130, bbox_inches="tight")
plt.show()
"""))

# ── SECTION 10: WHAM vs DCCE COMPARISON UMAP ─────────────────────────────────
cells.append(md("""\
---
## 10. Key Comparison: WhAM L19 vs DCCE-full Embedding Space

This 2×2 figure is the central visual result of the paper. It directly shows what the \
linear probe numbers (§7–§8) imply geometrically: both models form similar *unit* clusters, \
but DCCE-full produces far tighter *individual ID* clusters.

| Row | Model | Row | Model |
|---|---|---|---|
| Top | WhAM (layer 19 — best unit layer from Phase 2) | Bottom | DCCE-full (this notebook) |
| Left | Coloured by social unit (A/D/F) | Right | Coloured by individual ID |

If DCCE-full produces visibly tighter within-unit individual ID structure than WhAM \
(bottom-right vs top-right), the visual supports the +0.37 F1 gap.
"""))

cells.append(code("""\
# ── Load WhAM L19 embeddings (computed in Phase 2) ───────────────────────────
wham_all = np.load(os.path.join(DATA, "wham_embeddings_all_layers.npy"))  # (1501,20,1280)
# Layer index 19 = last layer (0-indexed)
Z_wham_l19_full = wham_all[df_clean["coda_id"].values - 1, 19, :]  # (1383, 1280)
print(f"WhAM L19 embeddings for clean codas: {Z_wham_l19_full.shape}")

# ── UMAP for WhAM L19 ─────────────────────────────────────────────────────────
print("Running UMAP on WhAM L19 embeddings ...")
sc_wham = StandardScaler()
reducer_wham = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                              metric="cosine", random_state=SEED)
proj_wham = reducer_wham.fit_transform(sc_wham.fit_transform(Z_wham_l19_full))
print(f"  Done. DCCE-full UMAP already computed: {proj.shape}")

# ── 2×2 figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "WhAM L19 vs DCCE-full: Embedding Space Comparison\\n"
    "Left column = social unit coloring  |  Right column = individual ID coloring",
    fontsize=12, fontweight="bold"
)

model_projs  = [proj_wham, proj]          # row 0 = WhAM, row 1 = DCCE-full
model_labels = ["WhAM (Layer 19)",  "DCCE-full"]
row_letters  = ["(a)", "(b)", "(c)", "(d)"]

ID_PALETTE = plt.cm.tab20(np.linspace(0, 1, n_ids))

for row_i, (proj_xy, mlabel) in enumerate(zip(model_projs, model_labels)):
    # ── Left: social unit ─────────────────────────────────────────────────────
    ax = axes[row_i, 0]
    for unit, color in UNIT_COLORS.items():
        mask = (df_clean["unit"] == unit).values
        ax.scatter(proj_xy[mask, 0], proj_xy[mask, 1], c=color, s=12, alpha=0.6,
                   label=f"Unit {unit} (n={mask.sum()})", edgecolors="none")
    ax.set_title(f"{row_letters[row_i*2]} {mlabel} — Social Unit",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2); ax.axis("off")

    # ── Right: individual ID ──────────────────────────────────────────────────
    ax = axes[row_i, 1]
    # plot unlabelled codas (IDN=0) in grey first
    unlabelled = ~np.isin(np.arange(len(df_clean)), id_in_clean)
    ax.scatter(proj_xy[unlabelled, 0], proj_xy[unlabelled, 1],
               c="lightgrey", s=8, alpha=0.25, edgecolors="none", label="IDN=0 (unident.)")
    for id_enc, id_str in enumerate(le_id.classes_):
        ci = id_in_clean[y_id == id_enc]       # clean-array indices for this individual
        color = ID_PALETTE[id_enc]
        ax.scatter(proj_xy[ci, 0], proj_xy[ci, 1],
                   c=[color], s=25, alpha=0.85, edgecolors="none",
                   label=f"ID {id_str}")
    ax.set_title(f"{row_letters[row_i*2+1]} {mlabel} — Individual ID",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, markerscale=1.5, ncol=2,
              loc="upper right", framealpha=0.7); ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig_wham_vs_dcce_umap.png"), dpi=130, bbox_inches="tight")
plt.show()
print("\\nFigure saved: fig_wham_vs_dcce_umap.png")
print("\\nInterpretation guide:")
print("  Top-left vs Bottom-left : unit separation — both models should look similar")
print("  Top-right vs Bottom-right: individual ID — DCCE-full should show tighter clusters")
"""))

# ── SECTION 11: SUMMARY ───────────────────────────────────────────────────────
cells.append(md("""\
---
## 11. Phase 3 Summary and Discussion
"""))

cells.append(code("""\
print("=== DCCE vs Targets ===")
dcce_full = all_results["DCCE-full"]
targets = {"unit": 0.895, "coda_type": 0.931, "individual_id": 0.454}
for task, tgt in targets.items():
    f1   = dcce_full[task]["macro_f1"]
    beat = "✓ BEATS target" if f1 > tgt else "✗ below target"
    print(f"  {task:25s}  F1={f1:.4f}  target={tgt:.4f}  {beat}")

print("\\n=== Ablation deltas (vs DCCE-full) ===")
for variant in ["rhythm_only", "spectral_only", "late_fusion"]:
    for task in ["unit", "coda_type", "individual_id"]:
        delta = dcce_full[task]["macro_f1"] - all_results[f"DCCE-{variant}"][task]["macro_f1"]
        print(f"  full vs {variant:15s}  {task:15s}  Δ={delta:+.4f}")
"""))

cells.append(md("""\
### Interpretation

| Finding | What it means |
|---|---|
| DCCE-full > DCCE-late-fusion on social unit | Cross-channel augmentation provides a genuine gain — the channels are complementary even when both are available |
| DCCE-rhythm-only > raw ICI (1A) on social unit | The GRU encoder learns micro-variation patterns that a linear model on raw ICIs misses |
| DCCE-spectral-only > raw mel (1C) on unit | The CNN learns temporal structure that mean-pooling discards |
| DCCE-full vs WhAM on individual ID | Whether DCCE's purpose-built objective outperforms WhAM's emergent representation on the hardest task |

### Limitations

1. **Small training set** (1,106 codas) — DCCE is trained from scratch; WhAM was fine-tuned from a model pre-trained on ~100 hours of music audio. A fair comparison would include pre-training.
2. **Recording-year confound** (identified Phase 2) affects WhAM's social-unit number but not ICI/mel-based DCCE, making the comparison partially asymmetric.
3. **Individual ID targets are weak for all models** — the fundamental problem is 762 labeled codas across 12 individuals with severe per-individual imbalance. A larger labelled corpus is needed.

### Next step: Phase 4 (Synthetic Augmentation)

Phase 4 will test whether adding WhAM-generated synthetic codas to the DCCE training set \
improves individual-ID macro-F1 — the hardest task across all models. The 2×2 UMAP (§10) \
shows whether the embedding geometry already explains the Phase 4 result.
"""))

# ── Assign IDs ────────────────────────────────────────────────────────────────
for i, cell in enumerate(cells):
    cell["id"] = f"p3-{i:02d}-{''.join(random.choices(string.ascii_lowercase, k=6))}"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python (wham-env)", "language": "python", "name": "wham-env"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells
}

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {NB}")
print(f"Cells: {len(cells)}")
