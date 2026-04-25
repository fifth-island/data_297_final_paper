"""Phase 3 — Dual-Channel Contrastive Encoder (DCCE)
Trains 4 variants: rhythm-only, spectral-only, late-fusion, full (cross-channel).
Evaluates via linear probes. Produces comparison figures and UMAP.
"""
import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import umap

BASE   = "/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project"
DATA   = f"{BASE}/datasets"
FIGDIR = f"{BASE}/figures/phase3"
os.makedirs(FIGDIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

UNIT_COLORS = {"A": "#2196F3", "D": "#FF9800", "F": "#4CAF50"}
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── load data ──────────────────────────────────────────────────────────────────
df    = pd.read_csv(f"{DATA}/dswp_labels.csv")
clean = df[df.is_noise == 0].reset_index(drop=True)

train_idx    = np.load(f"{DATA}/train_idx.npy")
test_idx     = np.load(f"{DATA}/test_idx.npy")
train_id_idx = np.load(f"{DATA}/train_id_idx.npy")
test_id_idx  = np.load(f"{DATA}/test_id_idx.npy")

X_ici_raw  = np.zeros((len(clean), 9), dtype=np.float32)
for i, row in enumerate(clean["ici_sequence"]):
    if pd.isna(row): continue
    vals = [float(x) * 1000 for x in str(row).split("|")]
    for j, v in enumerate(vals[:9]):
        X_ici_raw[i, j] = v
scaler_ici = StandardScaler()
X_ici = scaler_ici.fit_transform(X_ici_raw).astype(np.float32)

X_mel = np.load(f"{DATA}/X_mel_full.npy")  # (1383, 64, 128)
X_mel = X_mel[:, np.newaxis, :, :]          # (1383, 1, 64, 128)

le_unit = LabelEncoder(); le_unit.fit(clean["unit"].values)
y_unit  = le_unit.transform(clean["unit"].values)   # 0=A,1=D,2=F

le_type = LabelEncoder(); le_type.fit(clean["coda_type"].values)
y_type  = le_type.transform(clean["coda_type"].values)

idn_mask = clean["individual_id"].astype(str) != "0"
idn_positions = np.where(idn_mask.values)[0]
clean_idn = clean[idn_mask].reset_index(drop=True)
le_id = LabelEncoder(); le_id.fit(clean_idn["individual_id"].astype(str))
y_id_all = le_id.transform(clean_idn["individual_id"].astype(str))

n_units = len(np.unique(y_unit))   # 3
n_types = len(np.unique(y_type))   # 22
n_ids   = len(np.unique(y_id_all)) # 13 (after singleton removal, 762/763 in split)

print(f"n_units={n_units}  n_types={n_types}  n_ids={n_ids}")
print(f"Train={len(train_idx)}  Test={len(test_idx)}")

# ══════════════════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════════════════
class RhythmEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, emb_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, emb_dim), nn.ReLU())
    def forward(self, x):
        # x: (B, 9) → (B, 1, 9) as sequence
        out, h = self.gru(x.unsqueeze(1))
        return self.proj(h[-1])  # (B, emb_dim)

class SpectralEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # (16,32,64)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (32,16,32)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)), # (64,4,4)
        )
        self.proj = nn.Sequential(nn.Linear(64*4*4, 128), nn.ReLU(), nn.Linear(128, emb_dim))
    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.proj(h)

class FusionMLP(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )
    def forward(self, r, s):
        return self.net(torch.cat([r, s], dim=1))

class DCCE(nn.Module):
    def __init__(self, variant="full"):
        super().__init__()
        self.variant = variant
        if variant in ("rhythm_only", "late_fusion", "full"):
            self.rhythm_enc = RhythmEncoder()
        if variant in ("spectral_only", "late_fusion", "full"):
            self.spectral_enc = SpectralEncoder()
        if variant in ("late_fusion", "full"):
            self.fusion = FusionMLP(128, 64)
        # auxiliary heads
        if variant in ("rhythm_only", "late_fusion", "full"):
            self.head_type = nn.Linear(64, n_types)
        if variant in ("spectral_only", "late_fusion", "full"):
            self.head_id   = nn.Linear(64, n_ids)

    def encode(self, ici, mel):
        if self.variant == "rhythm_only":
            return self.rhythm_enc(ici)
        elif self.variant == "spectral_only":
            return self.spectral_enc(mel)
        else:
            r = self.rhythm_enc(ici)
            s = self.spectral_enc(mel)
            return self.fusion(r, s)

    def forward(self, ici, mel, ici_b=None, mel_b=None):
        z = self.encode(ici, mel)
        out = {"z": z}
        if self.variant in ("rhythm_only", "late_fusion", "full"):
            out["r_emb"] = self.rhythm_enc(ici) if self.variant != "rhythm_only" else z
            out["type_logits"] = self.head_type(out["r_emb"])
        if self.variant in ("spectral_only", "late_fusion", "full"):
            out["s_emb"] = self.spectral_enc(mel) if self.variant != "spectral_only" else z
            out["id_logits"] = self.head_id(out["s_emb"])
        if self.variant == "full" and ici_b is not None and mel_b is not None:
            # cross-channel positive: rhythm(a) + spectral(b) → cross_z
            r_a = self.rhythm_enc(ici)
            s_b = self.spectral_enc(mel_b)
            out["cross_z"] = self.fusion(r_a, s_b)
        return out

# ══════════════════════════════════════════════════════════════════════════════
# NT-Xent loss
# ══════════════════════════════════════════════════════════════════════════════
def nt_xent(z1, z2, tau=0.07):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / tau
    # mask self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # positives: i ↔ i+B
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class CodaDataset(Dataset):
    def __init__(self, ici, mel, y_unit, y_type, y_id, idn_mask, indices):
        self.ici    = torch.from_numpy(ici[indices])
        self.mel    = torch.from_numpy(mel[indices])
        self.y_unit = torch.from_numpy(y_unit[indices]).long()
        self.y_type = torch.from_numpy(y_type[indices]).long()
        self.y_id   = torch.from_numpy(y_id[indices]).long()   # -1 = no label
        self.unit_to_indices = {}
        units_here = y_unit[indices]
        for u in np.unique(units_here):
            self.unit_to_indices[u] = np.where(units_here == u)[0]
    def __len__(self): return len(self.ici)
    def __getitem__(self, i):
        return self.ici[i], self.mel[i], self.y_unit[i], self.y_type[i], self.y_id[i], i

def get_y_id_aligned(y_id_all, idn_positions, train_idx):
    """Build an id-label array aligned with train_idx (length = len(train_idx)), -1 for no label."""
    out = np.full(len(train_idx), -1, dtype=np.int64)
    idn_set = {pos: label for pos, label in zip(idn_positions, y_id_all)}
    for j, idx in enumerate(train_idx):
        if idx in idn_set:
            out[j] = idn_set[idx]
    return out

y_id_train = get_y_id_aligned(y_id_all, idn_positions, train_idx)
y_id_test  = get_y_id_aligned(y_id_all, idn_positions, test_idx)

# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train_dcce(variant, epochs=50, batch_size=64, lr=1e-3, lam1=0.5, lam2=0.5, tau=0.07):
    torch.manual_seed(42)
    model = DCCE(variant).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # weighted sampler for unit balance
    unit_train = y_unit[train_idx]
    class_counts = np.bincount(unit_train, minlength=n_units)
    weights = 1.0 / class_counts[unit_train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    # unit weight tensor for CE
    unit_wt = torch.tensor(1.0/class_counts, dtype=torch.float32).to(DEVICE)
    type_wt = torch.tensor(1.0/np.maximum(np.bincount(y_type[train_idx], minlength=n_types),1),
                           dtype=torch.float32).to(DEVICE)
    id_wt   = torch.tensor(1.0/np.maximum(np.bincount(y_id_all, minlength=n_ids),1),
                           dtype=torch.float32).to(DEVICE)

    # Build dataset manually for weighted sampling
    ici_tr  = torch.from_numpy(X_ici[train_idx]).to(DEVICE)
    mel_tr  = torch.from_numpy(X_mel[train_idx]).to(DEVICE)
    unit_tr = torch.from_numpy(y_unit[train_idx]).long().to(DEVICE)
    type_tr = torch.from_numpy(y_type[train_idx]).long().to(DEVICE)
    id_tr   = torch.from_numpy(y_id_train).long().to(DEVICE)

    dataset_size = len(train_idx)
    history = {"loss":[], "type_loss":[], "id_loss":[], "contrastive_loss":[]}

    # unit index map for cross-channel positives
    unit_arrays = {}
    for u in range(n_units):
        unit_arrays[u] = np.where(y_unit[train_idx] == u)[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.multinomial(torch.from_numpy(weights).float(), dataset_size, replacement=True)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, dataset_size, batch_size):
            idx_b = perm[start:start+batch_size]
            if len(idx_b) < 4: continue

            ici_b   = ici_tr[idx_b]
            mel_b   = mel_tr[idx_b]
            unit_b  = unit_tr[idx_b]
            type_b  = type_tr[idx_b]
            id_b    = id_tr[idx_b]

            # cross-channel: for each sample, pick a same-unit random sample
            if variant == "full":
                idx_cross = torch.zeros_like(idx_b)
                for j in range(len(idx_b)):
                    u = unit_b[j].item()
                    pool = unit_arrays[u]
                    idx_cross[j] = int(np.random.choice(pool))
                ici_cross = ici_tr[idx_cross]
                mel_cross = mel_tr[idx_cross]
                out_cross = model(ici_b, mel_b, ici_cross, mel_cross)
            else:
                out_cross = model(ici_b, mel_b)

            out = model(ici_b, mel_b)
            z   = out["z"]

            # 1. contrastive loss: same-unit pairs as positives
            # Build augmented view: same unit, random other sample
            idx_pos = torch.zeros_like(idx_b)
            for j in range(len(idx_b)):
                u = unit_b[j].item()
                pool = unit_arrays[u]
                idx_pos[j] = int(np.random.choice(pool))
            out2 = model(ici_tr[idx_pos], mel_tr[idx_pos])
            z2   = out2["z"]

            L_cont = nt_xent(z, z2, tau)

            # cross-channel contrastive
            if variant == "full" and "cross_z" in out_cross:
                L_cross = nt_xent(z, out_cross["cross_z"], tau)
                L_cont  = 0.5 * L_cont + 0.5 * L_cross

            # 2. auxiliary type head
            L_type = torch.tensor(0.0, device=DEVICE)
            if "type_logits" in out:
                L_type = F.cross_entropy(out["type_logits"], type_b, weight=type_wt)

            # 3. auxiliary id head (only for labeled samples)
            L_id = torch.tensor(0.0, device=DEVICE)
            if "id_logits" in out:
                labeled = id_b >= 0
                if labeled.sum() > 1:
                    L_id = F.cross_entropy(out["id_logits"][labeled], id_b[labeled], weight=id_wt)

            loss = L_cont + lam1 * L_type + lam2 * L_id
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        history["loss"].append(avg)
        if (epoch+1) % 10 == 0:
            print(f"  [{variant}] epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    return model, history

# ══════════════════════════════════════════════════════════════════════════════
# Linear probe evaluation
# ══════════════════════════════════════════════════════════════════════════════
def extract_embeddings(model, X_ici_all, X_mel_all, idx):
    model.eval()
    embs = []
    with torch.no_grad():
        for start in range(0, len(idx), 128):
            b_idx = idx[start:start+128]
            ici_b = torch.from_numpy(X_ici_all[b_idx]).to(DEVICE)
            mel_b = torch.from_numpy(X_mel_all[b_idx]).to(DEVICE)
            out   = model(ici_b, mel_b)
            embs.append(out["z"].cpu().numpy())
    return np.vstack(embs)

def linear_probe(Z_tr, Z_te, y_tr, y_te, label=""):
    sc = StandardScaler()
    Ztr_sc = sc.fit_transform(Z_tr)
    Zte_sc = sc.transform(Z_te)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             random_state=42, solver="lbfgs")
    clf.fit(Ztr_sc, y_tr)
    pred = clf.predict(Zte_sc)
    f1  = f1_score(y_te, pred, average="macro")
    acc = accuracy_score(y_te, pred)
    print(f"    {label}: F1={f1:.3f}  Acc={acc:.3f}")
    return f1, acc

def eval_model(model, variant):
    Z_tr = extract_embeddings(model, X_ici, X_mel, train_idx)
    Z_te = extract_embeddings(model, X_ici, X_mel, test_idx)

    f1_unit, acc_unit = linear_probe(Z_tr, Z_te, y_unit[train_idx], y_unit[test_idx], "unit")
    f1_type, acc_type = linear_probe(Z_tr, Z_te, y_type[train_idx], y_type[test_idx], "coda_type")

    # individual ID
    Z_id_tr = extract_embeddings(model, X_ici[idn_positions], X_mel[idn_positions], train_id_idx)
    Z_id_te = extract_embeddings(model, X_ici[idn_positions], X_mel[idn_positions], test_id_idx)
    f1_id, acc_id = linear_probe(Z_id_tr, Z_id_te, y_id_all[train_id_idx], y_id_all[test_id_idx], "individ_id")

    return {"unit_f1": f1_unit, "unit_acc": acc_unit,
            "type_f1": f1_type, "type_acc": acc_type,
            "id_f1": f1_id,    "id_acc": acc_id}

# ══════════════════════════════════════════════════════════════════════════════
# Train all 4 variants
# ══════════════════════════════════════════════════════════════════════════════
variants  = ["rhythm_only", "spectral_only", "late_fusion", "full"]
models    = {}
histories = {}
metrics   = {}

t0 = time.time()
for variant in variants:
    print(f"\n>>> Training DCCE-{variant} <<<")
    m, h = train_dcce(variant, epochs=50)
    models[variant]    = m
    histories[variant] = h
    print(f"  Evaluating {variant}...")
    metrics[variant] = eval_model(m, variant)

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Training curves
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"rhythm_only": "#7986CB", "spectral_only": "#4DB6AC", "late_fusion": "#FFB74D", "full": "#EF5350"}
for v, h in histories.items():
    ax.plot(h["loss"], label=f"DCCE-{v}", color=colors[v])
ax.set_title("DCCE Training Curves (total loss)", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_dcce_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Training curves saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — DCCE vs Baselines comparison
# ══════════════════════════════════════════════════════════════════════════════
# Load phase1 baselines
phase1 = pd.read_csv(f"{DATA}/phase1_results.csv")
p1 = {row["model"]: row for _, row in phase1.iterrows()}
ici_f1  = p1.get("ICI_LogReg_1A", {})
mel_f1  = p1.get("Mel_LogReg_1C", {})
wham_f1 = p1.get("WhAM_L10_1B",  {})

# DCCE numbers
dcce_labels = ["rhythm_only", "spectral_only", "late_fusion", "full"]
dcce_unit = [metrics[v]["unit_f1"] for v in dcce_labels]
dcce_type = [metrics[v]["type_f1"] for v in dcce_labels]
dcce_id   = [metrics[v]["id_f1"]   for v in dcce_labels]

tasks = ["Unit Macro-F1", "CodaType Macro-F1", "IndivID Macro-F1"]
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("DCCE vs Baselines — Macro-F1", fontsize=14, fontweight="bold")

for ax_i, (ax, task, dcce_vals, p1_ici, p1_mel, p1_wham, wham_l19) in enumerate(zip(
    axes,
    tasks,
    [dcce_unit, dcce_type, dcce_id],
    [ici_f1.get("unit_f1", 0.599), ici_f1.get("codatype_f1", 0.931), ici_f1.get("individ_f1", 0.431)],
    [mel_f1.get("unit_f1", 0.740), mel_f1.get("codatype_f1", 0.097), mel_f1.get("individ_f1", 0.250)],
    [wham_f1.get("unit_f1", 0.876), wham_f1.get("codatype_f1", 0.212), wham_f1.get("individ_f1", 0.424)],
    [0.895, 0.261, 0.493],  # Phase 2 WhAM best-layer
)):
    x = np.arange(len(dcce_labels))
    bars = ax.bar(x, dcce_vals, color=[colors[v] for v in dcce_labels], edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, dcce_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.axhline(p1_ici,  color="#7986CB", linestyle="--", linewidth=1.2, label=f"ICI={p1_ici:.3f}")
    ax.axhline(p1_mel,  color="#4DB6AC", linestyle="--", linewidth=1.2, label=f"Mel={p1_mel:.3f}")
    ax.axhline(p1_wham, color="#FF8A65", linestyle="--", linewidth=1.2, label=f"WhAM L10={p1_wham:.3f}")
    if ax_i == 0:
        ax.axhline(wham_l19, color="#FF8A65", linestyle=":", linewidth=1.2, label=f"WhAM L19={wham_l19:.3f}")
    ax.set_title(task); ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(["rhythm", "spectral", "late-fuse", "full"], rotation=20, fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_dcce_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("DCCE comparison saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — UMAP of DCCE-full embeddings
# ══════════════════════════════════════════════════════════════════════════════
print("Computing DCCE-full UMAP...")
model_full = models["full"]
Z_all = extract_embeddings(model_full, X_ici, X_mel, np.arange(len(clean)))
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
Z_umap = reducer.fit_transform(Z_all)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("DCCE-full Embeddings — UMAP", fontsize=14, fontweight="bold")
for unit in ["A","D","F"]:
    m = y_unit == le_unit.transform([unit])[0]
    axes[0].scatter(Z_umap[m,0], Z_umap[m,1], c=UNIT_COLORS[unit], label=f"Unit {unit}",
                    alpha=0.5, s=8, rasterized=True)
axes[0].set_title(f"Social Unit (F1={metrics['full']['unit_f1']:.3f})")
axes[0].legend(markerscale=3); axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

top5 = pd.Series(le_type.inverse_transform(y_type)).value_counts().head(5).index.tolist()
pal = sns.color_palette("tab10", 6)
y_type_names = le_type.inverse_transform(y_type)
for i, ct in enumerate(top5):
    m = y_type_names == ct
    axes[1].scatter(Z_umap[m,0], Z_umap[m,1], c=[pal[i]], label=ct, alpha=0.5, s=8, rasterized=True)
other = ~np.isin(y_type_names, top5)
axes[1].scatter(Z_umap[other,0], Z_umap[other,1], c=[pal[-1]], label="Other", alpha=0.3, s=5, rasterized=True)
axes[1].set_title(f"Coda Type (F1={metrics['full']['type_f1']:.3f})")
axes[1].legend(fontsize=8, markerscale=3); axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_dcce_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print("DCCE UMAP saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — 2×2: WhAM L19 vs DCCE-full × unit vs individ ID
# ══════════════════════════════════════════════════════════════════════════════
print("Computing WhAM vs DCCE 2×2 UMAP comparison...")
all_layers  = np.load(f"{DATA}/wham_embeddings_all_layers.npy")
coda_ids_clean = clean["coda_id"].values
wham_l19 = all_layers[coda_ids_clean - 1][:, 19, :]  # (1383, 1280)
sc_wham = StandardScaler(); wham_l19_sc = sc_wham.fit_transform(wham_l19)
reducer_w = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
Z_wham = reducer_w.fit_transform(wham_l19_sc)

# individual ID UMAP (only labeled)
Z_all_id = Z_all[idn_positions]
Z_wham_id = Z_wham[idn_positions]
id_names = le_id.inverse_transform(y_id_all)
unique_ids = np.unique(id_names)
id_pal = sns.color_palette("tab20", len(unique_ids))
id_color = {idn: id_pal[i] for i, idn in enumerate(unique_ids)}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("WhAM L19 vs DCCE-full — UMAP Comparison", fontsize=14, fontweight="bold")

for col, (Z, label, unit_f1, id_f1) in enumerate([
    (Z_wham, "WhAM L19",   0.895, 0.493),
    (Z_all,  "DCCE-full",  metrics["full"]["unit_f1"], metrics["full"]["id_f1"])
]):
    # row 0: social unit
    for unit in ["A","D","F"]:
        u_code = le_unit.transform([unit])[0]
        m = y_unit == u_code
        axes[0, col].scatter(Z[m,0], Z[m,1], c=UNIT_COLORS[unit], label=f"Unit {unit}",
                             alpha=0.5, s=7, rasterized=True)
    axes[0, col].set_title(f"{label} — Social Unit (F1={unit_f1:.3f})")
    axes[0, col].legend(markerscale=3, fontsize=9)
    axes[0, col].set_xlabel("UMAP 1"); axes[0, col].set_ylabel("UMAP 2")

    # row 1: individual ID
    Z_id = Z[idn_positions]
    for idn in unique_ids:
        m = id_names == idn
        axes[1, col].scatter(Z_id[m,0], Z_id[m,1], c=[id_color[idn]], label=str(idn),
                             alpha=0.6, s=12, rasterized=True)
    axes[1, col].set_title(f"{label} — Individual ID (F1={id_f1:.3f})")
    axes[1, col].legend(fontsize=7, markerscale=2, ncol=2)
    axes[1, col].set_xlabel("UMAP 1"); axes[1, col].set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_wham_vs_dcce_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print("WhAM vs DCCE comparison UMAP saved")

# ══════════════════════════════════════════════════════════════════════════════
# Save embeddings for Phase 4 reuse
# ══════════════════════════════════════════════════════════════════════════════
torch.save(models["full"].state_dict(), f"{DATA}/dcce_full_weights.pt")
np.save(f"{DATA}/dcce_full_embeddings.npy", Z_all)
print("DCCE-full weights and embeddings saved")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== PHASE 3 SUMMARY ===")
print(f"{'Model':<22} {'Unit F1':>8} {'Type F1':>10} {'IndivID F1':>12} {'Unit Acc':>9}")
for v in variants:
    m = metrics[v]
    print(f"  DCCE-{v:<16} {m['unit_f1']:>8.3f} {m['type_f1']:>10.3f} {m['id_f1']:>12.3f} {m['unit_acc']:>9.3f}")
print(f"  {'WhAM L10 (1B)':<20} {wham_f1.get('unit_f1', 0.876):>8.3f} {wham_f1.get('codatype_f1', 0.212):>10.3f} {wham_f1.get('individ_f1', 0.424):>12.3f}")
print(f"  {'WhAM L19 target':<20} {'0.895':>8} {'0.261':>10} {'0.493':>12}")
