"""Phase 4 — Synthetic Data Augmentation Sweep
Trains DCCE-full with N_synth ∈ {0, 100, 500, 1000} synthetic codas added to training.
Evaluates on real-only test set. Produces augmentation curve and visualization figures.
"""
import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
FIGDIR = f"{BASE}/figures/phase4"
os.makedirs(FIGDIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")
UNIT_COLORS = {"A": "#2196F3", "D": "#FF9800", "F": "#4CAF50"}
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── load real data ─────────────────────────────────────────────────────────────
df    = pd.read_csv(f"{DATA}/dswp_labels.csv")
clean = df[df.is_noise == 0].reset_index(drop=True)
train_idx    = np.load(f"{DATA}/train_idx.npy")
test_idx     = np.load(f"{DATA}/test_idx.npy")
train_id_idx = np.load(f"{DATA}/train_id_idx.npy")
test_id_idx  = np.load(f"{DATA}/test_id_idx.npy")

X_ici_raw = np.zeros((len(clean), 9), dtype=np.float32)
for i, row in enumerate(clean["ici_sequence"]):
    if pd.isna(row): continue
    vals = [float(x)*1000 for x in str(row).split("|")]
    for j, v in enumerate(vals[:9]): X_ici_raw[i, j] = v
sc_ici = StandardScaler()
X_ici = sc_ici.fit_transform(X_ici_raw).astype(np.float32)

X_mel = np.load(f"{DATA}/X_mel_full.npy")[:, np.newaxis, :, :]  # (1383, 1, 64, 128)

le_unit = LabelEncoder(); le_unit.fit(clean["unit"].values)
y_unit  = le_unit.transform(clean["unit"].values)
le_type = LabelEncoder(); le_type.fit(clean["coda_type"].values)
y_type  = le_type.transform(clean["coda_type"].values)

idn_mask     = clean["individual_id"].astype(str) != "0"
idn_positions = np.where(idn_mask.values)[0]
clean_idn    = clean[idn_mask].reset_index(drop=True)
le_id        = LabelEncoder(); le_id.fit(clean_idn["individual_id"].astype(str))
y_id_all     = le_id.transform(clean_idn["individual_id"].astype(str))

n_units = len(np.unique(y_unit)); n_types = len(np.unique(y_type)); n_ids = len(np.unique(y_id_all))
print(f"n_units={n_units}  n_types={n_types}  n_ids={n_ids}")

# ── load synthetic data ────────────────────────────────────────────────────────
X_ici_synth_raw = np.load(f"{DATA}/X_ici_synth_1000.npy")  # (1000, 9) — already in ms?
X_mel_synth     = np.load(f"{DATA}/X_mel_synth_1000.npy")[:, np.newaxis, :, :]  # (1000, 1, 64, 128)
y_unit_synth    = np.load(f"{DATA}/y_unit_synth_1000.npy")  # 0/1/2
y_type_synth    = np.load(f"{DATA}/y_type_synth_1000.npy")

# Synth ICI: scale same as real (apply the same StandardScaler fitted on real training data)
X_ici_synth = sc_ici.transform(X_ici_synth_raw.astype(np.float32))

# Check synth type labels — may have additional classes not in real training set; clip to known types
y_type_synth_clipped = np.clip(y_type_synth, 0, n_types-1)
print(f"Synth unit distribution: {np.bincount(y_unit_synth)}")
print(f"Synth ICI stats: mean={X_ici_synth.mean():.3f} std={X_ici_synth.std():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Architecture (same as Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
class RhythmEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(9, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
    def forward(self, x):
        _, h = self.gru(x.unsqueeze(1))
        return self.proj(h[-1])

class SpectralEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.proj = nn.Sequential(nn.Linear(64*4*4, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, x):
        return self.proj(self.conv(x).flatten(1))

class DCCE_Full(nn.Module):
    def __init__(self):
        super().__init__()
        self.rhythm_enc   = RhythmEncoder()
        self.spectral_enc = SpectralEncoder()
        self.fusion = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.ReLU())
        self.head_type = nn.Linear(64, n_types)
        self.head_id   = nn.Linear(64, n_ids)
    def encode(self, ici, mel):
        return self.fusion(torch.cat([self.rhythm_enc(ici), self.spectral_enc(mel)], 1))
    def forward(self, ici, mel):
        z   = self.encode(ici, mel)
        r   = self.rhythm_enc(ici)
        s   = self.spectral_enc(mel)
        return {"z": z, "r_emb": r, "s_emb": s,
                "type_logits": self.head_type(r),
                "id_logits":   self.head_id(s)}

def nt_xent(z1, z2, tau=0.07):
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], 0), 1)
    sim = z @ z.T / tau
    sim.masked_fill_(torch.eye(2*B, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)

def train_with_synth(n_synth, epochs=50, batch_size=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DCCE_Full().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Merge real + synthetic training data
    X_ici_tr   = X_ici[train_idx]
    X_mel_tr   = X_mel[train_idx]
    y_unit_tr  = y_unit[train_idx]
    y_type_tr  = y_type[train_idx]

    if n_synth > 0:
        sidx = np.arange(n_synth)  # first n_synth synthetic samples
        X_ici_tr  = np.vstack([X_ici_tr,  X_ici_synth[sidx]])
        X_mel_tr  = np.vstack([X_mel_tr,  X_mel_synth[sidx]])
        y_unit_tr = np.concatenate([y_unit_tr, y_unit_synth[sidx]])
        y_type_tr = np.concatenate([y_type_tr, y_type_synth_clipped[sidx]])

    n_tr = len(X_ici_tr)
    print(f"  N_synth={n_synth}: training on {n_tr} samples")

    ici_tr  = torch.from_numpy(X_ici_tr.astype(np.float32)).to(DEVICE)
    mel_tr  = torch.from_numpy(X_mel_tr.astype(np.float32)).to(DEVICE)
    unit_tr = torch.from_numpy(y_unit_tr).long().to(DEVICE)
    type_tr = torch.from_numpy(y_type_tr).long().to(DEVICE)

    # ID labels: only real codas have them
    idn_set = {pos: label for pos, label in zip(idn_positions, y_id_all)}
    y_id_tr = np.full(len(train_idx), -1, dtype=np.int64)
    for j, idx in enumerate(train_idx):
        if idx in idn_set: y_id_tr[j] = idn_set[idx]
    # Synthetic codas have no ID label: append -1s
    if n_synth > 0:
        y_id_tr = np.concatenate([y_id_tr, np.full(n_synth, -1, dtype=np.int64)])
    id_tr = torch.from_numpy(y_id_tr).long().to(DEVICE)

    cc = np.bincount(y_unit_tr, minlength=n_units)
    wts = (1.0 / cc[y_unit_tr]).astype(np.float32)
    type_wt = torch.tensor(1.0/np.maximum(np.bincount(y_type_tr, minlength=n_types),1), dtype=torch.float32).to(DEVICE)
    id_wt   = torch.tensor(1.0/np.maximum(np.bincount(y_id_all,  minlength=n_ids),1),  dtype=torch.float32).to(DEVICE)
    unit_arrays = {u: np.where(y_unit_tr==u)[0] for u in range(n_units)}

    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.multinomial(torch.from_numpy(wts).float(), n_tr, replacement=True)
        epoch_loss = 0.0; n_b = 0
        for start in range(0, n_tr, batch_size):
            idx_b = perm[start:start+batch_size]
            if len(idx_b) < 4: continue
            ici_b, mel_b, unit_b = ici_tr[idx_b], mel_tr[idx_b], unit_tr[idx_b]
            type_b, id_b = type_tr[idx_b], id_tr[idx_b]

            # same-unit positive
            idx_pos = torch.tensor([int(np.random.choice(unit_arrays[unit_b[j].item()])) for j in range(len(idx_b))])
            out1 = model(ici_b, mel_b)
            out2 = model(ici_tr[idx_pos], mel_tr[idx_pos])
            z1, z2 = out1["z"], out2["z"]

            # cross-channel
            idx_cross = torch.tensor([int(np.random.choice(unit_arrays[unit_b[j].item()])) for j in range(len(idx_b))])
            r_a = model.rhythm_enc(ici_b)
            s_b = model.spectral_enc(mel_tr[idx_cross])
            z_cross = model.fusion(torch.cat([r_a, s_b], 1))

            L_cont = 0.5*nt_xent(z1, z2) + 0.5*nt_xent(z1, z_cross)
            L_type = F.cross_entropy(out1["type_logits"], type_b, weight=type_wt)
            labeled = id_b >= 0
            L_id = F.cross_entropy(out1["id_logits"][labeled], id_b[labeled], weight=id_wt) if labeled.sum() > 1 else torch.tensor(0.0, device=DEVICE)

            loss = L_cont + 0.5*L_type + 0.5*L_id
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); n_b += 1
        history.append(epoch_loss / max(n_b,1))
        if (epoch+1) % 25 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={history[-1]:.4f}")

    return model, history

def eval_model(model):
    def extract(ici_data, mel_data, idx):
        model.eval(); embs = []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                b = idx[s:s+128]
                i = torch.from_numpy(ici_data[b].astype(np.float32)).to(DEVICE)
                m = torch.from_numpy(mel_data[b].astype(np.float32)).to(DEVICE)
                embs.append(model(i, m)["z"].cpu().numpy())
        return np.vstack(embs)

    def probe(Ztr, Zte, ytr, yte):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42, solver="lbfgs")
        clf.fit(sc.fit_transform(Ztr), ytr)
        pred = clf.predict(sc.transform(Zte))
        return f1_score(yte, pred, average="macro"), accuracy_score(yte, pred)

    Z_tr = extract(X_ici, X_mel, train_idx)
    Z_te = extract(X_ici, X_mel, test_idx)
    f1u, acu = probe(Z_tr, Z_te, y_unit[train_idx], y_unit[test_idx])
    f1t, act = probe(Z_tr, Z_te, y_type[train_idx], y_type[test_idx])

    Z_id_tr = extract(X_ici[idn_positions], X_mel[idn_positions], train_id_idx)
    Z_id_te = extract(X_ici[idn_positions], X_mel[idn_positions], test_id_idx)
    f1i, aci = probe(Z_id_tr, Z_id_te, y_id_all[train_id_idx], y_id_all[test_id_idx])

    return {"unit_f1":f1u, "unit_acc":acu, "type_f1":f1t, "id_f1":f1i, "id_acc":aci}, Z_te

# ══════════════════════════════════════════════════════════════════════════════
# Augmentation sweep: {0, 100, 500, 1000}
# ══════════════════════════════════════════════════════════════════════════════
n_synth_vals = [0, 100, 500, 1000]
sweep_results = []
sweep_histories = {}
models_by_n = {}

t0 = time.time()
for n_synth in n_synth_vals:
    print(f"\n>>> N_synth={n_synth} <<<")
    m, h = train_with_synth(n_synth, epochs=50, seed=42)
    res, Z_te = eval_model(m)
    res["n_synth"] = n_synth
    res["n_train"] = len(train_idx) + n_synth
    sweep_results.append(res)
    sweep_histories[n_synth] = h
    models_by_n[n_synth]  = (m, Z_te)
    print(f"  unit={res['unit_f1']:.3f}  type={res['type_f1']:.3f}  indivID={res['id_f1']:.3f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Augmentation curve
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("DCCE-full Augmentation Sweep: Effect of Synthetic Codas", fontsize=13, fontweight="bold")

n_vals = [r["n_synth"] for r in sweep_results]
unit_f1s = [r["unit_f1"] for r in sweep_results]
type_f1s = [r["type_f1"] for r in sweep_results]
id_f1s   = [r["id_f1"]   for r in sweep_results]

axes[0].plot(n_vals, unit_f1s, marker="o", color="#2196F3", linewidth=2)
axes[0].set_title("Social Unit Macro-F1")
axes[0].set_xlabel("N synthetic added"); axes[0].set_ylabel("Macro-F1")
axes[0].set_xticks(n_vals); axes[0].axhline(0.895, linestyle="--", color="gray", label="WhAM L19")
for x, y in zip(n_vals, unit_f1s): axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,8), fontsize=8, ha="center")
axes[0].legend()

axes[1].plot(n_vals, type_f1s, marker="s", color="#FF9800", linewidth=2)
axes[1].set_title("Coda Type Macro-F1")
axes[1].set_xlabel("N synthetic added"); axes[1].set_ylabel("Macro-F1")
axes[1].set_xticks(n_vals); axes[1].axhline(0.931, linestyle="--", color="gray", label="ICI baseline")
for x, y in zip(n_vals, type_f1s): axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,8), fontsize=8, ha="center")
axes[1].legend()

axes[2].plot(n_vals, id_f1s, marker="^", color="#E91E63", linewidth=2)
axes[2].set_title("Individual ID Macro-F1")
axes[2].set_xlabel("N synthetic added"); axes[2].set_ylabel("Macro-F1")
axes[2].set_xticks(n_vals); axes[2].axhline(0.424, linestyle="--", color="gray", label="WhAM L10")
for x, y in zip(n_vals, id_f1s): axes[2].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,8), fontsize=8, ha="center")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_augmentation_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Augmentation curve saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Training curves (loss vs epoch for each N_synth)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
pal = sns.color_palette("viridis", len(n_synth_vals))
for color, (n_s, h) in zip(pal, sweep_histories.items()):
    ax.plot(h, label=f"N_synth={n_s}", color=color, linewidth=1.5)
ax.set_title("Training Curves — Augmentation Sweep", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_aug_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Aug training curves saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Sample synthetic mel spectrograms
# ══════════════════════════════════════════════════════════════════════════════
meta = pd.read_csv(f"{DATA}/synthetic_meta.csv")
X_mel_synth_view = np.load(f"{DATA}/X_mel_synth_1000.npy")  # (1000, 64, 128)

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
fig.suptitle("Sample Synthetic Mel-Spectrograms by Unit", fontsize=14, fontweight="bold")
np.random.seed(42)
unit_names = le_unit.classes_  # ['A', 'D', 'F']
for row, unit_idx_i in enumerate(range(3)):
    unit_name = unit_names[unit_idx_i]
    pool = np.where(y_unit_synth == unit_idx_i)[0]
    chosen = np.random.choice(pool, size=4, replace=False)
    for col, sidx in enumerate(chosen):
        mel = X_mel_synth_view[sidx]
        axes[row, col].imshow(mel, aspect="auto", origin="lower", cmap="magma")
        axes[row, col].set_title(f"Unit {unit_name} synth #{sidx}", fontsize=9)
        axes[row, col].set_xlabel("Time frames")
        axes[row, col].set_ylabel("Mel bin")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_synth_spectrograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("Synthetic spectrograms saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Mean mel profiles: real vs synthetic by unit
# ══════════════════════════════════════════════════════════════════════════════
X_mel_real_view = np.load(f"{DATA}/X_mel_full.npy")  # (1383, 64, 128)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Mean Mel Profiles: Real vs Synthetic by Unit", fontsize=14, fontweight="bold")
for col, unit_name in enumerate(unit_names):
    unit_idx_i = col
    real_mask = y_unit == unit_idx_i
    synth_mask = y_unit_synth == unit_idx_i
    mean_real  = X_mel_real_view[real_mask].mean(axis=0)   # (64, 128)
    mean_synth = X_mel_synth_view[synth_mask].mean(axis=0)
    axes[0, col].imshow(mean_real,  aspect="auto", origin="lower", cmap="magma")
    axes[0, col].set_title(f"Real Unit {unit_name} (n={real_mask.sum()})")
    axes[0, col].set_xlabel("Time frames"); axes[0, col].set_ylabel("Mel bin")
    axes[1, col].imshow(mean_synth, aspect="auto", origin="lower", cmap="magma")
    axes[1, col].set_title(f"Synth Unit {unit_name} (n={synth_mask.sum()})")
    axes[1, col].set_xlabel("Time frames"); axes[1, col].set_ylabel("Mel bin")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_synth_mel_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Synth mel profiles saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — UMAP: N_synth=0 vs N_synth=1000
# ══════════════════════════════════════════════════════════════════════════════
print("Computing UMAPs for N_synth=0 and N_synth=1000...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("DCCE-full Embeddings — Effect of Augmentation (test set UMAP)", fontsize=13, fontweight="bold")

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1)
for ax, n_s in zip(axes, [0, 1000]):
    _, Z_te = models_by_n[n_s]
    Z_umap = reducer.fit_transform(Z_te)
    y_unit_te = y_unit[test_idx]
    for unit_i, unit_name in enumerate(unit_names):
        m = y_unit_te == unit_i
        ax.scatter(Z_umap[m,0], Z_umap[m,1], c=UNIT_COLORS[unit_name], label=f"Unit {unit_name}",
                   alpha=0.6, s=15, rasterized=True)
    r = next(r for r in sweep_results if r["n_synth"] == n_s)
    ax.set_title(f"N_synth={n_s}: unit F1={r['unit_f1']:.3f}, id F1={r['id_f1']:.3f}")
    ax.legend(markerscale=2); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_aug_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Augmentation UMAP saved")

# ══════════════════════════════════════════════════════════════════════════════
# Save results CSV
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(sweep_results)
results_df.to_csv(f"{DATA}/phase4_results.csv", index=False)
print("\nphase4_results.csv saved:")
print(results_df[["n_synth","n_train","unit_f1","type_f1","id_f1","unit_acc","id_acc"]].to_string(index=False))

print("\n=== PHASE 4 SUMMARY ===")
for r in sweep_results:
    print(f"  N_synth={r['n_synth']:>4d}  D_train={r['n_train']:>4d}  "
          f"unit={r['unit_f1']:.3f}  type={r['type_f1']:.3f}  indivID={r['id_f1']:.3f}")
best_id = max(sweep_results, key=lambda r: r["id_f1"])
print(f"\nBest individual ID: N_synth={best_id['n_synth']} → F1={best_id['id_f1']:.3f}")
print(f"WhAM L10 baseline: F1=0.424")
print("All Phase 4 figures saved.")
