"""
Extract WhAM (VampNet coarse transformer) embeddings for all 1,501 DSWP codas.

Usage:
    source wham_env/bin/activate
    python extract_wham_embeddings.py

Output:
    datasets/wham_embeddings.npy  — shape (1501, 768), layer-10 mean-pooled embeddings
    datasets/wham_embeddings_all_layers.npy — shape (1501, 20, 768), all layers

The vampnet_embed function follows the JukeMIR convention (layer 10 for downstream probes).
Embeddings are computed with no gradient, on MPS if available, falling back to CPU.
"""

import os, sys
import numpy as np
import torch
import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR  = os.path.join(BASE, "datasets", "dswp_audio")
MODELS_DIR = os.path.join(BASE, "wham", "vampnet", "models")
OUT_L10    = os.path.join(BASE, "datasets", "wham_embeddings.npy")
OUT_ALL    = os.path.join(BASE, "datasets", "wham_embeddings_all_layers.npy")

COARSE_CKPT = os.path.join(MODELS_DIR, "coarse.pth")
CODEC_CKPT  = os.path.join(MODELS_DIR, "codec.pth")

N_CODAS = 1501
LAYER   = 10     # JukeMIR convention: middle layer for downstream tasks

# ── device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ── verify weights ────────────────────────────────────────────────────────────
for path, name in [(COARSE_CKPT, "coarse.pth"), (CODEC_CKPT, "codec.pth")]:
    if not os.path.exists(path):
        print(f"ERROR: {name} not found at {path}")
        print("Download from: https://zenodo.org/records/17633708")
        sys.exit(1)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {name}: {size_mb:.0f} MB  OK")

# ── load interface ─────────────────────────────────────────────────────────────
print("\nLoading WhAM Interface (codec + coarse transformer)...")
from vampnet.interface import Interface
from audiotools import AudioSignal

interface = Interface(
    coarse_ckpt=COARSE_CKPT,
    codec_ckpt=CODEC_CKPT,
    coarse2fine_ckpt=None,   # not needed for embeddings
    wavebeat_ckpt=None,      # not needed for embeddings
    device=DEVICE,
)
interface.eval()
print("Interface loaded.")
print(f"  Coarse model layers: ", end="")
try:
    # get number of transformer layers
    n_layers = len(interface.coarse.transformer.layers)
    print(n_layers)
except Exception:
    print("unknown")

# ── embedding function ─────────────────────────────────────────────────────────
def vampnet_embed_all_layers(sig: AudioSignal, iface: Interface):
    """
    Returns mean-pooled embeddings for all transformer layers.
    Shape: (n_layers, hidden_dim)
    Follows visualize_embeddings.py vampnet_embed() from the WhAM repo.
    """
    with torch.inference_mode():
        sig = iface.preprocess(sig)
        vampnet = iface.coarse
        z = iface.encode(sig)[:, : vampnet.n_codebooks, :]
        z_latents = vampnet.embedding.from_codes(z, iface.codec)
        _z, embeddings = vampnet(z_latents, return_activations=True)
        # embeddings: [n_layers, batch=1, time, hidden_dim]
        embeddings = embeddings.squeeze(1)          # [n_layers, time, hidden_dim]
        embeddings = embeddings.mean(dim=1)         # [n_layers, hidden_dim]
        return embeddings.cpu().numpy()

# ── extraction loop ────────────────────────────────────────────────────────────
print(f"\nExtracting embeddings for {N_CODAS} codas (layer {LAYER} + all layers)...")

emb_l10      = None   # allocated after first coda (dimension unknown until then)
emb_all      = None   # allocated after first coda so we know n_layers
failed_ids   = []

for coda_id in tqdm.tqdm(range(1, N_CODAS + 1), desc="WhAM embed"):
    wav_path = os.path.join(AUDIO_DIR, f"{coda_id}.wav")
    if not os.path.exists(wav_path):
        print(f"  WARNING: {wav_path} not found, skipping")
        failed_ids.append(coda_id)
        continue
    try:
        sig = AudioSignal(wav_path)
        emb = vampnet_embed_all_layers(sig, interface)   # (n_layers, hidden_dim)

        # allocate arrays on first success — dynamic so we handle any hidden_dim
        if emb_all is None:
            n_layers, hidden_dim = emb.shape
            emb_l10 = np.zeros((N_CODAS, hidden_dim), dtype=np.float32)
            emb_all = np.zeros((N_CODAS, n_layers, hidden_dim), dtype=np.float32)
            print(f"\n  Embedding shape per coda: {emb.shape}  (n_layers={n_layers}, hidden={hidden_dim})")

        emb_l10[coda_id - 1]  = emb[LAYER]
        emb_all[coda_id - 1]  = emb

    except Exception as e:
        print(f"\n  ERROR on coda {coda_id}: {e}")
        failed_ids.append(coda_id)

# ── save ───────────────────────────────────────────────────────────────────────
if emb_l10 is not None:
    np.save(OUT_L10, emb_l10)
    print(f"\nSaved layer-{LAYER} embeddings → {OUT_L10}  shape={emb_l10.shape}")
else:
    print("\nERROR: no embeddings were produced — check errors above")

if emb_all is not None:
    np.save(OUT_ALL, emb_all)
    print(f"Saved all-layer embeddings  → {OUT_ALL}  shape={emb_all.shape}")

if failed_ids:
    print(f"\nFailed coda IDs ({len(failed_ids)}): {failed_ids}")
else:
    print(f"\nAll {N_CODAS} codas embedded successfully.")
