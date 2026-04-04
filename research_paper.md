# Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding

**A Research Paper Proposal built on Project CETI**

---

## Abstract

Project CETI (Cetacean Translation Initiative) has established the first
large-scale infrastructure and machine learning baseline for analyzing sperm whale
communication. Building on their most recent published result — the WhAM transformer
model (NeurIPS 2025) and the publicly available Dominica Sperm Whale Project (DSWP)
dataset — we propose a laptop-scale research program targeting three modest but
scientifically meaningful contributions: (1) a contrastive self-supervised objective
that jointly encodes *rhythm* (inter-click timing) and *spectral texture* (vowel)
channels of a coda, producing a richer representation than either alone; (2) a
systematic data-augmentation study using WhAM-generated synthetic codas to test
whether synthetic data improves downstream classification; and (3) a probing analysis
of WhAM's internal representations to understand what acoustic features determine
social-unit and coda-type clustering. All experiments run on a MacBook-class machine
(Apple MPS or CPU) using the CC-BY-4.0 DSWP dataset (1,501 codas, ~585 MB). We
outline concrete validation protocols against established biological ground truth,
providing a credible path to a publishable finding in bioacoustics or machine learning
for animal communication.

---

## 1. Introduction

Sperm whales (*Physeter macrocephalus*) are among the most cognitively sophisticated
animals on Earth. They possess the largest brain of any species, live in matrilineal
social units, and communicate through rhythmically patterned click sequences called
**codas**. Codas are believed to carry social information: groups of whales sharing
a coda repertoire form vocal clans, and family membership can be partially inferred
from acoustic style. Whether codas also carry information analogous to semantic
content — describing objects, actions, or states — is the central open question that
Project CETI is trying to answer.

CETI's approach is twofold: collect an unprecedented amount of in-context, tagged
whale audio (phase one), and apply state-of-the-art machine learning to find
regularities that might indicate meaning (phase two). As of early 2026, CETI has
published or preprinted several milestones across both phases.

However, an important gap remains. Existing ML work on codas either (a) focuses on
generation and treats classification as a secondary evaluation, or (b) focuses on
network-science metrics at the population level. No published work has performed a
controlled study of how the *two distinct information channels in a coda* — rhythm
(click timing) and spectral texture (vowel) — interact, complement, or conflict as
classification features. This gap is precisely where a researcher with a laptop and
public data can make a genuine contribution.

This paper outlines the intellectual context, the concrete experimental design, the
validation strategy, and the expected scope of findings.

---

## 2. Background: What Project CETI Has Achieved

### 2.1 Data Collection Infrastructure

CETI operates in the Caribbean Sea off the coast of Dominica, where a resident
population of ~20–25 sperm whales has been studied continuously since 2005 by
collaborator Shane Gero (Dominica Sperm Whale Project, DSWP). CETI augmented this
long-term monitoring with:

- **Bio-logging suction-cup tags** (CETI Tag v2/v2.5): Raspberry Pi Zero 2W-based
  devices with three synchronized hydrophones, pressure, orientation, IMU, temperature,
  and light sensors. Tags record up to 16.8 hours, survive depths to 560 m, and are
  recovered via GPS/ARGOS after shedding. Field deployments cover 10+ tag releases,
  44 hours of recording, and dives to 967 m.

- **Mooring arrays**: Up to ~200 synchronized underwater microphones providing
  far-field recordings and enabling passive acoustic monitoring (PAM) for
  localization and multi-whale tracking.

- **Aerial drones and swimming robots**: Used for behavioral context capture
  (surfacing video, group geometry).

- **Data pipeline** (`data-ingest` repo): Cloud-based ingestion, alignment, and
  archiving to produce a dataset consumable by downstream ML.

### 2.2 Core ML Publications (2022–2025)

#### 2.2.1 Theory of Unsupervised Translation (NeurIPS 2023)
*Goldwasser, Gruber, Kalai, Paradise (arXiv:2211.11081)*

This theoretical paper asks: *under what conditions is it possible to translate
between two languages without any parallel corpus?* The authors formalize the
problem as Unsupervised Machine Translation (UMT) and prove that sample complexity
bounds are inversely related to language complexity and the amount of shared
"common ground." The key practical implication: **UMT of animal communication may
be feasible if the communication system is rich enough** — motivating the empirical
program of characterizing that richness.

#### 2.2.2 Evidence of Social Learning Across Clan Boundaries (arXiv:2307.05304)
*Leitão, Lucas, Poetto, Hersh, Gero, Gruber, Bronstein, Petri*
*(submitted Jul 2023, revised Mar 2025)*

Using data from Atlantic and Pacific populations, this paper introduces a
computational model that encodes *rhythmic micro-variations* within codas — capturing
vocal style beyond just click count. Key findings:

- Vocal style-based clustering closely aligns with repertoire-based clan assignments.
- **Sympatry increases vocal style similarity for non-identity codas**, suggesting
  that whales learn vocal style from neighboring clans they interact with.
- Identity codas (35–60% of all codas) function as categorical cultural markers
  and *resist* style convergence; all other codas show cross-clan learning.

This provides the first quantitative evidence of cultural transmission beyond
clan boundaries, analogous to human dialect borrowing.

#### 2.2.3 Automatic Coda Detector and Annotator (arXiv:2407.17119)
*Gubnitsky, Mevorach, Gero, Gruber, Diamant (2024)*

The first automated pipeline for coda detection and annotation (coda type
classification), using graph-based clustering that exploits the expected similarity
between clicks within a single coda. Results include:

- Reliable detection at low SNR.
- Separation of codas from echolocation clicks.
- Discrimination between simultaneous vocalizers.
- Discovery of **new coda types** not in prior catalogs.
- Evidence of **synchronization** between communicating whales (both coda type and
  inter-coda timing).

This paper provides a scalable preprocessing baseline that any downstream ML work
can build on.

#### 2.2.4 WhAM — Whale Acoustics Model (NeurIPS 2025)
*Paradise, Muralikrishnan, Chen, Flores García, Pardo, Diamant, Gruber, Gero,
Goldwasser (arXiv:2512.02206)*

**WhAM is the current state of the art** in ML for sperm whale vocalization.
It is a transformer-based, masked acoustic token model (fine-tuned from VampNet, a
music audio generative model). Trained on ~10,000 coda recordings assembled from the
DSWP and additional CETI-tagged recordings, WhAM can:

1. **Generate** synthetic codas from any audio prompt (acoustic translation of
   speech, noise, or other animal sounds into the coda acoustic texture).
2. **Produce embeddings** used for downstream classification:
   - Social unit classification (which family group produced this coda?)
   - Rhythm / coda-type classification
   - Vowel (spectral texture) classification
3. **Synthesize** novel "pseudocodas" that fool expert marine biologists in
   perceptual studies.

**Key gap in WhAM**: it was optimized for *generation*, not *representation*. The
paper's classification results are presented as evidence of emergent representational
quality, but no ablation studies isolate which aspects of its training or
architecture drive performance. This leaves open: *What does WhAM actually encode?
Can a purpose-built representation do better?*

#### 2.2.5 Coda Vowel Phonology (GitHub: `Project-CETI/coda-vowel-phonology`)
*Beguš et al.*

A linguistically motivated analysis showing that click inter-pulse intervals within
a coda carry spectral variation analogous to vowels in human speech — hence the
term "coda vowels." This work establishes that codas have **two syntactically
independent information channels**:

- **Rhythm channel**: the pattern of inter-click intervals (e.g., 1+1+3 = one
  click, pause, one click, pause, three clicks) defines the coda **type**.
- **Spectral channel**: the spectral shape (formant-like structure) within the clicks
  carries **vowel** information, which correlates with individual and social-unit
  identity but is independent of coda type.

This decomposition is the key biological insight that motivates the present proposal.

### 2.3 Open Datasets

| Dataset | Size | Contents | License |
|---|---|---|---|
| DSWP (HF: `orrp/DSWP`) | 1,501 codas (~585 MB) | Isolated coda audio files, Dominica population, 2005–2018 | CC BY 4.0 |
| WMMS 'Best Of' Cut | ~hours | Cetacean audio, mixed species | Free research use |
| BirdSet | Large | Bird audio | Apache 2.0 |
| AudioSet (Animal subset) | Large | General animal audio | CC BY 4.0 |

The DSWP dataset is the most useful: it contains labeled codas with known social
unit provenance, enabling supervised validation. With 1,501 samples it is small
enough to train on a laptop in minutes.

---

## 3. The Scientific Gap This Paper Fills

Based on the literature above, the current frontier is:

> **WhAM produces embeddings that classify social unit, rhythm-type, and vowel-type
> well — but as an emergent byproduct of a generative objective trained on a music
> audio backbone. No work has purposefully designed a representation of codas that
> exploits the known biological decomposition (rhythm × vowel) to maximize
> individual/social-unit discriminability.**

Three concrete questions remain open:

**Q1:** Does jointly encoding both rhythm and spectral texture in a single low-dimensional
space produce classifiers that outperform using either channel alone, or using WhAM
embeddings?

**Q2:** Does augmenting the tiny DSWP training set with WhAM-generated synthetic codas
actually improve classifier generalization, or do synthetic artefacts hurt?

**Q3:** What are WhAM's embeddings geometrically doing? Do different dimensions encode
rhythm versus spectral information, and does this explain classification performance?

These questions are answerable with a laptop, public data, and publicly available
model weights. Positive or negative results on all three count as publishable
contributions to bioacoustic ML.

---

## 4. Proposed Research: Dual-Channel Contrastive Coda Representation

### 4.1 Biological Motivation

A coda can be fully described by two signals extracted from its waveform:

- **Rhythm feature** $r(\mathbf{x})$: the sequence of inter-click intervals (ICI),
  extracted via peak detection on the envelope. A 1+1+3 coda has ICIs
  $[t_{12}, t_{23}, t_{34}, t_{45}, t_{56}]$ where $t_{i,i+1}$ is the gap between
  consecutive click peaks. This is low-dimensional (typically 2–10 numbers) and
  has been used since the 1980s to define coda types.

- **Spectral feature** $s(\mathbf{x})$: a mel-spectrogram or MFCC summary of the
  waveform *within* each click inter-pulse interval, capturing the resonant structure
  (vowels). This has been formalized by Beguš et al. and shown to carry social
  information independent of coda type.

The rhythm channel encodes **what kind** of coda it is (categorical).
The spectral channel encodes **who** is talking (individual/social-unit identity).
Both channels are needed to fully identify a communication event.

### 4.2 The Proposed Model: Dual-Channel Contrastive Encoder (DCCE)

We propose a lightweight model with three components:

```
Waveform (x)
    │
    ├──── Rhythm Encoder (R) ──────► r_emb  ─┐
    │       (1D-CNN or GRU on ICIs)            │
    │                                          ├──► Fusion MLP ──► z (joint embedding)
    └──── Spectral Encoder (S) ────► s_emb  ─┘
            (small CNN on mel-spectrogram)
```

**Rhythm Encoder R**: Takes the sequence of inter-click intervals as input — a
vector of $k$ real numbers where $k$ varies by coda (typically 2–22). A single-layer
GRU or a small 1D-CNN produces a fixed-size rhythm embedding $r_{\text{emb}} \in \mathbb{R}^{64}$.

**Spectral Encoder S**: Takes a fixed-size mel-spectrogram crop of the coda audio
(e.g., 128 mel bins × 128 time frames, normalized). A ~5-layer CNN produces
$s_{\text{emb}} \in \mathbb{R}^{64}$.

**Fusion MLP**: Concatenates $[r_{\text{emb}}; s_{\text{emb}}]$ and projects to
$z \in \mathbb{R}^{64}$ via two linear layers with LayerNorm.

**Training Objective**: A combination of:

1. **Contrastive loss (SimCLR-style)** using augmented views of the same coda as
   positives and codas from different whales as hard negatives. The key novel choice
   is creating **cross-channel augmentation pairs**: the rhythm of coda A paired with
   the spectral texture of another coda from the *same whale* as coda A is still a
   "positive" — both channels identify the same speaker. This forces $z$ to capture
   speaker identity regardless of coda type.

2. **Coda-type prediction head** on $r_{\text{emb}}$ only (supervised with coda-type
   labels where available), to ensure R captures the rhythm/categorical dimension.

3. **Individual-ID contrastive loss** on $s_{\text{emb}}$ only, forcing S to
   be sensitive to vowel variation.

Mathematically, the full loss is:

$$\mathcal{L} = \mathcal{L}_{\text{contrastive}}(z) + \lambda_1 \cdot \mathcal{L}_{\text{type}}(r_{\text{emb}}) + \lambda_2 \cdot \mathcal{L}_{\text{id}}(s_{\text{emb}})$$

where $\lambda_1$ and $\lambda_2$ are hyperparameters (default: $\lambda_1 = \lambda_2 = 0.5$).

**Why this is feasible on a laptop**: The DSWP has 1,501 codas. Each training forward
pass processes mel-spectrograms of ~0.5–2 s long signals and a vector of $\leq 22$
ICI values. The CNN has ~500k parameters. Training for 100 epochs on 1,200 samples
takes ~10 minutes on CPU and ~2 minutes on Apple MPS.

### 4.3 Experiment 1: Representation Quality (Q1)

**Setup**: 80/20 train/test split of DSWP. Train DCCE with the dual objective. Train
ablated versions: rhythm-only ($z = r_{\text{emb}}$ alone), spectral-only
($z = s_{\text{emb}}$ alone), and late-fusion without cross-channel augmentation.
Also extract WhAM embeddings (using the publicly available Zenodo weights) for the
same test set as an additional baseline.

**Downstream tasks** (linear probes on frozen embeddings):
- Social unit classification (k = ~7 social units in DSWP)
- Coda-type classification (~50+ known types)
- Individual-whale identification (treat each named individual as a class)

**Primary metric**: Top-1 accuracy on linear probe (logistic regression). Secondary
metrics: macro-F1 (important due to class imbalance), and silhouette score in
t-SNE space.

**Expected finding**: We hypothesize that DCCE's joint embedding will outperform
single-channel baselines and match or exceed WhAM on social-unit and individual-ID
tasks (since it is purpose-built for identity-discriminative representation). We
expect WhAM to remain competitive on coda-type classification because it was trained
on far more data.

**Scientific claim**: If the joint embedding outperforms both single-channel models,
this provides direct evidence that rhythm and spectral channels carry *complementary*
social information in sperm whale codas — a novel empirical result.

### 4.4 Experiment 2: Synthetic Data Augmentation Study (Q2)

WhAM provides a coda generator conditioned on audio prompts. This enables the first
study of whether generative augmentation helps downstream bioacoustic classification.

**Protocol**:
1. Use WhAM to generate $N_{\text{synth}} \in \{0, 100, 500, 1000, 2000\}$ synthetic
   codas, conditioned on subsets of the training data (prompt = a real coda from
   each social unit).
2. Train a fresh DCCE (or a simpler CNN classifier flattened on spectrograms) on:
   - $D_{\text{train}}$ only
   - $D_{\text{train}} \cup D_{\text{synth}}$
3. Evaluate on the held-out $D_{\text{test}}$ (real codas only).
4. Measure accuracy as a function of $N_{\text{synth}}$.

**Expected finding**: The augmentation benefit curve may show an initial gain (more
diverse training signal) followed by a degradation at very high synthetic fractions
(domain shift from generative artefacts). Alternatively, we may find no benefit,
which is a negative result that bounds the reliability of WhAM's generation for
this downstream task.

**Why this matters**: This is the first controlled data augmentation experiment in
sperm whale bioacoustics. The result directly informs how the community should use
WhAM in future work with limited data.

### 4.5 Experiment 3: Probing WhAM Representations (Q3)

**Protocol**:
1. Extract WhAM (VampNet coarse + c2f model) intermediate-layer embeddings for all
   1,501 DSWP codas.
2. Run a set of linear probing classifiers targeting acoustic properties available
   from the DSWP labels and from automated feature extraction:
   - Number of clicks in coda → tests if WhAM encodes rhythm length
   - Mean ICI → tests tempo encoding
   - Dominant spectral centroid → tests vowel encoding
   - Social unit label → tests identity encoding
   - Recording system (towed hydrophone vs. D-tag) → tests domain confounding
3. Compare probe accuracy across different transformer layers (if the architecture
   exposes intermediate activations) to produce a **probing profile**.
4. Compute UMAP of the embedding space, colored by each label type.

**Expected finding**: Following probing analysis conventions from NLP (Tenney et al.,
2019), we expect early/shallow representations to capture low-level acoustic
properties (spectral centroid), while later representations capture higher-level
social structure. The recording-system probe tests whether WhAM is partly capturing
hardware domain shift rather than biological variation — a confound not tested in
the original paper.

**Why this matters**: It is the first interpretability analysis of WhAM, and
provides actionable guidance for task-specific fine-tuning.

---

## 5. Dataset & Feature Extraction Pipeline

### 5.1 Data Sources

**Training and evaluation**: DSWP dataset (HuggingFace: `orrp/DSWP`), CC BY 4.0.
1,501 codas extracted and isolated from longer recordings. Known social unit labels
are available from the original DSWP field catalog (Gero, 2005–2018) and are
used in WhAM's downstream tasks.

**Pre-training for domain adaptation** (optional, if extending WhAM fine-tuning):
WMMS 'Best Of' cut (cetacean audio), AudioSet Animal subset, BirdSet — all used in
WhAM's original training pipeline (available openly).

### 5.2 Feature Extraction

Implemented in Python using:

```python
# Dependencies
librosa          >= 0.10   # audio loading, STFT, mel-spectrogram
scipy            >= 1.11   # signal processing, peak detection for ICIs
torch            >= 2.2    # MPS-compatible DL framework
torchaudio       >= 2.2    # audio augmentation
datasets         >= 2.18   # HuggingFace dataset access
transformers     >= 4.39   # (optional) AVES embeddings
```

**ICI extraction (rhythm)**:
```python
import librosa, numpy as np, scipy.signal

def extract_icis(y, sr=44100):
    """Extract inter-click interval sequence from a coda waveform."""
    envelope = np.abs(librosa.effects.harmonic(y))
    peaks, _ = scipy.signal.find_peaks(
        envelope,
        height=np.percentile(envelope, 90),
        distance=int(0.002 * sr)     # min 2 ms between clicks
    )
    if len(peaks) < 2:
        return np.array([])
    icis = np.diff(peaks) / sr       # convert to seconds
    return icis
```

**Mel-spectrogram extraction (spectral)**:
```python
def extract_mel(y, sr=44100, n_mels=64, n_frames=64):
    """Extract fixed-size log-mel-spectrogram from a coda waveform."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    logS = librosa.power_to_db(S, ref=np.max)
    # Pad or crop to fixed length
    if logS.shape[1] < n_frames:
        logS = np.pad(logS, ((0, 0), (0, n_frames - logS.shape[1])))
    else:
        logS = logS[:, :n_frames]
    return logS.astype(np.float32)
```

**WhAM embedding extraction** (for Experiment 3):
```python
# Requires: pip install -e ./wham; download weights from Zenodo
from wham.embedding import extract_embeddings
embeddings = extract_embeddings(waveform_list, model_path="vampnet/models/")
```

### 5.3 Data Loading from HuggingFace

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("orrp/DSWP", split="train")

# Each example: {'audio': {'array': np.ndarray, 'sampling_rate': int},
#                'label': str  (e.g., coda type or social unit)}

# Precompute features
rhythms = [extract_icis(ex['audio']['array'], ex['audio']['sampling_rate'])
           for ex in ds]
mels    = [extract_mel(ex['audio']['array'], ex['audio']['sampling_rate'])
           for ex in ds]
```

---

## 6. Model Architecture (Full Detail)

### 6.1 Rhythm Encoder

```python
import torch
import torch.nn as nn

class RhythmEncoder(nn.Module):
    """GRU-based encoder for variable-length ICI sequences."""
    def __init__(self, input_size=1, hidden_size=64, output_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.GELU()
        )

    def forward(self, ici_padded, lengths):
        # ici_padded: (B, max_len, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            ici_padded, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h_n = h_n[-1]            # last layer, shape (B, hidden_size)
        return self.proj(h_n)    # (B, output_size)
```

### 6.2 Spectral Encoder

```python
class SpectralEncoder(nn.Module):
    """Small CNN for log-mel-spectrogram input."""
    def __init__(self, n_mels=64, n_frames=64, output_size=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, output_size)
        )

    def forward(self, mel):
        # mel: (B, 1, n_mels, n_frames)
        x = self.cnn(mel)
        return self.proj(x)   # (B, output_size)
```

### 6.3 Training Loop Sketch

```python
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.1):
    """Normalized temperature-scaled cross-entropy (SimCLR)."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.T / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def train_epoch(model, loader, optimizer, lambda1=0.5, lambda2=0.5):
    model.train()
    total_loss = 0
    for batch in loader:
        r_emb = model.rhythm_encoder(batch['ici'], batch['ici_lengths'])
        s_emb = model.spectral_encoder(batch['mel'])
        z     = model.fusion(torch.cat([r_emb, s_emb], dim=-1))

        # Augmented views (audio jitter in amplitude + time-stretch for spectral,
        # small gaussian noise on ICIs for rhythm)
        r_emb2 = model.rhythm_encoder(batch['ici_aug'], batch['ici_lengths'])
        s_emb2 = model.spectral_encoder(batch['mel_aug'])
        z2     = model.fusion(torch.cat([r_emb2, s_emb2], dim=-1))

        loss = (nt_xent_loss(z, z2)
                + lambda1 * F.cross_entropy(model.type_head(r_emb), batch['coda_type'])
                + lambda2 * nt_xent_loss(s_emb, s_emb2))   # individual-level

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

---

## 7. Validation Strategy

The key challenge in any ML work on animal communication is that there is no
"ground truth" meaning label to validate against. Instead, the field uses a
hierarchy of biological proxies:

### Level 1 — Social Unit Classification (Strongest Proxy)
DSWP has exhaustively catalogued which named whale appears in which recording.
Social units (matrilineal family groups, size ~3–10) are the natural classification
target for identity-bearing features. Social unit membership is the most externally
validated label because it is confirmed by visual photo-ID, not just acoustics.

**Validation**: Train/test split DSWP by coda occurrence date (not random), ensuring
the test set contains sessions recorded in years not seen during training. This tests
temporal generalization — critical for a real-world system that encounters new
recordings.

### Level 2 — Coda Type Matching (Medium Proxy)
The coda-type catalog (e.g., 1+1+3, 1+2, 4+1+1) is established behaviorally.
A good rhythm encoder should cluster same-type codas regardless of who produces
them. Validation: compare predicted cluster assignments to type labels using
Adjusted Rand Index (ARI).

### Level 3 — Consistency with Expert Bio-acoustic Feature Analysis
Beguš et al.'s coda vowel phonology paper provides quantitative acoustic feature
vectors for a subset of DSWP codas (formant frequencies, spectral centroid, etc.).
We can check whether $s_{\text{emb}}$ cosine similarities correlate with vowel
feature distances — a direct test of spectral encoding quality without requiring
full labels.

### Level 4 — Negative Controls
Any representation that achieves social-unit classification can do so by memorizing
recording artifacts (microphone, SNR, boat position). We run recording-domain probes
(recording system as the classification target) to verify the model captures biology,
not hardware.

### Summary of Metrics

| Task | Metric | Baseline |
|---|---|---|
| Social unit classification | Top-1 acc, Macro-F1 | WhAM embedding + LogReg |
| Coda type classification | Top-1 acc, ARI | ICI k-means baseline |
| Individual whale ID | Top-1 acc | WhAM embedding + LogReg |
| Vowel feature correlation | Pearson r with Beguš et al. features | MFCC cosine |
| Augmentation benefit | Δ accuracy vs. N_synth | No augmentation (N=0) |

---

## 8. Running Experiments on a Laptop

### 8.1 Environment Setup

```bash
# Create environment (Python 3.10 recommended for torch MPS)
conda create -n ceti python=3.10
conda activate ceti

# PyTorch with Apple Silicon MPS support
pip install torch torchvision torchaudio    # automatically uses MPS on M1–M4

# Audio and data
pip install librosa datasets huggingface_hub scipy scikit-learn

# WhAM (for baselines and augmentation)
git clone https://github.com/Project-CETI/wham.git
cd wham
pip install -e .
pip install -e ./vampnet
pip install --no-build-isolation madmom
conda install -c conda-forge ffmpeg

# Download WhAM weights from Zenodo
# (see https://zenodo.org/records/17633708)
```

### 8.2 Data Download

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Download the DSWP dataset (~585 MB)
ds = load_dataset("orrp/DSWP", cache_dir="./data/dswp")
print(f"Loaded {len(ds['train'])} codas")
```

### 8.3 Hardware and Runtime Estimates

| Stage | Hardware | Estimated Time |
|---|---|---|
| Feature extraction (all 1,501 codas) | MacBook Pro M2/M3 CPU | ~5 min |
| DCCE training (100 epochs, batch=32) | MPS | ~8 min |
| DCCE training (100 epochs, batch=32) | CPU only | ~25 min |
| WhAM embedding extraction (inference) | CPU | ~30 min |
| Linear probe evaluation (all tasks) | CPU/MPS | < 1 min |
| UMAP visualization (1,501 points) | CPU | ~2 min |
| Full paper experiment suite | MPS | ~2 hours |

All estimates are for a MacBook Pro with Apple Silicon (M2/M3). A plain Intel CPU
will be ~3–5× slower but still very tractable within a few hours.

### 8.4 Expected Disk Usage

| Item | Size |
|---|---|
| DSWP audio | ~585 MB |
| Extracted features (npy cache) | ~50 MB |
| WhAM model weights | ~800 MB |
| VampNet base weights | ~1.2 GB |
| Total | ~2.7 GB |

This is manageable on any modern laptop.

---

## 9. Expected Contributions and Novelty

### 9.1 Minimal Positive Result (publishable at a workshop or bioacoustics journal)

If the dual-channel encoder achieves even a +5% improvement over the single-channel
baselines on social-unit classification, that constitutes the **first principled
architecture designed around the known bimodal structure of sperm whale codas**. The
comparison itself (rhythm-only vs. spectral-only vs. joint vs. WhAM) is a useful
empirical contribution regardless of the sign of the delta.

### 9.2 Stronger Result (publishable at a ML conference, e.g., NeurIPS or ICLR)

If the joint embedding substantially outperforms WhAM on identity tasks, the
contribution is: **domain-specific architecture design that beats a much larger
generalist model** — a result of general interest to the bioacoustic ML community.

### 9.3 Augmentation Study (standalone contribution)

The WhAM synthetic augmentation study is independently publishable because:
- It is the first empirical test of CETI's generation pipeline as a data augmentation
  tool.
- It provides a lower bound on how useful generation-quality models are for
  downstream supervised tasks.
- Negative results are valuable: if synthetic data does *not* help, that informs
  future CETI priorities (collect more real data vs. improve generators).

### 9.4 Probing Study (interpretability contribution)

The WhAM probing analysis contributes to the growing field of representation
probing for audio models, applied for the first time to cetacean bioacoustics.

---

## 10. Limitations and Future Work

### 10.1 Dataset Size
1,501 codas from a single population (Caribbean, Eastern Clan) is a small and
geographically skewed sample. All results should be interpreted relative to the
Dominica population; generalization to Pacific clans (e.g., Regular, Whalers) would
require the Pacific dataset used in arXiv:2307.05304, which is not yet fully public.

### 10.2 Label Availability
Social unit labels for DSWP require cross-referencing with the original DSWP field
database (maintained by Shane Gero). The HuggingFace release includes audio but
metadata granularity (per-coda individual ID) may require contacting the authors.
WhAM's paper used "DSWP+CETI annotated" data that is partially restricted.

**Workaround**: Unsupervised evaluation (ARI against automatically labeled coda
types, silhouette score, UMAP visual inspection) does not require per-coda labels
and is sufficient for Experiment 1 and 3.

### 10.3 No Behavioral Context
Codas in DSWP are isolated snippets without behavioral metadata (dive depth at time
of production, social group composition at the moment, preceding codas in a
sequence). This limits the semantic analysis possible; future work connecting WhAM
or DCCE embeddings to CETI-tag behavioral data would be a natural next step once
more multi-modal data is released.

### 10.4 Ethical Considerations
All experiments use published, open-access data under CC BY 4.0. No new animal
interactions are required. CETI's data collection follows Dominica government
regulations and marine mammal protection guidelines; the analysis here does not
change the welfare footprint of the research.

---

## 11. Related Work (Extended)

| Paper | Key ML Technique | Relevance |
|---|---|---|
| Hersh et al. (2022), *Nat. Comms.* | Manual feature + statistics | Coda structure baseline |
| Leitão et al. (arXiv:2307.05304) | Network science, rhythm micro-variation model | Social learning measurement |
| Goldwasser et al. (NeurIPS 2023) | Theoretical UMT bounds | Feasibility framework |
| Gubnitsky et al. (arXiv:2407.17119) | Graph-based clustering | Coda detector pipeline |
| Paradise et al. / WhAM (NeurIPS 2025) | Transformer masked token modeling | State-of-the-art baseline |
| Beguš et al. (GitHub) | Phonological analysis | Vowel feature ground truth |
| Tenney et al. (2019), "BERT rediscovers NLP pipeline" | Probing classifiers | Methodology for Experiment 3 |
| Chen et al., SimCLR (ICML 2020) | Contrastive self-supervised learning | Core training algorithm |
| Kahl et al., BirdNET (2021) | CNN audio classification | Transfer learning baseline |
| MaCaulay Library / AVES (Hagiwara et al., 2022) | Transformer pre-trained on bird audio | Off-the-shelf embedding baseline |

---

## 12. References

1. **Goldwasser, S., Gruber, D.F., Kalai, A.T., Paradise, O.** (2023). A Theory of
   Unsupervised Translation Motivated by Understanding Animal Communication.
   *NeurIPS 2023*. arXiv:2211.11081.

2. **Leitão, A., Lucas, M., Poetto, S., Hersh, T.A., Gero, S., Gruber, D., Bronstein,
   M., Petri, G.** (2025, revised). Evidence of social learning across symbolic
   cultural barriers in sperm whales. arXiv:2307.05304.

3. **Gubnitsky, G., Mevorach, Y., Gero, S., Gruber, D.F., Diamant, R.** (2024).
   Automatic Detection and Annotation of Sperm Whale Codas. *arXiv:2407.17119*.

4. **Paradise, O., Muralikrishnan, P., Chen, L., Flores García, H., Pardo, B.,
   Diamant, R., Gruber, D.F., Gero, S., Goldwasser, S.** (2025). WhAM: Towards A
   Translative Model of Sperm Whale Vocalization. *NeurIPS 2025*. arXiv:2512.02206.

5. **Beguš, G. et al.** (2024). The phonology of sperm whale coda vowels.
   GitHub: Project-CETI/coda-vowel-phonology.

6. **Delpreto, J., et al.** (2024). An Open-Source Bio-Logging Sensor Tag for Marine
   Animals. *CETI TAG Documentation*. ceti-tag.csail.mit.edu.

7. **Gero, S., Gordon, J., Whitehead, H.** (2015). Individualized social preferences
   and long-term social fidelity in network associations of sperm whales.
   *Animal Behaviour*, 102, 15–23.

8. **Chen, T., Kornblith, S., Norouzi, M., Hinton, G.** (2020). A Simple Framework
   for Contrastive Learning of Visual Representations. *ICML 2020*.

9. **Flores García, H., et al.** (2023). VampNet: Music Generation via Masked
   Acoustic Token Modeling. *ISMIR 2023*.

10. **Tenney, I., Das, D., Pavlick, E.** (2019). BERT Rediscovers the Classical NLP
    Pipeline. *ACL 2019*.

11. **Hagiwara, M., et al.** (2022). AVES: Animal Vocalization Encoder Based on
    Self-Supervision. *ICASSP 2023*. arXiv:2210.14493.

---

## Appendix A: Project Timeline (for a single researcher on a laptop)

| Week | Task |
|---|---|
| 1 | Download DSWP, install dependencies, run WhAM inference, explore data |
| 2 | Implement ICI extractor and mel feature pipeline, cache to disk |
| 3 | Implement RhythmEncoder + SpectralEncoder, run sanity checks |
| 4 | Train DCCE, run linear probe evaluations (Experiment 1) |
| 5 | Conduct augmentation sweep (Experiment 2) |
| 6 | Extract WhAM embeddings, run probing classifiers, UMAP (Experiment 3) |
| 7 | Statistical analysis, write up results, produce figures |
| 8 | Draft paper, revise, submit to conference or journal |

---

## Appendix B: Key Coda Vocabulary

| Term | Definition |
|---|---|
| **Coda** | A stereotyped sequence of 2–23 broadband clicks produced by sperm whales during socializing |
| **Coda type** | Categorical classification by click count and rhythm pattern (e.g., 1+1+3 = 5 clicks) |
| **ICI** | Inter-Click Interval — time between consecutive click peaks in a coda |
| **Vowel** | Spectral variation within ICI; analogous to vowels in speech |
| **Social unit** | Matrilineal family group of ~3–10 females + juvenile males |
| **Clan** | Large cultural group sharing a coda repertoire; may span 100s of whales across thousands of km |
| **Identity coda** | Clan-specific coda type (35–60% of production) used as cultural markers |
| **Rhythm channel** | ICI sequence → encodes coda type (categorical, shared across clan) |
| **Spectral channel** | Click spectral content → encodes individual/unit identity |
| **PAM** | Passive Acoustic Monitoring — recording animal sounds without active sonar |
| **D-tag / CETI tag** | Suction-cup bio-logger attached to whale's back |

---

*Paper prepared: March 2026.*
*Author: [Your Name]*
*Course: CS 297 Final Project*
*Data: DSWP CC BY 4.0 (Gero / Project CETI, released 2025)*
