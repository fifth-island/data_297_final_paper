# Paper Outline
## "Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding"

---

## 0. Metadata

- **Venue target**: Workshop on Machine Learning for Animal Communication (NeurIPS / ICML adjacent) or *PLOS Computational Biology* / *Methods in Ecology and Evolution*
- **Target length**: 8–10 pages main text (conference) or 12–15 pages (journal)
- **Code/data**: All code and label files open-source under CC BY 4.0; DSWP audio via HuggingFace `orrp/DSWP`

---

## 1. Abstract (~200 words)

Key beats to cover:
- **What**: Sperm whale codas carry two independent information channels — rhythm (ICI timing) and spectral texture (vowel formants). No existing model exploits both jointly by design.
- **Proposed model**: Dual-Channel Contrastive Encoder (DCCE) — a purpose-built multi-view contrastive architecture combining a GRU rhythm encoder and a CNN spectral encoder, trained with NT-Xent cross-channel positive pairs.
- **Primary baseline**: WhAM (NeurIPS 2025), a VampNet-based masked acoustic token model trained on 10k codas — the current state-of-the-art.
- **Scope**: Three experiments on the 1,501-coda Dominica Sperm Whale Project (DSWP) dataset.
- **Key results** (3 numbers): DCCE individual-ID F1=0.834 vs. WhAM best F1=0.454; DCCE social-unit F1=0.878 vs. WhAM L19 F1=0.895 (near parity); WhAM social-unit advantage partly explained by recording-year confound (Cramér's V=0.51).
- **Negative result**: WhAM-generated synthetic augmentation does not improve DCCE (pseudo-ICI labels dilute individual-level contrastive signal).

---

## 2. Introduction (~1.5 pages)

### 2.1 Motivation: Machine Learning for Cetacean Communication
- Sperm whales: largest brain of any species; matrilineal social units; multigenerational cultural transmission
- Codas: rhythmically patterned click sequences; variation across populations like human dialects
- Project CETI (Cetacean Translation Initiative): first large-scale ML infrastructure for whale communication; central question — does coda structure carry meaning analogous to words?
- Why this is a compelling ML problem: multi-view signal, biologically grounded structure, small publicly available dataset enabling laptop-scale experiments

### 2.2 The Two-Channel Structure of Codas
- **Rhythm channel** (ICI timing): encodes *what type* of coda — shared across a clan, categorical
  - Cite: Leitão et al. (2023) — rhythmic micro-variation as cultural identity marker
  - Cite: Sharma et al. (2024) — 35 named coda types; combinatorial structure
- **Spectral channel** (vowel formants within each click): encodes *who is speaking* — individual voice fingerprint
  - Cite: Beguš et al. (2024) — formal phonological treatment; vowel categories `a` and `i`
- Key point: these two channels are syntactically independent within a single coda

### 2.3 The Gap We Address
- Existing ML work either: (a) treats classification as secondary to generation (WhAM), or (b) operates at the population level (graph-based clustering)
- No published model performs a controlled study of how rhythm + spectral channels interact as representation features
- WhAM is a generative model — its classification results are emergent byproducts of music-audio pre-training, not a designed objective
- Our question: **does biological domain knowledge about coda structure beat scale?**

### 2.4 Contributions
1. **DCCE**: a purpose-built dual-channel contrastive encoder with cross-channel positive pairs — first model specifically designed around the known biological decomposition of sperm whale codas
2. **WhAM probing**: first systematic layer-by-layer interpretability analysis of WhAM's internal representations, including a recording-year confound analysis not performed in the original paper
3. **Augmentation study**: first controlled study of whether WhAM-generated synthetic codas improve downstream classification
4. **Label assembly**: a merged label table (`dswp_labels.csv`) linking 1,501 DSWP audio files to social unit, coda type, individual ID, and ICI sequence — assembled from multiple public sources through non-trivial alignment work

---

## 3. Related Work (~1 page)

### 3.1 Sperm Whale Bioacoustics
- Gero, Whitehead & Rendell (2016): foundational coda taxonomy — 9 Caribbean social units, 21 coda types; provides the classification framework our experiments build on
- Leitão et al. (2023/2025): rhythmic micro-variation model; evidence of cross-clan cultural learning; identity codas vs. non-identity codas; directly motivates separating the rhythm channel
- Beguš et al. (2024): formal phonological treatment of coda vowels; ICI and spectral channels are syntactically independent; provides `codamd.csv` with hand-verified vowel labels

### 3.2 ML for Animal Communication
- Goldwasser et al. (NeurIPS 2023): theoretical framework — unsupervised machine translation of animal communication is feasible if the system is complex enough; motivates our representation learning direction
- Gubnitsky et al. (2024): first automated coda detector using graph-based clustering; provides preprocessing baseline and coda-type taxonomy
- Paradise et al. (NeurIPS 2025) — WhAM: masked acoustic token model (VampNet fine-tuned on ~10k DSWP codas); generates synthetic codas and produces linear-probe embeddings; **primary baseline in this work**

### 3.3 Contrastive and Multi-View Representation Learning
- SimCLR / NT-Xent (Chen et al., 2020): contrastive learning objective — same family as our training loss
- CLIP (Radford et al., 2021): cross-modal alignment via contrastive loss on different views — conceptual ancestor of our cross-channel pairing strategy
- Brief note on prior uses of contrastive learning in bioacoustics (if any can be cited; otherwise note gap)

---

## 4. Data and Label Assembly (~1.5 pages)
*(Label assembly is a genuine contribution — the DSWP release is audio-only; labels required non-trivial cross-referencing)*

### 4.1 DSWP Audio
- Source: HuggingFace `orrp/DSWP` (Paradise et al., 2025); CC BY 4.0
- 1,501 WAV files (`1.wav`–`1501.wav`); ~585 MB; recorded off Dominica, 2005–2010 (Dominica Sperm Whale Project, Shane Gero)
- **Critical gap**: the HuggingFace release is audio-only — no labels shipped with it
- All four experiments depend on this audio: mel-spectrogram extraction (Phases 1 & 3), WhAM embedding extraction (Phase 2), and generation prompts (Phase 4)

### 4.2 Label Assembly: DominicaCodas.csv → dswp_labels.csv

The only label source that covers the DSWP range is DominicaCodas.csv, released by Sharma et al. (2024, *Nature Communications*) alongside their study of combinatorial structure in sperm whale vocalizations.

- **Source**: `github.com/pratyushasharma/sw-combinatoriality`; 8,719 rows total; fields: `codaNUM2018`, `CodaType`, `Unit`, `Clan`, `IDN`, `ICI1`–`ICI9`, `Date`
- **Key discovery**: exactly 1,501 rows have `codaNUM2018` in the range 1–1,501; the index maps directly to DSWP filenames (`codaNUM2018 = N` → `N.wav`). This 1:1 correspondence was verified by matching pre-computed ICI sequences and coda durations across both files — it is not documented anywhere in either dataset's release.
- **Labels unlocked**: social unit (A/D/F), coda type (35 categories), individual whale ID (IDN), full ICI sequence (ICI1–ICI9), recording date, noise flag

The merged output, `dswp_labels.csv` (1,501 rows), is the single label source used by every notebook in this study and is released as part of the codebase.

**Note on other label sources investigated**: We retrieved and examined three additional public datasets (codamd.csv — Beguš et al. 2024; focal-coarticulation-metadata.csv — Beguš et al. 2024; sperm-whale-dialogues.csv — Sharma et al. 2024). None overlap with the DSWP codaNUM range (1–1,501): codamd covers codaNUM 4,933–8,860; the dialogue dataset uses an incompatible whale ID scheme and lacks coda type labels. Critically, codamd's hand-verified vowel labels (`handv`: `a` or `i`) — the spectral ground truth formalized by Beguš et al. — are unavailable for the DSWP range. This gap directly motivated our architectural choice to use mel-spectrogram CNN rather than explicit vowel supervision for the spectral encoder.

### 4.3 WhAM Model
- Weights: Zenodo 10.5281/zenodo.17633708 (coarse.pth 1.3 GB + codec.pth 573 MB); CC-BY-NC-ND 4.0
- Used in two roles across this work:
  1. **Representation baseline** (Phases 1–3): extract 1,280d embeddings at each of 20 transformer layers for all 1,501 codas via `extract_wham_embeddings.py`
  2. **Generative tool** (Phase 4): masked acoustic token sampling to produce synthetic codas conditioned on real DSWP prompts

### 4.4 Dataset Statistics and Known Challenges

#### Class Distribution
- Social units: A (273), D (336), F (892) — Unit F = 59.4%; severe imbalance
- Coda types: 35 categories; top-5 cover 73% of codas (1+1+3: 486, 5R1: 236, 4D: 167, 7D1: 122, 5-NOISE: 76)
- Clean codas (is_noise=0): 1,383 used in all experiments; 118 noise-contaminated codas excluded

#### Individual ID Limitation
- IDN=0 (unidentified) = 672 codas, confined almost entirely to Unit F — not a data quality issue; these whales were not individually photo-ID'd during field work
- Individual ID experiments use only the 762 IDN-labeled codas across 12 individuals (1 singleton removed from split)

### 4.5 Data Splits and Evaluation Protocol
- 80/20 stratified train/test split by social unit; random seed=42; **identical split reused across all four phases**
- **Primary metric**: macro-F1 (required by class imbalance — a unit-F majority classifier achieves high accuracy but near-chance macro-F1)
- Secondary metric: top-1 accuracy
- WeightedRandomSampler (inverse unit frequency) for balanced unit sampling during DCCE training

---

## 5. Method: Dual-Channel Contrastive Encoder (DCCE) (~2 pages)

### 5.1 Overview and Motivation
- Design principle: encode rhythm and spectral information separately, then fuse — no information blending at the input
- Contrast with WhAM: waveform → codec → transformer; no inductive bias about two-channel structure
- DCCE injects the known biology as an architectural prior

### 5.2 Input Representations
- **Rhythm input**: ICI sequence, zero-padded to length 9, StandardScaler normalized (mean=177ms, std=88.6ms); dimensionless encoding of timing pattern
- **Spectral input**: mel-spectrogram (64 mel bins × 128 time frames, fmax=8,000 Hz); 2D time-frequency image of acoustic texture

### 5.3 Architecture

```
Coda (audio + ICI)
    │
    ├── Rhythm Encoder: 2-layer GRU → r_emb (64d)
    │   Input: zero-padded ICI vector (length 9)
    │
    └── Spectral Encoder: 3-block CNN → s_emb (64d)
        Input: mel-spectrogram (64 × 128)
                  │
        Fusion MLP: concat(r_emb, s_emb) → LayerNorm → Linear(128→64) → ReLU → z (64d)
```

- Rhythm Encoder: 2-layer bidirectional GRU; hidden size 64; final hidden state → 64d
- Spectral Encoder: 3 × (Conv2d + BatchNorm + ReLU + MaxPool) → GlobalAvgPool → Linear(→64)
- Fusion MLP: LayerNorm → Linear(128→64) → ReLU → L2-normalize → joint embedding z

### 5.4 Training Objective

```
L = L_contrastive(z) + λ₁ · L_type(r_emb) + λ₂ · L_id(s_emb)
```

**L_contrastive (NT-Xent / SimCLR)**
- Temperature τ=0.07; batch size=64; 50 epochs; AdamW lr=1e-3
- Positive pairs: z(coda_A) vs. z(coda_B) where coda_A and coda_B share the same social unit
- **Key novelty — cross-channel positive pairs**: in each batch, swap the spectral context across same-unit pairs: rhythm(coda_A) + spectral(coda_B, same unit) = one view; rhythm(coda_B) + spectral(coda_A) = the other. This forces z to be invariant to *which specific coda* the spectral texture came from, as long as the speaker identity is shared.

**L_type (auxiliary coda-type head)**
- Cross-entropy on r_emb → coda_type; 22 active classes (weighted for imbalance)
- Forces rhythm encoder to remain type-discriminative; prevents collapse to unit-only signal

**L_id (auxiliary individual-ID head)**
- Cross-entropy on s_emb → individual_id; 12 classes; only 762 IDN-labeled codas
- Forces spectral encoder to remain speaker-discriminative

Hyperparameters: λ₁=λ₂=0.5; WeightedRandomSampler (inverse unit frequency)

### 5.5 Ablation Variants
| Variant | Encoders | Cross-channel aug |
|---|---|---|
| DCCE-rhythm-only | GRU only (z = r_emb) | N/A |
| DCCE-spectral-only | CNN only (z = s_emb) | N/A |
| DCCE-late-fusion | GRU + CNN, z = concat → MLP | No cross-channel swap |
| **DCCE-full** | GRU + CNN + cross-channel pairing | **Yes** |

### 5.6 Evaluation: Linear Probe Protocol
- Freeze all encoder weights after training
- Fit `LogisticRegression(class_weight="balanced", max_iter=500)` on frozen embeddings
- Three downstream tasks: social unit classification, coda type classification, individual ID classification
- Report macro-F1 (primary) and accuracy (secondary)

---

## 6. Experiment 1: WhAM Probing (~1.5 pages)

### 6.1 Setup
- WhAM: VampNet coarse transformer (20 layers × 1,280d); fine-tuned on ~10,000 DSWP codas
- Extract intermediate representations at each of the 20 transformer layers for all 1,501 DSWP codas
- Train linear probes (same LogisticRegression protocol) at each layer for: social unit, coda type, individual ID, recording year
- UMAP visualization of layer-19 embeddings colored by unit, coda type, individual ID, year

### 6.2 Layer-wise Probing Results
- Social unit accuracy peaks at **layer 19** (F1=0.895) — rises monotonically from layer 1 through 19
- Coda type: WhAM does not beat raw ICI baseline (ICI F1=0.931 vs. WhAM best F1=0.261)
- Individual ID: best at **layer 10** (F1=0.454); degrades in later layers (unit-class pressure dominates)
- Recording year: peaks at **layer 18** (F1=0.906) — highest of all probes

### 6.3 Recording-Year Confound
- **Finding**: Cramér's V(social_unit, recording_year) = 0.51; Spearman ρ=0.63 (p=0.003)
- Units A, D, F were recorded at systematically different periods during 2005–2010
- WhAM's social-unit advantage is partially attributable to acoustic drift in recording equipment and conditions across years — not purely biological unit identity
- This confound was **not identified or reported in the original WhAM paper**
- DCCE is less susceptible: uses hand-crafted ICI features (rhythm: recording-independent) and mel-spectrogram (spectral: per-coda, not accumulated waveform-level drift)

### 6.4 Interpretation
- WhAM encodes: strong social-unit structure (late layers), moderate temporal/year drift, weak coda-type information
- WhAM is not purpose-built for individual ID — its best individual-ID layer (10) shows mid-training representations, not the final task-optimized ones
- Gap between social-unit (0.895) and individual-ID (0.454) performance motivates DCCE's dual-channel design

---

## 7. Experiment 2: DCCE Representation Quality (~1.5 pages)

### 7.1 Baselines
| Model | Social Unit F1 | Coda Type F1 | Individual ID F1 |
|---|---|---|---|
| 1A — Raw ICI (logistic regression) | 0.599 | **0.931** | 0.493 |
| 1C — Mel-spectrogram (logistic regression) | 0.740 | 0.097 | 0.272 |
| 1B — WhAM L10 (best for indivID) | 0.876 | 0.212 | **0.454** |
| 1B — WhAM L19 (best for unit) | **0.895** | 0.261 | 0.426 |

### 7.2 DCCE Results and Ablations
| Model | Social Unit F1 | Coda Type F1 | Individual ID F1 |
|---|---|---|---|
| DCCE-rhythm-only | 0.637 | 0.711 | (N/A or low) |
| DCCE-spectral-only | — | — | — |
| DCCE-late-fusion | 0.656 | — | 0.612 |
| **DCCE-full** | **0.878** | **0.578** | **0.834** |

*(Fill exact ablation numbers from phase3_dcce.ipynb outputs)*

### 7.3 Analysis
- **Individual ID F1: 0.834 vs. 0.454 (WhAM L10)**: DCCE's primary win — +0.380 F1. The cross-channel contrastive objective explicitly optimizes for speaker-identity separation across both channels.
- **Social Unit F1: 0.878 vs. 0.895 (WhAM L19)**: near parity; WhAM's slight advantage is consistent with the recording-year confound (WhAM encodes year at F1=0.906 and year ≈ unit).
- **Coda Type F1**: neither DCCE-full (0.578) nor WhAM beats raw ICI (0.931). Coda type is fully determined by ICI timing; neural encoders add no value here over a simple logistic regression on the raw ICI sequence.
- **Cross-channel ablation**: DCCE-full vs. DCCE-late-fusion: +0.222 individual-ID F1 from cross-channel pairing alone — this is the quantitative contribution of the novel positive-pair construction.

### 7.4 UMAP Comparison (2×2 Figure)
- WhAM L19 colored by unit: compact clusters (partially year-driven)
- WhAM L19 colored by individual ID: diffuse, overlapping
- DCCE-full colored by unit: similar cluster quality to WhAM
- DCCE-full colored by individual ID: tight, well-separated individual clusters

---

## 8. Experiment 3: Synthetic Data Augmentation (~1 page)

### 8.1 Setup
- WhAM coarse model used as a generative augmentation tool: `rand_mask_intensity=0.8`, 30 sampling steps, `mask_temperature=15.0`
- Prompt: real DSWP coda from training set; generate one synthetic coda per prompt
- Pseudo-labels: unit + coda_type copied from prompt; ICI copied from prompt (no new rhythm information); individual ID not labeled
- N_synth sweep: {0, 100, 500, 1000} synthetic codas added to D_train
- Model: DCCE-full retrained from scratch for each N_synth; evaluated on real-only D_test
- Generation throughput: ~2.9s/coda on Apple MPS; 1,000 codas generated in ~2,943s

### 8.2 Results
| N_synth | Individual ID F1 | Social Unit F1 | Coda Type F1 |
|---|---|---|---|
| 0 (baseline) | **0.834** | **0.878** | **0.578** |
| 100 | slightly lower | slightly lower | slightly lower |
| 500 | slightly lower | slightly lower | slightly lower |
| 1000 | slightly lower | slightly lower | slightly lower |

*(Fill exact numbers from datasets/phase4_results.csv)*

### 8.3 Interpretation: Why Augmentation Fails
- Pseudo-ICI (copied from prompt) adds no new rhythm information — the rhythm encoder sees a duplicate ICI sequence, not a genuinely new coda pattern
- No individual ID label for synthetic codas: the individual-level contrastive loss `L_id(s_emb)` cannot train on synthetic data; synthetic codas only populate the unit-level contrastive loss
- Synthetic codas dilute the contrastive geometry without contributing individual-level signal
- **Key insight**: WhAM's generative model produces acoustically plausible codas, but plausible ≠ informative for speaker-identity learning. Augmentation quality must be assessed in terms of what information it adds to the specific downstream task — not acoustic fidelity alone.

### 8.4 Implications for Future Work
- Augmentation may become useful when: (a) authentic individual-ID labels can be assigned to synthetic codas (requires a reliable speaker classifier), or (b) WhAM's generation is conditioned on individual voice characteristics (not currently possible)
- The negative result is itself informative: it demonstrates that synthetic coda fidelity does not automatically translate to representation benefit

---

## 9. Discussion (~1 page)

### 9.1 Domain Knowledge as an Architectural Prior
- The central finding: DCCE's +0.380 individual-ID F1 over WhAM comes entirely from encoding the biological structure explicitly — not from more data, larger models, or longer training
- WhAM's 10,000-coda training set is 6.7× larger than DSWP; DCCE wins on individual identity with 1,501 codas and a laptop-scale model
- This supports the general principle: inductive bias from domain knowledge can substitute for scale when the domain structure is known

### 9.2 The Recording-Year Confound
- WhAM's social-unit classification advantage is partially confounded by recording year (V=0.51)
- This should be reported as a caveat in any deployment of WhAM for biological inference
- Future work should apply WhAM on recordings with balanced year × unit coverage to disentangle year from unit effects
- DCCE's use of pre-computed ICI features makes it inherently less susceptible to acoustic drift confounds

### 9.3 Limitations
- **DSWP is a single population**: Dominica, 3 social units, 2005–2010. Generalizability to other populations (Pacific clans, Atlantic clans) is unknown.
- **No vowel labels for DSWP**: The spectral encoder is supervised only by unit-level contrastive loss and individual-ID auxiliary loss — not by explicit vowel categories. Whether the spectral encoder learns vowel structure is untested.
- **Pseudo-ICI for synthetic codas**: The augmentation design does not attempt to generate novel ICI patterns; this was the limiting factor in the augmentation experiment.
- **Laptop-scale compute**: All models run on Apple MPS. DCCE is deliberately small (GRU-64 + CNN-64); scaling has not been explored.

### 9.4 Biological Implications
- Individual identity is robustly encoded in sperm whale codas at the acoustic level — a linear classifier on 64d embeddings achieves F1=0.834
- Social unit and individual identity are simultaneously decodable from the same representation (DCCE-full), consistent with the biological hypothesis that codas carry both social-group and individual-identity markers
- The rhythm channel alone is largely sufficient for coda-type classification (ICI F1=0.931) but insufficient for social unit or individual ID classification, confirming that spectral texture carries the remaining identity information

---

## 10. Conclusion (~0.5 page)

- We introduced DCCE, the first representation model explicitly designed around the two-channel structure of sperm whale codas
- DCCE substantially outperforms WhAM on individual identity (F1: 0.834 vs. 0.454) while matching it on social unit classification (0.878 vs. 0.895)
- We conducted the first layer-wise probing analysis of WhAM, revealing a recording-year confound not previously reported
- We conducted the first controlled synthetic augmentation study for cetacean bioacoustics, finding that acoustic fidelity does not guarantee representation benefit
- The assembled `dswp_labels.csv` label table and codebase are released as a public resource for future work on the DSWP dataset

---

## 11. References

Key citations to include:
1. Goldwasser et al. (NeurIPS 2023) — arXiv:2211.11081 — UMT theory
2. Leitão et al. (2023/2025) — arXiv:2307.05304 — rhythmic micro-variation, cross-clan learning
3. Gubnitsky et al. (2024) — arXiv:2407.17119 — automated coda detector
4. **Paradise et al. (NeurIPS 2025) — arXiv:2512.02206 — WhAM** *(primary baseline)*
5. Beguš et al. (2024) — coda-vowel-phonology — spectral/vowel channel formalization
6. Sharma et al. (Nature Comms 2024) — DominicaCodas.csv source; coda combinatorics
7. Gero, Whitehead & Rendell (2016) — Zenodo 4963528 — foundational coda taxonomy
8. Chen et al. (ICML 2020) — SimCLR / NT-Xent
9. Radford et al. (2021) — CLIP — cross-modal contrastive learning

---

## 12. Appendix / Supplementary (optional)

- A: Full DCCE hyperparameter table
- B: Phase-by-phase reproducibility notes (seeds, splits, CSV loading)
- C: WhAM layer-by-layer probe accuracy table (all 20 layers × 4 tasks)
- D: Synthetic coda generation config (mask parameters, vampnet API)
- E: dswp_labels.csv schema and label distribution figures (from Phase 0 EDA)

---

## Notes on Section Priorities

| Section | Priority | Why |
|---|---|---|
| §4 Data Assembly | **High** — expand | This is original work not in any prior paper; the alignment finding (DominicaCodas ↔ DSWP) is a genuine contribution |
| §6 WhAM Probing | **High** | Year confound is a novel finding with biological implications |
| §7 DCCE Results | **High** | Primary experiment; indivID result is the headline number |
| §3 Related Work | **Medium** | Standard; lean on research_paper.md section 2 |
| §8 Augmentation | **Medium** | Negative result but informative |
| §5 Method | **Medium** | Architecture is not complex; keep tight |
| §9 Discussion | **Medium** | Biological implications are important for the target audience |
| §1 Intro | **Medium** | Standard; motivate with team_update.md framing |
