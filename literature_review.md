# Literature Review
## Beyond WhAM — Sperm Whale Coda Representation Learning

**Project**: Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding  
**Course**: CS 297 Final Paper · April 2026  

> **How to use this document**: Each paper entry has (1) metadata + download link, (2) structured extraction of biological facts / quantitative results / methods / limitations, (3) a "→ Use in our paper" block mapping every finding to a specific section in our paper outline. Part II at the end is a thematic cross-reference index — use it when writing a specific paragraph to find all supporting citations quickly.

---

# PART I — PER-PAPER ENTRIES

---

## P1 · Paradise et al. (2025) — WhAM

**Title**: WhAM: Towards A Translative Model of Sperm Whale Vocalization  
**Authors**: Orr Paradise, Pratyusha Sharma, Piyush Muralikrishnan, Bella Chen, Alejandro Flores García, Bryan Pardo, Roee Diamant, David Gruber, Shane Gero, Shafi Goldwasser  
**Venue**: NeurIPS 2025  
**arXiv**: https://arxiv.org/abs/2512.02206  
**PDF**: https://arxiv.org/pdf/2512.02206  
**Code/Data**: https://github.com/Project-CETI/wham | Weights: Zenodo 10.5281/zenodo.17633708 | Audio: HuggingFace `orrp/DSWP`

---

### Biological Facts

- Sperm whales use "short sequences of clicks known as codas" for communication; variation is in "number, rhythm, and tempo"
- Whales inhabit "stable, female-led social units that form larger vocal clans based on dialect"
- Dialects are "believed to be learned socially rather than inherited genetically"
- Recent work reveals "vowel-like spectral properties of codas" in the 3.7–5.7 kHz band
- Natural codas have a "low-frequency bias" compared to synthetic versions
- Clicks possess "inter-pulse structures" that synthetic versions struggle to replicate; also have specific "onset and decay patterns"

### Quantitative Results

**Fréchet Audio Distance (FAD):**
- Baseline FAD between disjoint natural coda sets: **0.21**
- Synthetic codas from natural whale prompts: FAD-indistinguishable from natural samples

**Expert Perceptual Evaluation (5 marine biologists/acousticians, 3–20 years experience):**
- Audio-only 2AFC: **81% accuracy** (κ=0.41)
- Spectrogram-assisted 2AFC: **83% accuracy** (κ=0.41)
- Natural codas misidentified as synthetic: **36% of the time**
- Walrus-to-coda translation detection: only 75% accuracy (one expert: 50%)

**Downstream Classification (linear probe on WhAM embeddings):**

| Task | WhAM | AVES | BirdNET | Majority |
|---|---|---|---|---|
| Coda Detection | 91.3±0.2% | 60.9% | 92.8±0.1% | 60.9% |
| Rhythm Type | 87.4±1.6% | 66.3% | 90.4±1.6% | 60.9% |
| Social Unit | 70.5±5.6% | 42.5% | 92.0±0.7% | 35.1% |
| Vowel Classification | 85.2±2.5% | 66.3% | 91.8±2.9% | 66.3% |

> **Important note on our results vs. WhAM's**: WhAM reports social unit accuracy of 70.5%. On our exact split of 1,501 DSWP codas, we measure WhAM L19 macro-F1 = 0.895 on social unit. These are different metrics (accuracy vs. macro-F1) and different splits — not directly comparable but both valid. We consistently use macro-F1 throughout.

### Dataset Details

| Dataset | Codas | Purpose |
|---|---|---|
| DSWP (HuggingFace) | 2,507 codas | Species-specific finetuning |
| CETI-tagged recordings | 7,653 codas | Species-specific finetuning |
| **Total whale codas** | **~10,000** | WhAM training |
| FSD + AudioSet + WMMS + BirdSet | Hours of audio | Domain adaptation pre-finetuning |

### Architecture

- **Base model**: VampNet (masked acoustic token model pretrained on 797k music tracks)
- **Acoustic tokenizer**: Descript Audio Codec (DAC) with residual vector quantization
- **Masked Acoustic Token Model (MATM)**: Bidirectional transformer, iterative parallel decoding
- **Finetuning method**: LoRA (Low Rank Adaptation)
- **Training**: Phase 1: domain adaptation (500k iterations); Phase 2: species-specific finetuning (500k iterations)
- **Compute**: ~5 days on a single GPU

### Evaluation Methodology

- Quantitative: Fréchet Audio Distance using BirdNET embeddings (selected after calibration comparing 5 embedding models for sensitivity to rhythmic patterns)
- Perceptual: 4 tasks (audio-only 2AFC, spectrogram-assisted 2AFC, mixed collection classification, qualitative assessment)
- Inter-rater agreement: Fleiss's κ

### Stated Limitations

1. Only the MATM is finetuned; codec is fixed — "may limit ability to capture nuanced acoustic features specific to sperm whale vocalizations"
2. Expert feedback: "unnatural onset and decay patterns, inconsistent background noise, click properties more reminiscent of echolocation than communication codas"
3. Dataset "orders of magnitude smaller than typical in modern acoustic model training"
4. "The gap between generating vocalizations and understanding their meaning remains vast"

### Future Work They Identify

- Finetune the codec or develop specialized codecs for bioacoustic signals
- Adversarial components or specialized modules leveraging domain knowledge about click structure
- Unsupervised learning to uncover new coda features
- Scale to additional sperm whale datasets
- Adapt framework to other animal communication systems
- Bridge the semantic gap

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| WhAM classification results (Table above) | §6 WhAM Probing — primary comparison target; note accuracy vs. F1 distinction |
| "Designed for generation, not representation" | §2.3 The Gap (Introduction) — central motivation |
| Social unit 70.5% accuracy | §3 Related Work — WhAM subsection; contrast with our macro-F1 |
| DAC codec + bidirectional transformer architecture | §3 Related Work — WhAM architecture description |
| Expert FAD results | §3 Related Work — evidence WhAM generates realistic codas |
| Limitation: codec not finetuned | §9 Discussion — explains why WhAM may not encode fine-grained spectral texture |
| "Semantic gap remains vast" | §10 Conclusion — frames what remains open after our work |
| DSWP dataset (2,507 codas) and CETI (7,653) | §4 Data — establishes WhAM trained on ~6.7× our data |

---

## P2 · Goldwasser et al. (NeurIPS 2023) — UMT Theory

**Title**: A Theory of Unsupervised Translation Motivated by Understanding Animal Communication  
**Authors**: Shafi Goldwasser, David Gruber, Adam Tauman Kalai, Orr Paradise  
**Venue**: NeurIPS 2023  
**arXiv**: https://arxiv.org/abs/2211.11081  
**PDF**: https://arxiv.org/pdf/2211.11081  

---

### Core Theoretical Claims

- Central insight: "error rates are inversely related to the language complexity and amount of common ground"
- Formally: **more complex languages can be translated with less common ground** — complexity and common ground trade off
- Practical implication: **UMT of animal communication may be feasible if the communication system is sufficiently complex**

### Key Theorems

**Knowledge Graph Model (Theorem 3.2)** — for compositional languages:
```
err(θ̂) = O(log n / (α²d) + 1/α √(r log n / m))
```
where α = agreement between source/target, d = language complexity (avg degree), r = source nodes, m = samples

**Common Nonsense Model (Theorem 3.4)** — for unstructured languages with shared constraints:
```
err(θ̂) = O(log|Θ| / (α · min(m, |T|)))
```
Mirrors supervised learning bounds when common ground α is constant.

### Quantitative Experimental Results

For the knowledge graph model (n=10 target nodes, r=4/7/10, avg degree d≈5):
- r=4 source nodes: accuracy = **0.10 ± 0.05**
- r=7 source nodes: accuracy = **0.74 ± 0.08**
- r=10 source nodes: accuracy = **1.00 ± 0.00**

### Statements About Animal Communication / Sperm Whales

- "Whales do not 'talk' about smartphones" — proposes that a "broad prior" from language models can capture "plausible English translations of animal communication"
- Specifically discusses sperm whales performing echolocation
- Framework assumes a hypothetical "mermaid" fluent in both whale and English as the ground-truth translator
- Makes "few assumptions about the source language itself" beyond textual format — deliberately designed for systems without human-like linguistic structure

### Stated Limitations

1. Bounds are information-theoretic — "do not account for computational complexity of optimizing the translator"
2. Knowledge graph model assumes compositional structure
3. Realizability: requires ground-truth translator exists within hypothesis class
4. Requires access to a prior ρ over target translations (in practice: a language model)
5. "Not intended to accurately capture natural language" — stylized models

### Future Work They Identify

- Extend to k-ary relations and hypergraphs (beyond binary relations)
- Generalize to continuous parameter spaces
- Lossy translation (non-bijective f)
- Computationally efficient algorithms for translator optimization

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| "UMT feasible if system is sufficiently complex" | §2.1 Introduction — theoretical motivation for the whole CETI program |
| Error inversely related to language complexity | §2.1 Introduction — justifies studying coda structure carefully |
| Goldwasser et al. are co-authors of WhAM | §3 Related Work — shows theoretical + empirical CETI work are linked |
| Framework "few assumptions about source language" | §2.1 Introduction — supports bottom-up approach (don't assume human-like structure) |

---

## P3 · Leitão et al. (2023/2025) — Social Learning Across Clan Boundaries

**Title**: Evidence of Social Learning Across Symbolic Cultural Barriers in Sperm Whales  
**Authors**: António Leitão, Maxime Lucas, Andrea Poetto, Taylor A. Hersh, Shane Gero, David Gruber, Michael Bronstein, Giovanni Petri  
**arXiv**: https://arxiv.org/abs/2307.05304  
**PDF**: https://arxiv.org/pdf/2307.05304  
**Status**: Submitted Jul 2023; revised Mar 2025

---

### Biological Facts

- Sperm whale societies are multi-tiered: individual → social unit → vocal clan
- Social units: stable, matrilineally-based groups sharing vocal repertoires across years
- Vocal clans: higher-order groups of units sharing substantial parts of their repertoire; different clans maintain social separation even in geographic overlap
- Cultural transmission occurs via three pathways: vertical (adult kin to young), oblique (unrelated adults to young), horizontal (peer-to-peer)

### Identity vs. Non-Identity Codas

**Identity (ID) codas:**
- Account for **35–60% of all vocalizations**
- Function as "symbolic markers for each clan"
- Show **NO** relationship between geographic overlap and vocal style similarity
- Actually become **MORE dissimilar** with increasing overlap (consistent with symbolic marking theory — clans reinforce distinction when in contact)

**Non-identity (non-ID) codas:**
- Comprise **65% of vocalizations** (up to 93% when counting by coda type)
- Show **OPPOSITE** pattern: vocal style becomes **MORE similar** with increasing clan overlap
- No variation in usage frequency with overlap

### Cross-Clan Vocal Learning Finding

- "Sympatry increases vocal style similarity between clans for non-ID codas...suggesting social learning across cultural boundaries"
- Analogous to "accents aligning in human populations that share territory" — cultural borrowing at the micro-variation level while identity markers remain distinct

### Computational Model: Subcoda Trees (VLMCs)

- **Input**: Continuous ICI sequences → discretized into bins of width δt = 0.05 seconds
- **Model**: Variable Length Markov Chains (VLMCs) encoding transition probabilities — "rhythmic micro-variations within codas"
- **Distance**: Kullback-Leibler divergence between subcoda trees
- **Accuracy**: Synthetic codas generated from VLMCs achieve **~85% classification accuracy** vs. ~90% on real data

### Quantitative Results

- Dominica dataset: VLMCs correctly cluster social units into two known vocal clans
- Pacific dataset: vocal style clustering "closely matches the one obtained from coda usage"
- Statistical testing: KS tests confirm significant within-clan vs. between-clan vocal style differences (p < 0.01)

### Dataset Details

- **Dominica (Atlantic)**: 8,719 annotated codas, 2005–2019; 12 social units, 2 vocal clans; rich individual-level annotations
- **Pacific**: 23,555 codas from 1978–2017 across 23 locations; 57 coda samples; 7 vocal clans; spatial position only, no individual IDs

### Stated Limitations

1. Short clan anomaly: "more spread out in subcoda tree space" — causes clustering overlap
2. Spectral information absent: current model uses only temporal rhythm
3. Alternative hypotheses (environmental adaptation, genetics) not fully ruled out
4. Larger longitudinal datasets needed for improved statistical quality

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| Two-channel decomposition: rhythm (coda type) + spectral (vowels) | §2.2 The Two-Channel Structure; §5 Method motivation |
| VLMCs capture "rhythmic micro-variation within codas" | §2.2 — shows rhythm micro-variation is real and measurable |
| ID codas resist cross-clan convergence; non-ID codas converge | §2.2 — explains why rhythm alone is insufficient for unit ID |
| ~85% VLMC synthetic classification accuracy | §3 Related Work — establishes baseline for rhythm-only models |
| Limitation: "spectral information absent" | §2.3 The Gap — their own stated gap that our spectral encoder fills |
| 8,719-coda Dominica dataset | §4 Data — DominicaCodas.csv provenance |

---

## P4 · Gubnitsky et al. (2024) — Automated Coda Detector

**Title**: Automatic Detection and Annotation of Sperm Whale Codas  
**Authors**: Gal Gubnitsky, Tamir Avidor Mevorach, Shane Gero, David Gruber, Roee Diamant  
**arXiv**: https://arxiv.org/abs/2407.17119  
**PDF**: https://arxiv.org/pdf/2407.17119  
**Code**: Zenodo 10.5281/zenodo.14902261

---

### Detection Pipeline

Four stages processing 7-second audio buffers:
1. Bandpass filtering (2–24 kHz) + Teager-Kaiser energy operator → transient regions of interest
2. Graph-based clustering using structural + temporal similarity
3. Constraint filters: multipulse structure (𝒫) + resonant frequency (fᵣ < 12 kHz) to separate codas from echolocation
4. Automatic coda type annotation

Designed for "wide dynamic range" — works from both tag-attached near-field and boat/mooring far-field recordings.

### Graph-Based Clustering Approach

Optimization maximizing a utility function combining:
- **Structural likelihood (ℒˢ)**: waveform cross-correlation + IPI (inter-pulse interval) comparison + intensity similarity
- **Temporal likelihood (ℒᵗ)**: Generalized Gaussian Mixture Model against known coda type templates
- **Constraints**: multipulse structure, resonant frequency < 12 kHz, orthogonal clustering
- Solved via iterative greedy algorithm (the optimization is NP-hard)

### Quantitative Results

- Constrained detection reduces false alarm rate by **one order of magnitude** (slight detection rate reduction)
- FAR ≈ 0.0097 per minute (constrained, general noise)
- FAR ≈ 0.023 per minute (constrained, echolocation clicks)
- Click identification: "all clicks correctly identified in ~70% of codas; 80%+ identified in another 10–20%"

### New Coda Types Discovered

Processed 843 previously unanalyzed near-field codas from 2018 and discovered:
- **Type '1+5'**: descending rhythm with temporal variation
- **Type '1+1+4'**: distinct rhythm and tempo from established types
- These previously appeared as "contamination" in legacy databases due to sparse distribution

### Synchronization Findings

- Three focal whales (Atwood, Fork, Pinchy) showed distinct temporal patterns when interacting with non-focal whales
- "Whales often match coda type (and therefore ΔICI) during dyadic coda exchanges"
- "Identity cues at the level of the structure of the coda exchange" — not just within codas but in exchange timing

### Dataset Details

- **Near-field**: 42 tags on 25 individuals, 11 social units, 2014–2018; DTag gen. 3 at 120 kHz
- **Far-field**: Hydrophone recordings, 2005–2023; 96 kHz
- **Total processed**: 3,948 manually annotated codas (2014–2016) + 4,930 (2005–2012) + 843 detector-assisted (2018)
- **Location**: Eastern Caribbean, Dominica (~2000 km²)

### Stated Limitations

1. Misidentifying a single click may cause coda misclassification — affects ~10–30% of codas
2. Overlapping echolocation clicks from multiple whales may resemble coda structure
3. No individual-whale classifier — cannot attribute coda to specific whale within a group
4. Analysis limited to near-field data for reliable focal whale verification

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| First automated coda detection pipeline | §3 Related Work — preprocessing baseline our work builds on |
| Two new coda types discovered | §3 Related Work — shows corpus is not fully explored; 35 types in our data may undercount |
| Synchronization between whales in exchanges | §9 Discussion — future direction: model exchange-level structure |
| Coda type taxonomy provides labels | §4 Data — DominicaCodas.csv uses same 35-type taxonomy |
| Graph-based annotation approach | §3 Related Work — contrast with our contrastive learning approach |

---

## P5 · Beguš et al. (2024) — Coda Vowel Phonology

**Title**: The Phonology of Sperm Whale Coda Vowels (also: "Vowel- and diphthong-like spectral patterns in sperm whale codas")  
**Authors**: Gašper Beguš, Shane Gero, et al.  
**Repository**: https://github.com/Project-CETI/coda-vowel-phonology  
**Data**: `codamd.csv` (hand-verified vowel labels); `focal-coarticulation-metadata.csv` (per-click formant measurements)  

> Note: Full text not publicly available via arXiv at time of review. Content reconstructed from citations in other papers (WhAM, Leitão), the GitHub repository README, and the dataset files. Add the paper PDF once obtained from the authors.

---

### Core Claim: Two Independent Channels

- Sperm whale codas carry **two syntactically independent information channels**:
  1. **Rhythm channel**: the ICI pattern (timing between clicks) → encodes coda *type*
  2. **Spectral channel**: formant-like structure within each click → encodes *vowel* category, correlating with individual/social-unit identity
- These channels are **syntactically independent**: the same coda type can be produced with different vowel patterns, and the same individual produces multiple coda types

### Vowel Categories

- Two vowel categories: `a` and `i`, identified by hand-verified annotation (`handv` field)
- Vowel variation occurs in the spectral peak frequency (f1, f2) of individual clicks within a coda
- Coarticulation patterns (`aa`, `ai`, `ia`, `ii`) capture vowel transitions across clicks within a coda

### Spectral Measurements

From `focal-coarticulation-metadata.csv`:
- `pkfq`: spectral peak frequency (~3,000–9,000 Hz range)
- `f1pk`, `f2pk`: first and second formant peaks (Hz)
- `coart`: coarticulation type (aa=670, ii=336, ai=45, ia=46 — `aa` dominates)

### Dataset Details

- `codamd.csv`: 1,375 codas; `codanum` range 4,933–8,860; 13 named individual whales
- Vowel distribution: `a` = 745, `i` = 397, unlabeled = 233
- **Critical gap for our work**: does NOT overlap with DSWP range 1–1,501

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| Two syntactically independent channels (rhythm + spectral) | §2.2 (Introduction) — the foundational biological claim for DCCE architecture |
| Vowel categories `a`/`i` in clicks | §5 Method — spectral encoder target; acknowledge we cannot use these labels |
| Vowel gap for DSWP range | §4 Data — explains why spectral encoder uses contrastive loss, not vowel classification |
| Per-click formant measurements (f1, f2) | §6 WhAM Probing — cited as potential future spectral probe target |
| Coarticulation patterns | §9 Discussion — future work: model within-coda spectral transitions |

---

## P6 · Sharma et al. (2024) — Combinatorial Structure

**Title**: Contextual and combinatorial structure in sperm whale vocalisations  
**Authors**: Pratyusha Sharma, Shane Gero, Roger Payne, Ann Bowles, Hal Whitehead, Luke Rendell, David Gruber, Shafi Goldwasser  
**Venue**: *Nature Communications* 15, 3617 (2024)  
**DOI**: https://doi.org/10.1038/s41467-024-47221-8  
**Data**: https://github.com/pratyushasharma/sw-combinatoriality (DominicaCodas.csv, sperm-whale-dialogues.csv)  
**Zenodo**: 10.5281/zenodo.10817697  

> Note: Nature Communications paper is behind paywall. Content reconstructed from citations in WhAM, Gubnitsky, Leitão papers, and the public GitHub dataset release. Add PDF once accessed.

---

### Core Claim: Combinatorial Structure

- Sperm whale codas exhibit **contextual and combinatorial structure** — codas are not produced randomly but follow patterns analogous to combinatorial syntax
- Evidence for this: particular coda types are produced more or less often depending on the preceding coda type (contextual dependency) and social context

### Dataset Released (DominicaCodas.csv)

- **8,719 codas** from the complete Dominica corpus, 2005–2018
- Fields: `codaNUM2018` (1–8,878), `CodaType` (35 categories), `Unit` (13 social units), `Clan` (EC1/EC2), `IDN` (36 individual IDs), `ICI1–ICI9`, `Duration`, `Date`
- **DSWP alignment**: exactly 1,501 rows with codaNUM2018 1–1,501 (social units A, D, F only)
- **Clans**: EC1 = 7,770 codas; EC2 = 949 codas
- **NOISE-tagged codas**: 600 (clean: 8,119)

### Social Unit Statistics (full corpus)

| Unit | Codas |
|---|---|
| A | 273 (DSWP range) |
| D | 336 (DSWP range) |
| F | 892 (DSWP range) |
| J, K, N, P, R, S, T, U, V, ZZZ | Remaining ~7,218 codas |

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| DominicaCodas.csv as primary label source | §4 Data — the alignment finding is our data contribution |
| 35 coda type taxonomy | §5 Method — L_type auxiliary loss target |
| Combinatorial structure (coda sequences depend on context) | §9 Discussion — future: model coda sequences, not isolated codas |
| 8,719-coda corpus (vs. our 1,501 DSWP subset) | §4 Data — shows DSWP is a subset; other units could be included in future |

---

## P7 · Gero, Whitehead & Rendell (2016) — Identity Cues

**Title**: Individual, unit and vocal clan level identity cues in sperm whale codas  
**Authors**: Shane Gero, Hal Whitehead, Luke Rendell  
**Venue**: *Royal Society Open Science* 3, 150372 (2016)  
**DOI**: https://doi.org/10.1098/rsos.150372  
**Data**: Zenodo 4963528 (CC0)  

> Note: Paper is behind a paywall. Content reconstructed from citations and abstract. Add PDF once obtained.

---

### Core Findings

- Sperm whale codas carry identity information at **three hierarchical levels**: individual whale, social unit, and vocal clan
- The three levels are encoded independently — different acoustic features carry each level of identity
- Provides foundational coda taxonomy: 9 Caribbean social units, 21 coda types in the Atlantic

### ICI Analysis

- ICI sequences are the primary feature used to define coda types
- Confirmed that ICIs carry unit-level identity cues (in addition to coda type) — the micro-variation signal exploited by Leitão et al.
- Established methodology: zero-pad ICI sequences, compute pairwise similarities

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| Three-level identity hierarchy (individual, unit, clan) | §2.2 Introduction — biological grounding for our three classification tasks |
| Foundational coda taxonomy (21 types) | §3 Related Work — our 35 types are an extension of this taxonomy |
| ICIs as identity carriers | §5 Method — justifies ICI as rhythm encoder input |

---

## P8 · Chen et al. (2020) — SimCLR

**Title**: A Simple Framework for Contrastive Self-Supervised Learning  
**Authors**: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton  
**Venue**: ICML 2020  
**arXiv**: https://arxiv.org/abs/2002.05709  
**PDF**: https://arxiv.org/pdf/2002.05709  

---

### NT-Xent Loss (Normalized Temperature-Scaled Cross-Entropy)

$$\ell(i,j) = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

- `sim(u,v)` = cosine similarity between L2-normalized vectors
- `τ` = temperature parameter (we use τ=0.07)
- Within a batch of N original samples → 2N augmented views; each sample's positive pair is its augmented counterpart; all other 2(N-1) samples are negatives

### Architecture

- **Encoder f**: Standard ResNet; output h_i ∈ ℝ^d after average pooling
- **Projection head g**: 1-hidden-layer MLP with ReLU → projects to z_i for contrastive loss
- Key finding: "a nonlinear projection is much better than no projection (>10% improvement)"; the pre-projection representation h outperforms projected z for downstream tasks by >10%

### Data Augmentation

Effective combination: (1) random crop+resize, (2) color distortion (jitter 80% + grayscale 20%), (3) Gaussian blur (50%). "Composition of multiple augmentations is crucial."

### Quantitative Results (ImageNet)

- ResNet-50 (4×): **76.5% top-1 accuracy** on linear evaluation → matches supervised ResNet-50
- Semi-supervised 1% labels: 85.8% top-5; 10% labels: 92.6% top-5

### Hyperparameter Sensitivity

- Larger batch size substantially improves performance with fewer epochs (4096 optimal)
- Temperature τ and L2 normalization are essential — without them "performance is significantly worse"

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| NT-Xent loss formulation (exact formula) | §5.4 Training Objective — cite as source for our loss function |
| Positive pair construction (same sample, different augmentation) | §5.4 — contrast with our cross-channel positive pairs (novel contribution) |
| Projection head recommendation | §5.3 Architecture — justifies our Fusion MLP before the contrastive loss |
| Temperature τ sensitivity | §5.4 — cite for our choice of τ=0.07 |

---

## P9 · Radford et al. (2021) — CLIP

**Title**: Learning Transferable Visual Models From Natural Language Supervision  
**Authors**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever  
**Venue**: ICML 2021  
**arXiv**: https://arxiv.org/abs/2103.00020  
**PDF**: https://arxiv.org/pdf/2103.00020  

---

### Cross-Modal Contrastive Training Objective

- Given a batch of N (image, text) pairs, CLIP predicts "which of the N×N possible pairings actually occurred"
- Symmetric cross-entropy loss over cosine similarity scores
- Training data: 400 million (image, text) pairs from the internet (WebImageText/WIT dataset)
- Positive pair = image paired with its natural language description

### Architecture

- **Image encoder**: ResNet or ViT variants (best: ViT-L/14@336px)
- **Text encoder**: 12-layer Transformer (512-width, 8 heads, 63M parameters); BPE vocabulary size 49,152; max 76 tokens

### Key Results

| Task | Performance |
|---|---|
| ImageNet zero-shot | 76.2% top-1 |
| STL10 zero-shot | 99.3% (new SOTA) |
| Outperforms supervised ResNet-50 linear classifier | on 16 of 27 datasets |

### Core Innovation Relevant to Our Work

- **Cross-modal positive pairs**: image + text description of the same concept = positive pair; this is the structural inspiration for our **cross-channel positive pairs** (rhythm of coda A + spectral texture of another coda from the same unit = positive pair)
- Zero-shot transfer: learned representations generalize to novel tasks without fine-tuning

### Stated Limitations

- Poor on fine-grained classification (car models, flower species) — relevant: our fine-grained individual ID task may be similarly hard
- "Does not address poor data efficiency" — relies on scale; our model is designed for small-data regimes
- Distribution shift brittleness (88% on MNIST despite strong performance elsewhere)

### → Use in Our Paper

| Finding | Where in our paper |
|---|---|
| Cross-modal alignment via contrastive loss | §3.3 Related Work — conceptual ancestor of our cross-channel design |
| Symmetric cross-entropy over similarity matrix | §5.4 Training Objective — related loss formulation |
| N×N positive pair matrix | §5.4 — our batch construction follows same logic |
| Zero-shot generalization | §9 Discussion — future: test DCCE zero-shot on new units/codas |

---

# PART II — THEMATIC CROSS-REFERENCE INDEX

*Use this when writing a specific paragraph — find all supporting citations by theme.*

---

## Theme A: Sperm Whale Biology and Social Structure

| Claim | Source(s) |
|---|---|
| Largest brain of any species | team_update.md (general knowledge) |
| Matrilineal social units; multigenerational cultural transmission | P3 (Leitão), P7 (Gero 2016) |
| Vocal clans defined by shared coda repertoire | P3 (Leitão), P7 (Gero 2016), P1 (WhAM) |
| Dialects socially learned, not genetically inherited | P1 (WhAM), P3 (Leitão Rendell 2012 cited) |
| Social unit membership stable across decades | P3 (Leitão), P7 (Gero 2016) |
| 2 vocal clans in Caribbean Atlantic (EC1, EC2) | P6 (Sharma), P3 (Leitão) |

## Theme B: Coda Structure — The Two Channels

| Claim | Source(s) |
|---|---|
| Two syntactically independent channels (rhythm + spectral) | P5 (Beguš), P3 (Leitão limitation) |
| ICI timing encodes coda type (rhythm channel) | P7 (Gero 2016), P4 (Gubnitsky), P3 (Leitão) |
| Spectral texture within clicks encodes vowels / individual identity | P5 (Beguš), P1 (WhAM) |
| Same coda type produced by all units in a clan | Our EDA (heatmap), P6 (Sharma) |
| Rhythm and spectral channels are statistically independent | Our EDA (r≈0 centroid vs ICI), P5 (Beguš) |
| Identity codas resist cross-clan convergence | P3 (Leitão) |
| Non-identity codas show cross-clan style convergence | P3 (Leitão) |

## Theme C: ML Models for Coda Classification

| Claim | Source(s) |
|---|---|
| WhAM: VampNet fine-tuned on ~10k codas; SOTA generative+classification model | P1 (WhAM) |
| WhAM social unit accuracy: 70.5% (accuracy metric) | P1 (WhAM) |
| WhAM rhythm type accuracy: 87.4% | P1 (WhAM) |
| WhAM vowel classification: 85.2% | P1 (WhAM) |
| Bermant et al. 2019: first deep learning for sperm whale bioacoustics | P3 (cites Bermant), P1 (cites Bermant) |
| AVES: animal vocalization encoder via self-supervision | P1 (WhAM — used as baseline) |
| Gubnitsky 2024: first automated coda detection | P4 (Gubnitsky) |

## Theme D: Recording-Year Confound

| Claim | Source(s) |
|---|---|
| Units A, D, F recorded at different times (2005–2010) | Our EDA (phase2) |
| Cramér's V (unit, year) = 0.51; ρ = 0.63 (p=0.003) | Our Phase 2 results |
| WhAM year encoding peaks at layer 18 (F1=0.906) | Our Phase 2 results |
| WhAM not designed to separate unit from year effects | Our Phase 2 analysis; supported by P1 limitation (codec fixed) |

## Theme E: Contrastive Learning Methods

| Claim | Source(s) |
|---|---|
| NT-Xent loss formulation | P8 (SimCLR) |
| Positive pair construction: same sample, different augmentation | P8 (SimCLR) |
| Cross-modal positive pairs: image + text description | P9 (CLIP) |
| Projection head improves contrastive representation | P8 (SimCLR) |
| Temperature τ sensitivity; L2 normalization essential | P8 (SimCLR) |
| Contrastive learning reduces sample complexity | P2 (Goldwasser — theoretical bound) |

## Theme F: UMT / Animal Communication Theory

| Claim | Source(s) |
|---|---|
| UMT feasible if communication system is complex enough | P2 (Goldwasser) |
| Error inversely related to language complexity and common ground | P2 (Goldwasser) |
| No published work has decoded coda semantics | P2 (Goldwasser), P1 (WhAM) |
| Combinatorial / compositional structure in codas | P6 (Sharma) |

## Theme G: Evaluation Protocol

| Claim | Source(s) |
|---|---|
| Linear probe on frozen embeddings as standard SSL benchmark | P8 (SimCLR), P1 (WhAM) |
| Macro-F1 preferred over accuracy for imbalanced datasets | Our methodology; standard practice |
| Stratified train/test split by class | Our methodology |
| Fréchet Audio Distance for generative evaluation | P1 (WhAM) |

---

# PART III — ADDITIONAL PAPERS TO EXPLORE

*Papers cited by our core references that are directly relevant to our work, not yet reviewed in depth.*

## High Priority (directly relevant)

| Paper | Authors | Year | Link | Why Relevant |
|---|---|---|---|---|
| Deep Machine Learning Techniques for Detection and Classification of Sperm Whale Bioacoustics | Bermant, Bronstein, Wood, Gero, Gruber | 2019 | https://arxiv.org/abs/1901.07461 | First deep learning paper on sperm whale codas; establishes ML baseline we implicitly build on |
| Evidence from sperm whale clans of symbolic marking in non-human cultures | Hersh et al. | 2022 | https://www.pnas.org/doi/10.1073/pnas.2201692119 | Defines identity coda concept; explains why identity codas resist convergence |
| Towards understanding the communication in sperm whales | Andreas et al. | 2022 | https://arxiv.org/abs/2204.09483 (check) | Broad ML program for coda understanding; co-authors overlap with our team |
| Individual vocal production in a sperm whale social unit | Schulz, Whitehead, Gero, Rendell | 2011 | https://doi.org/10.1111/j.1748-7692.2010.00399.x | Shows individual-level ICI variation within units — directly motivates individual ID as a task |
| AVES: Animal Vocalization Encoder based on Self-Supervision | Hagiwara | 2023 | https://arxiv.org/abs/2210.07554 (check WhAM's citation) | Self-supervised SSL model for animal vocalizations; WhAM uses it as a baseline; related to our approach |
| VampNet: Music generation via masked acoustic token modeling | García et al. | 2023 | https://arxiv.org/abs/2307.04686 | WhAM's base model; understanding VampNet architecture helps understand WhAM's architecture |
| WhaleLM: Finding structure and information in sperm whale vocalizations and behavior | Sharma et al. | 2024 | Not yet on arXiv — search Google Scholar | Companion paper to Sharma 2024; may contain additional model results |

## Medium Priority (methodologically relevant)

| Paper | Authors | Year | Link | Why Relevant |
|---|---|---|---|---|
| Vocal clans in sperm whales | Rendell & Whitehead | 2003 | https://doi.org/10.1098/rspb.2002.2267 | Foundational paper defining vocal clans; cited in every sperm whale paper |
| Sperm whale codas | Watkins & Schevill | 1977 | *J. Acoust. Soc. Am.* 62, 1485 | Seminal paper: first description of codas as a communication system |
| Do sperm whales share coda vocalizations? | Rendell & Whitehead | 2004 | *Animal Behaviour* 67, 865 | Examines coda sharing at unit level; links to combinatorial structure |
| Using identity calls to detect structure in acoustic datasets | Hersh, Gero, Rendell, Whitehead | 2021 | *Methods Ecol. Evol.* 12, 1668 | Method for using identity codas as structural probes — related to our probing approach |
| Contrastive learning for audio (CLAP) | Wu et al. | 2023 | https://arxiv.org/abs/2206.04769 | Contrastive Language-Audio Pretraining — cross-modal contrastive learning specifically for audio |
| HuBERT: Self-supervised speech representation learning | Hsu et al. | 2021 | https://arxiv.org/abs/2106.07447 | Masked prediction for audio; conceptual ancestor of WhAM's masked token approach |
| Descript Audio Codec (DAC) | Kumar et al. | 2023 | https://arxiv.org/abs/2306.06546 | WhAM's tokenizer; understanding DAC helps explain what WhAM's tokens represent |

## Broader Bioacoustics ML (background)

| Paper | Authors | Year | Link | Why Relevant |
|---|---|---|---|---|
| BirdNET: A deep learning solution for avian diversity monitoring | Kahl et al. | 2021 | https://arxiv.org/abs/2306.09073 (check) | WhAM uses BirdNET for FAD metric; also relevant as bioacoustics ML baseline |
| BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics | Hamer et al. | 2023 | https://arxiv.org/abs/2312.07439 | Bioacoustics retrieval benchmark; shows gap in bioacoustic representation learning |
| Overlapping and matching of codas in vocal interactions between sperm whales | Schulz et al. | 2008 | *Animal Behaviour* 76, 1977 | Coda synchronization and turn-taking; relevant to Gubnitsky's synchronization findings |

---

# PART IV — DOWNLOAD LINKS SUMMARY

| Paper | PDF Link | Notes |
|---|---|---|
| P1 WhAM (Paradise et al. 2025) | https://arxiv.org/pdf/2512.02206 | Open access |
| P2 Goldwasser et al. (NeurIPS 2023) | https://arxiv.org/pdf/2211.11081 | Open access |
| P3 Leitão et al. (2023/2025) | https://arxiv.org/pdf/2307.05304 | Open access |
| P4 Gubnitsky et al. (2024) | https://arxiv.org/pdf/2407.17119 | Open access |
| P5 Beguš et al. (2024) | https://github.com/Project-CETI/coda-vowel-phonology | Check repo for PDF link |
| P6 Sharma et al. (Nature Comms 2024) | https://doi.org/10.1038/s41467-024-47221-8 | Open access (Nature Comms) |
| P7 Gero et al. (R. Soc. Open Sci. 2016) | https://doi.org/10.1098/rsos.150372 | Open access (RSOS) |
| P8 SimCLR (Chen et al. 2020) | https://arxiv.org/pdf/2002.05709 | Open access |
| P9 CLIP (Radford et al. 2021) | https://arxiv.org/pdf/2103.00020 | Open access |
| Bermant et al. 2019 | https://arxiv.org/pdf/1901.07461 | Open access |
| Hersh et al. 2022 (PNAS) | https://www.pnas.org/doi/10.1073/pnas.2201692119 | Open access |
| VampNet (García et al. 2023) | https://arxiv.org/pdf/2307.04686 | Open access |
| CLAP (Wu et al. 2023) | https://arxiv.org/pdf/2206.04769 | Open access |
| HuBERT (Hsu et al. 2021) | https://arxiv.org/pdf/2106.07447 | Open access |
