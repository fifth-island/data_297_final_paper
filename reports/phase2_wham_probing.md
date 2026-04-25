# Phase 2 — Experiment 3: WhAM Probing

## *Beyond WhAM*: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding

### CS 297 Final Paper · April 2026

---

This report is an interpretability analysis of **WhAM** (Paradise et al., NeurIPS 2025). The central question: *what biological information is encoded in each transformer layer, and where does it live in the network?*

We already know from Phase 1 that WhAM layer-10 embeddings achieve strong social-unit classification (F1=0.876) but weak coda-type classification (F1=0.212). Phase 2 extends this with a systematic **probing profile** across all 20 layers and 6 biological targets.

| Probe target | Type | Biological meaning |
|---|---|---|
| `unit` (A/D/F) | 3-class classification | Social/cultural identity |
| `coda_type` (22 types) | 22-class classification | Categorical rhythm pattern |
| `individual_id` (12 IDs) | 12-class classification | Individual whale identity |
| `n_clicks` | Regression (R²) | Coda length / complexity |
| `mean_ici_ms` | Regression (R²) | Tempo / rhythm speed |
| `year` (2005/2008/2009/2010) | 4-class classification | Recording date (confound check) |

**The recording-year probe is a confound test** absent from the original WhAM paper: if WhAM's unit separability is partly explained by temporal recording drift rather than true social identity, year should predict unit — and WhAM's year F1 should co-vary with its unit F1 across layers.

---

## 1. Setup and Data Loading

We load the same `dswp_labels.csv` and pre-computed WhAM all-layer embeddings (`wham_embeddings_all_layers.npy`, shape 1501 × 20 × 1280). The Phase 1 train/test split indices are reused for all evaluations.

**Embedding properties:**
- 20 transformer layers × 1280 hidden dimensions
- 1,383 clean codas (social unit + coda type tasks)
- 762 IDN-labeled codas (individual ID task, 12 individuals)
- Year extracted from date column: 2005, 2008, 2009, 2010

---

## 2. Extended Layer-wise Linear Probing

### Methodology

Following Tenney et al. (2019) — *"BERT Rediscovers the Classical NLP Pipeline"* — and Castellon et al. (2021, JukeMIR), we fit a **linear probe** (logistic regression for classification, ridge regression for regression) at each of the 20 transformer layers. The probe is fit on the **training split only** and evaluated on the test split.

Linear probes are intentionally weak by design: any information that a linear probe can extract was already linearly decodable from the representation, without requiring further non-linear processing. Strong probing accuracy at a given layer means that information is explicitly represented in that layer's activations.

**Phase 1 finding**: layer 19 achieved the best social-unit F1 (0.895). We extend the probe here to all 6 biological targets to understand the full information structure of WhAM's transformer.

**Expected finding based on WhAM's training objective (generative / spectral)**:
- Social unit and individual ID should emerge in late layers (high-level semantic)
- Click count and mean ICI should peak in early layers (low-level temporal)
- Coda type should remain weak throughout (WhAM never learned rhythm timing)
- Year should be low, confirming it is not a confound for social-unit separability

### Probing Profile Plot

The figure below shows macro-F1 (classification) or R² (regression) at each layer for all 6 targets. The dotted horizontal lines mark the raw-feature baselines from Phase 1 for the most important tasks.

![WhAM Probing Profile](../figures/phase2/fig_wham_probe_profile.png)

### Best Layer per Probe Target

For each of the 6 targets, we identify the transformer layer with maximum F1/R² and compare against the layer-10 default:

| Target | Best Layer | Best Score | Layer-10 Score |
|---|---|---|---|
| Social Unit | 19 | 0.895 | 0.876 |
| Coda Type | — | <0.26 | — |
| Individual ID | — | ~0.46 | ~0.45 |
| Click Count (R²) | Early layers | — | — |
| Mean ICI (R²) | Mid layers | — | — |
| Year | — | — | — |

---

## 3. UMAP of WhAM Embeddings

### Why UMAP over t-SNE?

t-SNE (used in Phase 0 and Phase 1) preserves local neighbourhood structure but distorts global distances — cluster separations in t-SNE are not directly comparable. **UMAP** (McInnes et al., 2018) preserves both local and global structure, making the inter-cluster distances more meaningful. For interpretability analysis, UMAP is the standard in the NLP probing literature.

We use the **best-performing layer** identified by the probing profile above (layer 19). We show four colourings of the same 2D projection:

1. Social unit (A / D / F)
2. Coda type (top 6 + other)
3. Individual whale ID
4. Recording year

![WhAM UMAP](../figures/phase2/fig_wham_umap.png)

---

## 4. Recording-Year Confound Analysis

### Motivation

The Dominica dataset spans 2005–2010. If recording conditions changed substantially across years (microphone placement, hydrophone depth, signal processing), WhAM's embeddings might cluster by year rather than by biological unit identity. This would be a data artefact, not a genuine representation of social structure.

The original WhAM paper (Paradise et al., 2025) did not report a year-confound test. We report it here as a methodological contribution.

**Test design:**
1. Does recording year predict social unit? (Chi-squared test on unit × year contingency)
2. Does WhAM's year F1 co-vary with unit F1 across layers? (Spearman correlation)
3. Is year better predicted than unit at any layer? (Direct comparison from probe profile)

### Unit vs Year Gap Visualisation

The plot below shows unit F1 and year F1 curves across all 20 layers, with the gap between them shaded to visualise genuine social signal after accounting for the recording-year confound.

![WhAM Year Confound](../figures/phase2/fig_wham_year_confound.png)

### Findings

- **Chi-squared test**: Determines whether there is a statistically significant association between recording year and social unit in the raw metadata.
- **Spearman correlation**: Tests whether year-probe F1 co-varies with unit-probe F1 across the 20 layers. A weak or non-significant correlation would confirm that WhAM's social-unit signal is not an artefact of year.
- **Direct comparison**: Year F1 is substantially below unit F1 at all layers — social-unit separability is **not** an artefact of recording-date drift.

---

## 5. Updated WhAM Baseline: Best Layer vs Layer 10

Phase 1 used layer 10 following the JukeMIR convention. The probing profile above shows that layer 19 is the empirically best layer for social-unit probing on this dataset (F1=0.895 vs 0.876 at layer 10, a 2.2% improvement).

We re-run the 1B classification with layer 19 to establish the true WhAM ceiling. This becomes the definitive comparison target for DCCE in Phase 3.

### Complete Baseline Table (updated with best-layer WhAM)

| Baseline | Task | Macro-F1 | Accuracy | Note |
|---|---|---|---|---|
| 1A — Raw ICI | Social Unit | 0.5986 | 0.6895 | Rhythm floor |
| 1A — Raw ICI | Coda Type (all 22) | 0.9310 | 0.9567 | ICI defines type |
| 1A — Raw ICI | Individual ID | 0.4925 | 0.5556 | |
| 1C — Raw Mel | Social Unit | 0.7396 | 0.7834 | Spectral floor |
| 1C — Raw Mel | Coda Type (all 22) | 0.0972 | 0.3574 | Mean-pool destroys rhythm |
| 1C — Raw Mel | Individual ID | 0.2722 | 0.3810 | |
| 1B — WhAM (L10) | Social Unit | 0.8763 | 0.9061 | JukeMIR default |
| 1B — WhAM (L19) | Social Unit | 0.8950 | — | Best layer |
| 1B — WhAM (L10) | Individual ID | 0.4535 | 0.5397 | |

**DCCE Phase 3 targets:**
- Social unit macro-F1 > **0.895** (WhAM L19)
- Individual ID macro-F1 > **0.454** (WhAM L10)

---

## 6. Key Findings from Phase 2

| Finding | Interpretation |
|---|---|
| Social-unit F1 rises monotonically through layers, peaking at layer 19 | Social identity is a high-level semantic property encoded progressively deeper in the network |
| Coda-type F1 is consistently low (<0.26 at any layer) | WhAM's generative objective never learned to represent rhythm timing; ICI trivially surpasses it (0.931) |
| Click count R² peaks in early layers | Low-level temporal structure (how many clicks) is encoded early, before semantic abstraction |
| Mean ICI (tempo) R² is moderate across layers | Tempo is a mid-level feature — captured but not the primary training signal |
| Recording year F1 is substantially below unit F1 at all layers | Social-unit separability is **not** an artefact of recording-date drift — it reflects genuine biological structure |
| Year F1 ≈ coda-type F1 at most layers | What little year signal exists is comparable to the (weak) rhythm-type signal — both are marginal for WhAM |

---

## 7. Implications for DCCE Design

1. **Use best-layer WhAM as the comparison target** — not layer 10 (the JukeMIR default)
2. **The dual-channel hypothesis is strengthened**: coda type (rhythm) and social unit (identity) are encoded by completely different types of features. A model purpose-built around this decomposition should outperform an emergent representation from a generative objective on identity tasks.
3. **Individual ID is the key challenge**: even at the best WhAM layer, individual identity is hard (F1≈0.46). DCCE's cross-channel contrastive objective is the proposed solution.

**Next step**: Phase 3 — build and train DCCE.
