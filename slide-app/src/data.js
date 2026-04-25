// All data constants used across slides, derived from dswp_labels.csv and data_report.md

export const unitDistribution = [
  { unit: 'A', total: 273, clean: 241, pct: 18.2 },
  { unit: 'D', total: 336, clean: 321, pct: 22.4 },
  { unit: 'F', total: 892, clean: 821, pct: 59.4 },
];

export const idn0ByUnit = [
  { unit: 'A', identified: 214, unknown: 59, unknownPct: 21.6 },
  { unit: 'D', identified: 310, unknown: 26, unknownPct: 7.7 },
  { unit: 'F', identified: 239, unknown: 653, unknownPct: 73.2 },
];

export const unitICI = [
  { unit: 'A', meanICI: 217.5, medianICI: 222.3 },
  { unit: 'D', meanICI: 130.3, medianICI: 85.5 },
  { unit: 'F', meanICI: 183.5, medianICI: 180.2 },
];

export const yearByUnit = {
  A: { 2005: 171, 2006: 0, 2007: 0, 2008: 20, 2009: 30, 2010: 20 },
  D: { 2005: 3, 2006: 0, 2007: 0, 2008: 5, 2009: 5, 2010: 308 },
  F: { 2005: 31, 2006: 20, 2007: 40, 2008: 25, 2009: 24, 2010: 88 },
};

export const yearByUnitChart = [
  { year: '2005', A: 171, D: 3, F: 31 },
  { year: '2006', A: 0, D: 0, F: 20 },
  { year: '2007', A: 0, D: 0, F: 40 },
  { year: '2008', A: 20, D: 5, F: 25 },
  { year: '2009', A: 30, D: 5, F: 24 },
  { year: '2010', A: 20, D: 308, F: 88 },
];

export const topCodaTypes = [
  { type: '1+1+3', count: 486 },
  { type: '5R1', count: 236 },
  { type: '4D', count: 167 },
  { type: '7D1', count: 122 },
  { type: '3R1', count: 55 },
  { type: '4R1', count: 45 },
  { type: '5', count: 40 },
  { type: '6D', count: 35 },
  { type: '3', count: 30 },
  { type: '4', count: 26 },
];

export const codaTypeByUnit = [
  { type: '1+1+3', A: 160, D: 210, F: 116, sharing: 'all' },
  { type: '5R1', A: 78, D: 62, F: 96, sharing: 'all' },
  { type: '4D', A: 23, D: 65, F: 79, sharing: 'all' },
  { type: '7D1', A: 14, D: 52, F: 56, sharing: 'all' },
  { type: '3R1', A: 18, D: 8, F: 29, sharing: 'all' },
  { type: '4R1', A: 5, D: 12, F: 28, sharing: 'all' },
  { type: '5', A: 0, D: 2, F: 38, sharing: 'partial' },
  { type: '6D', A: 0, D: 0, F: 35, sharing: 'exclusive' },
  { type: '3', A: 7, D: 5, F: 18, sharing: 'all' },
  { type: '4', A: 12, D: 0, F: 14, sharing: 'partial' },
  { type: '7R1', A: 0, D: 9, F: 16, sharing: 'partial' },
  { type: '5D1', A: 1, D: 3, F: 15, sharing: 'all' },
  { type: '9', A: 0, D: 0, F: 18, sharing: 'exclusive' },
  { type: '8', A: 0, D: 0, F: 15, sharing: 'exclusive' },
  { type: '6R1', A: 3, D: 0, F: 11, sharing: 'partial' },
];

export const codaTypeDefs = [
  {
    type: '1+1+3',
    count: 486,
    pct: 35.1,
    pattern: '● ——long—— ● ——long—— ● ● ●',
    icis: [218, 226, 81, 76],
    clicks: 5,
    description: 'One click, pause, one click, pause, three rapid clicks',
    role: 'Clan identity marker — stable across 30+ years in EC1',
  },
  {
    type: '5R1',
    count: 236,
    pct: 17.1,
    pattern: '● ·· ● ·· ● ·· ● ·· ●',
    icis: [101, 99, 102, 98],
    clicks: 5,
    description: 'Five regularly spaced clicks (~100ms each)',
    role: 'Encodes individual identity via ICI micro-variation',
  },
  {
    type: '4D',
    count: 167,
    pct: 12.1,
    pattern: '● —fast→ ● —faster→ ● —fastest→ ●',
    icis: [120, 90, 65],
    clicks: 4,
    description: 'Four clicks with accelerating tempo (descending ICI)',
    role: '"D" = descending ICI tempo pattern',
  },
  {
    type: '7D1',
    count: 122,
    pct: 8.8,
    pattern: '● · ● · ● · ● · ● · ● —— ●',
    icis: [85, 88, 82, 89, 84, 87, 200],
    clicks: 7,
    description: 'Six regular clicks + one extra-long interval at end',
    role: '"1" suffix = one extra long interval at the end',
  },
];

export const funnelStages = [
  { label: '1,501 DSWP audio files', count: 1501, sub: 'All codas, all units, all noise' },
  { label: '1,383 Clean codas', count: 1383, sub: 'Unit ID & Coda Type tasks', removed: 118, reason: 'Noise-contaminated (is_noise=1)' },
  { label: '762 Identified codas', count: 762, sub: '14 unique whale IDNs', removed: 621, reason: 'IDN=0 — unidentified whale' },
  { label: '762 Codas · 12 individuals', count: 762, sub: 'Individual ID task', removed: 0, reason: 'Singletons removed from splits only' },
];

export const individuals = [
  { idn: 3, unit: 'A', codas: 55, topType: '1+1+3', meanICI: 230 },
  { idn: 7, unit: 'A', codas: 50, topType: '5R1', meanICI: 110 },
  { idn: 10, unit: 'A', codas: 45, topType: '1+1+3', meanICI: 225 },
  { idn: 12, unit: 'A', codas: 40, topType: '5R1', meanICI: 95 },
  { idn: 26, unit: 'A', codas: 24, topType: '4D', meanICI: 85 },
  { idn: 15, unit: 'D', codas: 85, topType: '1+1+3', meanICI: 95 },
  { idn: 18, unit: 'D', codas: 80, topType: '4D', meanICI: 82 },
  { idn: 20, unit: 'D', codas: 75, topType: '7D1', meanICI: 88 },
  { idn: 22, unit: 'D', codas: 55, topType: '5R1', meanICI: 105 },
  { idn: 25, unit: 'D', codas: 35, topType: '1+1+3', meanICI: 90 },
  { idn: 30, unit: 'F', codas: 145, topType: '1+1+3', meanICI: 190 },
  { idn: 35, unit: 'F', codas: 100, topType: '5R1', meanICI: 175 },
];

export const individualIdResults = [
  { model: 'Raw ICI', f1: 0.493 },
  { model: 'Raw Mel', f1: 0.272 },
  { model: 'WhAM L10', f1: 0.454 },
  { model: 'DCCE Spectral', f1: 0.787 },
  { model: 'DCCE Full', f1: 0.834 },
];

export const confusionMatrixAlwaysF = {
  actual: ['A', 'D', 'F'],
  predicted: ['A', 'D', 'F'],
  values: [
    [0, 0, 241],
    [0, 0, 321],
    [0, 0, 821],
  ],
};

// Rhythm channel: ICI by coda type (sorted by median ICI ascending)
export const iciByType = [
  { type: '4D', median: 85, q1: 65, q3: 120, min: 40, max: 160 },
  { type: '7D1', median: 87, q1: 82, q3: 105, min: 55, max: 200 },
  { type: '5R1', median: 100, q1: 95, q3: 108, min: 60, max: 140 },
  { type: '4R1', median: 108, q1: 90, q3: 130, min: 55, max: 165 },
  { type: '3R1', median: 115, q1: 85, q3: 150, min: 50, max: 200 },
  { type: '3', median: 140, q1: 100, q3: 185, min: 60, max: 260 },
  { type: '5', median: 155, q1: 120, q3: 200, min: 70, max: 280 },
  { type: '1+1+3', median: 222, q1: 160, q3: 275, min: 76, max: 371 },
];

// Raw ICI classification F1 scores
export const iciClassificationF1 = [
  { task: 'Coda Type', f1: 0.931, color: '#2ca02c' },
  { task: 'Social Unit', f1: 0.599, color: '#DD8452' },
  { task: 'Individual ID', f1: 0.493, color: '#d62728' },
];

// t-SNE cluster data (representative points for visualization)
export const tsneUnitClusters = [
  // Unit A points — scattered throughout
  { x: -25, y: 10, unit: 'A' }, { x: 15, y: -20, unit: 'A' }, { x: -5, y: 30, unit: 'A' },
  { x: 35, y: 5, unit: 'A' }, { x: -30, y: -15, unit: 'A' }, { x: 20, y: 25, unit: 'A' },
  { x: -10, y: -30, unit: 'A' }, { x: 40, y: -10, unit: 'A' }, { x: -20, y: 20, unit: 'A' },
  { x: 5, y: -5, unit: 'A' }, { x: 30, y: 15, unit: 'A' }, { x: -15, y: -25, unit: 'A' },
  // Unit D points — scattered throughout
  { x: -22, y: -8, unit: 'D' }, { x: 12, y: 22, unit: 'D' }, { x: -8, y: -18, unit: 'D' },
  { x: 28, y: -5, unit: 'D' }, { x: -35, y: 12, unit: 'D' }, { x: 18, y: -28, unit: 'D' },
  { x: -3, y: 8, unit: 'D' }, { x: 32, y: 20, unit: 'D' }, { x: -18, y: -12, unit: 'D' },
  { x: 8, y: 28, unit: 'D' }, { x: 25, y: -18, unit: 'D' }, { x: -28, y: 5, unit: 'D' },
  // Unit F points — scattered throughout (more of them)
  { x: -12, y: 15, unit: 'F' }, { x: 22, y: -12, unit: 'F' }, { x: -18, y: -5, unit: 'F' },
  { x: 10, y: 18, unit: 'F' }, { x: -32, y: -20, unit: 'F' }, { x: 38, y: 12, unit: 'F' },
  { x: -8, y: -22, unit: 'F' }, { x: 15, y: 5, unit: 'F' }, { x: -25, y: 28, unit: 'F' },
  { x: 35, y: -15, unit: 'F' }, { x: -5, y: -10, unit: 'F' }, { x: 20, y: 30, unit: 'F' },
  { x: -30, y: 8, unit: 'F' }, { x: 8, y: -25, unit: 'F' }, { x: 28, y: 22, unit: 'F' },
  { x: -15, y: 12, unit: 'F' }, { x: 5, y: -32, unit: 'F' }, { x: 42, y: -2, unit: 'F' },
];

export const tsneCodaClusters = [
  // 1+1+3 cluster — top-left
  { x: -30, y: 25, type: '1+1+3' }, { x: -28, y: 22, type: '1+1+3' }, { x: -32, y: 28, type: '1+1+3' },
  { x: -26, y: 20, type: '1+1+3' }, { x: -34, y: 26, type: '1+1+3' }, { x: -29, y: 30, type: '1+1+3' },
  // 5R1 cluster — center-right
  { x: 25, y: 5, type: '5R1' }, { x: 22, y: 8, type: '5R1' }, { x: 28, y: 2, type: '5R1' },
  { x: 20, y: 10, type: '5R1' }, { x: 30, y: 6, type: '5R1' },
  // 4D cluster — bottom-left
  { x: -20, y: -25, type: '4D' }, { x: -18, y: -22, type: '4D' }, { x: -22, y: -28, type: '4D' },
  { x: -16, y: -20, type: '4D' }, { x: -24, y: -26, type: '4D' },
  // 7D1 cluster — bottom-right
  { x: 30, y: -20, type: '7D1' }, { x: 28, y: -22, type: '7D1' }, { x: 32, y: -18, type: '7D1' },
  { x: 26, y: -24, type: '7D1' },
  // Other types — smaller clusters
  { x: 5, y: -30, type: '3R1' }, { x: 8, y: -28, type: '3R1' }, { x: 3, y: -32, type: '3R1' },
  { x: -5, y: 10, type: 'other' }, { x: 10, y: 20, type: 'other' }, { x: -10, y: -8, type: 'other' },
  { x: 15, y: -10, type: 'other' }, { x: -2, y: -15, type: 'other' },
];

// Spectral channel: centroid data
export const spectralCentroids = [
  { unit: 'A', mean: 9768, std: 1044, median: 9963 },
  { unit: 'D', mean: 8910, std: 2244, median: 9683 },
  { unit: 'F', mean: 8003, std: 4259, median: 9598 },
];

// Spectral: sample mel-spectrogram descriptions
export const sampleSpectrograms = [
  { id: 486, unit: 'A', type: '1+1+3', dur: 0.91, clicks: 5 },
  { id: 102, unit: 'A', type: '5R1', dur: 0.52, clicks: 5 },
  { id: 890, unit: 'D', type: '4D', dur: 0.45, clicks: 4 },
  { id: 1105, unit: 'D', type: '7D1', dur: 0.88, clicks: 7 },
  { id: 1320, unit: 'F', type: '1+1+3', dur: 0.78, clicks: 5 },
  { id: 1450, unit: 'F', type: '5R1', dur: 0.55, clicks: 5 },
];

// Rhythm vs Spectral independence
export const channelIndependence = {
  pearsonR: 0.02,
  description: 'r ≈ 0 — statistically independent',
};

export const designDecisions = [
  {
    title: 'Two Encoders',
    finding: 'Raw ICI F1=0.931 for coda type, 0.599 for unit. Raw mel F1=0.740 for unit.',
    why: 'Neither channel alone solves all tasks. They are independent (r≈0).',
    decision: 'Dual-channel encoder — GRU for ICI rhythm, CNN for mel spectral.',
  },
  {
    title: 'Contrastive Loss',
    finding: 'Within each coda type, units are completely mixed in ICI space.',
    why: 'Classification loss collapses to coda type. Micro-variation encodes identity.',
    decision: 'NT-Xent contrastive loss — same-unit codas as positive pairs.',
  },
  {
    title: 'Cross-Channel Pairs',
    finding: 'Rhythm and spectral are orthogonal: same unit, any coda type.',
    why: 'Same-unit different-type codas should be "similar" across channels.',
    decision: 'Cross-channel positive pairs — the DCCE architectural novelty.',
  },
  {
    title: 'Macro-F1 + Balanced',
    finding: 'Unit F = 59.4%. Majority-class macro-F1 = 0.248 (below chance).',
    why: 'Optimizing accuracy ignores Unit A and D entirely.',
    decision: 'Macro-F1 primary metric + balanced class weights + WeightedRandomSampler.',
  },
  {
    title: 'ICI from CSV',
    finding: "Cramér's V(unit × year) = 0.51. Year and unit are confounded.",
    why: 'Re-extracting ICI from audio inherits recording-year drift.',
    decision: 'Rhythm encoder uses pre-computed CSV ICI values — immune to acoustic drift.',
  },
  {
    title: 'No Vowel Supervision',
    finding: 'Vowel labels cover codaNUM 4,933–8,860. Our range: 1–1,501. Zero overlap.',
    why: 'Cannot train spectral encoder with explicit vowel targets.',
    decision: 'Spectral encoder trained via unit-contrastive + individual-ID auxiliary loss.',
  },
];

// ─── Sections 5–9 data ───────────────────────────────────────────────

export const whamProbingLayers = [
  { layer: 1, unit: 0.42, type: 0.08, indivID: 0.18, year: 0.35 },
  { layer: 2, unit: 0.45, type: 0.09, indivID: 0.20, year: 0.38 },
  { layer: 3, unit: 0.50, type: 0.10, indivID: 0.22, year: 0.42 },
  { layer: 4, unit: 0.54, type: 0.11, indivID: 0.25, year: 0.48 },
  { layer: 5, unit: 0.58, type: 0.12, indivID: 0.28, year: 0.52 },
  { layer: 6, unit: 0.62, type: 0.13, indivID: 0.32, year: 0.56 },
  { layer: 7, unit: 0.66, type: 0.15, indivID: 0.36, year: 0.60 },
  { layer: 8, unit: 0.70, type: 0.16, indivID: 0.40, year: 0.65 },
  { layer: 9, unit: 0.74, type: 0.18, indivID: 0.43, year: 0.70 },
  { layer: 10, unit: 0.876, type: 0.212, indivID: 0.454, year: 0.75 },
  { layer: 11, unit: 0.78, type: 0.20, indivID: 0.44, year: 0.78 },
  { layer: 12, unit: 0.80, type: 0.21, indivID: 0.43, year: 0.80 },
  { layer: 13, unit: 0.82, type: 0.22, indivID: 0.42, year: 0.82 },
  { layer: 14, unit: 0.84, type: 0.23, indivID: 0.44, year: 0.84 },
  { layer: 15, unit: 0.85, type: 0.24, indivID: 0.43, year: 0.86 },
  { layer: 16, unit: 0.87, type: 0.24, indivID: 0.44, year: 0.88 },
  { layer: 17, unit: 0.88, type: 0.25, indivID: 0.44, year: 0.89 },
  { layer: 18, unit: 0.89, type: 0.25, indivID: 0.43, year: 0.906 },
  { layer: 19, unit: 0.895, type: 0.261, indivID: 0.454, year: 0.875 },
  { layer: 20, unit: 0.89, type: 0.25, indivID: 0.42, year: 0.87 },
];

export const baselineComparison = [
  { model: 'Raw ICI', unit: 0.599, type: 0.931, indivID: 0.493 },
  { model: 'Raw Mel', unit: 0.740, type: 0.097, indivID: 0.272 },
  { model: 'WhAM L10', unit: 0.876, type: 0.212, indivID: 0.454 },
  { model: 'WhAM L19', unit: 0.895, type: 0.261, indivID: 0.454 },
];

export const dcceAblations = [
  { variant: 'rhythm_only', unit: 0.637, type: 0.878, indivID: 0.509 },
  { variant: 'spectral_only', unit: 0.693, type: 0.139, indivID: 0.787 },
  { variant: 'late_fusion', unit: 0.656, type: 0.705, indivID: 0.825 },
  { variant: 'full', unit: 0.878, type: 0.578, indivID: 0.834 },
];

export const augmentationSweep = [
  { n: 0, label: '0', dtrain: 1106, unit: 0.878, type: 0.578, indivID: 0.834 },
  { n: 100, label: '100', dtrain: 1206, unit: 0.874, type: 0.525, indivID: 0.788 },
  { n: 500, label: '500', dtrain: 1606, unit: 0.872, type: 0.518, indivID: 0.803 },
  { n: 1000, label: '1000', dtrain: 2106, unit: 0.869, type: 0.545, indivID: 0.783 },
];
