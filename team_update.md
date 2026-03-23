# CS 297 Final Project — Team Update

I did some research to figure out what we should work on for the final project. Here's the full picture so we can align before we start writing.

---

## Why This Topic

I've been following **Project CETI** (Cetacean Translation Initiative) — a nonprofit that is literally trying to decode sperm whale communication using ML. Their question is one of the most fascinating open problems I've encountered: *can machine learning find structure in animal communication that is meaningful, the way words and sentences are meaningful to us?*

Sperm whales are a particularly compelling target. They have the largest brain of any known species, live in multigenerational matrilineal families with documented cultural transmission, and communicate through click sequences called **codas** — short, rhythmic patterns that vary across populations like dialects. What's exciting from an ML perspective is that recent work has shown these codas have **two independent information channels** baked into the same signal:

- a **rhythm channel** (the timing pattern of clicks) that encodes *what type* of coda it is — shared across a clan, like a word category
- a **spectral channel** (the acoustic texture within each click) that encodes *who is speaking* — like a voice fingerprint

No existing model exploits both channels together by design. That's the gap we're going after.

---

## What We Are Studying — Modality & Dataset

This is an **audio understanding** project, which puts us squarely in multimodal AI territory. The input modality is **raw acoustic waveforms** of whale vocalizations, and we'll work with two derived representations from the same signal:

- **Inter-Click Intervals (ICIs)** — the rhythm channel, a short variable-length numeric sequence
- **Mel-spectrograms** — the spectral texture channel, a 2D time-frequency image (same pipeline as most speech/audio deep learning)

This is a great fit for the class because we're fusing two different input representations of the *same physical event* — exactly the multi-view, multi-modal setup we've been studying.

**Dataset: Dominica Sperm Whale Project (DSWP)**
- Hosted publicly on HuggingFace: [`orrp/DSWP`](https://huggingface.co/datasets/orrp/DSWP)  
- **1,501 real sperm whale codas** (~585 MB audio), recorded off Dominica between 2005–2018
- License: **CC BY 4.0** — fully open for research
- Labels include coda type and social unit provenance (which family group produced it)
- Small enough to train on a laptop in minutes

---

## Papers We Are Building On

I used AI to do a systematic literature review of everything CETI and the broader field has published on this problem. Here are the key papers I found and what each one contributes:

| Paper | What It Does For Us |
|---|---|
| **Goldwasser et al., NeurIPS 2023** — *A Theory of Unsupervised Translation Motivated by Understanding Animal Communication* | Provides the theoretical justification: UMT of animal comms is feasible if the system is complex enough. Motivates the whole research direction. |
| **Leitão et al., 2023–2025** — *Evidence of Social Learning Across Symbolic Cultural Barriers in Sperm Whales* | Introduces a computational model encoding **rhythmic micro-variation** within codas; proves cross-clan cultural learning. Directly motivates separating the rhythm channel. |
| **Gubnitsky et al., 2024** — *Automatic Detection and Annotation of Sperm Whale Codas* | First automated coda detector using graph-based clustering. Gives us the preprocessing pipeline and baseline coda-type taxonomy we use as labels. |
| **Paradise et al., NeurIPS 2025** — *WhAM: Towards A Translative Model of Sperm Whale Vocalization* | **The current SOTA.** A transformer audio model (fine-tuned from VampNet) trained on 10k codas. Classifies social units, rhythm types, and vowels as a downstream evaluation. Also releases the DSWP dataset. This is our primary baseline. |
| **Beguš et al., 2024** — *The Phonology of Sperm Whale Coda Vowels* | Formalizes the spectral (vowel) channel linguistically — gives us acoustic feature ground truth to validate against. |

The key observation I came to after reviewing all of these: **WhAM was designed as a generative model, not a representation model.** Its classification results are emergent byproducts of music-audio pre-training. A model specifically designed around the known bimodal structure of codas should do better — and that's what we propose.

---

## Proposed Solution

We propose a **Dual-Channel Contrastive Encoder (DCCE)**. Here's exactly what that means, step by step.

---

### Step 1 — Learn a representation (no labels needed)

We train two small neural networks — one per information channel — to map each coda to a vector in 64-dimensional space. No labels are used. Training is driven by a single rule:

> **Two codas from the same whale → their vectors should be close.**  
> **Two codas from different whales → their vectors should be far apart.**

This is **contrastive learning** (SimCLR-style). The loss function is just a distance measurement in vector space. After training, we have an encoder that organizes codas by biological identity — without ever being told what "identity" means.

**Architecture:**

```
Coda waveform
    │
    ├── Rhythm Encoder (2-layer GRU on ICI sequence) ──► r_emb (64d)
    │                                                           │
    └── Spectral Encoder (small CNN on mel-spectrogram) ──► s_emb (64d)
                                                                │
                                   Fusion MLP ──► z (64d joint embedding)
```

- **Rhythm Encoder**: a 2-layer GRU that reads the sequence of inter-click intervals (a few numbers, like `[0.21s, 0.19s, 0.63s]`) and outputs a 64d vector capturing timing patterns.
- **Spectral Encoder**: a small CNN that reads the mel-spectrogram of the coda (a 2D image, like in speech recognition) and outputs a 64d vector capturing acoustic texture.
- **Fusion MLP**: concatenates both vectors and projects to a single 64d joint embedding `z`.

Two auxiliary losses keep each encoder honest:
- A **coda-type classification head** on `r_emb` only → rhythm encoder must stay type-aware
- A **per-individual contrastive loss** on `s_emb` only → spectral encoder must stay speaker-aware

The key novel idea: **cross-channel positive pairs**. The rhythm of coda A + the spectral texture of a *different coda from the same whale* = still a positive pair. This forces `z` to capture who is speaking, regardless of what coda type they happened to produce.

---

### Step 2 — Test what the representation learned (cheap, supervised)

We freeze the encoder entirely. We take the 64d vectors it produces for all 1,501 DSWP codas and train a **logistic regression** (a simple linear classifier) on top to predict:
- Which social unit produced this coda?
- What coda type is it?
- Which individual whale is it?

If a *linear* classifier on frozen embeddings does well, the encoder has genuinely learned meaningful biological structure. This evaluation technique is called a **linear probe** and is the standard way to benchmark self-supervised representations.

---

### Why this is better than just using WhAM

WhAM is a generative model — it was trained to *synthesize* codas, not to *represent* them. Its classification results are a side effect of music-audio pre-training on a much larger dataset. Our model is purpose-built to separate the two known information channels and fuse them explicitly. The question we're answering is: **does biological domain knowledge about coda structure beat scale?**

---

### Three experiments

1. **Representation quality** — does joint encoding beat rhythm-only, spectral-only, and WhAM embeddings on linear probe classification?
2. **Synthetic augmentation** — does adding WhAM-generated fake codas to our 1,501-sample training set actually help?
3. **WhAM probing** — what do WhAM's internal layers encode, layer by layer? (interpretability)

All experiments run on a MacBook (Apple MPS), estimated ~2 hours total.

---

### Why this fits our multimodal AI class

| Class concept | How we use it |
|---|---|
| Multi-view / multimodal fusion | Two encoders on different representations of the same signal |
| Contrastive self-supervised learning | Core training objective (SimCLR / NT-Xent loss) |
| Heterogeneous encoder architectures | GRU for sequences + CNN for 2D images |
| Linear probing for evaluation | Standard downstream benchmark for self-supervised models |
| Synthetic data generation | WhAM as a data augmentation tool |

---

Let me know what you think and what parts you'd each like to own. Happy to share the full draft paper if you want the deeper technical details on any section.

