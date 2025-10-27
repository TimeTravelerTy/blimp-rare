# Grammar-Features Pilot — Project Context & Instructions

**High-level goal:** Discover, stress-test, and causally validate **grammar-sensitive internal features** in open-weights LMs (e.g., Llama/Mistral) using **closed data**. We (1) build controlled evals (G/U × Typical/Rare), (2) learn **sparse autoencoders (SAEs)** to expose features, (3) link features to **linguistic phenomena** via probes, and (4) **intervene** (zero/clamper) to test causal responsibility.

---

## Timeline (pilot sprint = 4 weeks)

- **Week 1 — Data prep & stressors**
  - Build **Tier-A** minimal-pair datasets: **G/U × Typical/Rare**.
  - Rare = **multi-swap nouns** with **rare, real lemmas**; preserve tags (NN/NNS) and **COUNT/MASS** when quantifiers require it.
  - Aim for **nonce-ish but grammatical** G-Rare and **same violation** U-Rare.

- **Week 2 — Features (SAEs)**
  - Choose target model + layers. Train **SAEs** on hidden states (small shards).
  - Lock SAE hyperparams (sparsity, code size). Save dictionaries & activations.

- **Week 3 — Probing & analysis**
  - Train **simple probes** (e.g., logistic/linear) from SAE codes to Tier-A labels/phenomena.
    - Derive axes: (i) Polarity (G vs U) = sign of Delta-activation on minimal pairs; (ii) Typicality
    Robustness = sign consistency and >= alpha effect retention on Rare (e.g., alpha=0.6).
    - Mixed & Null: Mixed if a feature ranks top-k for >=2 phenomena with similar weight; Null if
    thresholds fail.
  - Run **Rare stress tests**: Typical→Rare deltas; identify **ConceptOnly** features.
  - Curate a short list of candidate features per phenomenon.

- **Week 4 — Causality & report**
  - **Causal zeroing/clamping** on candidate features; measure accuracy shifts on each cell.
  - Optional closed-model replicate (black-box only).
  - Draft report: methodology, feature cards, causal findings.

---

## Design principles

- **Closed-data, open-weights.** No train-data matching; focus on black-box-compatible methods and open activations.
- **Label integrity first.** All edits preserve the grammatical phenomenon; QC is structural (tags/COUNT/MASS), not fluency.
- **Determinism.** Seed per pair; Good/Bad share the same sampled replacements.
- **Lean first, deepen later.** Start with noun swaps; add verbs/names/resources incrementally.

---

## Week-1 data generation (what exists now)

- Targets: **SV-Agreement, Det–N Agreement, Reflexive/Anaphor, Quantifiers**.
- Edits: **Noun-only multi-swap** (or top-k), skip `ROOT`/entities, preserve **NN/NNS** via `lemminflect`.
- Rarity: `wordfreq` **Zipf < τ**.
- Countability: filter swaps with a **countability lexicon** (e.g., BECL lemma→COUNT/MASS/FLEX) **only** when quantifiers require it.
- Output: JSONL per item with **G-Typical, U-Typical, G-Rare, U-Rare** and swap metadata.

---

## Week-2/3/4 interfaces (so code stays future-proof)

- **SAE training I/O:** given `dataset.jsonl` + model + layer ids → saves `codes.pt`, `dict.pt`, `cfg.json`.
- **Probing I/O:** given `codes.pt` + labels → saves metrics, ROC, per-feature weights; emits **feature cards** (top examples, activation stats).
- **Causal I/O:** given feature indices + clamp/zero spec → runs eval on four cells; saves deltas.

---

## Immediate next steps (short)

1. **Verb randomization (opt-in module)**  
   - Swap **main lexical verb** only (exclude AUX set).  
   - Preserve morph tag (`VB/VBD/VBG/VBN/VBP/VBZ`) with `lemminflect`.  
   - **Transitivity guard:** prefer candidates matching presence/absence of direct object (use a light verb-frames list).  
   - Same per-pair RNG so Good/Bad align.  
   - QC stays structural (only VERB diffs; tag identical).

2. **Uncommon names** (optional benign audit or separate toggle)  
   - Replace PERSON entities with **rare real names** (cap-preserving, coref-consistent).  
   - Skip if gendered pronouns present to avoid agreement drift.  
   - Useful to reduce BLiMP’s frequent-name bias.

3. **Replace hardcoded lemma lists with datasets**  
   - **Rare nouns/verbs:** build from WordNet/other lexicons and filter by Zipf.  
   - **Countability:** use a public **countability lexicon** (collapse sense→lemma majority; FLEX if mixed).  
   - **Verb frames:** lightweight transitive/intransitive flags from VerbNet-style resources or a curated TSV.  
   - **Names:** uncommon given/surnames from public lists; rarity-filtered.

---

## Minimal repo (current → expandable)