# LUMINA

> **L**anguage-model **U**nified **M**eta-analysis with **I**ntegrated **N**umeric **A**ssembly

We piloted LUMINA in two distinct tasks—water (aquaculture) and fire—to construct a LLM-based emission factor database. All extraction tasks follow standard meta-analysis protocols.

## Pipeline

| Stage | Step | Description |
|-------|------|-------------|
| 0 | `prepare` | PDF → Markdown conversion, truncation before References, token audit |
| 1 | `examiner` | Multi-LLM structured JSON extraction per paper per question |
| 2 | `composite` | Aggregate all model outputs into per-question Excel files |
| 3 | `embeddings` | Chunk markdown and generate vector embeddings (cached as `.npy`) |
| 4 | `cross` | Cross-validation: verify evidence existence via embedding retrieval |
| 5 | `ensemble` | Consensus voting with `full` and configured, reachable cross-score thresholds |

```
PDFs ──→ [Prepare] ──→ Markdown ──→ [Examiner] ──→ Per-model CSV
                                                       ↓
                                                 [Composite] ──→ Unified Excel
                                                       ↓
                             [Embeddings] ──→ Chunk vectors ──┐
                                                                ↓
                                                      [Cross-Validation] ──→ Cross-scores
                                                                ↓
              [Ensemble] ← full + configured cross_score filtering ──→ Final consensus
```

### Stage 0 — Prepare

PDFs are converted to Markdown via [marker](https://github.com/VikParuchuri/marker), then truncated at the References / Acknowledgments / Appendix boundary to reduce context length. A token audit reports how many tokens remain after truncation.

### Stage 1 — Examiner

Each paper is sent to multiple LLMs, with each LLM answering a set of domain-specific questions. The LLM receives a 3-message prompt: system role (domain expert) → output instruction (JSON schema) → the specific question. Providers with `supports_json_mode=True` receive `response_format={"type": "json_object"}`; others rely on the prompt plus the existing JSON5 parser. Request and parse failures are retained in `_invalid.txt` and are retried on the next run rather than treated as completed CSVs.

### Stage 2 — Composite

All examiner CSVs are aggregated into per-question Excel files, merging results from every model for every paper. Invalid JSON from `_invalid.txt` is recovered when possible.

### Stage 3 — Embeddings

Each paper's markdown is chunked with `MarkdownTextSplitter` (chunk size 2048, 20% overlap) and embedded via the configured embedding model. Vectors are cached as `.npy` files for reuse.

### Stage 4 — Cross-Validation

For each evidence item produced in Stage 1, the evidence text is embedded and matched against paper chunks via cosine similarity. The best-matching chunk (with ±1 context extension) is sent to **other** LLMs for verification, asking: *"Does this evidence exist in the original context?"* Each verifying LLM returns `existing_flag` (0/1) and a `direct_quote`. These votes are aggregated into a `cross_score` (how many verifying models confirmed the evidence).

### Stage 5 — Ensemble

The composite data is filtered by the unfiltered `full` set and the configured `min_cross_scores` values. Each configured threshold must be between 1 and the number of independent verifiers (`selected models − 1`); LUMINA stops before a cross or ensemble run when the configuration violates that rule. Within each variant, answers are grouped by paper and voted on:

- **Meta items** (location, period, coordinates, species): text normalization → majority voting. Empty values win if they exceed 50%. Coordinates are merged by precision rounding (2 decimal places). Study periods support multiple date formats (`YYYY`, `Month YYYY`, `Month DD YYYY`, `May-December 2010`).
- **Numeric items** (flux values, emission factors): value normalization (stripping units, resolving `±` to center, range normalization) → voting by count, then confidence. When ≥50% of values have ≥2 decimal places, a cross-item aggregation mode groups answers by numeric value rather than by item name.

Each variant outputs two Excel files: a consensus result and an all-standardized-answers detail.

Cross-validation never lets the source model verify its own evidence. Legacy self-verification files are also ignored when scores are re-aggregated.

## Questions Per Domain

### Aqua — freshwater aquaculture GHG emissions

| Q | Topic | Output Fields |
|---|-------|---------------|
| Q1 | Study metadata | `Study_location`, `Study_location_detail`, `Study_period`, `Latitude`, `Longitude` |
| Q2 | Cultured species | `Specie` (fish / shrimp / crab / mixed / others) |
| Q3 | CH₄ flux values | `Flux-N` → `value`, `unit`, `evidence`, `confidence_lv` |

### Wildfire — biomass burning emissions

| Q | Gas | Output Fields |
|---|-----|---------------|
| Q1 | — | `Study_location`, `Study_period` |
| Q2 | CO₂ | `{Fuel}_{Flame}` → `value`, `mce`, `experimental`, `evidence`, `confidence_lv` |
| Q3 | CH₄ | Same structure as Q2 |
| Q4 | N₂O | Same structure as Q2 |

> Fuel types include forest, peatland, crop residue, etc. Flame types: `smoldering`, `flamming`, `mixed`, `None`.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp config.example.py config.py
# → Edit config.py with your API keys, endpoints, and local paths
# → Set supports_json_mode=False for providers that reject response_format
# → Keep every min_cross_scores value ≤ selected-model-count - 1

# Run full pipeline
python run_pipeline.py --domain aqua --stage all
python run_pipeline.py --domain wildfire --stage all

# Or run stages individually
python run_pipeline.py --domain wildfire --stage prepare
python run_pipeline.py --domain wildfire --stage examiner
python run_pipeline.py --domain wildfire --stage composite
python run_pipeline.py --domain wildfire --stage embeddings
python run_pipeline.py --domain wildfire --stage cross
python run_pipeline.py --domain wildfire --stage ensemble
```

## Files

```
github-codes/
├── run_pipeline.py
├── config.example.py
├── requirements.txt
├── README.md
└── lumina/
    ├── preparation.py
    ├── examiner.py
    ├── composite.py
    ├── cross_validation.py
    ├── ensemble.py
    ├── ensemble_utils_meta.py
    ├── ensemble_utils_value.py
    ├── prompts.py
    ├── llm.py
    ├── common.py
    └── utils.py
```

For the scientific workflow, quality gates, and the staged Agent architecture, see [LUMINA_AGENTIC_WORKFLOW.md](LUMINA_AGENTIC_WORKFLOW.md).
