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

PDFs are converted to Markdown via [marker](https://github.com/VikParuchuri/marker). The saved Markdown remains complete; downstream reads truncate the body before the References / Acknowledgments / Appendix boundary. A token audit reports full and retained tokens. Partial conversions resume missing or empty Markdown files. Missing/empty inputs fail explicitly. Markdown-only inputs do not require Marker.

### Stage 1 — Examiner

Each paper is sent to multiple LLMs, with each LLM answering a set of domain-specific questions. The LLM receives a 3-message prompt: system role (domain expert) → output instruction (JSON schema) → the specific question. Providers with `supports_json_mode=True` receive `response_format={"type": "json_object"}`; others rely on the prompt plus the existing JSON5 parser. Schema/request failures are retained in `_invalid.txt`. A stage with failed tasks raises an error and the CLI exits nonzero, after retaining successful work. A `.meta.json` sidecar records canonical task identities; successful TSVs carry request/body fingerprints. Resume only skips outputs matching the current paper, prompt, model endpoint, temperature and round. Missing SDK token usage remains unknown, not fabricated as zero.

### Stage 2 — Composite

Only selected models and `RUN.round_index` are aggregated into per-question Excel files. Recoverable `_invalid.txt` data require matching sidecar provenance and a valid schema; canonical model names come from metadata, not lowercased filenames. Each candidate has a content-based `candidate_id`; paper IDs remain strings. Empty results replace previous tables and incomplete expected tasks fail explicitly.

### Stage 3 — Embeddings

Each paper's markdown is chunked with `langchain_text_splitters.MarkdownTextSplitter` (chunk size 2048 characters, 20% overlap) and embedded via the configured embedding model. Vectors are cached as `.npy` files with content/model/endpoint/chunk fingerprints and dimension metadata. Both generation and cross-stage reads validate the cache. Embedding requests have bounded retry for connection/timeout/429/selected 5xx failures; malformed vectors fail explicitly.

### Stage 4 — Cross-Validation

For each evidence item produced in Stage 1, the evidence text is embedded and matched against paper chunks via cosine similarity. The best-matching chunk (with ±1 context extension) is sent to **other** LLMs for verification, asking: *"Does this evidence exist in the original context?"* Each verifying LLM returns `existing_flag` (0/1) and a `direct_quote`. These votes are aggregated into a `cross_score` (how many verifying models confirmed the evidence).

The scientific task remains evidence existence and topical relevance. Cross does not separately verify the extracted numeric value, unit or experimental identity. Cached votes match candidate and verifier-specification fingerprints; row order is not an identity. Unchanged resume performs neither embedding nor verifier API calls. Score aggregation is idempotent and excludes stale/failed/self votes. The CLI validates composite provenance before cross/ensemble; changed source, prompt or selected round requires rebuilding examiner/composite.

### Stage 5 — Ensemble

The composite data is filtered by the unfiltered `full` set and the configured `min_cross_scores` values. Each configured threshold must be between 1 and the number of independent verifiers (`selected models − 1`); LUMINA stops before a cross or ensemble run when the configuration violates that rule. Within each variant, answers are grouped by paper and voted on:

- **Meta items** (location, period, coordinates, species): text normalization → majority voting. Empty values win if they exceed 50%; pandas missing values and textual missing markers count as empty. Coordinates retain signs/hemispheres and are merged by precision rounding (2 decimal places). Study periods support English formats and ISO dates/ranges.
- **Numeric items** (flux values, emission factors): value normalization (scientific notation, signed ranges, `±` center) → voting by count, then confidence. The independent `unit` field participates in candidate identity/voting and is preserved in output; different units do not merge and no physical conversion is automatic. Explicit units retain SI prefix case and Unicode exponents. When ≥50% of values have ≥2 decimal places, the existing cross-item aggregation mode retains all candidate values. This mode is a candidate set, not a guaranteed single winner; its scientific policy is unchanged.

Each variant outputs two Excel files: a consensus result and an all-standardized-answers detail.

Configured threshold variants also write explicit empty tables when no rows pass, replacing stale results. A run status in `ensemble_dir/last_run.json` records the most recent stage's success/failure; it is not a complete scientific acceptance manifest. CLI runs sharing the same examiner parent/domain are serialized with an exclusive `.lumina-<domain>.lock`. After a hard interruption, verify the recorded PID before removing a stale lock. Writes are published atomically. Direct library calls must be coordinated by their caller.

Filtered ensemble requires completed current verification for every candidate with eligible evidence. Missing/failed/outdated verifier results are an incomplete stage, not negative votes or a successful empty filter. Candidates with no evidence remain in `full` and are excluded from filtered variants, as before. Composite validates the complete selected task batch before replacing any prior question table.

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
# If starting from PDF, separately install a compatible Marker release:
# pip install marker-pdf
# Its PdfConverter/create_model_dict/text_from_rendered API must be available.
# A real PDF converter smoke test is separate from the no-network tests below.

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

Tested with Python 3.12. Paths in config are resolved against the process working directory; use absolute paths when invoking from elsewhere. In PowerShell, use `Copy-Item config.example.py config.py`.

For embeddings, `EMBEDDING_MODEL.url` takes precedence; if blank, the embedding provider's `LLM_SETTINGS[...].url` is used. `DOMAINS.questions` must match the fixed domain indices (Aqua `[1,2,3]`, Wildfire `[1,2,3,4]`); arbitrary subsets/new questions are not silently enabled.

### Existing-output migration

Keep previous research outputs as an archive. Outputs from versions without fingerprints cannot be safely treated as current proof: rerun examiner/composite under a new output directory before cross/ensemble. Old `.npy` caches without metadata rebuild, and old row-index-only verifier TSVs are excluded. Numeric-leading paper names such as `01_title.md` retain ID `01`, but duplicate prefixes fail before model calls; other names use the full stem (e.g. `Smith_2020`). These corrections may change results previously affected by parser/unit/identity errors. This release does not reprocess existing research data automatically.

### Verification

```bash
python -B -m unittest discover -s tests -v
```

Tests use the real installed parser, splitter, numerical and Excel dependencies. Only external model/network or PDF-converter boundaries are mocked. Both domains are exercised end to end, including resume, repeated cross, stale inputs, units and failure handling. They do not establish real-provider compatibility, PDF conversion fidelity or scientific accuracy against a human gold standard.

## Agent 运行控制层（已实现，尚待正式验收）

除了上面的六阶段管线，这个仓库现在还有一层运行控制代码，负责在真正调用模型之前把研究条件冻结下来，并记录进度、请求、预算和人工批准。它不改变任何科学算法：六阶段、原文证据核验和共识投票仍由原来的核心模块执行。

控制层把一次运行放进独立目录 `runs/<run_id>/`：输入被复制并锁定哈希，任务、请求尝试、预算预留和人工门保存在 SQLite 账本里，`state.json`、`events.jsonl`、`errors.jsonl`、`metrics.json` 只是便于查看的导出视图。任何时候都以账本和磁盘上的实际产物为准，而不是终端滚动内容；账本损坏会明确失败，不会被当成空白重新开始。

主状态链固定为 `INIT → PREFLIGHT → INPUT_READY → SMOKE → HUMAN_GATE_SMOKE → BATCH → FINAL_QC → ARCHIVE → DONE`，运行中还可进入 `PAUSED`、`RECOVERABLE_ERROR`、`HUMAN_GATE`、`FATAL_ERROR`。先用 `smoke_size` 指定的少量论文走完六阶段，人工检查并批准 smoke 报告后才允许剩余论文批量执行；单篇通过不会自动授权批量。

每个真实 HTTP 请求都要先通过四项冻结上限（`max_calls`、`max_tokens`、`max_cost`、`max_runtime`）和每个模型的价格依据；`max_runtime` 从运行创建时刻起算，暂停和等待不会把它清零。请求结果未知时保留费用预留并交给人工判断，不做盲目重发。可选的诊断顾问默认关闭，只有在规格里写明模型和 provider 后才启用，只读取脱敏错误摘要，不能改数值、规格或预算。

需要说明边界：只有走这套控制层的运行才计入预算账本；直接用旧命令 `python run_pipeline.py --domain ... --stage ...` 仍按原管线运行，不经过预算控制。

完整命令、状态机、人工门、预算与恢复规则、跨运行导入和独立评价接口，见 [AGENT_GUIDE.md](AGENT_GUIDE.md)；研究规格模板见 research.example.json。commit、PR 与 CI 的交付证据统一记录在 [Plan/Goal](LUMINA_AGENTIC_PLAN_AND_GOALS.md) 中；真实论文与批量运行需要另行指定输入、端点、价格和预算，软件测试通过也不代表科学验收完成。

## Files

```
github-codes/
├── run_pipeline.py            # legacy CLI + agent subcommands
├── config.example.py
├── requirements.txt
├── README.md
├── LUMINA_AGENTIC_WORKFLOW.md
├── AGENT_GUIDE.md
├── research.example.json
├── lumina/
│   ├── preparation.py
│   ├── examiner.py
│   ├── composite.py
│   ├── cross_validation.py
│   ├── ensemble.py
│   ├── ensemble_utils_meta.py
│   ├── ensemble_utils_value.py
│   ├── prompts.py
│   ├── llm.py
│   ├── common.py
│   ├── utils.py
│   ├── evaluation.py          # read-only independent evaluation
│   └── agent/                 # run control layer
│       ├── contracts.py       # specification freeze + input isolation
│       ├── store.py           # SQLite ledger
│       ├── budget.py          # reservation, settlement, request bounds
│       ├── runtime.py         # per-request execution and reuse
│       ├── controller.py      # six-stage orchestration and human gates
│       ├── worker.py          # CLI wall-clock supervision
│       ├── advisor.py         # optional diagnosis advisor
│       ├── imports.py         # explicit cross-run import
│       ├── report.py          # run reports
│       ├── lease.py           # OS-owned run lock
│       └── cli.py             # subcommand parsing
```

For the scientific workflow, the implemented control-layer structure, and the remaining milestones, see [LUMINA_AGENTIC_WORKFLOW.md](LUMINA_AGENTIC_WORKFLOW.md). For operating the control layer, see [AGENT_GUIDE.md](AGENT_GUIDE.md). The single Goal and acceptance ledger is LUMINA_AGENTIC_PLAN_AND_GOALS.md.
