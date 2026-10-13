# Study configuration and migration / 研究配置与迁移

Use Python 3.12. Start with `configs/aqua`, `configs/wildfire`, or `configs/generic`.
Each copy is a self-contained research definition; no Python edits are needed for
new questions that use the existing text/numeric algorithms.

## Install into your own venv / 自定义虚拟环境安装

For interactive setup, run `python start.py --venv /path/to/your/venv` with Python 3.12. It opens the bilingual wizard after environment preparation, remembers language/config locally, and skips installation when the environment is healthy. Use `--lang en` or `--config PATH` as needed. Resume is separate from editing frozen research settings. In noninteractive jobs use `run_pipeline.py --config` instead.

For manual installation only, run `python install.py --venv /path/to/your/venv` with Python 3.12
(Windows example: `python install.py --venv "D:/envs/lumina"`). The installer creates
or reuses that venv and uses its interpreter for every pip operation. With no path,
it uses the active venv, otherwise the repository's `.venv`. Activate it afterwards
or use the printed interpreter path for all pipeline commands.

The default profile installs `requirements-pdf.txt`: runtime packages, pypdf and
Marker 2.0.0. If Marker cannot install or import, setup reports the pypdf-only
fallback explicitly. `--skip-marker` opts into that lightweight installation.
`pip check` and import probes must pass before setup reports success. `start.py` probes Marker in an existing core environment before calling it healthy.
A default installation writes `<venv>/.lumina-install-profile.json`; a recorded
`pypdf-fallback` is announced and reused without repeatedly retrying Marker.
To retry Marker, remove that receipt and rerun `start.py`, or use `install.py`
explicitly. Package installation does not download Marker weights or send model calls.
The first PDF's model preparation/download time counts toward `max_runtime`, so
include it in the initial run's time budget or prepare the resources beforehand.

## Files / 文件职责

| File | Contents |
|---|---|
| `project.toml` | Study, models/providers, retrieval, execution, budgets, price evidence |
| `papers.txt` | One source PDF per line by default; Markdown remains supported; blank lines and `#` comments are ignored |
| `questions.toml` | Prompt templates and an ordered list of question rules |
| `secrets.local.toml` | Local provider keys, copied from `secrets.example.toml`; never commit |

Relative paths resolve from the configuration directory, not the shell's current
directory. Paper paths may contain spaces and Unicode. Duplicate content is rejected
before any paid request. Questions use stable IDs; their order supplies the numeric
indices retained in existing workbooks.

相对路径以配置目录为基准。文章清单支持中文和空格；同一文章的重复内容会在请求前拒绝。
问题的 `id` 必须唯一，问题排列顺序对应现有结果表中的数字编号。

## Project / 主配置

- Root `selected_models`: aliases of the extraction models selected from `[models]`.
- `[project]`: `domain`, `domain_knowledge`, `run_id`, `runs_dir`, `papers`, `questions`,
  `secrets`, `rounds`, `smoke_size`, `stage`, `allow_pdf_resources`.
- `[models.<alias>]`: full `model` ID, provider `source`, optional `limit_token`,
  `max_input_tokens`, `max_output_tokens`, `timeout_seconds`, and `rate_limit_seconds`.
- `[providers.<source>]`: chat base `url` and `supports_json_mode`. Keys belong only
  in `[providers.<source>]` inside `secrets.local.toml`.
- `[embedding]`: `model`, `source`, and complete embedding request `url`, with optional
  timeout/rate-limit settings. Chat and embedding endpoints are distinct contracts.
- `[run]`: `round_index`, `temperature`, `chunk_size`, `overlap_percent`,
  `text_extension`, `min_cross_scores`, `min_consensus_models`.
- `[budget]`: optional positive `max_calls`, `max_tokens`, `max_cost`, `max_runtime`; omit any unset cap.
- `[pricing."full/model-id"]`: `input_per_million`, `output_per_million`,
  `max_input_tokens`, `max_output_tokens`, and dated/provider-specific `basis`.

Prices may be omitted. Use quoted TOML keys for IDs containing `/` or `.`.
Request limits belong on the model/embedding entry independently of prices.
Explicit legacy limits in pricing entries remain supported. New requests default
to 4096 output tokens (embedding: 0), 120-second timeout, and conservative input
reservation based on the actual payload when no explicit input bound is given.
Zero prices require evidence of a free arrangement; unknown rates are not zero.

`rounds` is the explicit list of requested rounds; `round_index` is the legacy
single-round default and is normalized to the first requested round. For M selected
models, verification thresholds are 1…M−1 and consensus thresholds are 1…M.
Omitting `min_consensus_models` uses the strict majority M // 2 + 1.

`stage` is the invocation's stop position, not a change to the scientific task.
Earlier prerequisite stages are checked/reused. Changing it to `all` continues the
same frozen run. Costs and the wall-clock deadline are never reset on resume.

四项预算可独立省略；省略表示该项未设上限。设置后的 `max_runtime` 从运行创建开始计算，包含等待审核时间。
阶段选择不会清空已花费用、重置时间或跳过试跑确认。程序默认不使用顾问模型。

## PDF preparation / PDF 转 Markdown

The examples take PDFs and automatically produce `outputs/prepared/<paper_uid>.md`.
Set conversion policy in the same main configuration:

```toml
[pdf]
backend = "marker"
fallback = "pypdf"
```

`marker` is the default. Import/model initialization failure, conversion failure or
empty Marker text can use `pypdf` when enabled. Disable fallback with `"none"`.
Choose `backend = "pypdf", fallback = "none"` for explicit text-only conversion.
The fallback performs no OCR: it refuses encrypted documents and any page without
extractable text instead of publishing an incomplete article. Review reading order
and tables even when extraction succeeds.

`project.allow_pdf_resources = true` in the shipped examples explicitly permits
Marker to prepare/download its local OCR/layout weights on first PDF use. For an
offline machine, prepare those model resources in advance or use the pypdf backend
for text-bearing PDFs. This permission does not authorize scientific API calls
outside the configured trial, approval and budget controls.

Conversion policy participates in the frozen specification. Metadata records the
source PDF hash, derived Markdown hash, converter name/version, fallback status and
the primary error type. Trial/final reports include these in `preparation`, and the
CLI highlights text-only conversion for review. Original PDF snapshots are retained.

Official APIs: [Marker](https://github.com/datalab-to/marker),
[pypdf extraction](https://pypdf.readthedocs.io/en/stable/user/extract-text.html).

## Questions and prompts / 问题及提示词

`[templates]` defines `system`, `instruction`, `verifier_system`,
`verifier_instruction`, and `verifier_full`. Preserve supported placeholders:
`{domain}` for extraction system context, `{content}` for article text, and
`{context}`, `{answer}`, `{key_topic}` for verification. Literal JSON braces in
formatted templates must be doubled (`{{` and `}}`); question prompt text is sent
literally, so its JSON braces do not need escaping.

Each `[[questions]]` entry contains:

| Field | Meaning |
|---|---|
| `id` | Stable unique identifier |
| `prompt` | Full question text; TOML multiline strings are supported |
| `kind` | `text` or `numeric`; selects existing normalization and consensus behavior |
| `items` | Allowed `item` names; an empty list permits variable item names |
| `allowed_values` | Optional permitted values, especially categorical answers |
| `require_unit` | Require a unit on nonmissing answers |
| `require_experimental` | Require an explicit experiment marker on nonmissing answers |
| `emission_type` | Optional output label used by the Wildfire example |

Model responses remain objects of named items containing scalar `value`, `evidence`,
and `confidence_lv`, plus `unit`/`experimental` when required. Keep prompt instructions
consistent with these rules. A changed prompt cannot override local validation.
Missing answers retain existing missing-value semantics, and scientific numeric
normalization is unchanged by this refactor.
If any nonmissing item violates a required field, the response task fails validation;
its raw response/receipt is retained for inspection rather than silently publishing a
partial response. The built-in studies retain their previous validation strength:
Wildfire's requested experimental marker is not newly made mandatory by this refactor.

新增主题可调整问题数量、顺序、字段名及提示词。不要通过提示词引入程序不支持的嵌套
回答格式；配置类型与输出结构不匹配时应修正配置，而不是放宽校验。

## Running, recovery and migration / 运行与迁移

```bash
python run_pipeline.py --config configs/my-study/project.toml --check
python run_pipeline.py --config configs/my-study/project.toml
```

The first command validates inputs and configuration without model requests or run
creation. The second creates or resumes `runs_dir/run_id`, stops for trial review,
then continues only after an explicit interactive `y`/`yes`. A noninteractive run
returns 2 at the review boundary. Detailed outputs and receipts stay under the run
directory; credentials are excluded from the manifest.

Prompts, field rules, articles, selected model identities, scientific parameters,
prices, budgets, and relevant code are checked against the frozen run. Changing them
requires a new run ID. Credential rotation with unchanged provider identity is
allowed. Configuration is not hot-reloaded during a processing step.

Legacy usage is retained:

```bash
python run_pipeline.py --config config.py --domain aqua --stage all
python run_pipeline.py run --spec research.json --config config.py --dry-run
```

To migrate, map `FULL_LLM_POOL` to `[models]`, `SELECTED_KEYS` to `selected_models`,
`LLM_SETTINGS` to `[providers]` and the secrets file, `EMBEDDING_MODEL` to `[embedding]`,
and `RUN` to `[run]`. Move the research JSON's article list, budgets and pricing into
the study files. Copy the built-in question file before modifying it. Use a new run
ID; existing workbooks and historical ledgers are not converted or overwritten.
Omit old descriptive `tag` fields. Put embedding input/output bounds under
`[embedding]`; legacy pricing bounds remain supported. The old embedding
`limit_token` metadata is not a study input.

Legacy runs freeze their code as well as their data. If they were created before
this change, resume them with the original revision, or create a new study. The
legacy direct-stage entry retains its original behavior; it does not acquire the
controlled study's budgets merely because it shares the scientific modules.

Both paths use the same scientific algorithms: Tvfy AND Tbsl, distinct generating
model votes, no acceptance of ties, no automatic unit conversion, and separate
experiment identities. Software completion is not independent scientific validation.

## Optional budgets and quotes / 可选预算与报价

The wizard accepts blank prices/caps for new studies, Enter to preserve an existing
value, and `-` to clear it. In TOML remove an unset key (TOML has no null literal).
Resolved snapshots use null for unset caps. No artificial large-number limit is used.
A money cap requires input/output rates for every selected chat and embedding model,
and comparable currency metadata. Mixed currencies are rejected, never converted.
Without full prices the report shows the priced subtotal and unpriced request count;
it does not claim a complete total. Unknown usage retains reservations. Single-request
timeouts, finite retries, unknown-outcome reconciliation, and trial approval remain.

报价导入支持本地 JSON 文件或无认证的公开 HTTP(S) JSON 地址，网络请求超时设为 10 秒、
最多读取 1 MiB。不是通用网页抓取器。界面预览匹配结果后要求确认；失败可手填或保留未知。
按实际接口和完整模型 ID 匹配。Embedding 的接口需填写完整请求 URL。
不使用官方价格推断代理价格，不自动换汇。修改币种或单价会将来源标记为手工输入。

```json
{
  "quotes": [
    {
      "base_url": "https://your-provider.example/v1",
      "model": "full/model-id",
      "currency": "USD",
      "input_per_million": 1.0,
      "output_per_million": 2.0,
      "as_of": "2026-10-13"
    }
  ]
}
```

These are schema examples, not actual provider prices. Imported entries freeze
`base_url`, `source`, `fetched_at`, `as_of`, `currency` and `basis` in the configuration.
Changing the provider endpoint invalidates its imported price binding. Rates never
refresh mid-run. For legacy entries with no currency, all participating prices must
use the same implicit unit; new imports always require a currency.

## Wizard files, editing and logs / 向导与日志

All scientific inputs stay in the four existing files. TOML Kit preserves comments
and untouched values when editing. The wizard displays all selected PDFs before
confirmation; imported manifest paths resolve against the manifest directory.
New selections are saved as absolute paths. An edit copied to another directory
keeps the original paper and output locations instead of silently rebasing them.

`VISUAL` then `EDITOR` selects the prompt editor; Windows falls back to Notepad,
Linux to an available nano/vi. Editor commands run without a shell. If unavailable,
the path is displayed for manual editing. Invalid edits remain in the setup editor
buffer for correction, and do not replace the last valid question definition.

`runs/_setup/<session>/` contains startup/install/error logs and redacted wizard events.
Draft exit/interruption saves four files under that session's `draft/`; it does not
overwrite the original study. Drafts contain your local secrets file, so keep the
Git-ignored setup directory private. `.lumina-ui.json` stores only language/config
location (and the original destination for a draft). Starting a run links redacted
logs under `reports/setup/<session>/`; drafts and secret files are never copied there.

运行显示使用只读账本观察；模型请求和科学状态仍由原控制层管理。等待提示包括
PDF/Marker 准备、请求响应、限速及阶段写入。未知总量只显示活动和耗时；有完成回执
才计入完成量。每轮按配置阈值展示正式结果数量、路径和最多 5 条预览。
显示失败不会重试模型请求。中断后使用相同配置续跑，未决请求需先核对。
