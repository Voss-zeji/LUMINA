# LUMINA 运行控制层操作指南

本文件写给真正要操作 LUMINA 控制层的人：怎么准备环境、怎么写研究规格、怎么运行和查看进度、什么时候需要人工批准、预算和失败如何处理、以及如何做独立的科学评价。命令名和参数保持英文原样，其余用普通话说明。

科学算法本身不在本文件重复。六阶段管线、证据核验和共识投票见 [README.md](README.md) 与 [LUMINA_AGENTIC_WORKFLOW.md](LUMINA_AGENTIC_WORKFLOW.md)。各 Goal 的正式验收状态以 [LUMINA_AGENTIC_PLAN_AND_GOALS.md](LUMINA_AGENTIC_PLAN_AND_GOALS.md) 为唯一台账，本文件不另立一份验收清单。

本指南描述 `feat/agent-control-v1` 的运行控制能力；最新 commit、PR 和 CI 证据统一记录在 Plan/Goal 文件中。真实论文与批量运行需另行指定输入、端点、价格和预算。软件能规划、查询、暂停、恢复、批准、报告和评价，但“跑通软件”不等于“科学结论已验证”。

## 1. 准备环境

环境用 Python 3.12（本地验证记录见 AGENT_FOUNDATION_VALIDATION.json，记录环境为 3.12.13）。依赖就是仓库已有的 requirements.txt，不需要额外安装：

```powershell
python -m pip install -r requirements.txt
```

只有从 PDF 开始时，才需要另外装一个兼容的 marker 版本：`pip install marker-pdf`，并确认 `marker.converters.pdf` 的 `PdfConverter` / `create_model_dict` / `text_from_rendered` 可用。只用 Markdown 输入时不需要 marker。

然后复制配置模板并填写本地内容：

```powershell
Copy-Item config.example.py config.py
```

config.py 保存密钥和端点，已经在 .gitignore 里，不要提交到 GitHub；仓库里的 config.example.py 刻意留空。填完以后重点核对三处：每个 provider 的 `key` / `url` / `supports_json_mode`；`SELECTED_KEYS` 与 `EMBEDDING_MODEL`；`RUN` 里的 `temperature`、`chunk_size`、`overlap_percent`、`text_extension`、`min_cross_scores`。两个模型时 `min_cross_scores` 只能是 `[1]`；换成 M 个模型时，每个阈值必须落在 `1..M-1`，否则控制层在 cross/ensemble 之前就停下。

控制层不会替你挑科学模型。`SELECTED_KEYS` 和 `EMBEDDING_MODEL` 是研究者明确选定的科研模型；开发时用的 Codex 子 agent 模型池是另一套东西，两者不能混为一谈，控制层也不会默认再塞进第三个模型。

## 2. 研究规格 research.json

`run` 需要一个 JSON 规格文件。仓库里的 research.example.json 是模板：它结构完整、可以被 JSON 解析，但预算和价格是 `null` 占位符（basis 写着 REPLACE），在校验时会被直接拒绝，因此不能拿去发付费请求。这样做是故意的，避免模板被误当成可运行配置。

规格只接受下面这些字段，多写或写错字段名都会被拒绝；凭据不属于规格，放在 config.py 里：

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `domain` | 必填 | `aqua` 或 `wildfire`，且必须在 config.py 里配置过 |
| `papers` | 必填 | 非空列表。每篇写成路径字符串，或 `{path, paper_uid}` 对象 |
| `rounds` | 可选 | 正整数轮次列表，默认取 `RUN.round_index`；各轮输出分开保存 |
| `smoke_size` | 可选 | 先跑几篇，默认 1 |
| `smoke_papers` | 可选 | 显式指定 smoke 用哪些论文（写 UID 或输入路径） |
| `budget` | 必填 | `max_calls`、`max_tokens`、`max_cost`、`max_runtime` 四项，缺一不可 |
| `pricing` | 必填 | 以模型 ID 为键，覆盖每个被选中的模型 |
| `advisor` | 可选 | `null` 或 `{model, source}`，默认关闭 |
| `allow_pdf_resources` | 可选 | 布尔值，默认 `false`；要下载 PDF 模型资源时才显式设真 |

`papers` 每个条目的 `paper_uid` 可以省略。省略时由内容哈希自动生成，写成对象时必须是可移植、已经规范化的字符串（建议字母开头）。同一份内容或同一个 UID 重复出现会直接失败，程序不会替你决定要不要去重。路径只是标签而不是科学身份：内容不变时改文件名不会改变规格指纹，但改内容会。

`smoke_size` 决定先跑几篇；不写 `smoke_papers` 时，默认取 `papers` 列表中排在前面的前 `smoke_size` 篇。`smoke_papers` 一旦写出，就必须是冻结输入清单里的条目，否则失败。

`pricing` 的键是模型 ID（例如 `DeepSeek-R1`），不是本地 `SELECTED_KEYS` 里的别名（如 `deepseek-r1`）。它必须覆盖每一个被选中的科研模型、embedding 模型，以及启用时的顾问模型。每个模型要写 `input_per_million`、`output_per_million`、`max_input_tokens`、`max_output_tokens` 和 `basis`。`basis` 是价格依据，必须写清 provider 和带日期的价格来源，不能只写“大概”。

## 3. 命令

入口仍然是 run_pipeline.py：当第一个参数是 `run`、`status`、`pause`、`resume`、`approve`、`report`、`import`、`evaluate` 之一时，它会转交给控制层；否则按原来的 `--domain/--stage` 老命令运行。两条入口各自独立，互不影响。

```text
python run_pipeline.py run      --spec research.json --config config.py --runs-dir runs [--run-id ID] [--dry-run]
python run_pipeline.py status   --run-id ID [--runs-dir runs]
python run_pipeline.py pause    --run-id ID [--runs-dir runs]
python run_pipeline.py resume   --run-id ID [--runs-dir runs] [--config config.py]
python run_pipeline.py approve  --run-id ID --gate-id GATE --reason TEXT [--runs-dir runs] [--outcome not_executed|executed_unknown]
python run_pipeline.py report   --run-id ID [--runs-dir runs]
python run_pipeline.py import   --run-id TARGET --from-run-id SOURCE --task-id TASK --reason TEXT [--runs-dir runs]
python run_pipeline.py evaluate --run-id ID [--runs-dir runs] [--gold FILE] [--output-dir DIR]
```

| 命令 | 必填参数 | 可选参数 | 作用 |
|---|---|---|---|
| `run` | `--spec` | `--config`（默认 config.py）、`--runs-dir`（默认 runs）、`--run-id`、`--dry-run` | 冻结规格、建立运行目录并规划任务 |
| `status` | `--run-id` | `--runs-dir` | 打印运行状态、任务/请求/人工门统计和预算 |
| `pause` | `--run-id` | `--runs-dir` | 请求暂停，在下一个请求前生效 |
| `resume` | `--run-id` | `--runs-dir`、`--config` | 在检查后继续，仍受墙钟上限监督 |
| `approve` | `--run-id`、`--gate-id`、`--reason` | `--runs-dir`、`--outcome` | 处理一个待决人工门 |
| `report` | `--run-id` | `--runs-dir` | 生成 `reports/status_report.json` |
| `import` | `--run-id`（目标）、`--from-run-id`、`--task-id`、`--reason` | `--runs-dir` | 把一个来源运行的任务显式导入目标 |
| `evaluate` | `--run-id` | `--runs-dir`、`--gold`、`--output-dir` | 只读科学评价，写到运行目录之外 |

只有 `run` 和 `resume` 会读取 config.py；`status`、`pause`、`approve`、`report`、`import`、`evaluate` 都不需要密钥就能运行。

### run：先冻结，再执行

`run --dry-run` 只做冻结和规划：复制输入、校验、写下计划报告，然后停在 `INPUT_READY`。这条路径不发任何 SDK 或 HTTP 请求，可以在没有密钥的情况下用来检查规格。

`run` 不带 `--dry-run` 时，先完成同样的冻结和规划，再由 worker 在墙钟上限内监督执行。`--run-id` 可选；不写时自动生成带时间戳的目录名。运行目录已存在会直接失败，不会静默接管旧运行。

### status / pause / report

`status` 打印当前状态快照和预算合计，是查看进度和待决人工门的常规手段。`pause` 是协作式暂停：不会中断已经发出的请求，只保证不再发新请求；终态运行不能被暂停。`report` 重新生成一份 `reports/status_report.json`，其中的 `scientific_validation` 永远是 `NOT_EVALUATED`。

### resume

`resume` 用于从 `PAUSED`、`RECOVERABLE_ERROR` 或 `HUMAN_GATE` 继续。开始前会核对冻结输入、账本和科学代码是否仍与冻结规格一致；只要有待决人工门没处理，就只打印待决门而不继续。改了被冻结的输入或科学代码后，必须新建一次运行，不能在原运行里继续。`DONE` 的运行再次 resume 只会重新校验归档产物；`FATAL_ERROR` 是终态，不能 resume，需要新建运行。

### approve

`approve` 每次只处理一个门，`--reason` 必填。只有 `smoke`、`unknown_request`、`confirmed_rejection`、`diagnosis` 四类门可以批准；其余门（预算、输入越界、请求变化、凭据完整性、阶段越权、PDF 资源）不能用批准绕过，只能解决原因或新建冻结运行。详细的处理规则见第 7 节。

### evaluate

`evaluate` 是只读评价。它读运行清单、状态、生产产物和原始候选表，不修改生产数据，也不打开运行时 API。`--gold` 是独立参照文件，`--output-dir` 是输出目录；不写 `--output-dir` 时默认写到 `runs/evaluations/<run_id>/`。输出目录必须在运行目录之外，也不能是它的上级目录。详见第 10 节。

### 退出码

每条命令都打印 JSON。脚本可以只根据退出码判断：

| 退出码 | 含义 |
|---|---|
| 0 | 成功，或 `--dry-run` 计划正常结束 |
| 2 | 需要关注：`HUMAN_GATE`、`HUMAN_GATE_SMOKE` 或 `PAUSED` |
| 1 | 失败：`RECOVERABLE_ERROR`、`FATAL_ERROR`，或命令本身抛错 |

## 4. 状态机与 smoke→batch

控制层把一次运行约束在固定主状态链上：

```text
INIT → PREFLIGHT → INPUT_READY → SMOKE → HUMAN_GATE_SMOKE
     → BATCH → FINAL_QC → ARCHIVE → DONE
```

运行中还可以进入 `PAUSED`（人工暂停或 Ctrl-C）、`RECOVERABLE_ERROR`（可恢复的运行错误）、`HUMAN_GATE`（需要人工决定）和 `FATAL_ERROR`（终态，需人工介入）。中断类状态会记住要回到哪个主状态，恢复不会丢进度。

真正的分界线是 `SMOKE` 和 `BATCH`：`SMOKE` 只执行 `smoke_size` 指定的少量论文，跑完六阶段后写出 smoke 报告并停在 `HUMAN_GATE_SMOKE`；只有人工批准 smoke 报告后，才允许进入 `BATCH` 执行剩余论文。单篇通过不会自动授权批量。

smoke 和 batch 用的是同一套逻辑任务身份，所以 smoke 里已经完成的论文在 batch 中直接复用，不会因为阶段标签不同而重复调用模型。多轮运行按轮次分开保存，投票也不跨轮合并。

## 5. 预算与请求边界

真实请求之前，规格必须给出四项冻结上限：`max_calls`、`max_tokens`、`max_cost`、`max_runtime`，加上每个模型的价格依据和输入/输出上界。缺了这些，控制层不会发付费请求。

每个真实 HTTP 请求在发出前先做一次保守预留（按冻结的 `max_input_tokens` + `max_output_tokens` 和对应费率计算），返回后再按 provider 报的 usage 结算。预留一直保留到请求有确定结果为止。**未知 usage 不会被记成 0**：无法测量的花费继续以保守值计入 held，恢复运行也不会重置任何计数。

`max_runtime` 是从运行创建时刻开始算的墙钟时间，暂停和等待都不会把它清零。CLI 的 worker 监督会在到点时终止本地子进程；它不承诺取消已经发出的远端请求，这类请求的结果由第 6 节的未知请求规则处理。

需要说清楚一个边界：**只有走这套控制层的运行才计入预算账本**。直接使用旧命令 `python run_pipeline.py --domain ... --stage ...` 仍然按原管线运行，不经过预算控制。所以“所有模型请求都被预算管住”这句话对控制层成立，对旧入口不成立。

## 6. 请求失败、恢复与重试

控制层关闭了 SDK 的内部重试（`max_retries=0`），由自己逐请求记账，避免把一次逻辑调用悄悄变成多次收费请求。失败分三类：

- **确定未执行**：请求在发出前就被明确拒绝（例如明确的 400 参数错误、401 无效密钥、403 无权限、404 模型不存在）。这类会记为 `rejected`，不收费，并写入 `confirmed_rejection` 门，由人工在有界次数内决定是否重试。
- **临时限流**：只有明确命名、可识别的 429（`rate_limit_exceeded` / `rate_limit`）才自动重试，同一请求最多 3 次，计数跨重启累计。超出上限后停下等人工。
- **结果未知**：超时、连接失败、未命名的 5xx，以及派发后进程中断，一律按“可能已执行并收费”处理，保留预留并进入 `unknown_request` 门，绝不盲目重发。

原始响应会先落盘再解析：完整响应写进账本凭据并保存到 `checkpoints/responses/<attempt_id>.json`，然后才做 JSON/结构解析。解析失败、`finish_reason` 不是正常结束时，原始凭据仍在，但不会自动重发。

向量（embedding）请求体如果**完全一致**，会在**同一次运行内**按精确签名复用，避免对同一段文本重复向量化；对话（chat）请求不做这种跨任务复用，跨运行也没有自动共享缓存——跨运行复用只能通过显式的 `import`（第 9 节）。

## 7. 人工门

控制层遇到需要人决定的情况会停下并建门；用 `status` 能看到待决门的 ID，用 `approve` 处理。可以批准的四类门：

- **smoke**：批准前会核对 `reports/smoke_report.json` 的哈希有没有变；报告被改过或状态不在 `HUMAN_GATE_SMOKE` 时会拒绝。批准后记录 `scope=batch`，即“允许进入批量”。
- **unknown_request**：必须用 `--outcome not_executed` 或 `--outcome executed_unknown` 明确交代 provider 结果。`not_executed` 释放预留、允许在原次数上限内重试；`executed_unknown` 继续保留预留且该请求不可重放。
- **confirmed_rejection**：针对有确定“未执行”凭据的失败请求，批准后允许在原上限内重试同一个请求。
- **diagnosis**：只记录“运行原因已解决”，批准后按原冻结规格继续。

其余门——预算（`budget`）、PDF 资源（`pdf_resources`）、请求变化（`request_changed`）、凭据完整性（`receipt_integrity`）、阶段越权（`phase_violation`）、输入越界（`input_bound`）——不能用 `approve` 绕过。它们要求解决原因，或新建一次冻结运行。

特别是 `diagnosis` 门：批准它只代表运维层面的原因已处理，**不会**顺带绕过其他待决门，也不改变任何冻结字段；如果还有别的门没处理，`resume` 仍会停下。

## 8. 可选诊断顾问

诊断顾问默认关闭。没有在规格里写明 `advisor: {model, source}` 时，控制层不会调用任何顾问模型，遇到异常直接把脱敏后的软件错误交给人工。

启用后，顾问只收到一份脱敏摘要（错误类型、目标、阶段、状态、有限的质检码和预算合计），**看不到论文正文或研究数据**。它只能从固定白名单里给出一个动作：`retry_task`、`reparse`、`rebuild_cache`、`pause`、`request_human`。顾问没有执行命令或 shell 的能力，不能批准人工门，不能改数值、单位、模型、提示词、阈值或预算；非法动作、改目标、改规格都会被拒绝并留记录。

自动恢复是**有界**的：每个任务最多自动恢复一次，且必须在不改变科学真值、不扩大预算、没有未决请求的前提下进行。`retry_task` 要求已有确定未执行的凭据且尝试次数未到上限；`rebuild_cache` 只适用于 embedding 缓存；其余恢复动作需要有可复用的已付费响应。任何不满足条件的建议都会被拒。

## 9. 跨运行导入

不同运行默认互相独立，没有自动共享的结果目录。要把一个已完成运行的任务搬到另一个运行，必须显式 `import`，而且目标运行必须还停在 `INPUT_READY`（尚未开始执行）。

导入要求来源和目标拥有完全相同的科学签名：输入、模型与端点、提示词、科学代码、轮次、阈值，以及请求的上界与价格。任何一项不同都会拒绝。导入可复制的阶段包括 prepare、examiner、embeddings、evidence_embedding 和 cross。

来源运行会保留它自己的原始凭据；目标运行把复制的请求记为 `imported`，**不重复计入目标预算**（不消耗新的调用次数和费用），并记录来源运行、任务和规格哈希作为出处。带未决请求的来源不能被导入。

## 10. 独立科学评价

`evaluate` 是只读的，绝不写入生产运行目录，也不猜它无法证明的匹配。它复用科研核心自己的数值和单位规范化，保证评价与生产口径一致：

- 数值按精确值比较，**不做单位换算、不给隐式容差**；
- 缺少显式单位，或缺少实验身份、或由多个实验合并得出的数值，判为 `UNMATCHED` 而不是判错；
- 候选集（小数多数模式）按集合评价，不按单一赢家；
- 正确答案缺失、漏提取、解析失败和被当前政策过滤的数值分开统计；
- 参照文件自己声明协议（`single_human`、`independent_double_read`、`adjudicated`、`synthetic`），这是“声明”而不是由程序独立验证；
- 没有提供 `--gold` 时，结果标为 `NOT_EVALUATED` 且指标为 `null`，绝不写成“准确率 0”。

评价结果写在运行目录之外（默认 `runs/evaluations/<run_id>/`），并记录评价前后生产文件的哈希。模型之间的一致不等于科学正确；合成参照只证明评价软件能跑，不证明科学准确率。

## 11. 产物与报告

控制层为每次运行建独立目录 `runs/<run_id>/`：

```text
runs/<run_id>/
  manifest.json          规格、指纹、创建时间（不含密钥）
  ledger.sqlite          权威账本：状态、任务、请求尝试、预算、人工门、事件
  state.json             账本的只读快照（导出视图）
  events.jsonl           事件流（导出视图）
  errors.jsonl           错误事件（导出视图）
  metrics.json           提取/核验/阈值/预算汇总（导出视图）
  inputs/                冻结输入的独立副本
  outputs/prepared/      准备后的 Markdown
  outputs/R01/...        第 1 轮的 examiner/composite/embeddings/cross/ensemble 产物
  checkpoints/           响应、阶段结果、advisor、final_qc 等检查点
  reports/               计划、smoke、状态、最终、截止与致命报告
```

`state.json`、`events.jsonl`、`errors.jsonl`、`metrics.json` 只是便于查看的导出，权威始终是 `ledger.sqlite` 加磁盘上的实际产物。任何“已完成”的判断都会重新校验产物哈希，而不是只信账本里的一行记录。

## 12. 常见错误与安全恢复

大多数失败都可以先 `status` 看状态和待决门，再按下面的方式处理：

| 现象 | 含义 | 安全动作 |
|---|---|---|
| 找不到 config.py | 没有复制模板 | `Copy-Item config.example.py config.py` 并填写 |
| 规格校验失败 | 字段名写错，或预算/价格为 null 占位 | 按第 2 节补齐四项预算和每个模型的价格 |
| 运行目录已存在 | 重复使用 `--run-id` | 换一个 `--run-id`，或按需删除旧目录 |
| 退出码 2，状态 `HUMAN_GATE_SMOKE` | 等待 smoke 批准 | 检查 smoke 报告后 `approve --gate-id ... --reason ...` |
| 退出码 2，状态 `HUMAN_GATE` | 预算、未知请求、确定拒绝或诊断门 | `status` 查看门，按第 7 节批准或解决原因 |
| 退出码 2，状态 `PAUSED` | `run`/`resume` 发现有待处理的暂停请求（含 Ctrl-C） | 检查后 `resume` |
| 退出码 1，`RECOVERABLE_ERROR` | 可恢复的运行错误 | 检查错误后 `resume` |
| 退出码 1，`FATAL_ERROR` | 终态错误（如账本损坏） | 不要在原运行继续；新建一次运行 |
| 退出码 1，`status: failed` | 命令本身抛错（如跑出 `runs-dir`） | 按 `error` 字段修正参数或路径 |

有两条安全原则始终成立：账本损坏会明确失败，不会被当成空账本重来；请求结果未知时保留预留并交人工判断，不做盲目重发。

## 13. 文档分工

- README.md：项目简介、六阶段管线、领域问题、旧命令快速开始。
- LUMINA_AGENTIC_WORKFLOW.md：科学流程与控制层的实现现状。
- AGENT_GUIDE.md（本文件）：控制层的操作手册。
- research.example.json：研究规格模板（占位符未填时不可运行）。
- LUMINA_AGENTIC_PLAN_AND_GOALS.md：唯一的 Goal 与验收台账，正式状态以它为准。
