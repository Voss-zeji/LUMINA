# LUMINA 核心流程与 Agent 化工作流

- 状态：六阶段科学管线与 V0 正确性护栏保持；Agent 运行控制层已实现。最新软件验收、commit、PR 和 CI 证据以同仓库的 Plan/Goal 为准，真实单篇与批量运行分别需要输入、预算与批准。
- 受众：维护 LUMINA 的科研人员、工程维护者和运行操作者。
- 修复基线：`71111c201d3d652d4237c3fc9f2d96f1208f6533`；本文件描述的实现现状对应 `feat/agent-control-v1` 分支上的本地代码。
- 文档分工：README 讲管线与旧命令，本文件讲科学流程与控制层实现现状，AGENT_GUIDE.md 讲操作，research.example.json 是规格模板，LUMINA_AGENTIC_PLAN_AND_GOALS.md 是唯一的 Goal 与验收台账。

## 一句话结论

LUMINA 的核心价值不是让一个 Agent 自由阅读并决定科学结论，而是把多模型独立提取、原文证据核验和共识集成固化为可复现的数据生产管线。Agent 化增加的是受策略约束的观察、质检、恢复、停止和审计能力，不替代六阶段科学算法。

换言之：**LUMINA Core 负责“如何计算”，Agent 控制层负责“何时允许计算、如何证明这次计算可信、何时停止”。**

## 1. 核心科学思路

LUMINA 将一篇论文的结构化信息提取拆为三个相互制衡的科学环节：

1. **Independent inquiry**：多个模型对同一篇论文和同一问题独立给出结构化候选答案、证据和置信度。
2. **Cross-examination**：针对每一条候选证据，从原文检索最相关的上下文，并由产生该答案以外的模型判断证据是否真实存在。
3. **Consensus confirmation**：以独立验证票数过滤候选答案，再对同一论文的候选值进行标准化和投票，形成可追溯的共识结果。

这三层分别压制了单模型幻觉、无原文支持的引用，以及单次模型输出对最终数据库的过度影响。控制层不改变这三层算法，只负责让每一次调用可复现、可计量、可停止。

## 2. 三层科学逻辑到六阶段软件管线

| 科学层 | 阶段 | 输入 | 主要动作 | 产物 |
|---|---|---|---|---|
| 输入准备 | `prepare` | PDF 或 Markdown | PDF 转 Markdown、截断参考文献后的内容、token 审计 | Markdown 与审计输出 |
| Independent inquiry | `examiner` | Markdown、领域问题、模型池 | 每篇×每模型×每问题的 JSON 提取 | 模型级 TSV |
| Independent inquiry | `composite` | 模型级 TSV | 合并为问题级候选表 | question-level Excel |
| Cross-examination | `embeddings` | 截断后的 Markdown | 分块、向量化、缓存 | `.npy` embeddings |
| Cross-examination | `cross` | 候选 evidence、embeddings | 检索上下文、独立模型核验、计票 | verifier TSV 与 `cross_score` |
| Consensus confirmation | `ensemble` | 带 `cross_score` 的候选表 | 阈值过滤、标准化、投票 | 共识 Excel 和明细 |

固定执行顺序为：

```text
prepare → examiner → composite → embeddings → cross → ensemble
```

六阶段是确定性编排的科学核心；其中模型调用本身有随机性，但输入、模型池、提示词、配置和输出版本必须被记录，才能复现运行。

## 3. 完整科研交付流程，而不只是 `run_pipeline.py`

一次可交付的 LUMINA 运行应包含下列十个阶段：

1. **研究规格冻结**：确定领域、纳入文献、问题、字段 schema、模型池和接受准则。
2. **仓库与配置审计**：记录代码 SHA、依赖、数据路径、模型端点能力和运行配置。
3. **输入 QC**：核对 PDF/Markdown 对应关系、正文截断、文献数、token 异常和 schema 覆盖。
4. **单篇 smoke test**：用少量论文走完六阶段，先检查字段、证据、检索和阈值语义。
5. **批量 independent inquiry**：运行 `examiner` 和 `composite`，检查 paper×model×question 覆盖率。
6. **批量 cross-examination**：生成或复核 embeddings，排除自验证，计算独立 `cross_score`。
7. **共识集成**：仅运行可由独立 verifier 数量达到的阈值，并保留 `full` 作为敏感性参照。
8. **结果 QC 与故障恢复**：区分可重跑的技术错误、需重建的缓存和需人工决策的科学异常。
9. **科学评价**：与人工专家或冻结的金标准比较 accuracy、completeness、阈值敏感性、模型差异和可重复性。
10. **归档与发布**：保存结果、配置、日志、数据与代码指纹；文稿、操作手册和版本说明同步更新。

第 9 步是独立的 evaluation pipeline。生产提取结果即使已经带有 `Final`、`cross_score` 或高模型一致性，也不能自动等价为人工验证完成。

## 4. 本次已实现的 V0 正确性护栏

| 风险 | V0 行为 | 作用范围 |
|---|---|---|
| 端点不支持 `response_format` | 每个 provider 显式声明 `supports_json_mode`；关闭时不发送该参数 | Examiner 与 cross 请求 |
| API/解析失败被伪装成完成 | 保存原始错误至 `_invalid.txt`，不生成可跳过的成功 TSV；旧的无效 TSV 也会重跑 | Examiner |
| 交叉核验包含来源模型本身 | 执行时排除输入模型；聚合历史结果时也过滤 `input_model == output_model` | Cross |
| 不可达到的阈值仍被运行 | 启动 `cross`、`ensemble` 或 `all` 前，要求每个阈值位于 `1..(M-1)` | Pipeline preflight |
| PDF 转换后下游找不到 Markdown | 转换产物写入配置的 `markdown_dir`，而非 PDF 源目录 | Prepare |
| `.npy` 缓存损坏或和当前分块不一致 | 文件不可读、不是二维向量或行数不同于 chunk 数时重建 | Embeddings |
| 错误的 cross TSV 永远被跳过或被当作反票 | 只跳过并聚合 `existing_flag ∈ {0,1}` 的核验；失败 TSV 保持可重试并保留原始错误 | Cross |

默认配置使用两模型，`min_cross_scores` 为 `[1]`，`min_consensus_models` 为 `2`。选择 `M` 个不同模型时，核验门槛在 `1..M-1`，共识门槛在 `1..M`。共识设置省略时按完整所选模型池的严格多数冻结；这是可配置的默认值，稿件没有规定固定的 T_bsl 数值。`MiniCrossNN` 只接受同时达到核验门槛和基础模型共识门槛的唯一最高频答案，同一模型重复行只计一票，指标、单位及明确实验标识分别判断；并列最高票暂不接受。`00_full` 保留为诊断基线。更改门槛或科学代码后新建 run，旧产物保留原有执行条件。

### 工程与数据修复的执行边界

本轮增加正文/请求/候选/核验规格指纹、选定轮次隔离、字符串论文 ID、幂等分数回写、原子文件发布、明确失败退出与空结果覆盖。输入转换支持补齐缺口；数值范围/指数、独立单位字段、负坐标、ISO 日期和缺失值的处理已修正。单独启动 cross/ensemble 时，过期的论文、prompt 或 round 会被拒绝并要求重建上游。

这些改动作用在核心阶段上，无论从哪条入口调用都生效。旧入口 `python run_pipeline.py --domain ... --stage ...` 仍用 `ensemble_dir/last_run.json` 记录最近一次阶段的成功或失败，并用相同输出位置的文件锁避免并发写；它只表示最近一次阶段结果，不是完整状态机，也不做预算控制。控制层的运行改用后面第 5–9 节描述的账本、租约和预算，不会自行改变研究规格。

科学边界保持：cross 检查证据存在与相关性，未扩展为数值/单位/实验对应关系的完整事实核验；单位不自动换算；小数多数模式仍保留跨 item 候选集。其单赢家/多实验输出契约仍需研究者确认，不能因修复工程问题而隐式改变。

旧产物缺少指纹时不作为当前已验证结果；先归档，再在新输出目录重跑 examiner/composite。工程回归测试及模拟全链路通过，不代表真实 PDF/API 验收或独立人工科学评价完成。

## 5. Agent 控制层的实现结构

控制层集中在 `lumina/agent/`，独立评价在 `lumina/evaluation.py`。它不搬迁科学核心，而是通过可选 `runtime=None` 的接入点调用现有六阶段：不传上下文时，核心保持旧行为；传入运行上下文时，核心的每一次真实请求都走预算与账本。

| 模块 | 责任 | 能否改变科学结果 |
|---|---|---|
| `contracts.py` | 冻结领域、论文、模型/端点、问题、prompt、轮次、阈值与预算，生成规格指纹；复制输入并锁定哈希 | 否 |
| `store.py` | SQLite 账本：运行状态、任务、请求尝试、预算预留、人工门、事件 | 否 |
| `budget.py` | 逐请求预留/结算，四项上限与价格依据校验，未知用量保留 | 否 |
| `runtime.py` | 逐请求执行（关闭 SDK 重试）、响应落盘、精确签名复用、阶段与输入边界检查 | 否 |
| `controller.py` | 六阶段编排、smoke→人工批准→batch、最终质检、有界自动恢复 | 否 |
| `worker.py` | CLI 墙钟监督：到点终止本地子进程并生成截止报告 | 否 |
| `advisor.py` | 可选诊断顾问：脱敏摘要 → 白名单动作 → 规则复核 | 否（只提建议） |
| `imports.py` | 显式跨运行导入，校验完整签名并记录出处 | 否 |
| `report.py` | 从规格、账本和磁盘产物汇总运行报告 | 否 |
| `evaluation.py` | 只读独立评价 | 不写生产值 |

不能把六阶段简单改名为六个自由 Agent：这会让模型决定其自身的输入、重试和阈值，增加成本与随机性，且破坏论文级可重复性。控制层因此是确定性程序，LLM 最多只做受限诊断。

## 6. 实际状态机（已实现）

控制层实现的主状态链是：

```text
INIT → PREFLIGHT → INPUT_READY → SMOKE → HUMAN_GATE_SMOKE
     → BATCH → FINAL_QC → ARCHIVE → DONE

运行中可进入：PAUSED、RECOVERABLE_ERROR、HUMAN_GATE、FATAL_ERROR
```

每一次状态转换都连同阶段、任务范围和理由写入账本，并导出到 `state.json` 和 `events.jsonl`。中断类状态会保存要返回的主状态，恢复时先复核冻结输入、账本和科学代码再继续；`DONE` 是完成，`FATAL_ERROR` 是终态，需要人工介入而不是继续恢复。

与早期“建议状态机”的差别要说明：早期草案把 smoke 拆成 `SMOKE_PREPARE → … → SMOKE_ENSEMBLE` 六个子状态，还包含一个独立的 `EVALUATION` 状态。实际实现把 smoke 的六阶段收进一个 `SMOKE` 状态，由内核顺序执行；评价不放进主状态链，而是用独立的只读 `evaluate` 命令在运行之外进行。这样既保留了“先单篇、再批量”的门，又不在状态机里伪造科学验收状态。

## 7. 人工门与停止规则（已实现）

控制层在需要人决定时停下并建门，用 `status` 查看、`approve` 处理。可以批准的四类门及其通过后的唯一动作：

| 门 | 触发 | 通过前检查 | 通过后 |
|---|---|---|---|
| `smoke` | smoke 六阶段跑完 | `smoke_report.json` 哈希与门一致，状态为 `HUMAN_GATE_SMOKE` | 记录 `scope=batch`，允许批量 |
| `unknown_request` | 请求结果未知 | 需显式 `--outcome` | `not_executed` 释放预留并允许有界重试；`executed_unknown` 继续保留且不可重放 |
| `confirmed_rejection` | 确定未执行的失败 | 有确定未执行凭据 | 在原次数上限内重试同一请求 |
| `diagnosis` | 顾问或规则判断原因可解决 | 状态为 `HUMAN_GATE`/`PAUSED` | 记录原因已解决，按原冻结规格继续 |

其余门不能用批准绕过，只能解决原因或新建冻结运行：预算（`budget`）、PDF 资源（`pdf_resources`）、请求变化（`request_changed`）、凭据完整性（`receipt_integrity`）、阶段越权（`phase_violation`）、输入越界（`input_bound`）。

`diagnosis` 门只记录运维层面的原因已解决，不改变任何冻结字段，也不顺带绕过其他待决门。自动恢复只适用于技术性、可逆、且不改变研究规格的情况，并且每个任务最多自动恢复一次。

## 8. 预算、成本与独立性政策（已实现）

真实请求前必须给出 `max_calls`、`max_tokens`、`max_cost`、`max_runtime` 四项上限，加上每个模型的价格依据与输入/输出上界。每个真实 HTTP 尝试先按冻结的保守上限预留，返回后按 provider 报的用量结算；`max_runtime` 从运行创建时刻起算，暂停和等待不重置。

用量未知时预留继续保留，**绝不记成 0**。CLI 的 worker 监督在墙钟到点时终止本地子进程，不承诺取消已经发出的远端请求，这类请求按未知请求处理。自动重试只限明确命名、可识别的 429（最多 3 次、跨重启累计）；确定的预执行拒绝进入人工门。

若 `M` 个模型，单条 evidence 最多获得 `M - 1` 张独立核验票；运行前应据此估计调用与费用。超出用户设定预算时，控制层只写出估算与门，等待人工决定，不自行换模型、减少问题或继续消耗额度。

独立性政策保持不变：cross 从不让来源模型核验自己的证据；跨运行复用必须显式导入，且要求来源与目标拥有完全相同的科学签名；控制层没有自动共享结果目录。

## 9. 运行目录与审计契约（已实现）

控制层为每次运行生成独立目录：

```text
runs/<run_id>/
  manifest.json       # 规格、指纹、创建时间（不含密钥）
  ledger.sqlite       # 权威账本
  state.json          # 账本快照（导出视图）
  events.jsonl        # 事件流（导出视图）
  errors.jsonl        # 错误事件（导出视图）
  metrics.json        # 提取/核验/阈值/预算汇总（导出视图）
  inputs/             # 冻结输入的独立副本
  outputs/prepared/   # 准备后的 Markdown
  outputs/R01/...     # 每一轮的六阶段产物
  checkpoints/        # 响应、阶段结果、advisor、final_qc
  reports/            # 计划、smoke、状态、最终、截止、致命报告
```

账本单独记录每一次请求尝试的预留、结算、原始响应、错误与计数；`state.json`、`events.jsonl`、`errors.jsonl`、`metrics.json` 只是便于查看的导出，权威始终是 `ledger.sqlite` 加磁盘上的实际产物。每份报告都会重算覆盖、核验完成、阈值分布、分歧与警告，并明确区分“完成任务”“证据支持”“核验完成”“阈值通过”和“费用已知/保留/未知”，不把它们合成一个数字。报告中的 `scientific_validation` 永远是 `NOT_EVALUATED`。

## 10. 能力分层：已实现与尚待完成

| 能力层 | 内容 | 状态 |
|---|---|---|
| V0 正确性护栏 | 能力声明、独立验证、阈值检查、可重试失败、路径/缓存修复 | 已实现，核心保持 |
| 规格冻结与运行档案 | ResearchSpecification、论文身份、manifest、独立输入副本 | 已实现 |
| 进度与请求账本 | 状态机、SQLite 账本、任务/请求/错误记录、检查点、status/pause/resume | 已实现 |
| 预算、质量门与恢复 | 四项预算、逐请求边界、人工门、有界恢复、运行报告、跨运行导入 | 已实现，正式验收见 Goal 台账 |
| 可选诊断顾问 | 独立配置、脱敏输入、白名单动作、越权拒绝；默认关闭 | 本地已实现，正式验收见 Goal 台账 |
| 独立科学评价 | 只读 evaluator、精确比较、不可比标记、无参照 `NOT_EVALUATED` | 本地已实现；合成验证只证明软件，不等于科学验收 |
| 真实单篇与批量 | 授权后的真实论文、端点、预算与人工批准 | 未启动 |

说明两件事。第一，本地代码实现不等于对应 Goal 已验收：正式状态统一记录在 LUMINA_AGENTIC_PLAN_AND_GOALS.md，本文件不另立验收清单。第二，本地回归通过、模拟全链路通过，都不等于真实 PDF/API 验收或独立人工科学评价完成。

## 11. 操作基线

1. 复制 `config.example.py` 为不入库的 `config.py`，填入本地密钥、路径和 provider capability；Python 3.12，依赖用现有 requirements.txt。
2. 用 `run --spec research.json --dry-run` 先冻结规格、只做规划，确认无误后再真正运行；研究规格模板见 research.example.json。
3. 确认模型数与 `min_cross_scores` 一致；两模型时只能使用 `1`。
4. 先跑 smoke，检查 smoke 报告和所有门；批准后才授权批量。
5. 每次批量结束后，检查失败 sidecar、覆盖率、cross-score 分布、`full` 对照结果，以及预算的已知/保留/未知三部分。
6. 在独立 evaluation 完成前，结果应标为机器辅助生产结果，而不是人工验证完成的科学数据库。

## 12. 不在本次范围内

- 不新增一个可以自行改 prompt、schema、模型池或阈值的自主研究 Agent。
- 不把高交叉分数或模型一致性表述为人工复核或金标准验证。
- 不自动替换模型、修改研究规格或在未知成本下继续批量调用。
- 不把控制层的报告或合成评价误称为已完成的科学验收。
