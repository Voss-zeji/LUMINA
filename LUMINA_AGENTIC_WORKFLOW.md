# LUMINA 核心流程与 Agent 化工作流

- 状态：V0 护栏及工程/数据错误修复已实现；V1–V4 的完整 Agent 控制层仍为后续设计。
- 受众：维护 LUMINA 的科研人员、工程维护者和运行操作者。
- 修复基线：`71111c201d3d652d4237c3fc9f2d96f1208f6533`；本轮保留六阶段和科学投票策略。

## 一句话结论

LUMINA 的核心价值不是让一个 Agent 自由阅读并决定科学结论，而是把多模型独立提取、原文证据核验和共识集成固化为可复现的数据生产管线。Agent 化应增加受策略约束的观察、质检、恢复、停止和审计能力，不能替代六阶段科学算法。

换言之：**LUMINA Core 负责“如何计算”，Agent 控制层负责“何时允许计算、如何证明这次计算可信、何时停止”。**

## 1. 核心科学思路

LUMINA 将一篇论文的结构化信息提取拆为三个相互制衡的科学环节：

1. **Independent inquiry**：多个模型对同一篇论文和同一问题独立给出结构化候选答案、证据和置信度。
2. **Cross-examination**：针对每一条候选证据，从原文检索最相关的上下文，并由产生该答案以外的模型判断证据是否真实存在。
3. **Consensus confirmation**：以独立验证票数过滤候选答案，再对同一论文的候选值进行标准化和投票，形成可追溯的共识结果。

这三层分别压制了单模型幻觉、无原文支持的引用，以及单次模型输出对最终数据库的过度影响。

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
4. **单篇 smoke test**：用一篇论文走完六阶段，先检查字段、证据、检索和阈值语义。
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

默认配置使用两模型，所以 `min_cross_scores` 默认值为 `[1]`。如果配置了 `M` 个不同模型，最大有效独立核验分数是 `M - 1`，而不是 `M`。

### 工程与数据修复的执行边界

本轮增加正文/请求/候选/核验规格指纹、选定轮次隔离、字符串论文 ID、幂等分数回写、原子文件发布、明确失败退出与空结果覆盖。输入转换支持补齐缺口；数值范围/指数、独立单位字段、负坐标、ISO 日期和缺失值的处理已修正。单独启动 cross/ensemble 时，过期的论文、prompt 或 round 会被拒绝并要求重建上游。

`last_run.json` 只记录最近一次 stage 的成功或失败，CLI 文件锁只避免相同输出位置的同时写入。它们不是 V2 的完整状态机、预算控制器或运行归档。没有新增会自行改变研究规格的 Agent。

科学边界保持：cross 检查证据存在与相关性，未扩展为数值/单位/实验对应关系的完整事实核验；单位不自动换算；小数多数模式仍保留跨 item 候选集。其单赢家/多实验输出契约仍需研究者确认，不能因修复工程问题而隐式改变。

旧产物缺少指纹时不作为当前已验证结果；先归档，再在新输出目录重跑 examiner/composite。工程回归测试及模拟全链路通过，不代表真实 PDF/API 验收或独立人工科学评价完成。

## 5. Agent 化后的目标架构

```text
Research specification
          ↓
LUMINA Supervisor ────── policy / budget / state
          ↓
Preflight and quality gates
          ↓
Deterministic LUMINA Core
  prepare → examiner → composite → embeddings → cross → ensemble
          ↓
PASS ── RECOVER ── HUMAN_GATE ── FATAL
          ↓
Evaluation and audit reporter
          ↓
Versioned archive
```

| 组件 | 类型 | 责任 | 能否改变科学结果 |
|---|---|---|---|
| Supervisor | 受策略约束的 Agent | 状态转移、成本与人工门控 | 否 |
| Preflight/QC | 优先确定性代码 | 路径、配置、模型能力、覆盖率、阈值校验 | 否 |
| Pipeline runner | 确定性代码 | 调用现有六阶段 | 否 |
| Recovery controller | 规则优先，必要时 Agent 辅助分类 | 有界重试、缓存重建、从 checkpoint 恢复 | 否 |
| Evaluation agent | 代码加受审查的解释 | 运行 benchmark、敏感性和异常报告 | 不直接写生产值 |
| Audit reporter | 确定性汇总，可选文字摘要 | manifest、指标、错误和决策记录 | 否 |

不能将六阶段简单改名为六个自由 Agent：这会让模型决定其自身的输入、重试和阈值，增加成本与随机性，且破坏论文级可重复性。

## 6. 建议状态机

```text
INIT
  → PREFLIGHT
  → INPUT_READY
  → SMOKE_PREPARE → SMOKE_EXAMINER → SMOKE_COMPOSITE
  → SMOKE_EMBEDDINGS → SMOKE_CROSS → SMOKE_ENSEMBLE
  → HUMAN_GATE_SMOKE
  → BATCH_RUN
  → FINAL_QC
  → EVALUATION
  → ARCHIVE
  → DONE

Any state → RECOVERABLE_ERROR → retry/resume → prior state
Any state → HUMAN_GATE
Any state → FATAL_ERROR
```

每一次状态转移都应记录输入指纹、输出摘要、执行时间、错误类别和作出的策略决定。状态不应只存在内存或终端滚动输出中。

## 7. 质量门与停止规则

| Gate | 检查 | 通过后 | 失败后的唯一允许动作 |
|---|---|---|---|
| G0 Preflight | 配置、路径、模型、阈值、预算 | 允许 smoke test | 停止或人工修改配置 |
| G1 Prepare | PDF/Markdown 覆盖、正文和 token | Examiner | 修复输入或人工审查 |
| G2 Examiner | 每个 paper×model×question 的成功覆盖 | Composite | 只重跑失败组合 |
| G3 Composite | 字段、模型和问题覆盖 | Embeddings/Cross | 返回 Examiner |
| G4 Embeddings | 每篇 chunk 数与 vector 数、向量维度 | Cross | 重建缓存 |
| G5 Cross | 证据存在、独立 verifier、合法 flag | Ensemble | 只重跑失败核验 |
| G6 Ensemble | 阈值可达、结果覆盖、空结果率 | Final QC | 修改配置后重跑，不静默降级 |
| G7 Final | 覆盖、错误率、票数分布、成本 | Evaluation/Archive | HUMAN_GATE |

自动恢复只适用于技术性、可逆且不会改动研究规格的情况，例如缺失组合的重跑、未完成缓存的重建和 checkpoint 恢复。下列情形必须进入 `HUMAN_GATE`：问题/schema 改变、模型池改变、embedding 模型改变、文献内容改变、阈值冲突、大规模 API 故障或异常科学结论。

## 8. 能力、成本和独立性政策

### Provider capability

provider 配置必须声明或继承清晰默认值，例如：

```python
"provider_1": {
    "key": "...",
    "url": "...",
    "supports_json_mode": True,
}
```

V0 只根据配置是否发送 `response_format`，并保留 JSON5 解析。V1 以后才可考虑“仅针对明确的 capability error 自动降级一次”的策略；必须记录该决策，且不得无限重试。

### 独立 verifier 与阈值

若有 `M` 个模型，单条 evidence 最多可获得 `M - 1` 张独立核验票：

```text
examiner calls = N_papers × M × N_questions
max independent cross_score = M - 1
cross calls ≈ N_evidence × (M - 1)
```

运行前应估计 token、调用数和费用。超过用户设定预算时，Supervisor 只能写出预估报告并等待 `HUMAN_GATE`，不得自行换模型、减少问题或继续消耗额度。

## 9. 运行审计契约（V2 目标）

建议每次运行生成独立目录：

```text
runs/<run_id>/
  manifest.json       # code SHA, config/data hashes, model and capability policy
  state.json          # current and completed states
  events.jsonl        # append-only state transitions
  errors.jsonl        # structured raw failures and retry decisions
  metrics.json        # coverage, tokens, calls, runtime, cost estimates
  checkpoints/        # resumable stage markers
  final_report.md     # human-readable run report
```

一个错误记录至少应包括 `run_id`、stage、paper、question、input/output model、provider、错误类型、原始错误文本、attempt、是否可重试、采取的动作和时间戳。当前 V0 的 `_invalid.txt` 是保留原始证据的过渡机制，不等价于完整 Error Ledger。

## 10. 分期实施路线

| 版本 | 目标 | 是否已实现 |
|---|---|---|
| V0 | 正确性护栏：能力声明、独立验证、阈值检查、可重试失败、路径/缓存修复 | 是 |
| V1 | Guarded pipeline：逐阶段 QC、覆盖检查和明确 PASS/FAIL | 否 |
| V2 | Agent orchestrator：状态机、Error Ledger、预算、checkpoint、规则化恢复和报告 | 否 |
| V3 | Evaluation agent：金标准比较、阈值敏感性、模型比较、重复运行与效率报告 | 否 |
| V4 | Research agent：新领域 schema/问题草案和小规模验证 | 否，且不得直连生产数据 |

## 11. 操作基线

1. 复制 `config.example.py` 为不入库的 `config.py`，填入本地密钥、路径和 provider capability。
2. 确认模型数与 `min_cross_scores` 一致；两模型时只能使用 `1`。
3. 先运行单篇 smoke test，再检查所有 Gate，最后才授权批量运行。
4. 每次批量运行结束后，检查失败 sidecar、覆盖率、cross-score 分布和 `full` 对照结果。
5. 在独立 evaluation 完成前，结果应标为机器辅助生产结果，而不是人工验证完成的科学数据库。

## 12. 不在本次范围内

- 不新增一个可以自行改 prompt、schema、模型池或阈值的自主研究 Agent。
- 不把高交叉分数表述为人工复核或金标准验证。
- 不自动替换模型、修改研究规格或在未知成本下继续批量调用。
- 不把 V0 的失败 sidecar 误称为完整的 runs/审计系统。
