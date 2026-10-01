# LUMINA 工程与数据错误修复记录

日期：2026-10-01。基线：`71111c201d3d652d4237c3fc9f2d96f1208f6533`。
本地分支：`fix/engineering-data-safeguards`。

## 保留的主要执行逻辑

`prepare → examiner → composite → embeddings → cross → ensemble`。

保留领域问题及其科学含义、来源模型不自验证、配置的可达分数阈值、元数据投票/并列/坐标精度规则、数值的票数/置信度选择规则，以及小数多数模式的跨 item 多候选策略。cross 仍检查证据存在与相关性，未扩展为数值/单位/实验对应关系的完整事实核验。单位未自动换算。

## 审查问题处理映射

| Finding | 本轮处理 | 主要验证 |
|---|---|---|
| F01 新安装无法导入 | 改为直接依赖/导入 langchain-text-splitters | 新环境仅安装 requirements；无 langchain 包仍正常导入 |
| F02 范围/指数/单位指数误解析 | 一次解析数值表达式，单位从剩余文本分离 | signed range、10-20、1e-3、m-2、± 等回归 |
| F03 独立 unit 丢失 | unit 参与候选分箱并保留；显式单位输出纯数值+独立 unit，旧内联单位字符串兼容 | mg/g 分离，SI 大小写、Unicode 指数、实际 ensemble 路由 |
| F04 坐标符号/半球错误 | 区分端点符号与范围符；保留负到正区间和共享 S/W 标记 | 主 agent 额外修正紧贴/Unicode 连接与末尾半球案例 |
| F05 论文 ID 碰撞 | 非数字前缀使用完整 stem；数字 ID 保留，重复前缀在请求前拒绝 | Smith_2020/Smith_2021、重复 01、001 前导零 |
| F06 非数字 ID 消失 | 标识全链路按字符串读写；避免 pandas 将 NA 标识认作缺失 | Smith_2020、字面 NA、001 全链路 |
| F07 新证据继承旧票 | content-based candidate_id + verifier fingerprint；不按行号关联 | 证据改变、删除/重排行、规格变化后旧票失效 |
| F08 向量缓存过期 | 内容、模型、端点与 chunk 指纹及维度校验；生成/消费共用检查 | 等 chunk 数内容变更、新 embedding 模型、损坏/错形状 |
| F09 聚合不幂等 | 重建派生 score/flag 列；按候选键回写 | 聚合两次相等，cross 重跑保持单一 cross_score |
| F10 多轮重复计票 | composite 只消费 RUN.round_index 和当前选定任务 | 两轮结果只汇总指定轮；旧 round 在 cross 前拒绝 |
| F11 输入准备不完整 | 补齐缺失/空 MD；检查转换覆盖，保留 Markdown-only | partial/empty/interrupted conversion，模型边界 mock |
| F12 保留旧空结果 | composite/每个阈值均明确写空表覆盖旧内容；失败 batch 不覆盖上一份完整汇总 | 无候选、零达标、损坏单项验证前不发布 |
| F13 请求全失败仍成功 | 汇总阶段失败并传播异常；CLI 非零；last_run 标记失败 | 6 请求全失败，实际子进程 exit 1、6 个 sidecar |
| F14 schema 验证不足 | value/evidence/confidence/unit 等字段检查，缓存身份检查 | 空/缺字段、坏 confidence、类型错误，不缓存为成功 |
| F15 小数多数模式多候选 | **科学规则保留**；文档明确其输出是多候选集 | 保留策略回归；没有自动选成单行或修改算法 |
| F16 缺失值成为文本 | 识别 NaN/NA/NaT、N/A、Not provided 等，不产生伪地点 | empty-majority、半数空、真实文本支持不变 |
| F17 ISO 时间丢失 | 先识别完整 ISO 单点/区间，再处理一般范围 | ISO day/month，原英文年月/精度投票保留 |
| F18 目录误截断 | 跳过实际 Abstract/Introduction 前的目录项；仍按文后段读取截断 | TOC 中文参考文献、正文保留、真实 References 截断 |
| F19 sidecar 模型身份错误 | 由匹配的 meta sidecar 恢复 canonical model，不拆猜文件名 | 大小写/下划线模型名的可恢复 raw |
| F20 测试掩盖依赖 | 移除生产依赖 sys.modules stubs，只 mock 外部边界 | 干净环境 real imports + unittest 全套 |
| F21 PDF 安装路径缺口 | README 给出 optional Marker 步骤，运行时提供明确缺依赖错误 | 未安装 Marker 的错误提示；**未执行真实 PDF/GPU 验收** |
| F22 null content 错误记录再失败 | 明确 null/refusal，错误 sidecar 可写，失败状态上抛 | null content、空 choices、处理器不再二次崩溃 |
| HC01 缺 usage 丢弃内容 | 可选 usage 保留 None，未知消耗不伪造为 0；client 关闭 | SDK adapter mock response |
| P01 提示词格式 | 示例改为合法 JSON，system/user 格式一致，科学问句保留 | prompt 示例/格式测试；实际模型输出率未评估 |
| P02 并发写入 | CLI 进程锁 + 原子文件发布；直接函数调用由调用者协调 | 第二 writer 拒绝、发布中断保留原文件 |

这些是实现层面的修正；错误值、漏文献或旧缓存导致的旧输出可能发生变化。未据此推断既有科学数据库全部错误或全部可信。

## 实现中新增的边界

- 成功 TSV 携带 request/body 指纹，`.meta.json` 保留 canonical paper/model/question/round 身份；旧无指纹文件不自动认作当前证据。
- composite 含 candidate_id，cross 文件按候选内容与 verifier 规格恢复。分数表 summary 明确保留 cross_score。
- 独立 cross/ensemble CLI 启动前检查当前论文、prompt、模型、温度和 round 是否与 composite 一致；过期时拒绝并要求重建上游。
- 失败任务保留已成功的工作与错误证据；修复配置后可恢复。部分/全部失败都不能冒充成功阶段。
- 向量为有限、非零、等维度矩阵；requests 的可重试异常限定为连接/超时、429 与指定 5xx，最多 3 次，不重试能力类 400。
- 原子文件发布覆盖 TSV、Excel、JSON、NPY、自动转换 Markdown；同 CLI 的异常会释放锁，硬中断残留锁须先核实 PID。

## 验证范围与证据

使用新建 Python 3.12 环境，只按修正后的 requirements 安装；未升级系统 Python。测试使用真实 pandas、JSON5、splitter、余弦相似度及 Excel 读写。模型/网络、token 计数的集成模拟与 PDF converter 边界被 mock，未调用真实付费 API，未下载 PDF 模型或处理用户科研数据。

最终全套 **132 个 unittest 通过**；包括原 11 个护栏（去除依赖 stub、补全真实 fixture）和工程/数据回归、两领域集成。基本错误/未定义名/无用导入检查通过，git diff --check 通过。

真实 CLI 子进程验收：

| Case | exit | 实际产物 / 调用 |
|---|---:|---|
| --help | 0 | 正常 usage |
| clean real imports | 0 | langchain 包不存在，所有实际阶段依赖可导入 |
| Aqua all | 0 | 12 ensemble Excel；6 examiner、6 verifier、7 embedding mock calls |
| Aqua resume | 0 | examiner/verifier/embedding 均 0 次 |
| Wildfire all | 0 | 16 ensemble Excel；8 examiner、8 verifier、9 embedding mock calls |
| Wildfire resume | 0 | examiner/verifier/embedding 均 0 次 |
| Aqua 请求全部失败 | 1 | 6 个错误 sidecar，last_run=failed，无成功结果 |

可移植的验收摘要、代码指纹和关键依赖版本见 [FIX_VALIDATION.json](FIX_VALIDATION.json)。完整本地日志作为审查归档保留；自动测试可按 README 的命令重新执行。

独立审核发现的两个重要问题已经修复并通过定向复核：整个预期汇总 batch 验证完成后才发布；需要核验的证据必须具有当前完整核验覆盖，缺失/部分失败不能变成成功空筛选。无证据的缺失答案仍按原规则处理，真实负票的零达标结果仍可正常写空表。定向复核未发现新的 critical/important 问题。已验收代码指纹与命令结果见 FIX_VALIDATION.json。

## 旧结果迁移与尚未执行的验收

原审查副本和研究数据未修改。建议保留历史输出作归档，使用新输出路径先做单篇真实 smoke；无指纹的旧结果需重建 examiner/composite，旧向量缓存会重建、旧行号核验票不参与当前结果。切换 ID/规格后不要只运行下游并复用旧输出。

显式单位的 Result 使用纯数值字符串与独立 unit；Detail 保留联合 normalized_value、scalar_value 及 unit_is_explicit。旧 value 内联单位输出格式保持。使用下游脚本时按 unit 字段解释物理量，不把数值字符串独自当作已统一单位。

真实模型端点能力、PDF 转换忠实度、Linux/macOS、科研语料发生率和独立人工/金标准准确率没有在本轮验证。工程测试通过不等于科学验收完成。

本报告记录工程实现与验收；后续提交、PR 和合并状态以 GitHub 历史为准。完整 Agent 状态机/预算/研究规格控制层仍属于后续设计。
