# AGENTS.md — Agent Operation Guidelines & Inviolable Rules

> **CRITICAL & INVIOLABLE RULE**: Every agent operating in this repository MUST strictly abide by the rules documented here. No exceptions.

---

## 1. ⚠️ Python Execution Environment: MANDATORY `uv run`

### ❌ ABSOLUTELY FORBIDDEN:
```bash
# NEVER RUN BARE PYTHON OR PIP DIRECTLY:
python scripts/train.py ...           # ❌ FORBIDDEN!
python -m pytest ...                  # ❌ FORBIDDEN!
python -c "import ..."                # ❌ FORBIDDEN!
```

### ✅ MANDATORY STANDARD:
```bash
# ALWAYS EXECUTE VIA UV RUN:
uv run python scripts/train.py ...    # ✅ MANDATORY
uv run pytest ...                     # ✅ MANDATORY
uv run python -c "..."                # ✅ MANDATORY
```

### 为什么必须严格执行 `uv run`？
1. **关键依赖缺失**：全局系统 Python 未安装该工程专属的底层 C/Rust 音频处理库（如 `df` / `DeepFilterNet3`）。
2. **延迟崩溃陷阱**：在启动阶段前几个 Epoch（如 Epoch 1~4），降噪强度为 0，裸 `python` 可以侥幸运行；但一旦进入 Epoch 5 数据增强课程（atten_db >= 0.5），裸 `python` 就会抛出 `ModuleNotFoundError: No module named 'df'`，导致跑了几个小时的实验半途夭折！
3. **环境绝对隔离**：项目专属虚拟环境已由 `uv` 完整管理（位于 `.venv`），必须且只能通过 `uv run` 执行。

---

## 2. 核心架构与训练规范

1. **WavLM 骨干网络绝对冻结**：
   - `model.fine_tune_layers: "none"`
   - `training.backbone_lr: 0.0`
   - 严禁意外解冻 WavLM 骨干参数。仅微调分类头、标量混合权重（`scalar_mix_weights`）以及 VSDC 背景抑制头。

2. **人工标注数据零覆盖保护（Human Labor Protection）**：
   - 任何 MLLM 预填、弱监督挖掘或脚本处理，严禁覆盖 `manual_annotation` 人工已标条目。
   - 人工已标条目在 weak manifest 中严格标记为 `redundant`，不可被弱监督逻辑覆盖。

3. **双任务独立性（Multi-Task Isolation）**：
   - 嘎裂声（Creaky Voice）主任务基准严禁受到气声（Breathy Voice）弱监督优化的扰动或负迁移。
   - 验证集早停与主指标仲裁以配置明确指定的 `eval.primary_task` 为准。

4. **磁盘与显存管理**：
   - 检查点与 TensorBoard 日志统一放置于高速盘（如 `R:\checkpoints\...`, `R:\runs\...`）。
   - 启动新实验前核查 `R:` 盘剩余空间，并在必要时安全清理废弃 run id 目录。

---

## 3. 🎯 学术严谨性与严禁过度宣称准则（Anti-Overclaiming Invariant）

1. **禁止仅凭单一点估计（Point Estimate）或平均值微小差异定性优劣**：
   - 严禁在看到均值指标（如 PR-AUC、ROC-AUC、F1、准确率等）出现微小数值差异时，直接断言“模型显著提升”、“某模块全面有效”或“性能显著优于基线”。
   - 必须通过**与数据分布特征相适配的严格统计检验方法**（如发音人/群组聚类非参数配对 Bootstrap 检验、置信区间估计、双尾假设检验、置换检验等，并统计胜率分布），评估观察到的差异是否超出随机扰动或发音人/样本间固有方差。
   - 若置信区间跨越 0、或 $p$ 值未达到统计学显著性阈值（如 $p > 0.05$）：**必须实事求是记录为“在当前统计检验下未观察到显著差异”，严禁使用主观夸大词汇（如“明显优于”、“大幅领先”等）**。

2. **结论与归因必须具备硬核数据支撑与领域机理解释**：
   - 任何针对模型行为、鲁棒性或消融实验的定性结论，必须同时具备两项支撑：
     1. **严谨的量化数据与对照实验支撑**（杜绝无对照、无测试集的无凭猜测）；
     2. **合理的领域物理/语音学生理学机理解释**（严禁给出违背声学常识或物理规律的牵强归因）。
   - 如果缺乏充分数据支撑，必须明确标注为“推测性假设（Hypothesis）”并指明后续验证方案，不得作为确凿结论呈现。

---

## 4. 🧹 脚本工程治理与防“死脚本”规范（Script Architecture Hygiene）

1. **单一权威入口原则（Canonical Entry Point）**：
   - 严禁为了临时单次验证在仓库中随意新增脱离体系的孤立脚本。
   - 每个专项功能模块（如表征与校准审计、层敏感度分析、显著性检验等）必须收敛至专门子目录下，且该子目录内必须指定**唯一的权威主入口脚本**。
   - 每个子目录必须配套维护自解释的 `README.md`，明确阐述学术对应图表、理论机制与标准 `uv run` 执行命令。

2. **核心算子公共内聚，杜绝私自复制代码**：
   - 声学环境退化、噪声混响合成、SNR 计算等核心运算，统一收敛于专属评估模块（如 `wavfry.eval.acoustic_degradation`）；
   - 帧级池化、动态 Softmax 掩码重归一化等算子统一调用核心损失/模型库，严禁在不同脚本中各自复制粘贴算法私有副本。

3. **历史脚本归档防腐机制**：
   - 任何因架构重构、功能合并而不再作为官方入口的历史脚本，必须移动至 `scripts/legacy/`；
   - 必须在 `scripts/legacy/README.md` 中详细登记该脚本的历史职能、废弃原因及当前标准替代方案。

---

## 5. ⏱️ 长耗时后台任务可观测性规范（Task Observability & Flushed I/O）

1. **控制台输出必须强制刷盘（Explicit Flushing）**：
   - 后台运行重定向至日志文件时，Python 在非交互终端下会启用 4KB~8KB 块缓冲（Block Buffering）。所有长耗时脚本的 `print()` 必须显式声明 `flush=True`，或直接使用标准 `logging` 模块，严禁将进度信息长时间滞留在内存缓冲区中。
2. **密集循环具备透明进度条/心跳反馈**：
   - 遍历全量验证集/测试集的大规模 DataLoader 循环，必须按批次（如每 100~150 batches）输出当前耗时与百分比进度，单次静默不得超过 60 秒。
3. **前置耗时预警与任务规模告知**：
   - Agent 在启动长耗时任务前，必须主动向用户同步计算规模（样本量、场景数、预计每轮耗时与总耗时），避免产生“程序死锁/无响应”的错觉。
