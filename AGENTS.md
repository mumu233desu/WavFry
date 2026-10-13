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
