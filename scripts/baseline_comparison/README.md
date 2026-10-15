# Baseline Comparison Evaluation Module

This directory contains the canonical baseline comparison pipeline corresponding to **Table 1** and **Figure 2** of the paper.

## 1. 核心架构与职责分工
- **`evaluate_qut_baselines.py`（唯一权威主入口）**：
  - 严格**只产出数据**（Data Producer），不包含任何绘图代码；
  - 自动对接 `wavfry.eval.cache_manager.IncrementalCSVCache`，实现基线特征秒级复用与增量补全；
  - 在 5 档退化层级（`Clean`, `15dB`, `5dB`, `0dB`, `Compound`）上评估模型与前人基线；
  - 产出规范数据成果：
    - `data/experiments/baselines/vowel_robustness_metrics.json`
    - `data/experiments/baselines/table1_baseline_comparison.csv`
- **绘图分离**：
  - 对应的出版级折线图由 `data/visualizations/plot_fig2_noise_robustness.py` 纯离线读取上述 JSON 渲染。

## 2. 标准运行命令
```bash
# 模式 A：直接从已有 Checkpoint 评估
uv run python scripts/baseline_comparison/evaluate_qut_baselines.py \
  --checkpoint exp_results/checkpoints/best_model.pt \
  --manifest manifests/segments_multi.csv

# 模式 B：纯离线秒级运行（基于已导出的预测 CSV）
uv run python scripts/baseline_comparison/evaluate_qut_baselines.py \
  --preds-csv data/experiments/cache/preds_tiered.csv \
  --manifest manifests/segments_multi.csv
```
