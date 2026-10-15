# Acoustic Interpretability Audit Module

This package contains the canonical acoustic-physical interpretability validation pipeline corresponding to **Table 8b** and **Figure 8** of the paper.

## 1. 核心学术论点与职责
- **消除 Cherry-picking 质疑**：不在测试集中单挑个别孤立音频展示语图，而是在全量测试集上进行传统声学物理指标（SHR, Jitter, Shimmer, H1*-H2*, HNR, CPPS）与模型后验概率的统计相关性校验（Spearman $\rho$ 与 Pearson $r$）。
- **`evaluate_acoustic_correlations.py`（唯一权威主入口）**：
  - 严格**只产出数据**（Data Producer），不包含任何绘图代码；
  - 自动对接 `wavfry.eval.cache_manager.IncrementalCSVCache`，对测试集元音区间进行 Parselmouth 特征增量提取与 CSV 缓存；
  - 产出规范数据成果：
    - `data/experiments/interpretability/acoustic_correlations_metrics.json`（Table 8b 相关系数表）
    - `data/experiments/interpretability/acoustic_correlations.csv`（Figure 8 分层小提琴图数据）
- **绘图分离**：
  - 对应的出版级多子图小提琴图由 `data/visualizations/plot_fig8_acoustic_correlations.py` 纯离线读取上述 CSV 渲染。

## 2. 标准运行命令
```bash
# 从已有 Checkpoint 评估
uv run python scripts/acoustic_interpretability_audit/evaluate_acoustic_correlations.py \
  --checkpoint exp_results/checkpoints/best_model.pt \
  --manifest manifests/segments_multi.csv

# 纯离线秒级运行（基于已导出的预测 CSV）
uv run python scripts/acoustic_interpretability_audit/evaluate_acoustic_correlations.py \
  --preds-csv data/experiments/cache/preds_tiered.csv \
  --manifest manifests/segments_multi.csv
```
