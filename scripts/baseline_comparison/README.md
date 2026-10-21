# Baseline Comparison Evaluation Module

This directory contains the canonical baseline comparison pipeline corresponding to **Table 1** and **Figure 2** of the paper.

## 1. 核心架构与职责分工
- **`evaluate_qut_baselines.py`（唯一权威主入口）**：
  - 严格**只产出数据**（Data Producer），不包含任何绘图代码；
  - 评测 5 个对比模型：**WavFry (Ours)** 与 4 项声学/深度学习基准（**Praat**, **Creapy**, **Kane**, **DeepFry**）；
  - 在 5 档规范退化梯次（`Clean`, `15dB`, `5dB`, `0dB`, `Compound`）上执行评测；
  - 自动对接 `wavfry.eval.cache_manager.IncrementalCSVCache`，实现基线预测秒级复用与增量补全；
  - 产出规范数据成果：
    - `data/experiments/baselines/vowel_robustness_metrics.json`（Figure 2 结构化数据）
    - `data/experiments/baselines/table1_baseline_comparison.csv`（Table 1 CSV）
    - `data/experiments/baselines/table1_baseline_comparison.md`（Table 1 Markdown）
- **`baseline_extractors.py`（核心基准特征与预测提取器）**：
  - 负责 16kHz 单声道实体音频的确定性退化池暂存（`stage_degraded_audio_pool`）；
  - 封装 Praat (Parselmouth)、CreaPy (Paierl & Röck)、DeepFry (Chernyak et al., 2022)、Kane (Kane et al., 2013 / Covarep) 的统一接口；
  - 统一区间重叠与连续概率计算（`compute_interval_overlap_ratio`）。
- **`plot_fig2_noise_robustness.py`（独立绘图与可视化入口）**：
  - 纯离线读取上述结构化数据，渲染满足 IEEE/Elsevier CSL 出版标准的 Figure 2 折线图；
  - 产出高分辨率矢量图与位图至 `data/visualizations/output/`。

## 2. 标准运行命令
```bash
# 模式 A：直接从已有 Checkpoint 评估全部 5 个模型
uv run python scripts/baseline_comparison/evaluate_qut_baselines.py \
  --checkpoint exp_results/checkpoints/best_model.pt \
  --manifest manifests/segments_multi.csv

# 模式 B：纯离线秒级运行（基于已导出的预测 CSV）
uv run python scripts/baseline_comparison/evaluate_qut_baselines.py \
  --preds-csv data/experiments/baselines/preds_best_model_tiered.csv \
  --manifest manifests/segments_multi.csv

# 模式 C：离线渲染 Figure 2 出版图
uv run python scripts/baseline_comparison/plot_fig2_noise_robustness.py
```
