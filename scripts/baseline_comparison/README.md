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
- **`evaluate_frame_event_map.py`（细粒度连续时序发声事件 Multi-IoU 评测权威主入口）**：
  - 依据 `docs/notes_frame_level_event_evaluation_methodology.md` 方法论备忘录实现；
  - 摆脱预切分元音区间的束缚，在连续自由时间轴上执行一维时序动作检测（Temporal Action Detection, TAD）；
  - 支持多阶梯容差及格线 $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$ 下的匈牙利二分图一对一配对与 PR 积分，输出宏观检出率（$\alpha=0.1\text{--}0.3$）、微观边界锐度（$\alpha=0.5\text{--}0.7$）及全局 $\text{mAP}$；
  - 支持固定阈值（验证集最优 $\tau^*$ 与开箱 0.5）下的二值化时间区间 Precision、Recall、F1-Score 评测（零泄漏协议）；
  - 产出规范数据成果至 `data/experiments/frame_level_eval/`：
    - `frame_event_map_metrics.json`
    - `frame_event_map_summary.csv`
    - `frame_event_map_report.md`

## 2. 标准运行命令
```bash
# 模式 A：闭合元音区间基准评测（Table 1 / Figure 2）
uv run python scripts/baseline_comparison/evaluate_qut_baselines.py \
  --checkpoint exp_results/checkpoints/best_model.pt \
  --manifest manifests/segments_multi.csv

# 模式 B：细粒度连续时序事件 Multi-IoU mAP 评测（Section 4.2 / Kappa Test）
uv run python scripts/baseline_comparison/evaluate_frame_event_map.py \
  --batch-dir "E:/Corpus/Creaky/MELI_processed/textgrid/_creaky_cache/batches/batch_1/manual_creaky_annotations.json.d" \
  --models "WavFry" "CreaPy" "DeepFry" \
  --tier clean

# 模式 C：离线渲染 Figure 2 出版图
uv run python scripts/baseline_comparison/plot_fig2_noise_robustness.py
```
