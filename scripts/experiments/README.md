# Baseline & Exploratory Experiments (`scripts/experiments/`)

本目录收敛了项目早期探索性实验、韵律学传统声学特征提取以及对比基线（Baselines）相关脚本。

---

## 脚本职责概览

| 脚本 | 功能定位 | 核心算法与输入输出 |
| :--- | :--- | :--- |
| **`run_prosodic_baseline.py`** | **传统韵律与声学物理特征基线**<br>(SVM / Random Forest / MLP) | 提取音高微扰（Jitter）、振幅微扰（Shimmer）、倒谱峰值突出度（CPPS）、低频谐波能量差（H1*-A3*）等传统语音学生理特征，作为与 WavLM 深度表征对比的纯物理特征基线。 |
| **`run_fewshot_experiment.py`** | **小样本与跨说话人泛化实验** | 评估在极少量人工标注（如 1-shot / 5-shot）条件下，模型依赖预训练声学特征进行发音人迁移的性能表现。 |
| **`align_wu_u2pp.py`** | **U2++ 深度声学模型对齐工具** | 早期基于 Wu's U2++ 框架的音素与音节时间边界强制对齐实现。 |
| **`extract_wu_syllable_nuclei.py`** | **音节核心（Syllable Nuclei）提取** | 基于能量包络极值与局部过零率，自动定位元音中心（Vowel Nucleus）区间。 |

---

## 执行说明

所有脚本均需使用 `uv run` 执行，例如：
```bash
uv run python scripts/experiments/run_prosodic_baseline.py
```
> **注**：核心模型主干训练与当前发表级评测推荐优先使用 `scripts/train.py`、`scripts/representation_and_calibration_audit/`、`scripts/layer_importance_and_truncation_audit/` 以及 `scripts/statistical_bootstrap_audit/`。
