# Multi-Backbone Comparison Ablation Suite

## 1. 学术定位与论文对应（Paper Mapping）

* **论文对应图表**：**Table 5: Self-Supervised Speech Backbone Sensitivity & Pretraining Inductive Bias Analysis**（第 5 节消融实验主表）
* **核心科学问题（Research Question）**：
  在保持下游架构（13 层标量混合加权 + 注意力池化 + 任务分类头）和参数完全冻结的前提下，**自监督预训练目标（Pretraining Objective）的归纳偏置（Inductive Bias）如何影响非常态发声态（嘎裂声与气声）的检测精度与声学抗噪鲁棒性？**
* **对比骨干网络**：
  1. **WavLM-Base+** (`microsoft/wavlm-base-plus`, 94.7M): 门控相对位置偏置 + **掩码语音降噪与重构（Masked Speech Denoising & Prediction）**；
  2. **HuBERT-Base** (`facebook/hubert-base-ls960`, 94.7M): 离线 k-means 声学聚类单元掩码预测（BERT 范式）；
  3. **wav2vec 2.0-Base** (`facebook/wav2vec2-base-960h`, 94.4M): Gumbel-Softmax 离散量化与对比学习（Contrastive Learning）。

---

## 2. 核心架构与学术严谨性保障（Inviolable Invariants）

1. **骨干绝对冻结（Frozen Backbone Invariant）**：
   * `model.fine_tune_layers: "none"`
   * `training.backbone_lr: 0.0`
   * 严禁微调预训练大模型本身。仅更新可学习的 13 层标量权重（`scalar_mix_weights`）以及下游分类头（`head_lr: 1e-4`），确保评价的纯粹性。
2. **结构严格等价（Exact Structural Equivalence）**：
   * 三者均为 16kHz 输入、7 层 CNN 特征抽取器（降采样步长 320，对应 20ms 一帧）；
   * 12 层 Transformer Encoder，输出 13 维 hidden states（$13 \times 768$）；
   * 下游分类头容量、损失函数（Focal Loss）、学习率调度及数据增强课程完全一致。

---

## 3. 标准执行命令（Mandatory `uv run` Execution）

### ① 一键运行完整对比（训练 + 测试评估并生成 Table 5）
```bash
uv run python scripts/backbone_comparison/run_backbone_comparison.py --mode train_and_eval
```

### ② 仅针对已训练好的检查点执行评测
```bash
uv run python scripts/backbone_comparison/run_backbone_comparison.py --mode eval_only
```

### ③ 单独跑指定骨干网络（例如仅测试 HuBERT）
```bash
uv run python scripts/backbone_comparison/run_backbone_comparison.py --backbones hubert --mode train_and_eval
```

---

## 4. 输出工件与产物（Artifacts）

* **日志与检查点**：
  * WavLM: `R:/runs/ablation_backbone_wavlm/`, `R:/checkpoints/ablation_backbone_wavlm/`
  * HuBERT: `R:/runs/ablation_backbone_hubert/`, `R:/checkpoints/ablation_backbone_hubert/`
  * wav2vec 2.0: `R:/runs/ablation_backbone_wav2vec2/`, `R:/checkpoints/ablation_backbone_wav2vec2/`
* **汇总指标与出版表格**：
  * `exp_results/backbone_comparison/backbone_comparison_results.json`（原始数据）
  * `exp_results/backbone_comparison/table5_backbone_comparison.md`（可直接贴入论文手稿的 Markdown/LaTeX 表格）
