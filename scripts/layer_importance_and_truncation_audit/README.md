# Layer Importance & Truncation Audit Suite (`scripts/layer_importance_and_truncation_audit/`)

本目录收敛了对 WavLM 骨干网络（13 层：L0 CNN 前端 + L1~L12 Transformer）在发声态多任务（嘎裂声 Creaky Voice、气声 Breathy Voice）下的**内部层重要度审计与结构剪枝敏感度评测**工具。

对应论文/技术报告中的 **层级重要度解耦与低延迟边缘剪枝分析（Layer Auditing & Depth Truncation）** 章节。

---

## 目录脚本与职责划分

| 脚本 | 论文对应图表 | 核心职能 | 审计视角 | 典型执行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **`evaluate_lolo_ablation.py`** | **Figure 4 / Table 3**<br>(LOLO 敏感度矩阵) | **微观逐层剥离消融 (Leave-One-Layer-Out)**<br>逐一屏蔽单个层（置 $w_l = -\infty$ 后动态 Softmax 归一化），量化单个隐藏层对任务不可或缺性（Indispensability）。 | **微观局部敏感度**<br>（排查哪一层是特定任务的关键表征瓶颈） | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py --checkpoint <ckpt>` |
| **`evaluate_topk_layer_truncation.py`** | **Figure 5 / Table 4**<br>(深度剪枝衰减曲线) | **宏观自顶向下层截断 (Top-$k$ Depth Truncation)**<br>依次自顶向下丢弃顶层 $k$ 个 Transformer 块（$k=1, 2, \dots, 12$），评估模型在仅保留浅层/中层时的性能衰减与算力-精度权衡。 | **宏观有效深度与部署上限**<br>（确定端侧模型可安全裁剪的层数与蒸馏目标层） | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py --checkpoint <ckpt>` |

---

## 核心算法与数学原理

### 1. 动态 Softmax 掩码重归一化 (Dynamic Softmax Re-Normalization)
在消融指定层 $l$ 时，严禁直接在加权后置零（这会导致特征模长萎缩与均方差失真）。本套工具统一在 Softmax 前置 logit 上施加掩码：
$$\tilde{w}_k = \begin{cases} -\infty, & k \in \mathcal{M} \\ w_k, & k \notin \mathcal{M} \end{cases}$$
$$h_{\text{fused}} = \sum_{k=0}^{12} \text{Softmax}(\tilde{\mathbf{w}})_k \cdot h_k$$
此设计严格保证了特征张量的数值尺度（Scale Invariance），杜绝了基线模长偏移带来的伪消融偏差。

### 2. 单次前向缓存优化 (Single-Pass Forward Caching)
WavLM 骨干网络参数量庞大（~94M 参数）。为避免针对 13 种消融条件重复调用 13 次 WavLM 前向，工具采用**单批次前向特征缓存与批量 GEMM 计算**：
- 对每个音频块，WavLM 骨干仅执行 **1 次** 前向提取全部 13 层特征；
- 在 GPU 显存内将 13 种消融掩码并行矩阵乘法融合（$13 \times B \times T \times D$），单卡可在数分钟内跑完数万片段的完整验证集。

---

## 详细参数说明与使用范例

### 1. LOLO 逐层消融 (`evaluate_lolo_ablation.py`)

```bash
# 1. 快速冒烟测试（300 个样本验证链路）
uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --quick

# 2. 全量验证集标准评测（输出 PR-AUC / ROC-AUC / F1 消融矩阵）
uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --split val \
  --task all \
  --output-dir data/experiments/lolo_ablation_evaluation

# 3. 施加真实环境噪声退化（评估在噪声压力下各层的抗扰韧性）
uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --noise-condition cafe_5db \
  --output-dir data/experiments/lolo_ablation_cafe_5db
```

### 2. Top-$k$ 深度截断 (`evaluate_topk_layer_truncation.py`)

```bash
# 1. 全量验证集自顶向下剪枝评估
uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --split val \
  --output-dir data/experiments/topk_layer_truncation_evaluation

# 2. 仅评估嘎裂声或气声单任务
uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --task breathy \
  --output-dir data/experiments/topk_breathy_only
```

---

## 产出物结构

运行后将在指定的 `--output-dir` 下自动生成：
- `*.csv`: 包含 Baseline 及每一层被剔除后的完整指标（PR-AUC, ROC-AUC, F1, Precision, Recall, $\Delta \text{PR-AUC}$）。
- `*.md`: 学术论文可直接贴入的 Markdown 汇总表与关键层排序分析。
- `*.json`: 结构化指标字典，方便自动绘图与下游脚本二次聚合。
