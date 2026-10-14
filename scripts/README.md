# WavFry 实验评测与论文图表映射全景指南 (`scripts/`)

本目录汇总了 WavFry 发声态多任务学习系统的全生命周期脚本。为了方便后续撰写学术论文、技术报告以及组织可复现实验，本指南建立了**论文图表（Tables & Figures）与评测脚本、输出资产、标准执行命令的一对一映射全景图**。

> **⚠️ 核心执行守则**：
> 全工程所有底层依赖由 `uv` 严格隔离。**严禁直接使用裸 `python`，必须统一使用 `uv run` 执行**。

---

## 1. 🎓 论文图表与测试评测脚本一对一映射总览 (Paper Roadmap)

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1: 主性能基准对比表**<br>(Main Benchmark) | 汇报 WavFry（Weak-On / Weak-Off）与经典对比基线（SVM 传统韵律、WavLM Zero-shot、纯线性头）在干净集与发音人泛化下的 PR-AUC、ROC-AUC、F1 指标。 | **`scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py`**<br>(基线对比配合 `scripts/experiments/run_prosodic_baseline.py`) | `data/experiments/statistical_bootstrap_audit/paired_bootstrap_report.csv`<br>`paired_bootstrap_audit_clean.md` | `uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py --model-a <weak_on> --model-b <weak_off>` |
| **Table 2: 发音人级非参数配对 Bootstrap 显著性检验**<br>(Significance Audit) | 从统计学严谨角度裁定气声弱监督增益是否超越随机扰动（95% 置信区间、双尾经验 $p$ 检验与 8/9 发音人胜率），同时硬核证明对嘎裂声主任务零负迁移。 | **`scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py`** | 同上，自动导出详细的配对差异统计表与发音人胜/负/平分布。 | `uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py --model-a <weak_on> --model-b <weak_off> --n-bootstrap 1000` |
| **Table 3 & Figure 4: LOLO 逐层消融敏感度矩阵**<br>(Micro Layer Ablation) | 剖析 WavLM 13 层（L0 CNN + L1~L12）的微观不可或缺性。揭示嘎裂声（底层微细脉冲 L1~L3）与气声（中层共振峰与高层语义 L4~L8）的**差异化语音学生理层级分布**。 | **`scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py`** | `data/experiments/lolo_ablation_evaluation/lolo_ablation_report.csv`<br>`lolo_ablation_summary.json` | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py --checkpoint <ckpt>` |
| **Table 4 & Figure 5: 宏观自顶向下层截断与轻量部署边界**<br>(Depth Truncation Pareto) | 自顶向下依次丢弃顶层 $k$ 个 Transformer 块（$k=1..12$）。论证保留浅层/中层（如前 6~7 层）即可无损保持精度，给出精度-算力/延迟 Pareto 前沿。 | **`scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py`** | `data/experiments/topk_layer_truncation_evaluation/topk_layer_truncation_report.csv`<br>`topk_layer_truncation_report.md` | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py --checkpoint <ckpt>` |
| **Table 5: 范式 A 12 场景声学退化因子校准矩阵**<br>(Paradigm A Calibration) | 覆盖 3 真实场景（Cafe/Car/Street）× 3 档 SNR（15/5/0 dB）+ 3 个对照组（Clean/Reverb/Reverb+Cafe 5dB）。量化各声学环境下的 ECE/MCE/Brier 置信度失调。 | **`scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py`** | `data/experiments/confidence_calibration_audit/weak_on_vsdc/paradigm_a_factorial_matrix_{task}.csv`<br>`calibration_audit_report.md` | `uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py --checkpoint <ckpt> --paradigm-a` |
| **Figure 6: 可靠性图与信噪比单调衰减曲线**<br>(Reliability & Decay Curves) | 呈现两栏式发布级可靠性图（经验准确率阶梯图 + 样本分箱直方图），以及 ECE 随信噪比由 15dB 恶化至 0dB 的单调衰减物理规律曲线。 | **`scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py`** | `data/experiments/confidence_calibration_audit/weak_on_vsdc/snr_ece_decay_{task}.pdf`<br>`reliability_diagram_{task}_clean.pdf` | 同上（随评测自动导出矢量 PDF 与高清 PNG） |
| **Table 6 & Figure 7: CKA 表征几何热力图与跨环境稳定性**<br>(Representation CKA) | 构建 13×13 线性 CKA 矩阵，展示 SSL 骨干内部特征模块化；计算 Clean vs. 真实噪声（如 5dB Cafe/Car）下的同一层表征保真度，证明声学抗扰稳定性。 | **`scripts/representation_and_calibration_audit/evaluate_representation_cka.py`** | `data/experiments/representation_cka_audit/layer_cka_heatmap_{task}.pdf`<br>`cka_cross_condition_stability.pdf` | `uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py --checkpoint <ckpt> --compare-noise --noise-condition cafe_5db` |
| **Figure 8: 典型案例 Praat 语图与时序轨迹对齐**<br>(Qualitative Case Studies) | 展示典型连续自然对话中嘎裂声声门脉冲周期、气声中高频湍流的声学语图，叠加模型逐帧预测输出与人工精标 Ground Truth 的完美契合。 | **`scripts/export_textgrid.py`** | 生成标准 Praat TextGrid 文件，供语音学专家在 Praat 中截取出版级波形与窄带语图。 | `uv run python scripts/export_textgrid.py --manifest manifests/segments_multi.csv` |
| **Table 7: 边缘部署实测速度与吞吐基准**<br>(Edge Runtime & RTF) | 汇报 ONNX 导出后在 CPU 与 GPU 上的单帧/单音频块推理延迟（Wall-clock ms）、实时率（RTF < 0.05）以及内存显存峰值。 | **`scripts/export_deployment.py`** 与 **`scripts/test_onnx_export.py`** | `deployments/wavfry_multitask.onnx` 及导出测试报告。 | `uv run python scripts/export_deployment.py --checkpoint <ckpt> --output-dir deployments/`<br>`uv run python scripts/test_onnx_export.py` |

---

## 2. 🗂️ 评测套件子目录架构与职责内聚

为了杜绝历史零散脚本与“死脚本”滋生，所有评测功能均收敛于专属包中，且各包具备独立的 `README.md`：

```
scripts/
├── statistical_bootstrap_audit/          # 对应 Table 1 & Table 2
│   ├── evaluate_paired_models_bootstrap.py # [唯一入口] 发音人级聚类非参数配对 Bootstrap 检验
│   └── README.md
├── layer_importance_and_truncation_audit/# 对应 Table 3, Table 4, Figure 4, Figure 5
│   ├── evaluate_lolo_ablation.py          # [唯一入口] LOLO 逐层剥离消融评测（微观层不可或缺性）
│   ├── evaluate_topk_layer_truncation.py  # [唯一入口] 自顶向下逐层截断评测（宏观计算深度 Pareto）
│   └── README.md
├── representation_and_calibration_audit/ # 对应 Table 5, Table 6, Figure 6, Figure 7
│   ├── evaluate_confidence_calibration.py # [唯一入口] 范式 A 12 场景置信度校准与可靠性图生成
│   ├── evaluate_representation_cka.py     # [唯一入口] 13x13 层级 CKA 热力图与跨退化环境稳定性
│   ├── calibration_core.py                # ECE / MCE / Brier 分数计算与绘图核心引擎
│   ├── cka_core.py                        # 快速 Frobenius 范数与无偏 HSIC 计算核心库
│   └── README.md
├── gemini_tts_synthesis/                 # 数据工程：日语副语言气声双段式 TTS 合成切分标注管线
│   └── README.md
├── mllm_annotation/                      # 数据工程：多模态大模型（MLLM）气声粗标预填与强制对齐管线
│   └── README.md
├── experiments/                          # 基准支撑：传统韵律物理特征（Jitter/CPPS）与小样本实验
│   └── README.md
└── legacy/                               # 历史防腐：已废弃或被重构的旧脚本归档与说明
    └── README.md
```

---

## 3. 🚀 论文完整评测复现流（One-Stop Reproduction Pipeline）

当最终模型检查点（`best_model.pt`）就绪后，按照以下标准流水线即可一站式生成论文所需的全部图表数据：

### Step 1: 主性能与配对显著性检验 (Table 1 & Table 2)
```bash
uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py \
  --model-a R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --model-b R:\checkpoints\final_weak_off_baseline\best_model.pt \
  --n-bootstrap 1000 \
  --output-dir data/experiments/statistical_bootstrap_audit
```

### Step 2: 层敏感度与自顶向下截断消融 (Table 3, Table 4, Figure 4, Figure 5)
```bash
# 1. LOLO 微观逐层剥离消融
uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --split val \
  --output-dir data/experiments/lolo_ablation_evaluation

# 2. Top-k 宏观自顶向下截断
uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --split val \
  --output-dir data/experiments/topk_layer_truncation_evaluation
```

### Step 3: 范式 A 声学退化置信度校准与衰减曲线 (Table 5 & Figure 6)
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --paradigm-a \
  --output-dir data/experiments/confidence_calibration_audit/weak_on_vsdc
```

### Step 4: CKA 表征几何与退化稳定性 (Table 6 & Figure 7)
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --compare-noise \
  --noise-condition cafe_5db \
  --output-dir data/experiments/representation_cka_audit
```

### Step 5: 部署模型导出与推断延迟验证 (Table 7)
```bash
uv run python scripts/export_deployment.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --output-dir deployments/

uv run python scripts/test_onnx_export.py
```
