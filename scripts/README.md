# WavFry 实验评测与论文图表映射全景指南 (`scripts/`)

本目录汇总了 WavFry 发声态多任务学习系统的全生命周期脚本。为了在后续论文写作（会议版扩展/期刊版）中全面展现 WavFry 的学术贡献，本指南建立了**论文图表（Tables & Figures）与评测脚本、输出资产、标准执行命令的一对一映射全景图**。

> **⚠️ 核心执行守则**：
> 全工程所有底层依赖由 `uv` 严格隔离。**严禁直接使用裸 `python`，必须统一使用 `uv run` 执行**。

---

## 1. 🎯 论文核心定位：双核发声态协同架构 (Dual-Core Phonation Pillars)

WavFry 并非单一的气声或嘎裂声模型，而是一个**面向自然连续对话、抗噪稳健的双发声态（Dual Phonation）深度分析框架**。论文的核心逻辑由两大支柱和一项协同基石构成：

```
                    ┌────────────────────────────────────────────────────────┐
                    │      WavFry: Dual-Core Conversational Phonation        │
                    └────────────────────────────────────────────────────────┘
                                   │                          │
                 ┌─────────────────┴────────┐       ┌─────────┴────────────────┐
                 │ 支柱一：嘎裂声主基准     │       │ 支柱二：气声弱监督拓展   │
                 │ (Creaky Voice Foundation)│       │ (Breathy Voice Expansion)│
                 ├──────────────────────────┤       ├──────────────────────────┤
                 │ • 对标4大前人SOTA检测器  │       │ • 多模态大模型弱标注挖掘 │
                 │   (Praat/Kane/DeepFry/   │       │ • VSDC-Net 背景抑制网络  │
                 │    CreaPy)，全面大幅领先 │       │ • 同发音人声纹对消对比   │
                 │ • 极强低频抗噪(Car 0dB)  │       │ • 突破 0.80 PR-AUC 瓶颈  │
                 │ • 性别与方言人口学公平性 │       │ • 8/9 发音人配对显著胜出 │
                 └──────────────────────────┘       └──────────────────────────┘
                                   │                          │
                                   └─────────────┬────────────┘
                                                 ▼
                    ┌────────────────────────────────────────────────────────┐
                    │ 基石：双任务特征解耦与语音学生理互补性                  │
                    │ • 嘎裂声受100%硬隔离保护，零负迁移 (PR-AUC 保持 0.923+) │
                    │ • 生理分层互补：嘎裂声(时域底层脉冲) vs 气声(频域中层气流)│
                    └────────────────────────────────────────────────────────┘
```

---

## 2. 🎓 论文图表与测试评测脚本一对一映射总览 (Paper Roadmap)

### 第一部分：嘎裂声（Creaky Voice）核心基准对比与鲁棒性

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1: 嘎裂声检测主流前人基线横向对比**<br>(Creaky Baseline Benchmark) | **WavFry 相比前人传统声学与浅层深度检测器（Praat, Kane 2013, DeepFry, CreaPy）展现出碾压性性能提升**（在无泄漏 45 人测试集上 F1 提升 +20%~+40%，PR-AUC 达 0.924）。 | **`scripts/legacy/eval_baselines_testset.py`**<br>(基线聚合于 `data/experiments/baselines/`) | `data/experiments/baselines/all_baselines_eval_results.json`<br>`wavfry_single_attn_eval_results.json` | 历史结果已归档，后续重评可调用 `scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py` |
| **Figure 2: 嘎裂声前人基线退化声学抗噪对比**<br>(Creaky Noise Robustness Curves) | 对比 WavFry 与 4 大基线在 Clean, 20dB, 10dB, 0dB, -10dB, -20dB, Reverb 下的性能衰减曲线，证明 WavFry 在严苛退化下的断层领先优势（NCMMSC 图形资产）。 | 历史绘图脚本：`paper/.../figures/fig_noise_robustness.pdf` | `data/visualizations/output/fig_noise_robustness_poster.pdf` | 已沉淀为矢量 PDF 资产 |
| **Table 2 & Figure 3: 性别与方言人口学公平性分析**<br>(Demographic Fairness across Subgroups) | 评估 WavFry 在性别（女性 vs. 男性）与三大汉语方言区（北京、广东、四川）下的检测公平性，证明模型对发音人群体无系统性偏见（偏置差异 < 2%）。 | 历史统计脚本：`scripts/legacy/annotation_stats.py` | `paper/.../figures/fig_demographic_heatmap.pdf`<br>`paper/.../figures/fig_wavfry_prob_violin.pdf` | 已沉淀为矢量 PDF 资产 |

---

### 第二部分：气声（Breathy Voice）弱监督拓展与统计显著性

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **Table 3: 气声弱监督与 VSDC-Net 演进消融表**<br>(Breathy Weak Supervision Progression) | 呈现气声从阶段 0（纯强监督 0.7712）$\to$ 阶段 1（朴素 MIL 0.7845）$\to$ 阶段 2（VSDC+粉红噪 0.7964）$\to$ 阶段 3（VSDC+真实噪声 **0.8017**）的逐级跃升过程。 | **`scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py`** | `docs/weak_label_training_technical_specification.md` 第 10 节 | `uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py --model-a <weak_on> --model-b <weak_off>` |
| **Table 4: 发音人级非参数配对 Bootstrap 显著性检验**<br>(Significance Audit) | **硬核统计推断**：$B=1000$ 轮重采样证实气声增益具备极高显著性（95% CI `[+0.0049, +0.0132]` 严格为正，$p \le 0.01$，8/9 发音人胜率）；同时证实嘎裂声主任务 $p=0.1558 > 0.05$（零负迁移）。 | **`scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py`** | `data/experiments/statistical_bootstrap_audit/paired_bootstrap_report.csv`<br>`paired_bootstrap_audit_clean.md` | `uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py --model-a <weak_on> --model-b <weak_off> --n-bootstrap 1000` |

---

### 第三部分：双任务解耦、层敏感度与声学物理机理剖析

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **Table 5 & Figure 4: LOLO 微观逐层消融敏感度矩阵**<br>(Micro Layer Ablation) | 揭示双任务在 WavLM 13 层中的**语音学生理层级差异**：嘎裂声强烈依赖底层时域微细脉冲（L1~L3）；气声更依赖中层共振峰与高层语义（L4~L8）。 | **`scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py`** | `data/experiments/lolo_ablation_evaluation/lolo_ablation_report.csv`<br>`lolo_ablation_summary.json` | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py --checkpoint <ckpt>` |
| **Table 6 & Figure 5: 宏观自顶向下层截断与剪枝边界**<br>(Depth Truncation Pareto) | 自顶向下依次丢弃顶层 $k$ 个 Transformer 块（$k=1..12$），论证保留前 6~7 层即可无损保持干净集双任务精度，给出端侧轻量化部署上限。 | **`scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py`** | `data/experiments/topk_layer_truncation_evaluation/topk_layer_truncation_report.csv`<br>`topk_layer_truncation_report.md` | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py --checkpoint <ckpt>` |
| **Table 7: 范式 A 12 场景声学退化因子校准矩阵**<br>(Paradigm A Calibration) | 覆盖 3 真实场景（Cafe/Car/Street）× 3 档 SNR + 3 个对照组共 12 种声学环境，对比 Weak-On 与 Weak-Off 的 ECE。量化环境退化引起的置信度失真。 | **`scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py`** | `data/experiments/confidence_calibration_audit/weak_on_vsdc/paradigm_a_factorial_matrix_{task}.csv`<br>`calibration_audit_report.md` | `uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py --checkpoint <ckpt> --paradigm-a` |
| **Figure 6: 可靠性图与信噪比单调衰减曲线**<br>(Reliability & Decay Curves) | 呈现两栏式发布级可靠性图，以及 ECE 随信噪比由 15dB 恶化至 0dB 的单调衰减物理规律曲线（证明 Car 低频轰鸣坚挺、Cafe 人声交谈恶化）。 | **`scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py`** | `data/experiments/confidence_calibration_audit/weak_on_vsdc/snr_ece_decay_{task}.pdf`<br>`reliability_diagram_{task}_clean.pdf` | 同上（自动导出矢量 PDF 与 PNG） |
| **Table 8 & Figure 7: CKA 表征几何热力图与跨环境稳定性**<br>(Representation CKA) | 构建 13×13 线性 CKA 矩阵，展示特征分块模块化；计算 Clean vs. 5dB 噪声下的同一层表征保真度，证明预训练声学几何结构的抗扰稳定性。 | **`scripts/representation_and_calibration_audit/evaluate_representation_cka.py`** | `data/experiments/representation_cka_audit/layer_cka_heatmap_{task}.pdf`<br>`cka_cross_condition_stability.pdf` | `uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py --checkpoint <ckpt> --compare-noise --noise-condition cafe_5db` |

---

### 第四部分：定性案例与部署基准

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 8: 典型案例 Praat 语图与双发声态对齐**<br>(Qualitative Case Studies) | 展示典型连续自然对话中嘎裂声声门脉冲周期、气声中高频湍流的窄带语图，叠加双任务逐帧预测概率轨迹与人工精标 Ground Truth 的完美契合。 | **`scripts/export_textgrid.py`** | 生成标准 Praat TextGrid 文件，供在 Praat 中直接导出出版级截图。 | `uv run python scripts/export_textgrid.py --manifest manifests/segments_multi.csv` |
| **Table 9: 边缘部署实测速度与吞吐基准**<br>(Edge Runtime & RTF) | 汇报 ONNX 导出后在 CPU 与 GPU 上的单音频块推理延迟（Wall-clock ms）、实时率（RTF < 0.05）以及内存显存峰值。 | **`scripts/export_deployment.py`** 与 **`scripts/test_onnx_export.py`** | `deployments/wavfry_multitask.onnx` 及导出验证报告。 | `uv run python scripts/export_deployment.py --checkpoint <ckpt> --output-dir deployments/`<br>`uv run python scripts/test_onnx_export.py` |

---

## 3. 🗂️ 评测套件子目录架构与职责内聚

为了杜绝历史零散脚本与“死脚本”滋生，所有评测功能均收敛于专属包中，且各包具备独立的 `README.md`：

```
scripts/
├── statistical_bootstrap_audit/          # 对应 Table 3 & Table 4（双模型配对显著性检验）
│   ├── evaluate_paired_models_bootstrap.py # [唯一入口] 发音人级聚类非参数配对 Bootstrap 检验
│   └── README.md
├── layer_importance_and_truncation_audit/# 对应 Table 5, Table 6, Figure 4, Figure 5（层重要度与深度裁剪）
│   ├── evaluate_lolo_ablation.py          # [唯一入口] LOLO 逐层剥离消融评测（微观层不可或缺性）
│   ├── evaluate_topk_layer_truncation.py  # [唯一入口] 自顶向下逐层截断评测（宏观计算深度 Pareto）
│   └── README.md
├── representation_and_calibration_audit/ # 对应 Table 7, Table 8, Figure 6, Figure 7（校准与 CKA 表征）
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
└── legacy/                               # 历史防腐：已废弃或被重构的旧脚本归档（含 NCMMSC 初版基线评估脚本）
    └── README.md
```

---

## 4. 🚀 论文完整评测复现流（Five-Step Reproduction Pipeline）

当未来训练好终版模型检查点（`best_model.pt`）后，只需依次在终端执行以下标准流水线即可一站式生成论文所需的全部图表数据：

### Step 1: 双模型基准与配对显著性检验 (Table 1, Table 3, Table 4)
```bash
uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py \
  --model-a R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --model-b R:\checkpoints\final_weak_off_baseline\best_model.pt \
  --n-bootstrap 1000 \
  --output-dir data/experiments/statistical_bootstrap_audit
```

### Step 2: 双任务层敏感度与自顶向下截断消融 (Table 5, Table 6, Figure 4, Figure 5)
```bash
# 1. LOLO 微观逐层剥离消融 (同时输出 Creaky 与 Breathy 的层级差异)
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

### Step 3: 范式 A 声学退化置信度校准与衰减曲线 (Table 7 & Figure 6)
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --paradigm-a \
  --output-dir data/experiments/confidence_calibration_audit/weak_on_vsdc
```

### Step 4: CKA 表征几何与退化稳定性 (Table 8 & Figure 7)
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --compare-noise \
  --noise-condition cafe_5db \
  --output-dir data/experiments/representation_cka_audit
```

### Step 5: 部署模型导出与推断延迟验证 (Table 9)
```bash
uv run python scripts/export_deployment.py \
  --checkpoint R:\checkpoints\final_weak_on_vsdc\best_model.pt \
  --output-dir deployments/

uv run python scripts/test_onnx_export.py
```
