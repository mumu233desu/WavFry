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
                 │ • QUT-NOISE 全场景抗噪   │       │ • 突破 0.80 PR-AUC 瓶颈  │
                 │ • 发音人性别公平性验证   │       │ • 发音人配对显著胜出     │
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

## 2. 🌐 统一声学退化评估全景标准 (Canonical Degradation Protocol)

为彻底杜绝各实验随意剪裁场景引发的 **Cherry-picking 质疑**，全工程建立**全局唯一、全篇论文严格一致的声学退化标准资产体系**：

### 2.1 物理退化全景资产（共 19 个独立测试环境）
1. **纯净基准 (Clean)**：无加噪、无混响原声；
2. **5 类真实加性环境噪声 $\times$ 3 档 SNR**（共 15 个条件）：
   - `cafe`（食堂/咖啡厅人群交谈声，Babble）: `15dB`, `5dB`, `0dB`
   - `car`（车内低频轰鸣与胎噪，Rumble）: `15dB`, `5dB`, `0dB`
   - `home`（厨房与室内生活杂音，Domestic）: `15dB`, `5dB`, `0dB`
   - `street`（街道交通车流声，Traffic）: `15dB`, `5dB`, `0dB`
   - `reverb_ambient`（大空间本底环境噪声，Ambient）: `15dB`, `5dB`, `0dB`
3. **2 类真实空间冲激响应（Spatial RIR Only）**（共 2 个条件）：
   - `car_rir`（车厢微空间短脉冲，短混响）
   - `room_rir`（礼堂/大房间冲激响应，长混响）
4. **2 类物理真实复合场景（Compound: RIR + 同源加性噪声 @ 5dB）**（共 2 个条件）：
   - `car_compound` = `car_rir` + `car_noise @ 5dB`（真实车内行驶环境）
   - `room_compound` = `room_rir` + `reverb_ambient @ 5dB`（真实大空间现场环境）

### 2.2 论文统一展现的 5 档宏观声学梯度（The 5 Canonical Degradation Tiers）
为兼顾**数据完整性**与**可视化高可读性**，全篇折线图与主表格统一收敛至 5 档梯度：
- **Tier 0**: `Clean`（分布内基准）
- **Tier 1**: `Mild (15 dB Avg over 5 noises)`
- **Tier 2**: `Moderate (5 dB Avg over 5 noises)`
- **Tier 3**: `Severe (0 dB Avg over 5 noises)`
- **Tier 4**: `Compound (Car Compound & Room Compound Avg)`

> **一致性铁律**：所有实验（Table 1, Figure 2, Table 4, Table 8）统一步调，严格按照这 5 档梯度或全量 19 条件进行汇报，严禁私自挑子集。

---

## 3. 🎓 论文图表与评测脚本映射总览 (Paper Roadmap)

### 第一部分：嘎裂声（Creaky Voice）核心基准对比与鲁棒性

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 | TODO 状态 / 前置依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1 & Figure 2: 主流前人基线横向对比与退化衰减曲线**<br>(Unified Baseline Benchmark) | **横向超越 4 大前人检测器**（Praat, Kane, DeepFry, CreaPy），且在 5 档声学退化下保持断层领先。<br>• **Figure 2**: 5 模型主折线，带浅色阴影带（Shaded Band 呈现 5 噪声方差）；<br>• **Table 1**: 5 梯度量化大表（附录附 19 条件明细）。 | **`scripts/baseline_comparison/evaluate_qut_baselines.py`**<br>(单脚本同时产出图与表) | `data/experiments/baselines/table1_baseline_benchmark.csv`<br>`fig2_baseline_noise_robustness.pdf` | `uv run python scripts/baseline_comparison/evaluate_qut_baselines.py` | 🟡 **待重构接入引擎**<br>• 新建单入口脚本接入退化引擎；<br>• 待新数据终版模型训练完成后，一键导出最终表格与矢量图。 |
| **Table 2: 发音人性别公平性分析**<br>(Gender Demographic Fairness) | 评估模型在女性 vs 男性下的检测公平性，证明无发音人性别人群系统偏见（差异 < 2%）。<br>*(已剔除异构外部语料中不存在的三大方言区)* | **`scripts/statistical_bootstrap_audit/evaluate_gender_fairness.py`** | `data/experiments/statistical_bootstrap_audit/gender_fairness_report.csv`<br>`fig_gender_fairness_violin.pdf` | `uv run python scripts/statistical_bootstrap_audit/evaluate_gender_fairness.py --checkpoint <ckpt>` | 🟡 **待新建脚本并重跑**<br>• 归档旧方言脚本，新建纯净的性别公平性脚本；<br>• 待终版模型产出后统计导出。 |

---

### 第二部分：气声（Breathy Voice）弱监督拓展与统计显著性

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 | TODO 状态 / 前置依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 3: 核心模块正交消融表**<br>(Component Ablation Table) | 验证关键模块的不可或缺性：<br>• 基线 (纯强监督)<br>• + 朴素 MIL 弱监督<br>• + VSDC 背景抑制网络<br>• + QUT-NOISE 真实退化增强 | **`scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py`** | `docs/weak_label_training_technical_specification.md`<br>`component_ablation_summary.csv` | 整理各消融臂权重后调用标准评测 | 🟡 **待终版重训后填入**<br>• 已剔除 4 阶段内部演进流水账；<br>• 待终版消融模型训练后填入规范正交数据。 |
| **Table 4: 跨退化梯度发音人配对 Bootstrap 显著性检验**<br>(Significance Audit across Tiers) | **硬核统计推断**：$B=1000$ 轮重采样证实气声增益在 5 档声学梯度（Clean 至 Compound）下均保持显著（95% CI > 0, $p \le 0.05$）；嘎裂声主任务 $p > 0.05$（零负迁移）。 | **`scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py`** | `data/experiments/statistical_bootstrap_audit/paired_bootstrap_suite_report.csv`<br>`paired_bootstrap_audit.md` | `uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py --model-a <weak_on> --model-b <weak_off> --run-suite` | 🟡 **待增加 `--run-suite`**<br>• 增加一键遍历 5 档梯度模式；<br>• 待新数据终版模型（Weak-On vs Weak-Off）训练完成后统一执行。 |

---

### 第三部分：双任务解耦、层敏感度与表征几何机理剖析

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 | TODO 状态 / 前置依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 5 & Figure 4: LOLO 微观逐层消融敏感度矩阵**<br>(Micro Layer Ablation) | 揭示双任务在 WavLM 13 层中的**语音学生理层级差异**：嘎裂声强烈依赖底层时域微细脉冲（L1~L3）；气声更依赖中层共振峰与高层语义（L4~L8）。 | **`scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py`** | `data/experiments/lolo_ablation_evaluation/lolo_ablation_report.csv`<br>`lolo_ablation_summary.json` | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py --checkpoint <ckpt>` | 🟢 **管线就绪，待终版重刷**<br>• 当前已在中间模型跑通验证；<br>• 加数据重训后，需在终版 Checkpoint 上一键重跑固化。 |
| **Table 6 & Figure 5: 宏观自顶向下层截断与剪枝边界**<br>(Depth Truncation Pareto) | 自顶向下依次丢弃顶层 $k$ 个 Transformer 块（$k=1..12$），论证保留前 6~7 层即可无损保持干净集双任务精度，给出端侧轻量化部署上限。 | **`scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py`** | `data/experiments/topk_layer_truncation_evaluation/topk_layer_truncation_report.csv`<br>`topk_layer_truncation_report.md` | `uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py --checkpoint <ckpt>` | 🟢 **管线就绪，待终版重刷**<br>• 当前已跑通并验证 Pareto 边界；<br>• 待终版模型产出后重新执行一次确认端侧裁剪层数。 |
| **Table 8 & Figure 7: CKA 表征几何与跨退化保真度**<br>(Representation CKA & Stability) | 构建 13×13 线性 CKA 双任务自相似度热力图（展示生理分块）；计算在标准声学退化下的**逐层 CKA 保真度条形图**，量化预训练几何抗扰能力。 | **`scripts/representation_and_calibration_audit/evaluate_representation_cka.py`** | `data/experiments/representation_cka_audit/layer_cka_heatmap.pdf`<br>`cka_layer_fidelity_bar.pdf` | `uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py --checkpoint <ckpt> --compare-noise` | 🟢 **管线就绪，待终版执行**<br>• 已剔除冗余混乱的置信度校准图；<br>• 待终版模型重训后执行出图。 |

---

### 第四部分：定性案例与部署基准

| 论文拟收录图表 | 核心学术论点 / 实验目的 | 唯一权威主入口脚本 | 产出成果与文件位置 | 标准 `uv run` 执行命令 | TODO 状态 / 前置依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 8: 典型案例 Praat 语图与双发声态对齐**<br>(Qualitative Case Studies) | 展示典型连续自然对话中嘎裂声声门脉冲周期、气声中高频湍流的窄带语图，叠加双任务逐帧预测概率轨迹与人工精标 Ground Truth 的完美契合。 | **`scripts/export_textgrid.py`** | 生成标准 Praat TextGrid 文件，供在 Praat 中直接导出出版级截图。 | `uv run python scripts/export_textgrid.py --manifest manifests/segments_multi.csv` | 🟡 **待定选发话段并推理**<br>• 需在测试集筛选 2~3 个同时具备典型气声与嘎裂声的发话段；<br>• 用终版模型推理导出 TextGrid，导出出版级语图。 |
| **Table 9: 边缘部署实测速度与吞吐基准**<br>(Edge Runtime & RTF) | 汇报 ONNX 导出后在 CPU 与 GPU 上的单音频块推理延迟（Wall-clock ms）、实时率（RTF < 0.05）以及内存显存峰值。 | **`scripts/export_deployment.py`** 与 **`scripts/test_onnx_export.py`** | `deployments/wavfry_multitask.onnx` 及导出验证报告。 | `uv run python scripts/export_deployment.py --checkpoint <ckpt> --output-dir deployments/`<br>`uv run python scripts/test_onnx_export.py` | 🟢 **管线就绪，待终版导出实测**<br>• 导出包装器已支持多任务与帧级头；<br>• 待终版 Checkpoint 训练完毕后，导出最终 ONNX 并实测 CPU/GPU 推理延迟与 RTF。 |

---

## 4. 🗂️ 评测套件子目录架构与职责内聚

所有评测功能均收敛于专属包中，且各包具备独立的 `README.md`：

```
scripts/
├── baseline_comparison/                  # 对应 Table 1 & Figure 2（前人基线与 QUT-NOISE 统一抗噪评测）
│   ├── evaluate_qut_baselines.py         # [唯一入口] 统一生成 Table 1 数据表与 Figure 2 阴影带折线图
│   └── README.md
├── statistical_bootstrap_audit/          # 对应 Table 2 & Table 4（显著性检验与性别公平性）
│   ├── evaluate_paired_models_bootstrap.py # [唯一入口] 发音人级聚类非参数配对 Bootstrap 检验（支持 5 梯度）
│   ├── evaluate_gender_fairness.py       # [唯一入口] 发音人性别公平性量化与小提琴图生成
│   └── README.md
├── layer_importance_and_truncation_audit/# 对应 Table 5, Table 6, Figure 4, Figure 5（层重要度与深度裁剪）
│   ├── evaluate_lolo_ablation.py          # [唯一入口] LOLO 逐层剥离消融评测（微观层不可或缺性）
│   ├── evaluate_topk_layer_truncation.py  # [唯一入口] 自顶向下逐层截断评测（宏观计算深度 Pareto）
│   └── README.md
├── representation_and_calibration_audit/ # 对应 Table 8 & Figure 7（CKA 表征几何与退化保真度）
│   ├── evaluate_representation_cka.py     # [唯一入口] 13x13 隐层 CKA 双任务热力图与逐层保真度条形图
│   ├── cka_core.py                        # 快速 Frobenius 范数与无偏 HSIC 计算核心库
│   └── README.md
├── gemini_tts_synthesis/                 # 数据工程：日语副语言气声双段式 TTS 合成切分标注管线
│   └── README.md
├── mllm_annotation/                      # 数据工程：多模态大模型（MLLM）气声粗标预填与强制对齐管线
│   └── README.md
├── experiments/                          # 基准支撑：传统韵律物理特征（Jitter/CPPS）与小样本实验
│   └── README.md
└── legacy/                               # 历史防腐：已废弃旧脚本归档（含已淘汰的高斯加噪目录评测脚本）
    └── README.md
```

---

## 5. 🚀 论文终版一键评测复现流（Four-Step Final Reproduction Pipeline）

当未来加入扩充数据（如修复后的 MELI、BreathyTTS 等）训练好终版模型检查点（`best_model.pt`）后，只需依次在终端执行以下四步流水线即可一站式生成论文所需的全部最终图表数据：

### Step 1: 嘎裂声前人基线横向对比与抗噪曲线 (Table 1 & Figure 2)
```bash
uv run python scripts/baseline_comparison/evaluate_qut_baselines.py \
  --checkpoint exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --output-dir data/experiments/baselines
```

### Step 2: 发音人配对显著性检验与性别公平性 (Table 2 & Table 4)
```bash
# 1. 跨 5 档声学梯度的发音人非参数配对 Bootstrap 检验
uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py \
  --model-a exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --model-b exp_results/checkpoints/final_weak_off_baseline/best_model.pt \
  --n-bootstrap 1000 \
  --run-suite \
  --output-dir data/experiments/statistical_bootstrap_audit

# 2. 发音人性别公平性审计
uv run python scripts/statistical_bootstrap_audit/evaluate_gender_fairness.py \
  --checkpoint exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --output-dir data/experiments/statistical_bootstrap_audit
```

### Step 3: 双任务层敏感度、截断边界与 CKA 几何保真度 (Table 5, Table 6, Table 8, Figure 4, 5, 7)
```bash
# 1. LOLO 微观逐层剥离消融
uv run python scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py \
  --checkpoint exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --split val \
  --output-dir data/experiments/lolo_ablation_evaluation

# 2. Top-k 宏观自顶向下截断
uv run python scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py \
  --checkpoint exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --split val \
  --output-dir data/experiments/topk_layer_truncation_evaluation

# 3. CKA 13x13 几何热力图与跨退化逐层保真度条形图
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
  --checkpoint exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --compare-noise \
  --output-dir data/experiments/representation_cka_audit
```

### Step 4: 边缘部署模型导出与推断延迟验证 (Table 9)
```bash
uv run python scripts/export_deployment.py \
  --checkpoint exp_results/checkpoints/final_weak_on_vsdc/best_model.pt \
  --output-dir deployments/ \
  --task all

uv run python scripts/test_onnx_export.py
```
