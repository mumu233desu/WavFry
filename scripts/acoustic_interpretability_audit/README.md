# Acoustic Interpretability Audit Module (`scripts/acoustic_interpretability_audit/`)

本模块包含面向期刊（Elsevier *Computer Speech & Language* / *Journal of Phonetics*）的**权威声学-物理可解释性验证流水线**，对应论文中的 **Table 8b**（生理声学特征相关性表）与 **Figure 8**（连续后验概率 vs. 物理声学特征散点回归与置信带分布图）。

---

## 1. 核心职责与架构划分（Decoupled Pipeline Architecture）

本模块严格遵循工程治理中的**单一权威入口原则**与**数据/绘图解耦原则**，包含两个职能明确的权威脚本：

```
scripts/acoustic_interpretability_audit/
├── evaluate_acoustic_correlations.py   # [声学数据生产] CS-Dialogue 元音特征、后验匹配、相关性统计
├── plot_acoustic_scatter_regressions.py # [声学绘图] Figure 8 出版级 4 面板散点回归与 95% 置信区间
├── evaluate_egg_correlations.py        # [EGG 数据生产] AVID 真实电子声门图 DECOM 开商/接触商相关性审计
├── plot_egg_correlations.py            # [EGG 绘图] Figure 9 出版级 4 面板 EGG 物理效度验证图
└── README.md                           # 本文档
```

### 1.1 `evaluate_acoustic_correlations.py`（数据生产主入口）
- **职责**：严格**只产出数据**，不包含 GUI/绘图依赖。
- **机制**：
  - 对接 `wavfry.eval.cache_manager.IncrementalCSVCache`，对测试集元音区间进行 Parselmouth / Praat 物理特征提取并缓存，避免重复跑昂贵的声学校验；
  - 提取传统黄金标准物理声学参数：
    - **嘎裂声（Creaky Voice）**：次谐波谐波比（SHR）、局部基频抖动（Jitter local）、局部振幅微扰（Shimmer local）、共振峰修正谐波差（$H_1^*-H_2^*$，负斜率）；
    - **气声（Breathy Voice）**：平滑倒谱峰突显度（CPPS）、谐噪比（HNR）、共振峰修正谐波差（$H_1^*-H_2^*$，正斜率）；
  - 修复历史时间轴漂移漏洞（采用 `preserve_times=False` 确保子片段局部基频追踪精确有效）；
  - 计算双尾检验的 Pearson 相关系数 $r$、Spearman 秩相关系数 $\rho$ 及其显著性 $p$ 值。
- **产出成果**：
  - `data/experiments/interpretability/table8b_acoustic_correlations.csv` & `.md`
  - `data/experiments/interpretability/acoustic_correlations_metrics.json`
  - `data/experiments/interpretability/acoustic_correlations.csv`（包含每个有效元音的后验概率与对应声学物理值）

### 1.2 `plot_acoustic_scatter_regressions.py`（出版级绘图主入口）
- **职责**：纯离线读取上述结构化数据，渲染符合 Elsevier 期刊审稿标准的矢量与栅格图版。
- **特点**：
  - 四面板科学图布局：
    - (a) Creaky Posterior vs. Jitter (local)
    - (b) Creaky Posterior vs. Shimmer (local)
    - (c) Breathy Posterior vs. CPPS (dB)
    - (d) Breathy Posterior vs. Formant-Corrected $H_1^*-H_2^*$ (dB)
  - 自动绘制带有 **95% Bootstrap 置信区间带（95% CI Bands）** 的线性回归拟合线；
  - 标注精准统计检验量：Pearson $r$、Spearman $\rho$ 与 $p$ 值（$p < 0.001$）；
  - 自动剔除极端噪点（99分位数微调）保证出版视觉清晰度。

---

## 2. 无声区间与清音区间的处理规范（Silence & Unvoiced Phonation Invariant）

### 2.1 语音学生理学第一性原理
发声态（Phonation Types，如常态、嘎裂、气声）在生理学与物理学上**严格依托于声带的周期性张弛振动**。
在语流中的**无声停顿（Silence/Pause）**或**清辅音闭合/摩擦段（Voiceless Consonants，如 /s/, /t/, /k/, /p/ 等）**，声带处于未振动状态，**在物理上根本不存在发声态的概念**。因此，无声区间绝不应该被纳入发声态的评估、相关性拟合或散点可视化。

### 2.2 工程实现方案：TextGrid 元音掩码（TextGrid Vowel Masking）
针对学术对话语料库（CS-Dialogue、AVID 等）具备强时间对齐文本网格（Forced-Aligned TextGrid）的特点：
1. **直接基于 TextGrid 做元音掩码**：
   - 在评测入口（`wavfry/eval/unified.py:build_test_manifest`）和推理模块（`scripts/predict.py`）中，直接按 TextGrid 的音素层过滤 `phone_type == "vowel"`（或匹配标准音标元音集）；
   - 静音段（`sil`, `sp`, `""`）和清辅音区间在数据载入阶段就被 100% 隔离剔除，不传入发声态评估链路；
   - **无需引入外置复杂 VAD**，既杜绝了额外模型依赖，又避免了传统 VAD 能量阈值误伤微弱气声（Breathy）或低沉嘎裂声（Vocal Fry）的缺陷。
2. **声学物理端双重防御（Acoustic Defense-in-Depth）**：
   - 考虑强制对齐在边界处可能存在 10~20ms 的轻微抖动，在 `evaluate_acoustic_correlations.py` 的物理特征提取器中设置了底层安全过滤：
     - 若切片 RMS 能量 $< 10^{-4}$，直接判定为数字静音/环境底噪，返回 `NaN`；
     - 若 Praat 基频检测有效周期比例 $< 20\%$，判定为清音或未起振段，返回 `NaN`；
   - 绘图脚本 `plot_acoustic_scatter_regressions.py` 严格执行 `dropna(subset=[x, y])`，所有无声/无效段被绝对排除在散点图与回归线之外。

---

## 3. 标准运行命令（Standard Execution Commands）

> **规则要求**：必须在项目虚拟环境下使用 `uv run` 执行。

### 步骤 1：运行声学相关性评估并生成数据
```bash
# 模式 A：从训练好的 Checkpoint 运行（自动提取特征并缓存）
uv run python scripts/acoustic_interpretability_audit/evaluate_acoustic_correlations.py \
  --checkpoint exp_results/checkpoints/best_model.pt \
  --manifest manifests/segments_multi.csv \
  --output-dir data/experiments/interpretability

# 模式 B：秒级离线复现（基于已导出的统一测试集预测结果）
uv run python scripts/acoustic_interpretability_audit/evaluate_acoustic_correlations.py \
  --preds-csv data/experiments/cache/preds_tiered.csv \
  --manifest manifests/segments_multi.csv \
  --output-dir data/experiments/interpretability
```

### 步骤 2：渲染出版级四面板散点回归图（Figure 8）
```bash
uv run python scripts/acoustic_interpretability_audit/plot_acoustic_scatter_regressions.py \
  --input-csv data/experiments/interpretability/acoustic_correlations.csv \
  --output-dir data/visualizations/output \
  --format both \
  --dpi 300
```
产出物将自动保存至：
- `data/visualizations/output/fig8_acoustic_physical_scatter_regression.png`
- `data/visualizations/output/fig8_acoustic_physical_scatter_regression.pdf`

---

## 4. 声门生理物理效度与 EGG 外部验证（Glottal Physical Validity & Figure 9）

为了彻底摆脱“感知主观标注循环自证（Perceptual Circularity）”，本模块引入 AVID 双通道（Speech + EGG 同步记录）语料库，通过电子声门图（Electroglottography, EGG）客观提取声带开闭生理指标，直接验证 WavFry 模型后验概率与真实声门物理状态的外部效度（External Physiological Validity）。

### 4.1 核心算法与整段长录音直接提取机制
1. **直接提取原始双通道长录音（`E:\Corpus\EGG_RAW\AVID\Repository 1`）**：
   - 针对 50 位发音人的原始 44.1kHz 完整连续录音（`Spk1_1406.wav` ~ `Spk50_2806.wav`，每段约 15~20 分钟）；
   - **首部 94dB 纯音标定信号剔除**：自动跳过前 5.0 秒（`skip_lead_s = 5.0`），彻底隔离标定音对生理声门周期的污染；
   - **分块滑动推理防显存溢出（Chunked GPU Inference）**：对 16kHz 语音流采用 30 秒滑动窗口前向推理，在时间维度无缝拼接帧级后验，彻底规避 WavLM 自注意力矩阵 $O(T^2)$ 带来的 CUDA OOM 陷阱。
2. **DECOM 声门接触商/开商提取**：
   - 算法采用国际金标准 **Henrich et al. (2004) DECOM**（双通道微商极值检测法），依托 GNU Octave + COVAREP 工具箱；
   - 提取开商（Open Quotient, $O_q$）与接触商（Contact Quotient, $CQ = 1 - O_q$）；
   - 生理基准门控保护：$50\text{ Hz} < F_0 < 800\text{ Hz}$，物理有效开商区间 $0.20 \le O_q \le 0.95$。
3. **权威官方性别对齐（Ground-Truth Biological Sex Mapping）**：
   - 严格依据 AVID 官方元数据 `AVID/Repository 2/Metadata_with_labels_SENT.csv`，对齐 50 位发音人的生理性别（25 位男性、25 位女性，如 Spk1 为女性，Spk2~Spk8 为男性等），杜绝任何编号奇偶或前后分段的主观臆测。
4. **Figure 9 双面板科学图版布局**：
   - **面板 (a) 声门收缩物理连续谱（Glottal Constriction Spectrum）**：
     - 统一纵坐标：EGG 开商（$O_q \in [0.2, 0.9]$）；统一横坐标：模型后验概率（$0.0 \to 1.0$）；
     - 同图呈现气声（Teal 蓝绿向上斜率，反映声门闭合不全 $O_q \uparrow$）与嘎裂声（Coral 珊瑚红向下斜率，反映声带内收挤压 $CQ \uparrow \iff O_q \downarrow$）的对立运动学极点；
     - 附带 95% Bootstrap 置信区间带与生理机制标注。
   - **面板 (b) 全样本发音人相关性分布（Population-Wide Consistency, All 50 Speakers）**：
     - 展示全部 50 位发音人的 Pearson $r$ 分布（Violin + Boxplot + Jittered 说话人散点）；
     - 按男女真实生理性别区分散点颜色（男：Steel Blue `#457B9D`，女：Coral Rose `#E76F51`）；
     - 标注总体中位数、胜率（$r > 0$ 比例）及性别间高度一致性。

### 4.2 EGG 评测标准运行命令
```bash
# 步骤 1：运行 EGG 物理相关性评测（全量 50 发音人，8 线程并行加速 + 增量缓存）
uv run python scripts/acoustic_interpretability_audit/evaluate_egg_correlations.py \
  --all-speakers \
  --duration-per-speaker 120 \
  --num-workers 8 \
  --output-dir data/experiments/interpretability/egg_audit

# 步骤 2：渲染出版级双面板 EGG 物理效度验证图（Figure 9）
uv run python scripts/acoustic_interpretability_audit/plot_egg_correlations.py \
  --output-dir data/visualizations/output \
  --format both \
  --dpi 300
```
产出物将自动保存至：
- `data/visualizations/output/fig9_egg_physical_validation.png`
- `data/visualizations/output/fig9_egg_physical_validation.pdf`
- `data/experiments/interpretability/egg_audit/table_egg_correlations.md`
- `data/experiments/interpretability/egg_audit/egg_speaker_correlations.csv`
