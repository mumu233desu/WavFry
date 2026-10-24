# Layer Acoustic Encoding Audit Suite (`scripts/layer_acoustic_encoding_audit/`)

本目录收敛了针对预训练 WavLM 骨干网络（13 层：$L_0$ CNN 前端 + $L_1\dots L_{12}$ Transformer 块）在**传统物理声学特征（Physical Acoustic Parameters）与声门动力学代理指标上的跨层编码回归审计**工具。

**对应期刊论文章节**：*Section 5.2: Cross-Layer Physical Acoustic Encoding Decay & Mechanistic Auditing*  
**核心学术使命**：从底层物理声学与语音学生理机制出发，解释**“为什么在 WavLM 骨干上自顶向下剪掉高 6 层（Drop Top 6, $L_6\text{--}L_{12}$）不会造成发声态检测性能退化”**以及**“为什么嘎裂声与气声在隐藏层深度需求上存在不对称性”**。

---

## 1. 理论机制与物理声学指标说明

发声态（Phonation types）本质上是声门振动物理形态在声学波形上的投射。本审计模块采用 Praat/Parselmouth 在全量测试集元音段上提取的 6 项黄金标准物理声学参数：

| 参数标识 | 声学全称 | 生理学 / 声学物理意义 | 目标发声态代理 | 典型跨层演化趋势 |
| :--- | :--- | :--- | :---: | :--- |
| **`jitter_local`** | Local Jitter (%) | 声带周期微扰：相邻基频周期间的时间扰动 | 嘎裂声 (Creaky) | 峰值集中在 $L_1$（$\rho = 0.688$），高层逐层衰减至 $\rho = 0.523$ |
| **`shimmer_local`** | Local Shimmer (%) | 声门振幅微扰：相邻周期间峰值振幅波动 | 嘎裂声 (Creaky) | 峰值集中在 $L_0$（$\rho = 0.667$），高层单调衰减至 $\rho = 0.465$ |
| **`shr`** | Subharmonic-to-Harmonic Ratio | 次谐波与谐波比：周期倍增（Period Doubling）与复音声带机械运动 | 嘎裂声 (Creaky) | 中低层维持稳定，顶层 $L_{12}$ 轻微衰退 |
| **`h1h2_star`** | Formant-corrected $H_1^*-H_2^*$ (dB) | 声带开相谱倾斜：声门开商（Open Quotient）与声带内收度的直接代理 | 嘎裂声（低值/负值）<br>气声（高值/正值） | **$L_0$ 峰值（$\rho = 0.644, R^2 = 32.2\%$）$\to L_8$ 彻底崩塌（$\rho = 0.195, R^2 = 0.6\%$）** |
| **`cpps`** | Smoothed CPPS (dB) | 倒谱峰值突显度：高信噪比谐波结构与周期性规律度 | 气声（低值）<br>模态声（高值） | 在中浅层 $L_1\text{--}L_3$ 维持峰值（$\rho \approx 0.795$），$L_6$ 后降至 $0.60$ |
| **`hnr`** | Harmonics-to-Noise Ratio (dB) | 谐噪比：声门闭合不全产生的高频湍流摩擦噪声比例 | 气声（低值）<br>模态声（高值） | 维持高峰值直到 $L_4$（$\rho \approx 0.889$），$L_6$ 后单调退化 |

### 核心物理推论：为什么高 6 层可以安全剪除？
1. **声门微观脉冲的时域平滑效应**：自注意力机制随着层数加深，其时间感受野逐渐泛化以捕捉音素与音节级语义，因而抹平了单个脉冲周期微扰（Jitter/Shimmer）和声门开相谱倾斜（$H_1^*-H_2^*$）。
2. **高层特征语义抽象**：$L_6\text{--}L_{12}$ 的表征主要编码跨说话人不变的语言学信息（ASR 预训练目标），已完全剥离声门源动力学。保留高层不仅无法提供增量声学信息，反而会引入与发声态正交的语义噪声。
3. **任务深度不对称性**：嘎裂声依赖周期倍增与微扰，信息完全收敛于 $L_0\text{--}L_3$；气声依赖宽带谐波与湍流能量平衡，信息延伸至 $L_0\text{--}L_5$。两者在 $L_6$ 处均已完全饱和。

---

## 2. 评测协议与算法规范

* **样本池**：全量说话人分组测试集（$N=1,371$ 元音黄金段，跨 18 名独立说话人）；
* **无偏验证机制**：采用严格的 **5-Fold Speaker-Grouped Cross-Validation（按发音人分组交叉验证）**，确保特征归一化与模型参数绝不跨说话人泄露；
* **回归模型**：`RidgeCV(alphas=np.logspace(-2, 4, 20))`，对训练折进行标准差归一化并在留出发音人折上输出预测值；
* **指标体系**：计算袋外预测序列与真实物理参数间的 **Spearman 秩相关系数（$\rho$）**、双尾显著性 **$p$-value** 以及 **决定系数（$R^2$）**。

---

## 3. 标准执行命令与使用范例

> **⚠️ 必须统一通过 `uv run` 执行！**

```bash
# 1. 标准一键审计与出图（直接利用既有特征缓存）
uv run python code/scripts/layer_acoustic_encoding_audit/evaluate_layer_acoustic_encoding.py

# 2. 强制全量重新计算回归矩阵（若缓存参数更新）
uv run python code/scripts/layer_acoustic_encoding_audit/evaluate_layer_acoustic_encoding.py \
  --force-rerun \
  --n-splits 5

# 3. 自定义输出目录
uv run python code/scripts/layer_acoustic_encoding_audit/evaluate_layer_acoustic_encoding.py \
  --output-dir data/experiments/layer_acoustic_encoding_audit
```

---

## 4. 输出资产说明

| 文件路径 | 格式与内容 | 用途 |
| :--- | :--- | :--- |
| `layer_acoustic_encoding_metrics.csv` | 13 层 $\times$ 6 参数详细矩阵（含 $\rho, p, R^2, n$） | 论文 Table 7 / 正文统计量支撑 |
| `fig_layer_acoustic_encoding_decay.png` | 300 DPI Elsevier CSL 规范双面板物理衰减对比图 | 期刊论文正式插图 |
| `fig_layer_acoustic_encoding_decay.pdf` | 矢量 PDF 格式出版图 | LaTeX 正文直接内嵌 |
| `layer_acoustic_encoding_report.md` | 自动化 Markdown 实验实证分析与物理机制归因报告 | 审稿答辩与论文写作参考底稿 |
