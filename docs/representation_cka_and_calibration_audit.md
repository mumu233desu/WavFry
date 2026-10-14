# WavFry 表征相似度（CKA）与置信度校准误差（ECE）审计技术规范

> **版本**: 1.0.0  
> **更新日期**: 2026-10-14  
> **归属工程**: WavFry (06_wavfry)  
> **文件存放规范**: 核心工具集中归档于 `scripts/representation_and_calibration_audit/`，文档集中归档于 `docs/`。

---

## 1. 概述与理论背景

在语音副语言表征学习（嘎裂声 Creaky Voice、气声 Breathy Voice）任务中，WavLM 骨干网络的多层隐层表征具有丰富的分层抽象特征。为了在不依赖线性分类头探针（Linear Probing）的前提下深入度量模型的内部几何结构演变与声学物理退化稳定性，同时在域偏移（Domain Shift）条件下严谨评估模型预测置信度与真实发生率之间的对齐程度，本规范制定并实现了两大评估工具体系：

1. **Centered Kernel Alignment (CKA) 表征几何相似度审计体系**：
   - 基于 Kornblith et al. (ICML 2019) 与 Cortes et al. (JMLR 2012) 的核对齐理论。
   - 度量 13 层（CNN 编码层 $L_0$ + 12 层 Transformer）两两之间的特征流相似度矩阵（$13 \times 13$ Heatmap）。
   - 在真实声学环境（QUT-NOISE 真实噪声、房间冲激响应 RIR 混响）干扰下，度量跨条件几何表征稳定性 $\text{CKA}(H_{\text{clean}}^{(l)}, H_{\text{degraded}}^{(l)})$。

2. **Expected Calibration Error (ECE / Reliability Diagram) 置信度校准审计体系**：
   - 基于 Guo et al. (ICML 2017) 与 Niculescu-Mizil & Caruana (ICML 2005) 的置信度校准理论。
   - 评估模型预测概率 $\hat{p}$ 与真实正例率 $y$ 的经验期望校准误差（ECE）、最大校准误差（MCE）与 Brier Score。
   - 自动生成出版级双面板（Two-Panel）Reliability Diagram，并支持干净域与噪声/混响域之间的置信度漂移压测对比。

---

## 2. 核心数学原理与算法优化

### 2.1 线性 CKA（Linear Centered Kernel Alignment）

给定两组隐层特征矩阵 $X \in \mathbb{R}^{N \times D_1}$ 和 $Y \in \mathbb{R}^{N \times D_2}$，其核矩阵分别为 $K = X X^T$ 和 $L = Y Y^T$。
定义中心化矩阵 $H = I_N - \frac{1}{N} \mathbf{1} \mathbf{1}^T$，则中心化 Gram 矩阵为 $\tilde{K} = H K H$ 与 $\tilde{L} = H L H$。
Hilbert-Schmidt 独立性准则（HSIC）定义为：
$$\text{HSIC}(K, L) = \frac{1}{(N-1)^2} \operatorname{tr}(\tilde{K} \tilde{L})$$

Centered Kernel Alignment (CKA) 即归一化的 HSIC：
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}} \in [0, 1]$$

#### 算法计算优化（大样本 $O(N D_1 D_2)$ vs 传统 $O(N^2)$）：
在语音长音频序列中，总帧数 $N$ 通常可达数万乃至数十万。直接计算 $N \times N$ 的 Gram 矩阵会导致内存暴涨（RAM/VRAM OOM）。
本工具实现了中心化特征矩阵的 Frobenius 范数等价变换：
1. 对特征列直接均值归零：$\tilde{X} = X - \frac{1}{N} \mathbf{1} (\mathbf{1}^T X)$，$\tilde{Y} = Y - \frac{1}{N} \mathbf{1} (\mathbf{1}^T Y)$。
2. 利用矩阵迹的循环性质：
   $$\operatorname{tr}(\tilde{K} \tilde{L}) = \|\tilde{Y}^T \tilde{X}\|_F^2$$
3. 空间复杂度从 $O(N^2)$ 降低为 $O(D_1 D_2)$，时间复杂度为 $O(N \cdot D_1 \cdot D_2)$，毫秒级即可完成数十万样本的高精度 CKA 计算。
4. 针对小样本严谨科研场景，亦提供可选的 `--unbiased` 有限样本无偏估计器（Song et al., 2012）。

### 2.2 期望校准误差（ECE）与 Reliability Diagram

将模型预测概率区间 $[0, 1]$ 划分为 $M$ 个等宽区间（Bins）$B_m = (\frac{m-1}{M}, \frac{m}{M}]$。
对于每个区间 $B_m$：
- 经验置信度（Mean Confidence）：$\operatorname{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$
- 经验准确率（Empirical Accuracy / Positive Rate）：$\operatorname{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$
- 校准差（Calibration Gap）：$\operatorname{gap}(B_m) = |\operatorname{acc}(B_m) - \operatorname{conf}(B_m)|$

核心统计指标定义为：
- **Expected Calibration Error (ECE)**：
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \operatorname{gap}(B_m)$$
- **Maximum Calibration Error (MCE)**：
  $$\text{MCE} = \max_{m=1, \dots, M, |B_m| > 0} \operatorname{gap}(B_m)$$
- **Brier Score**：
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

#### 评测区间 Logit 平均聚合机制：
与统一评估流水线（`Trainer.validate()`、`wavfry/eval/unified.py`）严格保持一致：在连续音频切片滑窗预测中，跨块重叠的多帧由其 `interval_id` 唯一索引。工具首先在 Logit 空间进行算术平均（Logit-averaging），再通过 Sigmoid 转换为最终样本级置信度 $\hat{p} \in [0, 1]$，彻底杜绝数据样本虚假膨胀（Inflation）。

---

## 3. 代码结构与工程实现

所有实现代码均集中在 `scripts/representation_and_calibration_audit/` 目录下，禁止散落：

```
scripts/representation_and_calibration_audit/
├── __init__.py                                 # 导出核心函数与模块
├── cka_core.py                                 # CKA/HSIC 数学核心算法与稳定性分析
├── calibration_core.py                         # ECE/MCE/Brier 计算与双面板图表渲染
├── evaluate_representation_cka.py              # CKA 评估与热力图生成 CLI 工具
└── evaluate_confidence_calibration.py         # 校准评估与域偏移压测 CLI 工具
```

### 严格的 Fail-Fast 检查点策略（Zero Silent Fallback）
遵循工程准则，两个 CLI 脚本均实施**零静默回退（Fail-Fast）**机制：
- `--checkpoint` 参数为强制必填项；
- 脚本启动时立即使用 `Path.is_file()` 校验检查点物理存在性；
- 若检查点不存在或路径非法，立即抛出 `FileNotFoundError` 并中断退出，严禁任何猜测、自动寻找或默认回退逻辑。

---

## 4. CLI 工具使用指南

> **运行环境强制约束**：必须严格通过 `uv run` 执行！禁止裸 `python`。

### 4.1 CKA 表征相似度评估 (`evaluate_representation_cka.py`)

#### 快速烟雾测试（验证管线顺畅）：
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
    --checkpoint exp_results/checkpoints/ablation_weak_on_vsdc_full/best_model.pt \
    --split val \
    --task creaky \
    --quick
```

#### 全量生产评估（带真实噪声与混响稳定性压测）：
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
    --checkpoint exp_results/checkpoints/ablation_weak_on_vsdc_full/best_model.pt \
    --split test \
    --task creaky \
    --device cuda \
    --noise-snr-db 5.0 \
    --output-dir exp_results/audit_reports/cka_analysis/
```

#### 关键参数说明：
- `--checkpoint` (必需): 模型权重路径（如 `best_model.pt`）。
- `--config` (可选): 指定配置文件，默认使用检查点同目录配置或 `config/_defaults.yaml`。
- `--split` (可选): 评估数据集切分，`val`（默认）或 `test`。
- `--task` (可选): 目标任务，`creaky`（默认）或 `breathy`。
- `--unbiased` (可选): 启用有限样本无偏 HSIC 估计器。
- `--quick` (可选): 快速抽样测试模式（仅取前 5 个 batch）。
- `--output-dir` (可选): 审计图表与报告保存目录（默认为 `exp_results/audit_reports/representation_cka/`）。

---

### 4.2 置信度校准误差评估 (`evaluate_confidence_calibration.py`)

#### 基础校准度评估（单面板 Reliability Diagram）：
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
    --checkpoint exp_results/checkpoints/ablation_weak_on_vsdc_full/best_model.pt \
    --split val \
    --task creaky
```

#### 跨域偏移校准压测（Domain Shift Stress Test）：
```bash
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
    --checkpoint exp_results/checkpoints/ablation_weak_on_vsdc_full/best_model.pt \
    --split test \
    --task creaky \
    --test-domain-shift \
    --device cuda \
    --output-dir exp_results/audit_reports/calibration_analysis/
```

#### 关键参数说明：
- `--checkpoint` (必需): 模型权重路径。
- `--split` (可选): 数据切分，`val` 或 `test`。
- `--task` (可选): 评估任务，`creaky` 或 `breathy`。
- `--num-bins` (可选): 校准分箱数，默认 10。
- `--test-domain-shift` (可选): 自动压测 5 种声学域（Clean、Cafe 0dB、Car 0dB、Cafe 10dB、Reverb RIR），并输出对比漂移曲线。
- `--output-dir` (可选): 结果保存目录。

---

## 5. 产出文件规范

评估完成后，审计工具将在 `--output-dir` 目录下输出期刊发表级别的结构化产物：

### 5.1 CKA 输出文件
- `cka_similarity_matrix.pdf` / `.png`: 13×13 隐层表征自相似度热力图（Publication-grade Seaborn Whitegrid, 300 DPI）。
- `cka_stability_curves.pdf` / `.png`: 声学退化下的逐层表征几何漂移曲线。
- `cka_audit_metrics.json`: 结构化数值记录（层间自相似度矩阵、跨条件退化 CKA 值）。
- `cka_audit_summary.md`: 自动生成的 Markdown 格式实验分析报告，便于直接引用。

### 5.2 置信度校准输出文件
- `reliability_diagram_{domain}.pdf` / `.png`: 双面板可靠性图表（上：经验准确率与 Gap 梯级，下：预测置信度样本频数分布）。
- `domain_shift_calibration_curves.pdf` / `.png`: 多声学域置信度漂移对比曲线。
- `calibration_metrics_{domain}.csv`: 每个分箱的详细样本数、平均置信度、实际正例率与 Gap。
- `calibration_audit_summary.md`: ECE、MCE、Brier Score 的量化汇总表格与分析。

---

## 6. 自动化测试体系

本工具配备完备的单元测试，已全部集成至 `tests/`：
- `tests/test_representation_cka.py` (9 passed):
  - 矩阵自对齐（$X = X \implies \text{CKA} = 1.0$）
  - 正交旋转不变性（$Y = X R, R^T R = I \implies \text{CKA} = 1.0$）
  - 各向同性缩放不变性（$Y = \alpha X \implies \text{CKA} = 1.0$）
  - 独立高维随机表征极小相似度（$\text{CKA} < 0.05$）
  - 多层矩阵对称性与对角线约束
  - 跨条件衰减计算
- `tests/test_calibration.py` (8 passed):
  - 完美校准（$\text{ECE} = 0.0$）
  - 极度过度自信（Overconfident $\hat{p} = 0.99, y = 0 \implies \text{ECE} \approx 0.99$）
  - 极度缺乏自信（Underconfident $\hat{p} = 0.02, y = 1 \implies \text{ECE} \approx 0.98$）
  - 极端 Brier Score 检验
  - 空箱（Empty Bins）与稀疏区间零除防护
  - 出版级图表渲染与文件系统输出验证

测试命令：
```bash
uv run pytest tests/test_representation_cka.py tests/test_calibration.py -v
```
