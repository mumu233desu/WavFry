# Representation & Calibration Audit Suite (`scripts/representation_and_calibration_audit/`)

本目录收敛了对 WavFry 模型在不同声学退化环境下的**表征几何结构（Centered Kernel Alignment, CKA）**与**预测置信度校准（Confidence Calibration & Reliability Diagrams）**的学术级审计套件。

对应论文/技术报告中的 **声学退化下的表征不变性（Representation Invariance under Acoustic Degradation）** 与 **模型预测不确定性与置信度校准（Uncertainty & Confidence Calibration）** 章节。

---

## 目录模块与脚本职能

| 模块 / 脚本 | 论文对应图表 | 核心职能 | 输出形式 |
| :--- | :--- | :--- | :--- |
| **`evaluate_confidence_calibration.py`** | **Figure 6 / Table 5**<br>(校准可靠性图与退化 ECE 表) | **置信度校准与范式 A 评估**<br>计算期望校准误差（ECE）、最大校准误差（MCE）、Brier 分数，绘制高精度两栏式可靠性图，并在真实环境噪声与混响下进行单调衰减分析。 | - `reliability_diagram_{task}_{domain}.pdf / .png`<br>- `paradigm_a_factorial_matrix_{task}.csv`<br>- `snr_ece_decay_{task}.pdf`<br>- `calibration_audit_report.md` |
| **`evaluate_representation_cka.py`** | **Figure 7 / Table 6**<br>(层级 CKA 热力图与退化稳定性) | **线性 CKA 表征相似度审计**<br>构建 13×13 层级 CKA 相似度矩阵，剖析 SSL 骨干内部特征的语义演进；并提取 Clean vs. 退化声学环境下的同一层表征相似度，量化声学抗扰稳定性。 | - `layer_cka_heatmap_{task}.pdf / .png`<br>- `cka_cross_condition_stability.pdf`<br>- `layer_cka_matrix_{task}.json`<br>- `cka_audit_report.md` |
| **`calibration_core.py`** | 核心计算库 | **校准计算与绘图引擎**<br>纯函数实现，提供分箱 ECE/MCE/Brier 计算、自适应空箱处理、可靠性图绘制与范式 A 梯次汇总格式化。 | 供主脚本及单元测试无缝复用 |
| **`cka_core.py`** | 核心计算库 | **CKA 与 HSIC 矩阵计算库**<br>提供快速中心化 Frobenius 范数法（$O(N \cdot D^2)$）与严格无偏估计量（Debiased HSIC, $O(N^2)$），确保数学无偏性。 | 供主脚本及单元测试无缝复用 |

---

## 范式 A（Paradigm A）声学环境矩阵

校准评估原生支持**范式 A 完整因子矩阵**（3 真实场景 × 3 档 SNR + 3 个基准对照，共 12 种条件）：
1. **纯净对照组**：`clean`
2. **咖啡馆人声干扰**：`cafe_15db`, `cafe_5db`, `cafe_0db`
3. **车载低频轰鸣**：`car_15db`, `car_5db`, `car_0db`
4. **街头突发噪声**：`street_15db`, `street_5db`, `street_0db`
5. **真实房间混响**：`reverb` (8 组真实物理空间 RIR 卷积)
6. **复合极限退化**：`reverb_cafe_5db` (空间混响 + 5dB 真实人声交谈)

---

## 核心执行命令与使用范例

### 1. 置信度校准评估 (`evaluate_confidence_calibration.py`)

```bash
# 1. 快速冒烟测试（300 个样本验证绘图与表格导出）
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --quick

# 2. 全量验证集标准 12 场景范式 A 审计
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --paradigm-a \
  --output-dir data/experiments/confidence_calibration_audit/weak_on_vsdc

# 3. 指定特定声学场景
uv run python scripts/representation_and_calibration_audit/evaluate_confidence_calibration.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --eval-domains clean cafe_5db car_5db reverb
```

### 2. CKA 表征相似度与退化稳定性 (`evaluate_representation_cka.py`)

```bash
# 1. 计算 13x13 层级 CKA 矩阵与热力图
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --split val \
  --output-dir data/experiments/representation_cka_audit

# 2. 结合真实噪声进行跨声学域稳定性对比 (Clean vs. Cafe 5dB)
uv run python scripts/representation_and_calibration_audit/evaluate_representation_cka.py \
  --checkpoint R:\checkpoints\ablation_weak_on_vsdc_real_noise\best_model.pt \
  --compare-noise \
  --noise-condition cafe_5db \
  --output-dir data/experiments/representation_cka_audit_cafe5db
```

---

## 单元测试保障

本模块所有数学指标与绘图函数均已配备完备的自动化单元测试：
```bash
uv run pytest tests/test_calibration.py tests/test_representation_cka.py -v
```
覆盖分箱边界防御、零样本空箱保护、HSIC 无偏估计量检验与图表输出断言。
