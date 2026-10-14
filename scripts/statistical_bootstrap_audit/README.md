# Statistical Bootstrap Audit (统计显著性检验工具包)

本目录为 WavFry 论文全周期实验提供发音人级非参数聚类 Bootstrap（Speaker-Level Cluster Bootstrap, $B=1000$）显著性推断工具。

---

## 核心脚本与职责

| 脚本 | 论文对应图表 | 核心功能 | 运行方式 |
| :--- | :--- | :--- | :--- |
| **`evaluate_paired_models_bootstrap.py`**<br>**(推荐核心入口)** | **Table 1 / Table 2 / 消融对比主表** | **双模型配对显著性检验**：输入任意两个模型 Checkpoint（如 Weak-On vs Weak-Off，或 Baseline vs WavFry），在未见发音人上进行 1000 轮配对重采样，输出 95% 置信区间、经验 $p$ 值（Davison & Hinkley 1997 连续性平滑）与发音人胜率细分。支持 `--noise-condition`。 | `uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py --model-a <path> --model-b <path>` |
| `evaluate_robustness_and_bootstrap.py`<br>*(历史单模型评测)* | 历史单模型截断表 | 针对单个 Checkpoint 测试不同保留层数。其核心功能大部分已迁移至更高效的 `scripts/layer_importance_and_truncation_audit/`。 | 仅供单模型特定基准复现 |

---

## 典型论文复现工作流

### 1. 检验弱监督对气声与嘎裂声的统计显著性
```powershell
uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py `
  --model-a R:/checkpoints/ablation_weak_on_vsdc_real_noise/best_model.pt `
  --model-b R:/checkpoints/ablation_weak_off_real_noise/best_model.pt `
  --label-a "Weak-On (VSDC)" `
  --label-b "Weak-Off (Baseline)" `
  --noise-condition clean `
  --device cuda
```

### 2. 在严苛真实噪声（Cafe 5dB）下检验显著性
```powershell
uv run python scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py `
  --model-a R:/checkpoints/ablation_weak_on_vsdc_real_noise/best_model.pt `
  --model-b R:/checkpoints/ablation_weak_off_real_noise/best_model.pt `
  --noise-condition cafe_5db `
  --device cuda
```
