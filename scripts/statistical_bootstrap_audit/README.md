# Statistical Bootstrap Audit (统计显著性检验工具包)

本目录为 WavFry 论文全周期实验提供发音人级非参数聚类 Bootstrap（Speaker-Level Cluster Bootstrap, $B=1000$）显著性推断工具。

---

## 核心脚本与职责

- **`evaluate_paired_models_bootstrap.py` (核心通用入口)**：
  - **服务论文**：Table 1 / Table 2 / 消融对比主表、噪声鲁棒性对比表。
  - **功能**：输入任意两个模型 Checkpoint（如 Weak-On vs Weak-Off、Baseline vs WavFry、含/不含某一架构模块），在未见发音人上进行 1000 轮配对整群重采样，输出 95% 置信区间、经验 $p$ 值（Davison & Hinkley 1997 连续性平滑）与发音人胜率细分。
  - **环境支持**：原生支持 `--noise-condition`（如 `clean`, `cafe_5db`, `car_5db`, `reverb` 等），通过动态退化引擎完成严苛环境下的统计学推断。

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
