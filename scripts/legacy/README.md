# Legacy Scripts（归档）

这些脚本自 2026-09 的配置体系重构（`90f8962` 删除 `data.annotated_speakers` /
`data.val_speakers` / `data.sampler`，`7ecdb6a`/`8aaaccd` 迁移标注存储路径，
`084cac4` 统一评测管线）后已失效。归档于此仅供历史协议参考，**不再维护**：
它们引用的配置键已从 YAML 中删除（部分会被 `--override` 键名校验直接
KeyError，部分静默读取空值），部分脚本的标注/评测路径也已失效。

| 脚本 | 原功能 | 失效原因 |
|:---|:---|:---|
| `evaluate_baselines.py` | 调用外部 `04_creaky_tools` 旧 pipeline 跑基线 | 引用已删除的 `data.annotated_speakers` + 外部旧路径 |
| `evaluate_setup_b_testset.py` | DeepFry/Creapy 在 LOOCV 发音人上评测出 md 报告 | 金标标注单体路径已迁移为 per-batch 存储（静默空转）；`data.annotated_speakers` 已删除。被 `eval_baselines_testset.py` 取代 |
| `evaluate_setup_b_noise.py` | 同上，噪声测试集版本 | 同上 + 已知缓存哈希缺陷（audit M-07） |
| `run_long_training.py` | 40 epoch 长训 CLI 包装 | 硬编码 8 speakers + 传已删除的 override（必崩） |
| `run_ablation_full.py` | 旧版全量消融（subprocess 调 train.py） | `data.sampler`/`data.annotated_speakers` 已删除；消融臂会被 train.py 静默忽略退化为与基线相同。被 `run_frozen_wavlm_ablation.py` 取代 |
| `run_ablation_smoketest.py` | 旧版消融冒烟 | 同上 |
| `run_ablation_fixed_2speakers.py` | 固定 2 发音人验证的消融 | `data.val_speakers` 已删除，固定验证协议名实不符 |
| `run_ablation_batch2_val.py` | Batch2 作验证集的消融 | 同上 |
| `run_ablation_temporal.py` | 时序类消融 | `data.annotated_speakers` 已删除 |
| `visualize_pos_encoder.py` | 可视化位置编码 | 硬编码旧 checkpoint 路径 |
| `generate_noisy_testset.py` | 离线生成高斯加噪测试集 | 被统一动态声学退化引擎 `wavfry/eval/acoustic_degradation.py` 取代，避免占用数 GB 离线磁盘且消除人造高斯白噪失真 |
| `generate_reverb_testset.py` | 离线生成固定混响测试集 | 同上，被动态声学退化引擎及 QUT-NOISE / RIR 卷积架构取代 |
| `eval_wavfry_testset.py` | 基于离线目录的 WavFry 测试集评测 | 被统一多任务与动态退化评测体系取代 |
| `eval_baselines_testset.py` | 基于离线目录的外部基线评测 | 依赖已弃用的离线加噪文件夹；正在逐步接入统一动态退化引擎 |

当前有效的替代入口：

- 训练：`scripts/train.py`（配置见 `config/`，消融用 `run_frozen_wavlm_ablation.py`）
- 声学退化引擎：`wavfry/eval/acoustic_degradation.py`（支持 3x3 真实 QUT-NOISE 与真实空间混响）
- 校准与 CKA 审计：`scripts/representation_and_calibration_audit/`
- 层重要性与深度裁剪：`scripts/layer_importance_and_truncation_audit/`
- 统计显著性 Bootstrap：`scripts/statistical_bootstrap_audit/`
- 韵律基线：`scripts/experiments/run_prosodic_baseline.py`
- Few-shot 基准：`scripts/experiments/run_fewshot_experiment.py`
