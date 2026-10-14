# WavFry 脚本库全景导航索引 (`scripts/`)

本目录汇总了 WavFry 发声态多任务学习系统的全生命周期脚本，涵盖**数据构建、模型训练、推断部署、论文图表审计评测与人机协同标注管线**。

> **⚠️ 核心执行规范**：
> 本工程专属底层音频与声学依赖环境由 `uv` 全权管理。**严禁直接使用全局裸 `python` 执行任何脚本，必须统一使用 `uv run`**：
> ```bash
> # ✅ 正确范式：
> uv run python scripts/train.py --config config/ablation_weak_on_vsdc.yaml
> uv run pytest tests/
> ```

---

## 1. 论文发表级审计与评测核心套件（子目录规范）

项目遵循**单一职责、高度内聚**的原则，将学术评测与实验验证划分为专门的子包，每个子包均配有详细的独立 `README.md` 与自动化测试：

```
scripts/
├── statistical_bootstrap_audit/          # [学术审计] 发音人级非参数聚类 Bootstrap 检验
│   ├── evaluate_paired_models_bootstrap.py # 唯一官方双模型配对 Bootstrap 显著性检验入口
│   └── README.md
├── layer_importance_and_truncation_audit/# [学术审计] WavLM 层重要度与深度裁剪敏感度分析
│   ├── evaluate_lolo_ablation.py          # LOLO 逐层剥离消融评测（微观不可或缺性）
│   ├── evaluate_topk_layer_truncation.py  # 自顶向下逐层截断评测（宏观有效计算深度）
│   └── README.md
├── representation_and_calibration_audit/ # [学术审计] 表征 CKA 几何相似度与置信度校准
│   ├── evaluate_confidence_calibration.py # 范式 A（12 场景）置信度校准与可靠性图生成
│   ├── evaluate_representation_cka.py     # 13x13 层级 CKA 热力图与跨退化环境稳定性
│   ├── calibration_core.py                # ECE / MCE / Brier 分数计算与绘图引擎
│   ├── cka_core.py                        # 快速 Frobenius 范数与无偏 HSIC 计算引擎
│   └── README.md
├── gemini_tts_synthesis/                 # [数据合成] 日语副语言气声双段式 TTS 合成切分标注管线
│   ├── script_generator.py                # 语音学 Prompt 驱动的对话剧本生成
│   ├── audio_dialogue_slicer.py           # 三阶段无爆音智能切分（TIFA+RMS能量谷+零交叉）
│   ├── textgrid_labeler.py                # 元音抽取与分片标注导出
│   └── README.md
├── mllm_annotation/                      # [人机协同] 多模态大模型（MLLM）气声粗标预填
│   ├── mllm_batch_prefill.py              # 并发预填、原子分片写入与人工劳动成果零覆盖保护
│   ├── mllm_alignment_core.py             # TextGrid 音素强制对齐时间戳抽取
│   └── README.md
├── test_annot_mllm/                      # [前期探索] MLLM 气声检测基准指标对比
│   └── README.md
├── experiments/                          # [对比基线] 传统韵律物理特征（Jitter/CPPS）与小样本实验
│   └── README.md
└── legacy/                               # [历史归档] 已废弃/被重构的历史脚本归档库
    └── README.md
```

---

## 2. 核心生产与训练脚本 (`scripts/` 根目录)

| 脚本 | 核心职能 | 核心参数与执行示例 |
| :--- | :--- | :--- |
| **`train.py`** | **主模型训练启动程序**<br>支持多任务学习、VSDC-Net 背景抑制头、QUT-NOISE 动态加噪/混响退化课程调度、动态早停与断点恢复。 | `uv run python scripts/train.py --config config/ablation_weak_on_vsdc.yaml`<br>`uv run python scripts/train.py --resume R:\checkpoints\ablation_weak_on_vsdc\latest.pt` |
| **`predict.py`** | **端到端批量与单音频推断**<br>支持对任意外部 `.wav` 预测嘎裂声与气声时序区间。 | `uv run python scripts/predict.py --checkpoint <path> --audio test.wav` |
| **`export_deployment.py`** | **部署模型导出程序**<br>将 PyTorch 训练权重导出为优化后的 ONNX 格式，支持动态批处理与变长音频输入。 | `uv run python scripts/export_deployment.py --checkpoint <path> --output-dir deployments/` |
| **`export_textgrid.py`** | **Praat TextGrid 格式导出**<br>将模型预测出的发声态区间写入标准 TextGrid，供语音学专家在 Praat 中直观可视化波形与语图。 | `uv run python scripts/export_textgrid.py --manifest manifests/segments_multi.csv` |
| **`process_qut_noise.py`** | **真实噪声与物理 RIR 预处理**<br>切除 QUT-NOISE 扫频标定段、48k $\to$ 16k 重采样、提取第 1 列并施加 Parseval 单位能量归一化。 | `uv run python scripts/process_qut_noise.py --qut-dir data/qut_noise_raw` |

---

## 3. 数据集构建与质量管控脚本

| 脚本 | 核心职能 |
| :--- | :--- |
| **`build_manifest.py`** | 汇总清洗后的强标注短音频与标注分片，生成主清单 `manifests/segments_multi.csv`。 |
| **`build_weak_manifest.py`** | 汇总无人工标注、经由 MLLM/TTS 产生的弱标签清单 `manifests/weak_breathy.csv`。 |
| **`review_annotation_quality.py`** | 校验标注数据质量，检查时间戳合法性与置信度分布。 |
| **`audit_annotation_integrity.py`** | 审计人工标注资产保护（Human Labor Protection），确保无数据污染。 |
| **`annotate_non_breathy_ishi.py`** | 基于 Carlos T. Ishi 语音学启发式规则标注常态（Modal）对照语音段。 |
| **`annotate_non_breathy_rules.py`** | 基于停顿与能量阈值规则提取常态音频片段。 |
| **`convert_textgrid_to_breathy_cache.py`** | 将 TextGrid 格式对齐文件预先转换为轻量二进制缓存。 |
| **`diagnose_breathy_checkpoint.py`** | 针对特定 Checkpoint 的多任务头进行权重范数与注意力热力图快速诊断。 |
| **`download_ucla_breathy.py`** | 下载并校验 UCLA 经典发声态语音学基准语料。 |

---

## 4. 声学统计与实验打包辅助

| 脚本 | 核心职能 |
| :--- | :--- |
| **`stat_esd_cpps.py`** | 统计 ESD 语料库的倒谱峰值突出度（CPPS）基线分布。 |
| **`stat_jvnv_cpps.py`** | 统计 JVNV 语料库的 CPPS 声学物理分布。 |
| **`stat_jvnv_voiced_segments.py`** | 统计发音人浊音段比例与平均音长。 |
| **`generate_loocv_2fold_report.py`** | 聚合留一发音人交叉验证（LOOCV）2-Fold 实验指标并生成报告。 |
| **`package_experiment.py`** | 自动化归档指定 run 的 checkpoint、log、tensorboard 与 config 为独立交付压缩包。 |
| **`test_augmentation.py`** | 针对音频加噪与混响数据增强流水线的快速单音频声学校验脚本。 |
| **`test_onnx_export.py`** | 验证导出的 ONNX 模型与 PyTorch 原生模型输出误差（Torch vs. ONNX Runtime 数值对齐）。 |

---

## 5. 快速导航建议

- **日常训练** $\to$ 参考 `scripts/train.py` 与 `config/` 目录；
- **论文评测与图表生成** $\to$ 直接进入对应子包：
  - Table 1 / Table 2 / 显著性 $p$ 检验 $\to$ `scripts/statistical_bootstrap_audit/`
  - Figure 4 / Figure 5 / 消融实验 $\to$ `scripts/layer_importance_and_truncation_audit/`
  - Figure 6 / Figure 7 / 校准与 CKA $\to$ `scripts/representation_and_calibration_audit/`
- **语料生产与标注** $\to$ `scripts/gemini_tts_synthesis/` 与 `scripts/mllm_annotation/`；
- **排查历史弃用脚本** $\to$ 参考 `scripts/legacy/README.md`。
