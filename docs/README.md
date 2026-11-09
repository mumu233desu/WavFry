# WavFry 技术文档中心 (Documentation Hub)

本文档中心归集了 **WavFry** 核心模型的理论推导、系统架构设计、数据工程流水线、实验评估规范及历史技术演进底稿。

---

## 📚 目录导航

```
docs/
├── README.md                                                   # 本文档中心主页
│
├── 1. 核心理论与论文文稿 (Theory & Manuscripts)
│   ├── discussion_unidentifiability_of_phonation_decoupling.md # 论文 Section 6 核心论稿：发声态解耦不可识别性极限
│   ├── breathy_tts_synthesis_methodology.md                    # BreathyTTS 日语副语言气声合成与标注方法论
│   ├── weak_label_training_technical_specification.md          # 弱监督多示例学习与 VSDC-Net 背景抑制技术规范 (v2.0)
│   └── notes_frame_level_event_evaluation_methodology.md       # 连续音频时序事件 TAD Multi-IoU mAP 评测协议备忘
│
├── 2. 工程实现与评测规范 (Engineering & Evaluation Specifications)
│   ├── data_collection_and_processing_pipeline.md             # 数据集全生命周期管理指南（切片/对齐/分片标注）
│   ├── cloud_environment_setup_and_execution_guide.md         # A100 云端分布式训练、镜像加速与网络环境搭建
│   ├── layer_importance_and_truncation_audit.md               # 跨层重要性审计规范（LOLO 微观消融与 Top-k 深度截断）
│   └── representation_cka_and_calibration_audit.md            # 表征几何（CKA）与温度校准（AURC）审计规范
│
└── archive/                                                    # 历史技术里程碑与消极结论归档区
    ├── 202607_initial_prd_creaky_vowel_detection.md           # 2026-07 早期立项产品需求文档 (PRD)
    ├── 20260726_extended_phoneme_vad_rework_plan_v1.2.md       # 2026-07 扩展音素白名单与静音 VAD 自动标注实施规范
    ├── 202608_ncmmsc_conference_technical_report.md           # 2026-08 NCMMSC 会议投稿版原始技术报告
    ├── 20260907_codebase_core_vulnerability_audit_report.md   # 2026-09 代码库核心缺陷复核与防御性重构终审报告
    ├── 20260907_u1089_fewshot_adaptation_benchmark.md         # 2026-09 单发音人 U1089 Few-shot 早期基准表
    ├── 20261012_weak_breathy_supervision_v1_draft.md          # 2026-10 气声弱监督初版草稿（已被 v2.0 规范取代）
    ├── evidential_head/                                        # 对应归档 Tag: archive/evidential-confidence-head
    │   ├── 20261015_neph_model_architecture_report.md         # NEPH 自然证据态模型架构技术报告
    │   └── 20261015_neph_development_notes_and_pitfalls.md    # NEPH 开发备忘与实证陷阱记录
    └── vad_reports/                                            # 早期过程数据报告
        ├── 20260726_vad_anchor8_dry_run_report.json
        ├── 20260726_vad_u1001_dry_run_report.json
        └── 20260726_vad_zh_u1001_dry_run_report.json
```

---

## 一、 核心活跃文档 (Active Documentation)

### 1. 理论与论文文稿
- **[发声态解耦不可识别性极限证明](discussion_unidentifiability_of_phonation_decoupling.md)**  
  *文档定位*：论文第 6 节（Discussion）核心论据。  
  *核心贡献*：严格证明在单标注者二值感知体系下，将分类后验概率无监督解耦为“物理严重度（Degree）”与“认知置信度（Confidence）”属于病态欠定问题（Ill-Posed Unidentifiable Limit），确立发声态不确定性的四维分解理论边界。
- **[BreathyTTS 副语言气声合成与标注方法论](breathy_tts_synthesis_methodology.md)**  
  *文档定位*：论文数据集构建章节依据。  
  *核心贡献*：基于 Carlos T. Ishi (ATR) 分类学（8 大类 18 子类 540 场景），详述基于 Gemini TTS 的剧本设计、端到端生成、TIFA 强制对齐与声学边界切分机制。
- **[弱监督多示例学习技术规范 (v2.0)](weak_label_training_technical_specification.md)**  
  *文档定位*：多任务弱监督与背景抑制模型设计规范。  
  *核心贡献*：系统形式化句子级非对称正例 MIL（Positive-Only MIL）、VSDC-Net 背景抑制与声纹对消网络、动态装箱批处理与 QUT-NOISE / RIR 课程学习。
- **[连续音频时序事件 Multi-IoU mAP 评测协议](notes_frame_level_event_evaluation_methodology.md)**  
  *文档定位*：时序动作检测（TAD）连续发声事件评测方法学。  
  *核心贡献*：详述基于真实物理声学区间与金标 TextGrid 的 Dual-Scale Multi-IoU 平均精度（mAP）及 F1-Score 仲裁协议。

### 2. 工程实现与评测规范
- **[数据全流程管理与处理指南](data_collection_and_processing_pipeline.md)**  
  *文档定位*：语料工程标准作业程序（SOP）。  
  *核心贡献*：系统覆盖 CS-Dialogue、MELI、YSJ、BreathyTTS 等多语料库的预加重、去混响、MFA 音素对齐、双轨制 Kappa 裁定与统一 Manifest 生成。
- **[A100 云端分布式训练与网络环境指南](cloud_environment_setup_and_execution_guide.md)**  
  *文档定位*：计算基座工程规范。  
  *核心贡献*：涵盖国内镜像源加速、ZeroTier 跨网直连、容器 `/dev/shm` 总线防崩溃配置及多卡显存调优参数。
- **[跨层重要性与深度截断审计规范](layer_importance_and_truncation_audit.md)**  
  *文档定位*：模型可解释性与边缘推理优化规范。  
  *核心贡献*：规范 Leave-One-Layer-Out (LOLO) 微观敏感度与 Top-$k$ 前缀/稀疏截断评估，支持单次前向多条件缓存。
- **[表征几何（CKA）与置信度校准规范](representation_cka_and_calibration_audit.md)**  
  *文档定位*：表征审计与安全拒识规范。  
  *核心贡献*：线性/RBF 核 CKA 相似度矩阵度量、温度缩放校准（Expected Calibration Error, ECE）以及选择性预测风险-覆盖率曲线（Risk-Coverage / AURC）。

---

## 二、 历史归档区 (Archive)

历史文档存放在 [`docs/archive/`](archive/) 目录下，主要包括：
1. **会议底稿与早期需求**：包含立项初期仅支持 5 发音人的 [早期 PRD](archive/202607_initial_prd_creaky_vowel_detection.md) 以及 [NCMMSC 2026 会议投稿技术报告底稿](archive/202608_ncmmsc_conference_technical_report.md)；
2. **早期实施计划与初版草稿**：包含 [扩展音素与 VAD 返修规范](archive/20260726_extended_phoneme_vad_rework_plan_v1.2.md)、[气声弱监督初版草稿](archive/20261012_weak_breathy_supervision_v1_draft.md) 以及 [2026-09-07 代码库核心缺陷复核报告](archive/20260907_codebase_core_vulnerability_audit_report.md)；
3. **探索性消融分支（Negative Result）**：在 [`docs/archive/evidential_head/`](archive/evidential_head/) 中完整保留了自然证据态发声头（NEPH）的技术报告与开发踩坑备忘，对应 Git 标签 `archive/evidential-confidence-head`；
4. **过程数据报告**：在 [`docs/archive/vad_reports/`](archive/vad_reports/) 中收敛了早期 3 份历史 VAD dry-run 统计 JSON。
