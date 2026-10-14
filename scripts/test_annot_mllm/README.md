# MLLM Breathy Detection Evaluation Benchmark (`scripts/test_annot_mllm/`)

本目录包含了早期针对多模态大语言模型（MLLM，如 Gemini 2.5/3.8 Flash High）在日语对话气声（Breathy Voice）检测能力的**基准评测（Benchmarking & Evaluation）**与**提示词有效性验证**脚本。

---

## 脚本职责概览

| 脚本 | 核心功能 | 依赖与评测基准 |
| :--- | :--- | :--- |
| **`test_annot_mllm_eval.py`** | **MLLM 气声检测基准评测**<br>通过 OpenAI 兼容 API 并发请求音频大模型，支持分片缓存（`api_cache/`）与断点恢复，计算气声二分类混淆矩阵、Precision、Recall、F1。 | `benchmark_manifest.json`<br>严格禁止模型静默回退（Fail-Fast）。 |
| **`test_annot_mllm_eval_ysj.py`** | **YSJ 人工标注黄金集对齐评测**<br>以语音学专家（YSJ）的高精度逐音素标注为黄金真值（Ground Truth），细粒度评估 MLLM 粗标的一致性（Cohen's Kappa / Krippendorff's Alpha）。 | YSJ 专家标注 TextGrid / JSON |
| **`test_annot_mllm_prepare.py`** | **评测切片与 Manifest 准备脚本**<br>从原始长音频中裁剪基准测试集音频切片，组织元数据。 | `manifests/` |
| **`test_annot_mllm_coarse_textgrid.py`** | **粗粒度 TextGrid 区间投影**<br>将 MLLM 给出的整句/半句定性判断投影回 TextGrid 候选区间。 | TextGrid 解析与区间过滤 |
| **`test_annot_mllm_prompts.py`** | **评测提示词库**<br>包含 System Instruction 与副语言气声 Few-shot 对话模板。 | - |

---

## 执行说明

```bash
uv run python scripts/test_annot_mllm/test_annot_mllm_eval.py \
  --manifest manifests/test_annot_mllm/benchmark_manifest.json \
  --endpoint http://127.0.0.1:8045 \
  --model gemini-3.8-flash-high
```
> **注**：大规模生产级预填请使用 `scripts/mllm_annotation/mllm_batch_prefill.py`。本目录专门用于算法前期小规模基准检验。
