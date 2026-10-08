# Gemini 日语气声（Breathy Voice）TTS 合成与验证管线

本目录实现了一套基于 **Gemini 3.8 Flash TTS** 的多说话人日语对齐与声学质检实验管线，旨在为模型补充高质量、多音色的日语会话气声正例与常态负例，解耦单一说话人音色对帧级分类头的干扰。

## 理论与语音学基础

* **Carlos Ishi（ATR）副语言功能体系**：
  在非音位性气声语言（如日语）中，气声承载 5 类核心交际功能：
  1. `prominence`: 注意の喚起・強調（Attention / Prominence）
  2. `real_feelings`: 本気の気の表出（Expression of Real Feelings / Sincerity）
  3. `polite_backchannel`: 相槌・配慮・丁寧（Politeness / Modesty / Empathy）
  4. `gentle_tone`: 童話の読み聞かせ・優しさ（Gentle Tone / Storytelling）
  5. `surprise_admiration`: 感嘆・意外・驚き（Surprise / Admiration）

* **Gemini 3.8 Flash TTS 控制机制**：
  * **`<breath>`**：属于非语言声音爆发（Non-speech Vocal Burst），生成独立的吸气/喘息声音频，不会改变后续文字的发声态。
  * **结构化 `parts` 分段**：通过将单句回复切分为常态部分（`style: "casual"`）与气声部分（`style: "soft, breathy intimate tone"`），实现语句内部常态到气声的自然过渡，保证气声在整句中占比合理（40%~70%）。

* **TIFA 强制对齐与标签投影**：
  * 剥离 `<breath>` 标签生成纯文本 TXT，由 OpenVPI TIFA 进行音素级对齐生成 TextGrid。
  * 脚本根据台词分段位置，自动将 TextGrid 中对应的元音区间打标为 `breathy=True`（正例）或 `breathy=False`（负例）。

## 目录结构

```
scripts/gemini_tts_synthesis/
├── configs/
│   └── ishi_contexts.yaml       # 5 大语境设定与 Prompt 模板
├── script_generator.py          # 调用本地 8045 (gemini-3.8-flash-high) 产出分段台词 JSON
├── tts_client.py                # Gemini TTS 客户端（Bearer 认证、多音色轮换、指数退避重试）
├── tifa_align_wrapper.py        # 封装 YouTube-Collector 的 TIFA 对齐器生成 TextGrid
├── textgrid_labeler.py          # 投影台词分段至 TextGrid 元音区间
├── acoustic_validator.py        # 声学特征质检（CPPS、HNR、周期性、谱倾斜）
└── run_smoke_test.py            # 10 条冒烟测试总控脚本
```

## 运行方式

在项目根目录使用 `uv` 运行冒烟测试：

```bash
uv run python scripts/gemini_tts_synthesis/run_smoke_test.py
```

结果输出于 `exp_results/tts_smoke_test/`：
* `smoke_scripts.json`：生成的结构化台词
* `audio/`：合成的 24kHz 单声道 WAV 文件及清洗后的 TXT 对齐文本
* `textgrids/`：TIFA 生成的 TextGrid 对齐文件
* `smoke_intervals.csv`：WavFry 格式的标注区间
* `acoustic_validation_report.md`：声学特征对比验证报告
