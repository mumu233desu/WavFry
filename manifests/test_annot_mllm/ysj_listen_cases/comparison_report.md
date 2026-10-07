# YSJ 自发语音（Batch 3）人工精标 vs. Gemini 3.8 Flash 对比听测报告

本报告针对您在 YSJ Batch 3 中人工精标的 22 条真实播客切片，提取出 **双方共识气声**、**LLM单独检出**、**用户单独标注** 以及 **共识常态** 四类对比案例，并切出短音频片段供您点击试听核查。

## 宏观切片级指标概览

- **总测试音频**: 22 条长切片（平均时长 22.0 秒，总长 8.1 分钟）
- **切片精确率 (Precision)**: **100.00%**
- **切片特异度 (Specificity)**: **100.00%**（6 条纯常态切片仅 0 条报阳）
- **切片召回率 (Recall)**: **25.00%**

---

## 🟢 象限 1：双方高度共识的气声（Agree Positive - TP）
人工精标为气声，Gemini 3.8 Flash 也以高置信度捕捉并精准定位。

### 案例 TP-1: `F_00_JP_JA_JA-JP_C0105_S0_19` (区间: 6.6s - 11.1s)
- 🎧 **试听音频**: [case_tp_1_F_00_JP_JA_JA-JP_C0105_S0_19.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_tp_1_F_00_JP_JA_JA-JP_C0105_S0_19.wav)
- 👤 **人工精标**: 元音 `a` (时间: 7.48s ~ 7.8s)
- 🤖 **LLM 判定**: `breathy` (时间: 7.2s ~ 10.5s)
- 🧠 **LLM 声学理由**: 说话人模拟回应语境（'あーすごいね、よかったねーって'），声门后部未完全闭合，在维持基频周期的同时伴随明显的呼气性湍流噪声，听感呈现显著的气流感与漏气发声态（Breathy voice）。
- 📜 **切片台词**: に住もうと思ってるんだよねみたいななんかお金ありますアピールばっかりしてきてあすごいねよかったねって思いながら話を聞いてましたそれであの注文したご飯なんですけど三品くらい頼んでくれててそれが順番に来た...

### 案例 TP-2: `F_00_JP_JA_JA-JP_C0105_S0_19` (区间: 6.6s - 11.48s)
- 🎧 **试听音频**: [case_tp_2_F_00_JP_JA_JA-JP_C0105_S0_19.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_tp_2_F_00_JP_JA_JA-JP_C0105_S0_19.wav)
- 👤 **人工精标**: 元音 `o` (时间: 10.81s ~ 10.88s)
- 🤖 **LLM 判定**: `breathy` (时间: 7.2s ~ 10.5s)
- 🧠 **LLM 声学理由**: 说话人模拟回应语境（'あーすごいね、よかったねーって'），声门后部未完全闭合，在维持基频周期的同时伴随明显的呼气性湍流噪声，听感呈现显著的气流感与漏气发声态（Breathy voice）。
- 📜 **切片台词**: に住もうと思ってるんだよねみたいななんかお金ありますアピールばっかりしてきてあすごいねよかったねって思いながら話を聞いてましたそれであの注文したご飯なんですけど三品くらい頼んでくれててそれが順番に来た...

### 案例 TP-3: `F_00_JP_JA_JA-JP_C0041_S0_10` (区间: 0.0s - 15.5s)
- 🎧 **试听音频**: [case_tp_3_F_00_JP_JA_JA-JP_C0041_S0_10.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_tp_3_F_00_JP_JA_JA-JP_C0041_S0_10.wav)
- 👤 **人工精标**: 元音 `a` (时间: 0.6s ~ 0.64s)
- 🤖 **LLM 判定**: `whispery` (时间: 0.0s ~ 15.0s)
- 🧠 **LLM 声学理由**: 全程缺乏周期性声带振动和稳定基频谐波结构，所有元音和音节均由强烈的声门湍流气流摩擦噪声驱动，呈现出典型的耳语发声态（whisper/whispery voice）。
- 📜 **切片台词**: 書いたんですよ。だからそこから飛んできてくださった方が多いのかなと思いつつ、うん、まあけどね、わからないですからね。どこから見つけていただけたかは、...

## 🟡 象限 2：LLM 检出气声，但人工未标（LLM-Only / 潜在漏标或轻微气声）
大模型判定含有显著气声/耳语，但您在此区间未打标为气声。请重点试听：是模型虚警，还是人工听觉疲劳漏标？

## 🔴 象限 3：人工标注了气声，但 LLM 漏报（User-Only / 模型保守漏报）
您打标为气声，但大模型判定为常态发声。通常发生在音素极其短促、或气流感极为轻微的区间。

### 案例 User-Only-1: `F_00_JP_JA_JA-JP_C0105_S0_10` (区间: 3.41s - 5.09s)
- 🎧 **试听音频**: [case_user_only_1_F_00_JP_JA_JA-JP_C0105_S0_10.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_user_only_1_F_00_JP_JA_JA-JP_C0105_S0_10.wav)
- 👤 **人工精标**: 元音 `e` (时间: 4.21s ~ 4.29s)
- 🤖 **LLM 判定**: 整句判定为常态发声，未报阳
- 📜 **切片台词**: きっと予約してくれてるし、いやですって言えないまま店内に入りました。余談なんですけど、私は本当にタバコが苦手なので、アプリのプロフィールにもタバコ苦手です。タバコ吸う方はごめんなさいっていう風に書いて...

### 案例 User-Only-2: `F_00_JP_JA_JA-JP_C0105_S0_10` (区间: 3.55s - 5.25s)
- 🎧 **试听音频**: [case_user_only_2_F_00_JP_JA_JA-JP_C0105_S0_10.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_user_only_2_F_00_JP_JA_JA-JP_C0105_S0_10.wav)
- 👤 **人工精标**: 元音 `a` (时间: 4.35s ~ 4.45s)
- 🤖 **LLM 判定**: 整句判定为常态发声，未报阳
- 📜 **切片台词**: きっと予約してくれてるし、いやですって言えないまま店内に入りました。余談なんですけど、私は本当にタバコが苦手なので、アプリのプロフィールにもタバコ苦手です。タバコ吸う方はごめんなさいっていう風に書いて...

### 案例 User-Only-3: `F_00_JP_JA_JA-JP_C0105_S0_10` (区间: 11.49s - 13.33s)
- 🎧 **试听音频**: [case_user_only_3_F_00_JP_JA_JA-JP_C0105_S0_10.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_user_only_3_F_00_JP_JA_JA-JP_C0105_S0_10.wav)
- 👤 **人工精标**: 元音 `e` (时间: 12.29s ~ 12.53s)
- 🤖 **LLM 判定**: 整句判定为常态发声，未报阳
- 📜 **切片台词**: きっと予約してくれてるし、いやですって言えないまま店内に入りました。余談なんですけど、私は本当にタバコが苦手なので、アプリのプロフィールにもタバコ苦手です。タバコ吸う方はごめんなさいっていう風に書いて...

## ⚪ 象限 4：双方高度共识的纯常态（Agree Negative - TN）
全切片 100% 为人工常态标注，大模型也完全认定为常态，未产生任何误报。

### 案例 TN-1: `F_00_JP_JA_JA-JP_C0025_S0_11` (截取区间: 9.03s - 12.03s)
- 🎧 **试听音频**: [case_tn_1_F_00_JP_JA_JA-JP_C0025_S0_11.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_tn_1_F_00_JP_JA_JA-JP_C0025_S0_11.wav)
- 👤 **人工精标**: 100% 纯常态发声
- 🤖 **LLM 判定**: Modal voice (无气声/耳语)
- 📜 **切片台词**: 体が縮んでしまっていた。工藤新一が生きているとやつらにバレたら、また命を狙われ、周りの人間にも被害が及ぶ。アガサ博士の助言で正体を隠すことにした俺は、ランに名前を聞かれて、とっさに江戸川コナンと名乗り...

### 案例 TN-2: `F_00_JP_JA_JA-JP_C0025_S0_2` (截取区间: 8.9s - 11.9s)
- 🎧 **试听音频**: [case_tn_2_F_00_JP_JA_JA-JP_C0025_S0_2.wav](file:///E:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/manifests/test_annot_mllm/ysj_listen_cases/case_tn_2_F_00_JP_JA_JA-JP_C0025_S0_2.wav)
- 👤 **人工精标**: 100% 纯常态发声
- 🤖 **LLM 判定**: Modal voice (无气声/耳语)
- 📜 **切片台词**: 劇場のあの公開日が四月の十五日でしたよね。土曜日の公開でございまして、その公開当日に見に行ってきたわけでございますけれども、その日はですね、あの同時に私の母が好きな嵐の大野君の映画の。...
