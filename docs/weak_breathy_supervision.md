# Breathy 弱标签训练与人工质检

## 标签含义

本功能默认关闭。只接收明确的 MLLM breathy 正例，监督范围是完整音频切片。
MLLM 的 false、未命中、uncertain 都不是训练负例；纯 whispery 不算 breathy。
原人工音素标签保持独立，预填 mllm_breathy 不会转换为人工音素正标签。

## 1. 离线构建清单

```bash
uv run python scripts/build_weak_manifest.py \
  --dataset-id ysj_japanese \
  --cache-dir <_breathy_cache/mllm_cache> \
  --sampling-state <_breathy_cache/sampling_state.json> \
  --audio-dir <audio> --textgrid-dir <textgrid> \
  --vowel-phones a,i,u,e,o,um \
  --manual-manifest manifests/segments_multi.csv \
  --output manifests/weak_breathy_v1.csv
```

使用与 MLLM 推理相同的音频与 TextGrid 目录。split 和 speaker 来自 sampling_state，
不是从标签或文件夹猜测。已有人类正音素的切片列为 redundant，不重复训练。

可选 `--reviews clip_reviews.csv`，字段为 file_key,task,scope,label；
scope=utterance 表示人类复核的是整段。部分音素负例不能推导整段负。

CSV 保存 eligible / unknown / review / redundant / human_reviewed 全部导入决策。
只有 eligible + train 会进入机器弱监督。清单附带缓存、音频与对齐信息指纹；
变更标注、缓存、TextGrid 或 sampling_state 后需要重新构建清单。
旧输出不会自动覆盖。不同语料分别导入，合并时必须保持 supervision_id 唯一。

## 2. 质检（可先用已有模型）

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1 uv run python scripts/review_annotation_quality.py \
  --checkpoint <checkpoint.pt> --annotation-cache <_creaky_cache> \
  --audio-dir <audio_directory> --batches 0,1 --dataset-id cs_dialogue \
  --task creaky --device cpu --output-dir <new_report_directory>
```

气声使用 `--task breathy` 和对应的 `_breathy_cache`、音频目录、dataset ID。
主入口直接读取原始 batch 的 `manual_*_annotations.json.d`，不依赖训练 manifest；
保留所有原始音素（不按训练时长/音素白名单过滤）、原状态、批次、分片路径与复核时间。
可用 `--annotation-shards <manual_creaky_annotations.json.d>` 仅检查一个分片目录；
`--manual-manifest` 是兼容的 CSV 入口，与原始分片入口互斥。
`--audio-dir` 可接多个目录（例如英语与汉语），同名音频有歧义时明确报错。
可额外加 `--weak-manifest` 检查 MLLM 切片标签。

嘎裂声和气声都要求 checkpoint 有已启用、保存了权重的目标帧头；
第一版支持 LSE 池化。缺帧头权重或声明该任务训练禁用时明确报错，不用随机帧头评分。
质检使用 checkpoint 的 block 长度、步长和 LSE temperature。
自动静音与机器预填单列来源；uncertain 不获得正负一致性 Q。
短于一帧步长或越出音频的标注保留，但标记无法可靠评分。
精确嘎裂正范围可以评分，不从其余空白区域自动生成负标签。

- all_scores.csv：人工标签、区间头/帧头分数、帧时间曲线、上下文差异、音频及原候选位置。
- review_queues.csv：按来源/任务/正负类别分别选择低一致性、不稳定、随机和高一致性对照。
- report.json：模型版本和训练来源标记。

Q 是模型和标签的一致程度，不是标签正确率。低 Q 可能是错标，也可能是真实难例。
整段获得支持但机器候选位置不获支持时，单列 location_only_review。

`--fold-map map.json` 可指定 dataset_id|speaker_id -> checkpoint 路径。
只有 checkpoint 的完整 training_provenance 能证明没见过该说话人时才标 held_out。
旧 checkpoint、来源不明的 warm-start 链标 unknown；不会自行训练交叉验证模型。
不要仅按低 Q 自动改标。人类确认仍通过原标注 APP；本工具不写入原标注缓存。

## 3. 混合训练

在一份自包含的 block 训练 YAML 的 training 中加入：

```yaml
weak_supervision:
  enabled: false                 # 确认清单后再改 true
  manifest: manifests/weak_breathy_v1.csv
  task: breathy
  candidate_scope: vowels
  loss_weight: 0.05               # 起始实验值，不保证最优
  start_epoch: 2
  samples_per_epoch: 64
  max_windows: 16
```

模型必须包含 breathy，block.frame_head.enabled=true，mil_mode=lse；
train_heads.breathy 不能为 false，epochs 必须覆盖 start_epoch。

保持 2s block / 1s hop。完整切片作为一个训练样本，内部窗口扁平批处理后
按绝对时间合并重复帧，只在全部 TextGrid 元音范围池化；不局限于机器候选。
没有元音的记录需复核。单句超过 max_windows 会报错，不能截断后仍算整句正例。

混合 batch 保留全部人工 batch，每轮最多加入 samples_per_epoch 条弱切片，
按窗口预算组批。每个切片得到一个正例 BCE 项，不改变人工区间 pos_weight。
没有独立弱验证集早停；仍用人工验证数据选择最佳模型。弱分支不作音频增强，
避免未经核实的增强改变发声方式。

总更新步数按混合 batch / gradient_accumulation_steps 计算；最后不足的累计组
按实际组大小缩放。weak 模式记录清单指纹、配置、完整 epoch 边界与训练来源；
恢复时改变清单或抽样日程会报错，请新开实验。学习率总步数/有效训练量发生改变，
对比原训练时要明确记录更新预算，不能只比较相同 epoch 数。

本版实现完整切片 batch，不对单句进行梯度分块；显存不足先减少 max_windows 或
排查超长记录。若要训练超长音频，应另行实现经过验证的梯度分块，而不是随机选块。

## 验证范围

使用 CPU 小型随机 WavLM 测试时间覆盖、尾部事件、帧合并梯度、padding、
文件/说话人隔离、正例过滤、调度步数、断点恢复与指纹变化拒绝。
不自动调用 MLLM API，不自动长训或占用 GPU。
实际效果需要后续在固定人工验证集上比较，尤其检查 creaky 是否受共享参数影响。
