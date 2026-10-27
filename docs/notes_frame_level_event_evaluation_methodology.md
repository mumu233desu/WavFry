# 帧级发声态事件评测方法论备忘录
## Methodology Note: Frame-Level Phonation Event Evaluation (Multi-IoU mAP & Zero-Leakage Protocol)

> **创建日期**：2026-10-24  
> **状态**：方案定稿 / 等待 CS-Dialogue 53 句人工帧级标注数据返回后实施  
> **适用范围**：WavFry 与外部基线（CreaPy, DeepFry）的无边界/细粒度时间段评测  

---

### 一、 核心痛点与理论动机 (Motivation)

在语音学与语音信号处理中，对嘎裂声（Creaky voice）与气声（Breathy voice）进行细粒度帧级（Frame-level）评测面临两大核心挑战：

1. **发声器官的生理连续性与感知渐变**：
   声门从模态发声（$M_1$ 状态，全层均匀振动）过渡到嘎裂发声（$M_0$ 状态，杓状软骨内收、短厚弛缓、周期倍增），通常需要 2~5 个声门周期（约 20~50 ms）的渐进过程。不存在毫秒级、方波般的绝对物理硬开关。
2. **人工标注的心理声学主观性与时域抖动**：
   人类标注员在音频波形和频谱图上手动拉取区间时，起止边界必然存在 $\pm 10\sim 30\text{ ms}$ 的随机偏差与认知差异。如果进行逐点（Pointwise）0/1 逐帧硬比对，会极度放大边缘偏差，甚至强迫模型去过拟合单个标注员的手抖习惯。
3. **坚决摒弃人工启发式后处理规则（Reject Heuristics）**：
   部分传统工作采用“$<30\text{ ms}$ 缝隙自动缝合、$<30\text{ ms}$ 脉冲强制丢弃”等硬编码规则。但在同行评审中，此类规则极易被质疑为“针对测试集调参的 Heuristic Tricks”。因此，本方案坚持**端到端无规则平滑，完全依靠客观数学框架**来吸收时域抖动。

---

### 二、 核心方案：Multi-IoU mAP（多阶梯容差平均精度）

借鉴计算机视觉目标检测（COCO mAP）与音视频时序动作检测（Temporal Action Detection, TAD）的成熟范式，将连续发声态视为一维时间轴上的**发声事件（Phonation Event）**。

#### 1. 一维时域交并比（Temporal IoU）
对于任意人工标注真值事件 $G_k = [s_k, e_k]$ 与模型预测事件 $P_j = [s_j, e_j]$，其交并比定义为：
$$\text{IoU}(G_k, P_j) = \frac{\text{Duration}(G_k \cap P_j)}{\text{Duration}(G_k \cup P_j)} = \frac{\max(0, \min(e_k, e_j) - \max(s_k, s_j))}{\max(e_k, e_j) - \min(s_k, s_j)}$$

#### 2. 多阶梯及格线设计（Multi-IoU Thresholds）
为了彻底避免“单挑某一个 IoU（如 0.3 太松、0.5 太严）”的争端，设置多阶梯容差阈值集合：
$$\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$$
* **低容差阶梯（$\alpha \in [0.1, 0.3]$）**：考核模型对“嘎裂声事件在宏观上是否存在（Macro Event Detection）”的捕获能力；
* **高容差阶梯（$\alpha \in [0.5, 0.7]$）**：考核模型对“声带振动起止点微观瞬变（Micro Boundary Sharpness）”的精确定位锐度。

#### 3. 数学上的“完全无阈值依赖（Threshold-Free via PR Curve）”
Multi-IoU mAP 在分类概率层面**完全不需要人工指定二值化阈值（如 0.5）**：
1. 在固定某个时域及格线 $\alpha$ 下，模型预测出的所有事件区间携带各自的置信度得分（Confidence $\in (0, 1)$），按置信度从高到低全局降序排列；
2. 遵循 COCO 与 ActivityNet/THUMOS 标准，采用**置信度降序单射二分图配对（Greedy Confidence-Sorted Injective Matching）**：置信度最高的预测优先挑选最佳未被消耗的真值，一旦真值配对即告锁定，杜绝后续重叠预测重复占有，同时严格保证 Precision-Recall 曲线关于置信度门限的单调积分有效性；
3. 沿置信度列表逐步降低过滤门槛，计算完整的 Precision-Recall 曲线；
4. 对该曲线进行连续插值求积分（COCO/PASCAL VOC 连续包络面积）得到该阶梯下的平均精度 $\text{AP}_{\alpha}$；
5. 对所有阶梯的 $\text{AP}_{\alpha}$ 取算术平均，得到全局综合得分：
   $$\text{mAP} = \frac{1}{|\mathcal{A}|} \sum_{\alpha \in \mathcal{A}} \text{AP}_{\alpha}$$

---

### 三、 具象二值化区间的 F1 评测准则 (Zero-Leakage Protocol)

如果实际语音学转写或下游应用必须输出具体的二值化时间区间（从而需要计算具体的 F1-Score、Precision 与 Recall），必须严格执行**零泄漏冻结阈值协议**：

1. **WavFry 决策阈值标定**：
   * 仅在纯净验证集（$\mathcal{D}_{\text{val}}$）上进行网格搜索，寻找到使验证集事件级 $\text{F1}$ 达到全局最大的最优阈值 $\tau^*$；
   * 将该 $\tau^*$ 绝对冻结（Frozen），在未见说话人测试集（$\mathcal{D}_{\text{test}}$）上直接应用，严禁根据测试集表现反向调整；
2. **外部对比基线（CreaPy, DeepFry）的阈值处理**：
   * 若使用开箱即用的默认阈值（0.5），同步汇报其表现；
   * 同时计算其在测试集上的理论最优阈值（Oracle Threshold），并在报告与论文中显式注明：*“该 Oracle 阈值赋予了基线额外的先知优势，反映的是基线的能力理论上界”*，展现学术严谨度与从容度。

---

### 四、 待实施路线图 (Implementation Roadmap)

待后续 CS-Dialogue 53 句的精细帧级人工标注返回后，按以下三步快速落地：

1. **输入数据就绪检查**：
   * 人工真值：解析 TextGrid / CSV 得到每句音频的连续嘎裂声真实区间 $[(s_k, e_k)]$；
   * 模型输出：直接复用现已落盘缓存的 5 档退化连续帧概率 `.npz`（路径：`data/experiments/cache/temp_frame_probabilities/{model}/{tier}/{file_key}.npz`），**无需重新前向推理**。
2. **脚本实现**：
   * 编写专用权威评测入口：`code/scripts/baseline_comparison/evaluate_frame_event_map.py`；
   * 实现无规则候选区间生成（按局部概率峰值与迟滞阈值生成 candidate intervals）；
   * 接入匈牙利匹配与多阶梯 IoU 积分模块；
3. **交付产物**：
   * 生成 `data/experiments/frame_level_eval/frame_event_map_metrics.json`；
   * 生成不同退化梯队下的 Multi-IoU 衰减曲线图与表格，直接支撑论文 Section 4.2 细粒度评测小节。
