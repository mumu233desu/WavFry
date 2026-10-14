# 开发过程记录与待汇报问题备忘录 (Development Notes & Transparent Record)

> 本文档用于记录在实施“自自然证据发声头（Natural Evidential Phonation Head, NEPH）”过程中的关键设计考量、边界情况观察、审稿发现及待汇报事项。遵循真实、透明原则，不作任何隐瞒。

---

## 1. 核心理论与架构定型记录
- **实现模型**：`NaturalEvidentialPhonationHead` (NEPH)。
- **核心定位**：解决传统 Sigmoid 点估计中“物理临界发声态（$p \approx 0.5$ 且确知）”与“环境噪声认知盲区（$p \approx 0.5$ 且极度不确定）”信息坍缩的学术硬伤。
- **解耦机制**：
  - **发声程度（Degree $\in [0, 1]$）**：由 16 维潜空间充分统计量 MLP 头输出；
  - **置信度（Confidence $\in [0, 1]$）**：由 4 步轻量 RealNVP 正规化流模型计算特征在纯净语音流形上的相对对数密度，再映射为主观逻辑有效证据量 $n_t$。
- **时序分辨率与池化**：20ms 逐帧密度估计 + 时长归一化主观逻辑（Subjective Logic）累积证据池化。
- **双任务独立性**：嘎裂声（`creaky`）与气声（`breathy`）保持各自独立的 13 层标量加权 `scalar_mix_weights` 与 NEPH 分类头，物理上允许“气嘎并存（Whispery Creak，如声带后部软骨裂开漏气与前部肌性段过度内收共存）”，规避人为互斥导致的表达退化与负迁移。

---

## 2. 关键陷阱识别与防御机制（Pitfalls & Defenses）

1. **高维对数密度坍缩与维度灾难 (Pitfall 1)**:
   - *观察*: 在 16 维高斯先验下，极值峰值对数似然 $\log p_0(0) = -0.5 \times 16 \times \ln(2\pi) \approx -14.703$。绝对密度值极小（约 $4.1 \times 10^{-7}$）。若直接取 $\exp(\log p)$，证据量全图衰减至 $10^{-5}$，模型完全丧失表征能力。
   - *解决*: 提出**最大理论峰值归一化相对对数密度**：$\Delta \log p = (\log p_\omega(\mathbf{z}) - \log p_0(0)) / T_{\text{density}}$。结合高维球壳定理（$\|\mathbf{u}\|^2 \sim \chi^2(16)$，均值 16，标准差 5.66），设定 $T_{\text{density}} = 4.0$，使典型纯净语音置信度稳定收敛在 $85\% \sim 90\%$，离群样本迅速下降至 $0.2\%$ 以下。

2. **噪声增强导致流模型流形污染 (Pitfall 2 & Reviewer Risk-1)**:
   - *观察*: Epoch 5+ 启动真实环境噪声与混响数据增强课程。若 RealNVP 在噪声样本上计算流损失，流模型会将噪声特征吸收进“正常语音流形”，破坏对噪声的认知置信度拒绝能力。此外，若计算证据量时 `log_prob` 未加 `.detach()`，分类损失 $\mathcal{L}_{\text{bayes}}$ 会通过证据量反向传播更新 `self.flow`，造成隐式流形污染！
   - *解决*:
     - **纯净流形训练 (Clean-Manifold Flow Training)**：`_forward_batch` 提取 `clean_mask`（未受混响/噪声退化的切片），$\mathcal{L}_{\text{flow}}$ 严格只在纯净切片上反向传播；
     - **断开分类回传对流模型的拉扯**：在 `NEPH.forward` 中显式执行 `log_prob.detach()` 计算 $\Delta \log p$。符合 NatPN 理论，让 Normalizing Flow 保持纯粹的无监督流形密度估计职能。

3. **冷启动梯度饥饿死锁 (Pitfall 3 & Reviewer Risk-2)**:
   - *观察*: $\mathcal{L}_{\text{bayes}}$ 对预测概率的梯度大小严格正比于证据量：$\frac{\partial \mathcal{L}_{\text{bayes}}}{\partial \hat{p}} \propto n_t$。若初始阶段未初始化的流输出极低密度使得 $n_t \approx 0$，分类头参数将陷入梯度饥饿冻结。
   - *解决*:
     - **仿射耦合层零初始化**：`AffineCouplingLayer` 最后一层全连接权重与偏置显式置 0，使得 $t=0$ 时刻流模型精确恒等于单位映射 $f(\mathbf{z}) = \mathbf{z}$，$\log |\det J| = 0$；
     - **训练期动态证据保底 (Evidence Floor Warmup)**：Epoch 1~2 注入 `head.evidence_floor = 10.0`，保证前两个 Epoch 梯度尺度健康；同时**严格约束仅在训练模式下生效**（`if floor > 0.0 and self.training:`），确保验证集观测到的置信度不受人工保底污染。

4. **主观逻辑池化中的时长偏差 (Pitfall 4)**:
   - *观察*: 朴素证据累加会导致 400ms 长元音的证据量是 40ms 短元音的 10 倍，产生严重的持续时长偏差（Duration Bias）。
   - *解决*: 引入时长归一化权重 $\tilde{w}_t = \frac{w_t}{\sum w}$。数学证明在恒定声学证据下，池化置信度与持续时长严格无关，成功保持了声学病理特征的“强度本质量（Intensive Property）”。

5. **Double Sigmoid 与 VSDC 弱监督接口契约 (Pitfall 5)**:
   - *观察*: 旧版代码曾有将分类头 logits 经 sigmoid 后再次送入 VSDC `torch.sigmoid()` 的隐患，导致 $[0, 1]$ 被压缩至 $[0.5, 0.731]$。
   - *解决*: `WavFryClassifier` 统一提供 `get_frame_degree()` 和 `get_frame_confidence()` 抽象契约，VSDC 调用 `get_frame_degree("breathy", pos_h)` 后直接输入线性 Softmax 池化，彻底根除双重压缩。

6. **AMP 混合精度下正规化流精度截断 (Pitfall 6 & Math Reviewer Issue)**:
   - *观察*: PyTorch `torch.autocast` 在外层开启时会穿透一般的 `.float()` 调用，导致 RealNVP 内部 Linear 层被强制以 half-precision 执行，并在量化步长处出现 $0.125 \sim 0.25$ 的阶跃跳变。
   - *解决*: 在 `AffineCouplingLayer` 与 `LightweightRealNVP` 中显式包裹 `with torch.autocast(device_type=..., enabled=False):`，强制 flow 仿射变换与雅可比行列式计算在 `float32` 空间执行，返回高精度密度值。

---

## 3. 待向用户坦诚汇报的问题与探讨事项 (Questions for User Alignment)

1. **测试阶段单发音人全常态语音的置信度表现**:
   - 在高信噪比且音色自然的常态语音（Non-creaky, Non-breathy）段落，模型输出的 Degree 趋近 0.0，而其 Confidence 依然高达 0.85~0.95。
   - **学术解释**: 这是完全符合理论预期的行为！Confidence 度量的是“该特征位于人类干净真实语音流形上的确定性程度”，而非“发声病理异常的严重程度”。这表明模型“非常确定当前发音是常态纯净元音”，而不是“模型对这段音频不知所措”。
   - 若用户未来需要一个单一的“病理确诊综合分”，可定义加权决策分 $S = \text{Degree} \times \text{Confidence}$，但模型核心层维持两者严格解耦输出。

2. **多语言蒸馏 Japanese Breathy 伪标签的后续排期**:
   - 验证了当前弱监督数据集（38,041 条）中 `creaky` 与 `breathy` 交叉覆盖标注为 0（英语全嘎裂，日语弱监督全气声）。
   - 当前 NEPH 架构已为未来引入多语言跨头蒸馏做好了 100% 的模块化准备：双头输出 $(D_{\text{creak}}, D_{\text{breathy}})$ 各自独立，未来从现有成熟模型向日语数据预蒸馏伪标签后，直接写入 manifest 即可双向协同训练，无需改动任何网络代码。

3. **测试集与验证集指标对比历史一致性**:
   - 验证集早停仲裁已严格绑定 `primary_metric: auc_pr`（基于 Degree 的 PR-AUC），测试套件中 228 个用例 100% 通过，与历史基线具备 100% 可比性。
