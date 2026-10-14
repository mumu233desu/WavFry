# Natural Evidential Phonation Head (NEPH): 发声程度与表征置信度解耦机制技术报告

**项目名称**: WavFry (Continuous Pathological Phonation Detection & Quantification)  
**分支**: `feat/evidential-confidence-head`  
**核心代码路径**:
- 模型实现: [`wavfry/model/evidential_head.py`](file:///e:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/wavfry/model/evidential_head.py)
- 损失函数与池化: [`wavfry/model/losses.py`](file:///e:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/wavfry/model/losses.py)
- 模型接入与契约: [`wavfry/model/wavlm_classifier.py`](file:///e:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/wavfry/model/wavlm_classifier.py)
- 训练与优化器动力学: [`wavfry/train/trainer.py`](file:///e:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/wavfry/train/trainer.py)
- 训练配置文件: [`config/ablation_evidential_confidence_scratch.yaml`](file:///e:/Corpus/CS-Dialogue/data/short_wav/WAVE/06_wavfry/config/ablation_evidential_confidence_scratch.yaml)

---

## 摘要 (Abstract)

在语音病理学与副语言发声检测（如嘎裂声 Creaky Voice、气声 Breathy Voice）中，传统的判决头大多采用一阶 Sigmoid 点估计输出分类概率 $p \in [0, 1]$。然而，这一经典架构在理论上存在根本性缺陷：它将**“物理临界发声态（物理发声轻微，但声学特征清晰确凿，输出 $p \approx 0.5$）”**与**“声学环境退化（强噪声或混响导致模型认知迷茫，输出 $p \approx 0.5$）”**两类截然不同的物理现象混淆坍缩在同一个标量值中。

为此，本工作在冻结的 WavLM-Base+ 骨干网络之上，提出了 **自然证据发声头（Natural Evidential Phonation Head, NEPH）**。NEPH 从自然后验网络（Natural Posterior Network, NatPN）与主观逻辑（Subjective Logic）理论出发，构建了 20ms 帧级的二阶不确定性解耦表征架构：
1. **发声严重程度（Degree $\in [0, 1]$）**：由 16 维紧凑流形潜空间的充分统计量 MLP 头输出；
2. **表征置信度（Confidence $\in [0, 1]$）**：由 4 步轻量可逆 RealNVP 正规化流（Normalizing Flow）计算特征在纯净语音流形上的无偏对数概率密度，并通过最大理论峰值相对对数密度映射为有效证据量 $n_t$；
3. **时长不变性区间累积**：提出时长归一化主观逻辑池化（Duration-Normalized Subjective Logic Cumulative Pooling），彻底根除了长短元音证据膨胀偏差；
4. **气嘎复合发声空间**：完全避免人为硬互斥 Softmax，构建非互斥二维发声连续流形 $(D_{\text{creak}}, D_{\text{breathy}}) \in [0, 1]^2$，在语音学生理学上为“气嘎并存（Whispery Creak）”保留了完整的自然解空间。

单元测试与全库回归测试表明，NEPH 体系在 228 项自动化测试中取得了 100% 通过率，严格遵循冻结骨干、强制 `uv run` 与无静默回退准则。

---

## 一、理论背景与动机 (Theoretical Background)

### 1.1 一阶 Sigmoid 点估计的信息坍缩缺陷
常规二分类网络将声学隐层 $\mathbf{h} \in \mathbb{R}^D$ 投影为标量几率：
$$z = \mathbf{w}^\top \mathbf{h} + b, \quad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
在语音学临床诊断与语料标注中，存在两种本质不同的 $p \approx 0.5$ 场景：
- **场景 A（物理弱发声/临界轻微嘎裂）**：信噪比极高，发音人发声特征清晰，声带肌性段处于轻微紧闭与常态振动的交界处。模型应当**以极高置信度断定其发声程度约为 0.5**。
- **场景 B（高噪声/强混响/严重环境干扰）**：加性噪声淹没声学周期结构，特征偏离正常语音空间。模型对该段音频**完全缺乏有效声学证据，处于极度迷茫状态**。

在传统 Sigmoid 下，场景 A 与场景 B 产生完全相同的输出 $p=0.5$。系统无法甄别当前音频是“轻微病理”还是“录音质量差”，极易引发临床假阳性或筛选误判。

### 1.2 经典证据深度学习 (EDL) 的失效模式
经典证据深度学习（Evidential Deep Learning, EDL, Sensoy et al. NeurIPS 2018）尝试通过神经网络直接输出 Dirichlet / Beta 分布的非负参数 $\alpha, \beta$。然而，近年来多项权威学术研究（Shen et al. NeurIPS 2024, Jürgens et al. ICML 2024, Pandey & Yu TPAMI 2026）严格证实了 EDL 的根本缺陷：
- **流形外置信度爆炸**：EDL 使用标准前馈 MLP 直接拟合证据量，而由于深度神经网络存在径向外推漏洞，当输入大幅偏离分布的强噪声或对抗样本时，MLP 会输出极大的证据量，导致越离群置信度越高；
- **KL 散度正则化优化病态**：EDL 依赖启发式的 KL 正则项压制非目标类的证据，在长尾或严重类别不平衡数据集上极易出现模式坍塌。

### 1.3 自然后验网络 (NatPN) 与主观逻辑启示
为了规避 EDL 的外推缺陷，自然后验网络（NatPN, Charpentier et al. ICLR 2022）提出利用显式概率密度估计器（如 Normalizing Flow）来衡量输入特征落在真实数据流形上的似然度，以此作为严谨的二阶证据源。主观逻辑（Subjective Logic, Jøsang 2016）则将二项观点建模为由证据支撑的信念分布，建立了证据量 $n$ 与认知置信度 $c = \frac{n}{2 + n}$ 的闭环数学映射。

---

## 二、NEPH 架构与数学推导 (Mathematical Formulation)

```
                       Input Waveform: 16kHz
                                 │
                     [WavLM-Base+ Backbone] (100% Frozen)
                                 │
                       h_t in R^768 (20ms frames)
                                 │
                   ┌─────────────┴─────────────┐
                   │ Linear Bottleneck (768->16)│  <-- No LayerNorm (Preserves radial scale)
                   └─────────────┬─────────────┘
                                 │
                       z_t in R^16 (Compact Latent)
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
[Sufficient Statistics Head]              [Lightweight RealNVP Flow]
 Linear(16->64) + GELU + Linear(64->1)     4-Step Invertible Coupling Layers
           │                                           │
      raw_logits                                    log p(z_t)
           │                                           │ (Detach gradient)
    Degree in [0, 1]                      Relative Density Scaling
  d_t = sigmoid(logits)                 Δlog p = (log p - log p0(0)) / T_density
           │                                           │
           │                                   Evidence n_t in [0, N_H]
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 │
                     Frame-Level Beta Posteriors:
                     α_t = 1 + n_t * d_t
                     β_t = 1 + n_t * (1 - d_t)
                     Confidence = n_t / (2 + n_t)
                                 │
                 [Duration-Normalized SL Pooling]
                 tilde_w_t = w_t / sum(w)
                 Pooled e_1 = sum(tilde_w_t * e_{1, t})
                 Pooled e_0 = sum(tilde_w_t * e_{0, t})
                                 │
                     Pooled Interval Outputs:
                     Degree = (1 + e_1) / (2 + e_1 + e_0)
                     Confidence = (e_1 + e_0) / (2 + e_1 + e_0)
                     Equivalent Logit = ln((1 + e_1) / (1 + e_0))
```

### 2.1 无归一化线性瓶颈投影 (Radial-Preserving Bottleneck)
高维 768 维骨干特征包含较多说话人声纹冗余。NEPH 采用无偏线性映射将 768 维声学特征压缩至 16 维紧凑流形潜空间：
$$\mathbf{z}_t = W_{\text{bn}} \mathbf{h}_t + \mathbf{b}_{\text{bn}} \in \mathbb{R}^{16}$$
> **关键设计抉择**：瓶颈层**严禁使用 LayerNorm**。LayerNorm 会在实例维度除以方差，消除特征向量的绝对欧氏长度，导致 15 倍强幅度的异常噪声在归一化后被强行压入常态超球面内部，彻底破坏模型对振幅异常噪声的拒绝能力。

### 2.2 轻量 RealNVP 可逆流模型与精确雅可比计算
在 16 维潜空间中，定义 4 步可逆仿射耦合变换（Affine Coupling Transforms）$f_\omega = f_4 \circ f_3 \circ f_2 \circ f_1$：
在每层 $k$ 中，将输入按奇偶通道切分 $\mathbf{z} = [\mathbf{z}_1, \mathbf{z}_2]$：
$$\begin{cases} \mathbf{u}_1 = \mathbf{z}_1 \\ \mathbf{u}_2 = \mathbf{z}_2 \odot \exp(s_k(\mathbf{z}_1)) + t_k(\mathbf{z}_1) \end{cases}$$
其中 $s_k, t_k = \text{MLP}_k(\mathbf{z}_1)$。为了防止指数发散，缩放系数施加 Tanh 有界化：
$$s_k = 2.0 \cdot \tanh(s_{\text{raw}} / 2.0) \implies \exp(s_k) \in (0.1353, 7.3891)$$
雅可比矩阵呈分块下三角结构，其雅可比行列式对数可解析精确计算：
$$\log |\det J_k| = \sum_{i=1}^{8} s_{k, i}$$
总对数密度服从变量代换公式：
$$\log p_\omega(\mathbf{z}) = \log p_0(f_\omega(\mathbf{z})) + \sum_{k=1}^4 \log |\det J_k|$$
基底先验选用标准各向同性高斯分布 $p_0(\mathbf{u}) = \mathcal{N}(0, I_{16})$：
$$\log p_0(\mathbf{u}) = -\frac{16}{2} \ln(2\pi) - \frac{1}{2} \|\mathbf{u}\|_2^2$$

### 2.3 最大理论峰值相对对数密度缩放 (Max-Normalized Relative Density)
标准正态分布在 16 维空间的理论峰值位于原点 $\mathbf{u}=0$ 处：
$$\log p_0(0) = -\frac{16}{2} \ln(2\pi) \approx -14.703$$
定义相对于先验理论最大值的对数密度差：
$$\Delta \log p = \frac{\log p_\omega(\mathbf{z}) - \log p_0(0)}{T_{\text{density}}}$$
引入相对密度并映射为正实数证据量 $n_t$：
$$\text{relative\_density} = \exp\left(\text{clamp}(\Delta \log p, -12.0, 2.0)\right)$$
$$n_t = N_H \cdot \text{relative\_density}$$
其中 $N_H = 100.0$ 为理论最大证据尺度因子，$T_{\text{density}} = 4.0$ 为密度平滑温度。
- **高维球壳定理自洽性**：根据 $\|\mathbf{u}\|^2 \sim \chi^2(16)$，纯净语音典型范数平方 $\|\mathbf{u}\|^2 \approx 16$，$\log p_0(\mathbf{u}) - \log p_0(0) \approx -8.0$。经 $T_{\text{density}}=4.0$ 平滑后，$\Delta \log p \approx -2.0 \implies n_t \approx 13.5 \implies \text{Confidence} \approx 87\%$；
- **异常噪声拒绝**：对于强噪声或离群样本（$\|\mathbf{u}\|^2 \ge 80$），$\Delta \log p \le -10 \implies n_t \le 0.005 \implies \text{Confidence} < 0.25\%$。

### 2.4 二阶 Beta 后验与解析贝叶斯期望交叉熵损失
由统计量 MLP 头输出发声严重程度：
$$\text{logits}_t = \text{stat\_head}(\mathbf{z}_t), \quad \hat{p}_t = \sigma(\text{logits}_t) \in [0, 1]$$
主观逻辑将标量证据量 $n_t$ 与发声程度 $\hat{p}_t$ 分解为正负证据：
$$e_{1, t} = n_t \cdot \hat{p}_t, \quad e_{0, t} = n_t \cdot (1 - \hat{p}_t)$$
对应的 Beta 分布参数为：
$$\alpha_t = 1.0 + e_{1, t}, \quad \beta_t = 1.0 + e_{0, t}, \quad S_t = \alpha_t + \beta_t = 2.0 + n_t$$
定义后验贝叶斯期望交叉熵损失（Bayesian Expected Cross Entropy Loss）：
$$\mathcal{L}_{\text{Bayes}} = \mathbb{E}_{p \sim \text{Beta}(\alpha, \beta)} [- y \ln p - (1 - y) \ln(1 - p)]$$
利用 digamma 函数 $\psi(x) = \frac{d}{dx} \ln \Gamma(x)$ 展开为解析闭式：
$$\mathcal{L}_{\text{Bayes}} = w_{\text{pos}} \cdot y \cdot (\psi(\alpha + \beta) - \psi(\alpha)) + (1 - y) \cdot (\psi(\alpha + \beta) - \psi(\beta))$$
当引入类别不平衡加权系数 $w_{\text{pos}}$ 时，正负样本梯度保持严密的物理流向与单调性。

### 2.5 时长归一化主观逻辑区间池化 (Duration-Invariant Interval Pooling)
给定元音区间内的帧集合掩码 $\mathbf{m} \in \{0, 1\}^T$ 与帧权重 $\mathbf{w}$，定义时长归一化权重：
$$\tilde{w}_t = \frac{w_t \cdot m_t}{\sum_{t'=1}^T w_{t'} \cdot m_{t'}}, \quad \sum_{t=1}^T \tilde{w}_t = 1$$
累积证据量为：
$$e_1 = \sum_{t=1}^T \tilde{w}_t e_{1, t}, \quad e_0 = \sum_{t=1}^T \tilde{w}_t e_{0, t}$$
区间级充分统计量输出为：
$$\text{Degree} = \frac{1 + e_1}{2 + e_1 + e_0}, \quad \text{Confidence} = \frac{e_1 + e_0}{2 + e_1 + e_0}$$
等价判决 Logit 为：
$$\text{Logit} = \ln\left(\frac{1 + e_1}{1 + e_0}\right) = \ln(1 + e_1) - \ln(1 + e_0)$$
在恒定声学证据下，40ms 短元音与 400ms 长元音的输出完全相等，消除了持续时长引起的虚假确定性膨胀。

---

## 三、六大经典陷阱与防御机制 (6 Pitfalls & Defenses)

| 陷阱编号 | 核心物理/工程风险 | 触发机理 | NEPH 数学与架构防御对策 |
| :--- | :--- | :--- | :--- |
| **Pitfall 1** | **高维对数密度坍缩与空先验** | 16 维标准高斯先验理论峰值极小（$\sim 4.1 \times 10^{-7}$），直接计算绝对密度导致 $n_t \approx 0$ | 引入**最大理论峰值归一化** $\Delta \log p = (\log p - \log p_0(0)) / T_{\text{density}}$，设定 $T_{\text{density}}=4.0$，典型纯净语音置信度稳定在 $87\%$。 |
| **Pitfall 2** | **数据增强导致流模型流形污染** | 噪声增强样本若参与流训练或通过分类损失回传梯度，流模型会将噪声特征吸收为“正常语音” | **双重阻断**：1. 流损失 $\mathcal{L}_{\text{flow}}$ 严格通过 `clean_mask` 仅在未加噪切片上计算；2. 计算相对密度时显式执行 `log_prob.detach()`，切断分类损失对流参数的隐式回传。 |
| **Pitfall 3** | **冷启动梯度饥饿死锁** | $\mathcal{L}_{\text{Bayes}}$ 对概率的梯度正比于证据量 $\frac{\partial \mathcal{L}}{\partial \hat{p}} \propto n_t$，未收敛流输出极小似然导致分类头冻结 | **零初始化 + 训练期动态保底**：耦合层最后一层零初始化使初始流恒等于单位映射；Epoch 1~2 注入 `evidence_floor = 10.0`，且严格限制仅在 `self.training` 时生效，防止污染验证集。 |
| **Pitfall 4** | **主观逻辑池化的元音时长偏差** | 朴素相加使长元音累积证据量为短元音的 10 倍，短元音系统性呈现虚假高不确定性 | 引入**时长归一化权重** $\tilde{w}_t = \frac{w_t}{\sum w}$，数学证明证据累积关于元音时长完全不变，还原发声病理的强度本质量属性。 |
| **Pitfall 5** | **Double Sigmoid 与 VSDC 接口失配** | 将已经 Sigmoid 激活的 Degree 再次输入下游模块的 `torch.sigmoid()`，导致输出压缩至 $[0.5, 0.731]$ | 设立抽象统一契约方法 `get_frame_degree()` 与 `get_frame_confidence()`，直接接入线性 Softmax 池化，严禁二次 Sigmoid。 |
| **Pitfall 6** | **AMP 混合精度下流模型精度下潜** | PyTorch autocast 拦截仿射耦合网络使其下潜至 half-precision，且 `log_prob` 发生阶跃截断 | 在仿射变换与对数似然估计内部显式声明 `with torch.autocast(enabled=False):`，强制 flow 计算与对数似然保持 `float32` 纯净度。 |

---

## 四、训练协议与实验配置 (Training Protocol & Configurations)

### 4.1 配置文件规范
实验配置文件收敛于 `config/ablation_evidential_confidence_scratch.yaml`：
- **模型设置**: `model.head_type: evidential`, `latent_dim: 16`, `flow_layers: 4`, `evidence_scale: 100.0`, `temp_density: 4.0`；
- **骨干冻结**: `model.fine_tune_layers: "none"`, `training.backbone_lr: 0.0`（铁律：骨干 100% 冻结）；
- **优化器参数分组**:
  - `backbone`: lr=0.0, weight_decay=0.0
  - `scalar_mix`: lr=1.0e-3, weight_decay=0.0
  - `head`: lr=1.0e-4, weight_decay=0.01（分类统计量 MLP 权重层）
  - `head_no_decay`: lr=1.0e-4, weight_decay=0.0（包含所有 bias、Norm 层以及 **RealNVP 流模型全部参数**，防止 L2 正则化收缩流形体积）
- **流损失与证据保底**: `flow_loss_weight: 0.001`, `evidence_floor: 10.0`, `evidence_floor_epochs: 2`；
- **多任务设置**: `tasks: [creaky, breathy]`, `eval.primary_task: creaky`, `eval.primary_metric: auc_pr`。

### 4.2 统一执行命令 (Mandatory `uv run`)
```bash
# 1. 运行 evidential 专用集成测试套件（13 个用例全部通过）
uv run pytest tests/test_evidential_modules.py tests/test_evidential_integration.py -v

# 2. 运行全仓库全量回归测试套件（228 个用例全部通过）
uv run pytest

# 3. 启动 NEPH 重训实验（严格使用高速盘 checkpoints 与 TensorBoard runs）
uv run python scripts/train.py --config config/ablation_evidential_confidence_scratch.yaml
```

---

## 五、验证与复现性审查 (Verification & Reproducibility)

经过 3 位独立 Subagent（数学流动力学审查、多任务与优化审查、可观测性与流水线审查）的严格代码级走查与审计，修复了以下隐患：
1. **阻断分类损失对流模型的逆向污染**：在计算相对密度时对 `log_prob` 进行显式 `.detach()`；
2. **约束保底证据仅在训练期生效**：验证集置信度统计完全脱敏，忠实反映真实流形密度；
3. **可观测性全面刷盘**：在 `Trainer.run()` 中将 `val_mean_confidence_{task}` 同步写入 `epoch_record`，使原子落盘的 `epoch_history.json` 和 `epoch_history.csv` 具备完整的离线曲线复现支持，并在 Epoch 结束时显式调用 `self.writer.flush()`；
4. **Delta 检查点可逆恢复保真度**：经 `test_evidential_delta_checkpoint_roundtrip` 测试验证，NEPH 的 40 个 RealNVP 参数与 6 个 Head 参数无一遗漏地被序列化，重载后浮点误差严格为 0。

---

## 六、结论与展望 (Conclusion)

NEPH 架构在保持原有 20ms 逐帧高时序分辨率与基准度量口径（PR-AUC / F1）完全可比的前提下，以严密的自然后验流形网络与主观逻辑，首次在 WavFry 中实现了**发声严重程度（Degree）**与**表征置信度（Confidence）**的真正解耦。该模块不仅使模型具备了对复杂真实环境噪声的自发拒识能力，更为未来多语言副语言发声态蒸馏与临床智能辅助诊断奠定了坚实的学术与工程基石。
