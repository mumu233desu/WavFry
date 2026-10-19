# 云端算力平台（端砚 / A100）从零搭建与多 Backbone 评测执行权威指南

> 本指南针对书生·端砚（Intern Inkstone）、AutoDL 等云端 GPU 算力平台编写，规范化指导从裸实例启动到 4 款 SSL Backbone 训练与 18 场景声学退化评测的全流程。

---

## 架构原则：计算、存储与数据分离设计

为兼顾**极速训练 I/O 吞吐**与**成果资产持久化安全**，采用分层存储设计：
- **持久化云盘 (`/data`)**：
  - 存放数据集原始压缩包（`Corpus_*.zip`）；
  - 存放训练输出物（Checkpoints、TensorBoard 日志、Table 5 评测结果 `table5_backbone_comparison.md`）；
  - **特点**：实例关机、销毁重建后依然永久保留。
- **本地高速 NVMe 盘 (`/root`)**：
  - 解压后的数据集实体目录（`/root/Corpus`）；
  - 代码仓库与 Python 虚拟环境（`/root/WavFry`）；
  - **特点**：单盘吞吐达数 GB/s，彻底消除云端网络共享文件系统的随机小文件寻址延迟。

---

## 一键式端到端搭建流程

### 第一步：系统基础依赖与 Rust/Cargo 工具链安装

`deepfilternet` 包含基于 Rust 编写的底层 C-ABI 扩展库 `deepfilterlib`，`maturin` 构建工具必须依赖 `cargo` 进行本地编译。

# 1. 扩容容器共享内存（防止多进程 DataLoader 报 Bus Error）
mount -o remount,size=16G /dev/shm 2>/dev/null || true

# 2. 更新 APT 并安装基础工具
apt-get update && apt-get install -y unzip git tmux build-essential curl

# 3. 安装 uv（超高速 Python 包与环境管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# 4. 安装 Rust 与 Cargo（彻底解决 deepfilterlib==0.5.6 构建报错）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 验证 cargo 与 uv 是否就绪
cargo --version
uv --version
```

---

### 第二步：解压数据集到本地高速盘 `/root/Corpus`

原始压缩包位于云盘 `/data`，直接解压到本地高速盘 `/root/Corpus`：

```bash
# 1. 创建本地高速数据集目录与持久化结果目录
mkdir -p /root/Corpus
mkdir -p /data/exp_results

# 2. 从 /data 解压 6 个数据集 zip 到本地 NVMe /root/Corpus
cd /data
unzip -q -o Corpus_CS_Dialogue.zip -d /root/Corpus/
unzip -q -o Corpus_Creaky_MELI.zip -d /root/Corpus/
unzip -q -o Corpus_Breathy_YSJ.zip -d /root/Corpus/
unzip -q -o Corpus_Breathy_ITA.zip -d /root/Corpus/
unzip -q -o Corpus_Breathy_BreathyTTS.zip -d /root/Corpus/
unzip -q -o Corpus_Noise_QUT.zip -d /root/Corpus/

echo "[OK] 6 个数据集已全部解压至本地高速盘 /root/Corpus！"
```

---

### 第三步：克隆代码仓库并配置持久化软链接

```bash
# 1. 配置 Git 全局使用 ghfast 国内镜像加速 GitHub 访问（彻底解决 clone / pull 网络波动）
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"

# 2. 克隆代码主仓库（自动走 ghfast 加速）
cd /root
git clone https://github.com/mumu233desu/WavFry.git
cd /root/WavFry

# 若仓库已存在，可直接切换加速远端：
# git remote set-url origin https://ghfast.top/https://github.com/mumu233desu/WavFry.git

# 3. 检出多 Backbone 对照专项分支
git checkout feat/backbone-comparison

# 4. 将实验输出目录软链接至持久化云盘 /data
# （模型权重与日志将自动穿透写入云盘，即使实例被释放也不丢失）
ln -s /data/exp_results exp_results

# 5. 配置持久化环境变量（写入 ~/.bashrc）
echo 'export WAVFRY_CORPUS_ROOT=/root/Corpus' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

### 第四步：一键同步虚拟环境与依赖

```bash
cd /root/WavFry

# 同步虚拟环境与依赖（已安装 Cargo，uv 会自动拉起 maturin 完成 deepfilterlib 本地编译）
uv sync
```

---

### 第五步：运行前快速健康检查（5秒）

运行简短自检脚本，核验 CUDA 与声学路径对齐状态：

```bash
uv run python -c "
import torch
from wavfry.data.path_resolver import resolve_corpus_path
print('=' * 60)
print('CUDA Available  :', torch.cuda.is_available())
print('Compute Device  :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('VRAM Total (GB) :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0)
print('Corpus Resolved :', resolve_corpus_path('E:/Corpus/CS-Dialogue/data/short_wav/WAVE/01_en_dereverb'))
print('Noise Resolved  :', resolve_corpus_path('E:/Corpus/Noise/QUT-NOISE_processed/noise'))
print('=' * 60)
"
```
**期望输出**：
- `CUDA Available : True`
- `Compute Device : NVIDIA A100-SXM4-80GB`（或对应显卡）
- `Corpus Resolved: /root/Corpus/CS-Dialogue/...`
- `Noise Resolved : /root/Corpus/Noise/QUT-NOISE_processed/noise`

---

### 第六步：正式启动多 Backbone 对照全流程

使用 `tmux` 会话运行，避免浏览器或 SSH 会话断开导致实验中断：

```bash
# 1. 启动并进入 tmux 会话
tmux new -s wavfry

# 2. 进入项目主目录
cd /root/WavFry

# 3. 清理历史所有运行产物，从零全新启动：
rm -rf /data/exp_results/* exp_results/* 2>/dev/null || true
mkdir -p /data/exp_results

# 4. 权威执行 4 款 Backbone 训练 + 18 场景全声学退化评测
uv run python scripts/backbone_comparison/run_backbone_comparison.py \
    --backbones wavlm hubert wav2vec2 wavlm_plus \
    --mode train_and_eval \
    --output-dir /data/exp_results/backbone_comparison
```

#### 会话恢复与断连查看：
- 离开会话（后台保持运行）：按 `Ctrl + B`，然后按 `D`；
- 重新连接会话：`tmux attach -t wavfry`。

---

## 结果交付物（Table 5）

实验执行完毕后，以下文件将持久保存在 `/data/exp_results/backbone_comparison/`：
1. `table5_backbone_comparison.md`：论文级 IEEE/Interspeech 格式汇总大表（包含 18 种声学退化场景与均值）；
2. `table5_backbone_comparison.csv`：各 Backbone 在全部条件下的详细数值度量；
3. `ablation_backbone_{backbone_key}/best_model.pt`：各模型最佳检查点权重。
