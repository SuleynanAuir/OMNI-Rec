# 🧩 SID Construction — Semantic ID Generation via RQ-VAE

> 本模块负责将 Item 的连续向量表示（Embedding）转化为多层离散的 **Semantic ID（语义编码）**，为后续推荐模型的序列建模提供结构化输入。

---

## 📌 目录

- [概述](#-概述)
- [环境依赖](#-环境依赖)
- [快速开始](#-快速开始)
- [核心组件](#-核心组件)
- [算法原理：残差向量量化（RVQ）](#-算法原理残差向量量化rvq)
  - [1. 数据输入与批处理](#1-数据输入与批处理)
  - [2. 前向计算过程](#2-前向计算过程)
  - [3. 损失函数](#3-损失函数)
  - [4. 反向传播与优化](#4-反向传播与优化)
- [关键设计决策](#-关键设计决策)

---

## 🔍 概述

本模块的核心任务是训练一个 **RQ-VAE（Residual Quantized Variational Autoencoder）** 模型，将由 [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) 提取的高维连续 Embedding 向量，离散化压缩为多层 Token 序列（即 Semantic ID），用于驱动后续的 RQ-KMeans 聚类与推荐序列建模。

**整体流程：**

```
Item 文本 / 图像
       │
       ▼
[Qwen-8B VL-Embedding]  ← 初级嵌入编码器（多模态）
       │  连续向量 X ∈ ℝ^(N×D)
       ▼
[RQ-VAE 训练]           ← 残差向量量化
       │  离散编码 [c₁, c₂, ..., c_K]
       ▼
  Semantic ID           ← 输入推荐模型
```

---

## 🛠 环境依赖

| 依赖项 | 说明 |
|--------|------|
| Python ≥ 3.9 | 基础运行环境 |
| PyTorch ≥ 2.0 | 深度学习框架 |
| NumPy | 向量数据加载（`.npy` 格式） |
| Qwen3-VL-Embedding-8B | 初级 Embedding 提取模型 |

---

## 🚀 快速开始

### Step 1：提取 Item Embeddings

使用 Qwen3-VL-Embedding-8B 对 Item 文本 / 图像进行编码，生成 `.npy` 格式的 Embedding 文件：

```bash
# 参考 rq/text2emb/ 目录下的脚本
bash rq/text2emb/amazon_text2emb.sh
```

### Step 2：训练 RQ-VAE

```bash
bash rq/rqvae.sh \
    --data_path data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_dir  ./output/Industrial_and_Scientific \
    --lr        1e-3 \
    --epochs    10000 \
    --batch_size 20480
```

**关键参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | — | Item Embedding 文件路径（`.npy`） |
| `--ckpt_dir` | `./output/` | 模型权重及日志的保存目录 |
| `--lr` | `1e-3` | 初始学习率 |
| `--epochs` | `10000` | 训练轮数 |
| `--batch_size` | `20480` | 批大小（建议尽量大，提升码本利用率） |

训练完成后，模型权重将保存至 `--ckpt_dir` 指定目录，包含：
- 模型权重文件（`.pt` / `.safetensors`）
- 训练损失日志

### Step 3：生成 Semantic IDs

```bash
bash rq/generate_indices_plus.sh
```

---

## 🧱 核心组件

| 文件 | 说明 |
|------|------|
| `rq/models/rq.py` | `ResidualVectorQuantizer` — 残差向量量化核心类 |
| `rq/models/rqvae.py` | RQ-VAE 完整模型（Encoder + RVQ + Decoder） |
| `rq/models/vq.py` | 单层 Vector Quantizer 基础实现 |
| `rq/models/layers.py` | MLP、归一化等通用网络层 |
| `rq/rqvae.py` | 训练入口脚本 |
| `rq/trainer.py` | 训练循环与日志管理 |
| `rq/generate_indices_plus.py` | 推理脚本：使用训练好的模型生成 Semantic IDs |
| `rq/datasets.py` | 数据集加载与预处理 |

---

## 📐 算法原理：残差向量量化（RVQ）

核心实现位于 [`rq/models/rq.py`](models/rq.py) 中的 `ResidualVectorQuantizer` 类。

### 1. 数据输入与批处理

**输入：** Item Embedding 矩阵 $X \in \mathbb{R}^{N \times D}$，从 `.npy` 文件读取。

每个训练步从中采样一个 Mini-Batch：

$$x \in \mathbb{R}^{B \times D}$$

其中 $B$ 为批大小（推荐 20480），$D$ 为原始 Embedding 维度。

---

### 2. 前向计算过程

#### 2.1 编码器（Encoder）— 潜空间映射

将原始高维 Embedding 压缩映射至低维潜空间：

$$z = E(x) \in \mathbb{R}^{B \times d}$$

> 注：$d$ 可与 $D$ 保持同维，也可通过 MLP 进行降维压缩。

#### 2.2 逐层残差量化（RQ Core）

设共有 $K$ 层码本，每层码本 $\mathcal{C}^{(k)}$ 包含 $M$ 个可学习向量：

$$\mathcal{C}^{(k)} = \bigl\{e^{(k)}_1,\ e^{(k)}_2,\ \dots,\ e^{(k)}_{M}\bigr\}$$

**递归量化流程：**

1. **初始化残差：** $r^{(1)} = z$
2. **逐层迭代** （$k = 1, 2, \dots, K$）：

   - **最近邻搜索** — 在第 $k$ 层码本中寻找最近向量的索引：
     $$q^{(k)} = \operatorname{NN}\!\left(r^{(k)},\ \mathcal{C}^{(k)}\right)$$

   - **更新残差** — 减去当前层的量化表示：
     $$r^{(k+1)} = r^{(k)} - q^{(k)}$$

3. **最终输出：**
   - **连续量化表示：** $z_q = \displaystyle\sum_{k=1}^{K} q^{(k)}$
   - **离散 Semantic ID：** $[c_1,\ c_2,\ \dots,\ c_K]$，其中 $c_k$ 为第 $k$ 层码本中最近邻的索引

#### 2.3 解码器（Decoder）

将量化后的潜空间向量重建回原始维度：

$$\hat{x} = D(z_q) \in \mathbb{R}^{B \times D}$$

---

### 3. 损失函数

总损失 $\mathcal{L}$ 由三项组成：

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction Loss}} + \sum_{k=1}^{K} \left( \underbrace{\bigl\|\,\text{sg}[r^{(k)}] - q^{(k)}\bigr\|^2}_{\text{Codebook Loss}} + \beta\,\underbrace{\bigl\|r^{(k)} - \text{sg}[q^{(k)}]\bigr\|^2}_{\text{Commitment Loss}} \right)$$

| 损失项 | 作用 | 备注 |
|--------|------|------|
| **Reconstruction Loss** | 约束 Decoder 重建质量，保证 Embedding 还原度 | 基本重建目标 |
| **Codebook Loss** | 驱动码本向量向 Encoder 输出靠拢 | 优化码本 |
| **Commitment Loss** | 约束 Encoder 输出稳定性，防止其在码本向量间频繁跳变 | $\beta$ 通常取 $0.25$ |

> **符号说明：** $\text{sg}[\cdot]$ 为 Stop-Gradient 算子，在前向传播时透明传递，反向传播时阻断梯度流。

---

### 4. 反向传播与优化

#### 4.1 Straight-Through Estimator（STE）

由于量化操作（最近邻搜索）本质上不可微，采用 **直通估计器（STE）** 近似梯度：

$$\frac{\partial \mathcal{L}}{\partial z} \approx \frac{\partial \mathcal{L}}{\partial z_q}$$

即反向传播时，将 $z_q$ 处的梯度直接复制给 $z$，绕过不可导的量化操作。

#### 4.2 码本更新策略

| 方式 | 描述 | 适用场景 |
|------|------|----------|
| **方式 A — 梯度下降** | 通过 Codebook Loss 计算梯度，直接更新码本向量 | 简单、通用 |
| **方式 B — EMA 更新** | 使用指数移动平均（EMA）平滑更新码本，收敛更稳定 | **推荐，默认使用** |

EMA 更新公式：

$$e^{(k)} \leftarrow m \cdot e^{(k)} + (1-m) \cdot \overline{r^{(k)}}$$

其中 $m$ 为动量系数（通常取 $0.99$），$\overline{r^{(k)}}$ 为当前 Batch 中分配到该码本向量的残差均值。

---

## 💡 关键设计决策

### ❗ 码本利用率（Codebook Utilization）

**问题：** 当训练数据分布不均匀时，码本中的大量向量可能从未被选中（"死码本"问题），导致有效编码容量浪费。

**解决方案：**

- **超大 Batch Size（20480）：** 确保每个训练步中有足够多的样本覆盖不同区域，最大化码本各条目被激活的概率。
- **EMA 码本更新：** 相比梯度下降，EMA 更新对码本向量的调整更平滑，避免剧烈振荡，有助于维持高利用率。
- **码本重置策略（可选）：** 对长期未被激活的码本向量，用当前 Batch 中的随机样本进行重新初始化。

### 🔗 与推荐系统的接口

RQ-VAE 训练完成后，每个 Item 将被表示为一个长度为 $K$ 的离散 Token 序列：

$$\text{Item}_i \mapsto [c_1^{(i)},\ c_2^{(i)},\ \dots,\ c_K^{(i)}]$$

这一 Semantic ID 序列将作为推荐模型（如 OMNI-Rec）的输入，支持基于 Transformer 的序列化推荐建模。

---

## 🗂 输出目录结构

```
output/
└── Industrial_and_Scientific/
    ├── model_best.pt          # 最优模型权重
    ├── model_final.pt         # 最终模型权重
    ├── train_log.json         # 训练损失记录
    └── semantic_ids.npy       # 生成的 Semantic ID 矩阵（N × K）
```

---

## 💻 核心逻辑伪代码

```python
def residual_quantize(z, codebooks):
    """
    残差向量量化核心流程。

    Args:
        z         (Tensor): Encoder 输出，形状为 [B, d]
        codebooks (List[Tensor]): K 层码本，每层形状为 [M, d]

    Returns:
        z_q     (Tensor): 量化后的连续表示，形状为 [B, d]
        indices (List[Tensor]): 各层最近邻索引，每项形状为 [B]
    """
    residual = z
    z_q = torch.zeros_like(z)
    indices = []

    for k, codebook in enumerate(codebooks):
        # Step 1: 计算残差与码本各向量的 L2 距离
        distances = torch.cdist(residual, codebook)  # [B, M]

        # Step 2: 最近邻搜索
        idx = distances.argmin(dim=-1)               # [B]

        # Step 3: 获取对应的量化向量
        q_k = codebook[idx]                          # [B, d]

        # Step 4: 更新残差
        residual = residual - q_k

        # Step 5: 累加量化表示
        z_q = z_q + q_k
        indices.append(idx)

    return z_q, indices  # indices 即为 Semantic ID: [c_1, c_2, ..., c_K]
```