<div align="center">

<img src="./assets/ChatGPT Image Apr 26, 2026, 11_57_26 PM_copy.png" width="500em" ></img> 


**OMNI-Rec 基于首个开源 OneRec生成式推荐框架，SID + SFT + 面向推荐 Rec-RL 的E2E工作流 💼**

![Kuaishou Rec](https://img.shields.io/badge/Inspired-KuaishouRecSystem-orange)
![ByteDance](https://img.shields.io/badge/Inspired-ByteDance-blue)
![TikTok Style](https://img.shields.io/badge/Design-TikTokRec-critical)
![Short Video Rec](https://img.shields.io/badge/Application-ShortVideoRec-red)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)


<a href="https://arxiv.org/abs/2510.24431">📄 技术报告</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖 Modelscope</a>

<img src="./assets/frame.png" width="1000em" ></img> 



</div>

🌟 整体模型学习流程: SFT → RL → Eval 是递进关系，但不是“数据传递”，而是“模型能力逐步升级”

基础编码方式进化：
[![RQ-VAE + Balanced K-Means Operation](https://img.shields.io/badge/RQ--VAE-Documentation-blue)](./rq/README.md)

| 阶段 | 全称 | 核心目标 | 学习内容 | 输入 | 输出 | 是否训练 |
| -------- | ---------------------- | ------ | ----------------------- | -------------- | -------- | ---- |
| SFT | Supervised Fine-Tuning | 拟合用户行为 | P(item｜user, history) | 数据集（用户序列） | 初始化推荐模型（Root Model）🌟 | ✅ |
| RL | - | 优化长期收益 | Policy π(a｜s) | SFT模型 + reward | 优化后的策略模型 （RL-Enhanced Model）🌟 | ✅ |
| Evaluate | - | 模型评估 | 无（推理） | 训练好的模型 | 指标 + 图表 | ❌ |

- SFT：基础推荐 root model 仅模拟模仿历史数据（behavior cloning），不会考虑RL中的长期收益
- RL: 学习“推荐策略（policy）”，最大化长期收益:
	- 将 SFT训练好的 root Model + 模拟用户交互环境【真实数据：(user, history, recommended_item, clicked, dwell_time, ...) —— root model ——> 模型输出推荐列表 a_t】 + 设计好的reward（点击、停留、转发）
	- 得到一个“更聪明”的推荐策略模型 （RL-Enhanced Model）
- Eval: 评估行为阶段

---

🔥 SFT → RL 衔接机制（KL 约束 + PPO）

在生成式推荐系统（如 MiniOneRec / OneRec）中，  
SFT → RL 的衔接核心在两条主线：

- KL 散度约束（Stability）
- PPO（Policy Optimization）

---

🧠 1. 核心思想

SFT（监督学习）$\pi_{\text{SFT}}(a \mid s) \approx \text{User Behavior}$

👉 本质：根据用户历史（history + user）预测下一步行为（推荐序列）

---

Rec-RL（强化学习）$\max \mathbb{E}_{\pi_\theta}[R]$

👉 本质：不只是“用户会点什么”，而是“推荐什么更好”

---

⚠️ 2. 为什么需要 KL 约束？

如果只优化 Reward：

- 模型可能偏离真实用户分布  
- 出现极端推荐（标题党）  
- 出现 hallucination  

👉 解决方案：在 RL 中加入 KL 散度约束，让模型不要偏离 SFT 太远

---

🔥 3. RL + KL 统一目标

$\mathcal{L}_{\text{RL}} = - \mathbb{E}_{\pi_\theta}[R] + \beta \cdot \mathrm{KL}(\pi_\theta \parallel \pi_{\text{ref}})$

变量说明：

- \(R(s,a)\)：Reward（点击 / 停留等）
- \(\pi_{\text{ref}}\)：参考模型（SFT）
- KL：分布偏移惩罚
- \(\beta\)：惩罚系数（越大越保守）

👉 直觉：让 reward 高，同时不要偏离 SFT

---

🧠 4. Token-Level KL（关键）

$\mathrm{KL}_t =\log \pi_\theta(a_t \mid s_t)-\log \pi_{\text{ref}}(a_t \mid s_t)$

融入 reward：$R'_t = R_t - \beta \cdot \mathrm{KL}_t$

👉 本质：把 KL 当成负奖励（penalty）

---

🚀 5. PPO（策略优化）

$\mathcal{L}_{\text{PPO}} = -\mathbb{E} \Big[\min(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\Big]$

说明：

- \(r_t = \frac{\pi_\theta}{\pi_{\text{old}}}\)
- \(A_t\)：优势函数（基于 \(R'_t\)）
- \(\epsilon\)：控制更新幅度

👉 作用：防止策略更新过大，保证稳定训练

---

🔗 6. SFT → RL 完整流程

(1) 训练 SFT：得到初始化策略 π_ref  
        ↓  
(2) 初始化 π_θ ← π_ref  
        ↓  
(3) 用当前 π_θ 生成推荐序列（tokens）  
        ↓  
(4) 计算 reward R（点击 / 停留 / 指标）  
        ↓  
(5) 计算 KL：  
        KL_t = logπθ(at|st) − logπref(at|st)  
        ↓  
(6) 得到修正奖励：  
        R' = R − β·KL  
        ↓  
(7) 用 PPO 更新策略 π_θ  
        ↓  
(8) 重复  

---

🧠 7. 直观理解

- SFT：用户“会怎么做”  
- RL：系统“应该怎么做”  
- KL：不要偏离用户太远  
- PPO：稳定更新策略  

---

🔥 8. 一句话总结

SFT learns to imitate user behavior, RL optimizes long-term reward, and KL regularization ensures the policy remains aligned with the original behavior distribution while PPO guarantees stable updates.

---

🚀 9. 核心认知

RL 不是替代 SFT，而是在 SFT 基础上的“受约束优化（Constrained Optimization）”


---

# 第一部分：快速上手 · 环境配置与文件运行

## 📢 更新日志

- **2026-04-26** — 提升训练/评估的可观测性与稳定性：
  - SFT 进度日志改为每 100 步输出一次（不再逐步打印）。
  - 新增 `visualize_training_metrics.py`，可从 `trainer_state.json` 绘制 `loss`、`grad_norm`、`learning_rate` 训练曲线。
  - RL 日志新增诊断指标：`loss_raw`、`loss_abs`、`policy_loss`、`kl_loss` 以及 advantage 统计量。
  - 评估阶段增加实时进度打印，并支持自适应 OOM 重试（自动缩减 batch size / beam size）。
  - `evaluate.sh` 现在默认使用显式的 `MiniOneRec` 环境 Python，避免环境不匹配问题。

- **2026-01-04** — 若基于 Instruct 模型的复现结果与论文指标存在差异，请检查评估日志中 CC 指标是否非零（参见 `calc.py`）。若非零，说明模型仍在生成大量无效物品，约束解码未能正常生效。该问题可能与 transformer 等依赖版本有关，正在排查中；临时方案是将 Instruct 模型替换为 base 模型（如 Qwen2.5-base）。

- **2025-12-04** — 新增脚本，支持处理 Amazon23 数据集。

- **2025-12-01** — 修复 `data.py` 中的一个 Bug：该 Bug 可能导致 SID–物品对齐任务提前看到答案，不影响最终模型性能。

- **2025-11-20** — 更新 **RQ-Kmeans+** 中的 SID 构建方法（首次开源复现，原方法来自 **GPR**）。

- **2025-11-19** — 基于 Accelerate 实现多 GPU 并行 text-to-embedding，效率大幅提升：`rq/text2emb/amazon_text2emb.py`。

- **2025-11-19** — 更新 **Constrained RQ-Kmeans** 中的 SID 构建方法。

- **2025-11-07** — 根据 Issue 反馈，发布全新实现版本。遇到问题请先升级至**最新版本**再排查。

- **2025-11-07** — 支持在 SFT 阶段冻结 LLM 参数，仅训练新增 SID 词表的 Embedding。

- **2025-10-31** — 可直接下载 MiniOneRec 模型的实现 **checkpoints**。

- **2025-10-31** — 更新 **RQ-Kmeans** 中的 SID 构建方法。

---

## 🗂️ 仓库目录结构

| 文件 / 目录 | 说明 |
| --- | --- |
| `sft.sh` | 启动监督微调（SFT）阶段的 Shell 脚本 |
| `sft.py` | SFT 训练循环的 Python 实现 |
| `sft_gpr.py` | 基于 GPR 思路的 SFT，含价值感知微调（VAFT）：通过物品价值模拟实现加权损失 |
| `rl.sh` | 启动强化学习（RL）阶段的 Shell 脚本 |
| `rl.py` | RL 训练循环的 Python 实现 |
| `rl_gpr.py` | 基于 GPR 思路的 RL，含层次增强策略优化（HEPO） |
| `minionerec_trainer.py` | MiniOneRec 自定义 Trainer，基于 GRPO，专为生成式推荐设计 |
| `configs/` | YAML 配置文件目录 |
| `evaluate.sh` | 一键离线 Top-K 评估脚本 |
| `evaluate.py` | 计算 HR@K 和 NDCG@K 的评估工具 |
| `visualize_training_metrics.py` | 训练指标可视化（读取 `trainer_state.json`，绘制 `loss` / `grad_norm` / `learning_rate`） |
| `LogitProcessor.py` | 约束解码的 Logit 处理器（Python 实现） |
| `data.py` | SFT 和 RL 训练的数据流水线 |
| `convert_dataset.py` | 将 RQ 训练数据集转换为 SFT-then-RL 格式 |
| `convert_dataset_gpr.py` | GPR 风格数据集转换，注入异质 token（U/E/I/O）以模拟统一输入表示 |
| `data/amazon18_data_process.sh` | 过滤并预处理 Amazon18 数据为 RQ 就绪格式的 Shell 脚本 |
| `data/amazon18_data_process.py` | Amazon18 数据预处理流水线的 Python 实现 |
| `data/amazon18_data_process_gpr.py` | GPR 风格 Amazon18 预处理，提取异质特征用于统一输入表示 |
| `data/amazon23_data_process.sh` | 过滤并预处理 Amazon23 数据为 RQ 就绪格式的 Shell 脚本 |
| `data/amazon23_data_process.py` | Amazon23 数据预处理流水线的 Python 实现 |
| `rq/text2emb/amazon_text2emb.sh` | 通过 emb_model 为 Amazon 数据集生成物品（标题+描述）嵌入的 Shell 脚本 |
| `rq/text2emb/amazon_text2emb.py` | 上述嵌入生成的 Python 实现 |
| `rq/text2emb/amazon_text2emb_gpr.py` | GPR 风格 text-to-embedding |
| `rq/generate_indices.py` | 训练 RQ-VAE 模型后生成 SID 文件 |
| `rq/rqvae.sh` | 在 Amazon 物品嵌入上训练 RQ-VAE 的 Shell 脚本 |
| `rq/rqvae.py` | RQ-VAE 训练的 Python 实现 |
| `rq/rqkmeans_faiss.py` | 基于 faiss 的 RQ-Kmeans 训练 Python 实现 |
| `rq/rqkmeans_constrained.py` | Constrained RQ-Kmeans 的 Python 实现 |
| `rq/rqkmeans_constrained.sh` | 在 Amazon 物品嵌入上训练 Constrained RQ-Kmeans 的 Shell 脚本 |
| `rq/rqkmeans_plus.py` | RQ-Kmeans+ 的 Python 实现 |
| `rq/rqkmeans_plus.sh` | 在 Amazon 物品嵌入上训练 RQ-Kmeans+ 的 Shell 脚本 |
| `rq/generate_indices_plus.py` | 训练 RQ-Kmeans+ 模型后生成 SID 文件 |
| `rq/generate_indices_plus.sh` | 生成 RQ-Kmeans+ SID 文件的 Shell 脚本 |
| `requirements.txt` | Python 依赖列表 |

---

## 🚀 快速开始

使用我们提供的预训练 Industrial / Office SID，即可快速上手！
仅需 4–8 张 A100/H100 GPU 即可完成复现。

### 步骤 1：创建独立 Python 环境

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 步骤 2：安装依赖包

```bash
pip install -r requirements.txt
```

### 步骤 3：监督微调（SFT）

在 SID → 目标物品预测任务上进行监督微调训练。

```bash
bash sft.sh
```

**日志行为：**
- SFT 进度回调每 `100` 步（及最后一步）打印一次，减少冗余日志。

**核心参数（在 `sft.sh` 中配置）：**

| 参数 | 说明 |
| --- | --- |
| `base_model` | 基座 LLM 路径（如 Qwen 或 Llama） |
| `batch_size` | 每 GPU 批次大小（如 1024） |
| `micro_batch_size` | 梯度累积微批次大小（如 16） |
| `train_file` | 训练 CSV 路径（如 `./data/Amazon/train/Industrial_and_Scientific_*.csv`） |
| `eval_file` | 验证 CSV 路径 |
| `output_dir` | SFT 检查点保存路径 |
| `sid_index_path` | 预计算的 SID 索引 JSON（如 `./data/Amazon/index/Industrial_and_Scientific.index.json`） |
| `item_meta_path` | 物品元数据 JSON（如 `./data/Amazon/index/Industrial_and_Scientific.item.json`） |
| `freeze_LLM` | 是否冻结 LLM 参数、仅训练 Embedding（默认 False） |

**自定义方式：** 编辑 `sft.sh`，更新路径、模型名和超参数。

### 步骤 4：面向推荐的强化学习（RL）

基于 GRPO（群体相对策略优化）对 SFT 模型进一步精调。

```bash
bash rl.sh
```

**日志行为：**
- RL 日志包含诊断字段：`loss_raw`、`loss_abs`、`policy_loss`、`kl_loss`、`adv_mean`、`adv_abs_mean`、`adv_std`。
- GRPO 中 `loss` 可能显示为 `0.0`（格式精度问题），但 `loss_raw` 非零，属正常现象。

**核心参数（在 `rl.sh` 中配置）：**

| 参数 | 说明 |
| --- | --- |
| `model_path` | 步骤 3 产生的 SFT 检查点路径 |
| `train_batch_size` | 训练批次大小（如 64） |
| `eval_batch_size` | 评估批次大小（如 128） |
| `num_train_epochs` | RL 训练轮数（如 2） |
| `train_file` / `eval_file` / `info_file` | 同 SFT 配置 |
| `num_generations` | 每个 prompt 的候选生成数量（如 16） |
| `reward_type` | 奖励信号类型（如 `ranking` 表示排名感知奖励） |
| `learning_rate` | RL 学习率（如 1e-5） |
| `beta` | KL 惩罚系数（如 1e-3） |
| `beam_search` | 是否使用约束束搜索（可在 `rl.sh` 中配置） |
| `output_dir` | RL 检查点保存路径 |

**自定义方式：** 编辑 `rl.sh`，更新检查点路径和超参数。

### 步骤 5：离线评估

```bash
bash evaluate.sh
```

**推荐运行方式（带日志保存）：**
```bash
set -o pipefail
bash evaluate.sh | tee logs/eval_$(date +%F_%H-%M-%S).log
```

**评估流程：**
1. **数据切分**：将测试数据分配到各 GPU
2. **并行推理**：各 GPU 独立执行预测
3. **结果合并**：汇总所有 GPU 的预测结果
4. **指标计算**：计算 HR@K 和 NDCG@K
5. **深度分析**：计算多维推荐指标并生成可发表质量的可视化图表

**核心参数（在 `evaluate.sh` 中配置）：**

| 参数 | 说明 |
| --- | --- |
| `exp_name` | 训练好的模型检查点路径 |
| `test_file` | 测试 CSV 路径（如 `./data/Amazon/test/Industrial_and_Scientific_*11.csv`） |
| `info_file` | 类目信息文件 |
| `EVAL_BATCH_SIZE` | 推理批次大小（默认 `2`） |
| `EVAL_NUM_BEAMS` | 生成束大小（默认 `20`） |
| `EVAL_MAX_NEW_TOKENS` | 最大生成 token 数（默认 `128`） |
| `temperature` | 采样温度（默认 1.0） |
| `cudalist` | 使用的 GPU ID（默认 0–7） |

**运行时特性：**
- 实时进度日志：`evaluate.sh` 打印进程启动，`evaluate.py` 打印阶段和批次进度
- 自适应 OOM 重试：CUDA OOM 时自动依次缩减 batch size、beam size 并重试
- 环境稳定性：默认使用 `MiniOneRec` 环境的显式 Python（`ENV_PYTHON`）

**输出文件：**
- `./results/{model_name}/final_result_{category}.json`：合并后的预测结果
- `./results/{model_name}/analysis_{category}/metrics_summary.json`：多指标汇总
- `./results/{model_name}/analysis_{category}/metrics_table.csv`：逐 K 值指标表
- `./results/{model_name}/analysis_{category}/top1_frequency.csv`：Top-1 预测频率统计
- `./results/{model_name}/analysis_{category}/figures/ranking/`：排名质量图（置信区间曲线、Precision/Recall/F1/MAP）
- `./results/{model_name}/analysis_{category}/figures/quality/`：质量与安全图（新颖性、流行度偏差）
- `./results/{model_name}/analysis_{category}/figures/distribution/`：分布图（长尾分布、首次命中排名分布、洛伦兹曲线）
- `./results/{model_name}/analysis_{category}/figures/diagnostics/`：诊断图（指标相关热力图、熵/基尼曲线）

---

## 📜 完整流水线详解

### 0. 硬件/软件前置要求

- GPU：推荐 4–8 张 A100/H100 80 GB 或同等规格
- Python：3.11

### 1. 环境搭建

**1.1 克隆仓库**
```bash
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```

**1.2 创建并激活 conda 环境**
```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

**1.3 安装依赖**
```bash
pip install -r requirements.txt
```

### 2. 数据准备

**2.1 下载原始数据集（可选）**

从官方页面获取：
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

> 注意：Industrial 和 Office 数据集包含在 Amazon 2018 中；使用 Amazon 2014 或 2023 版本需对 `data/amazon18_data_process.py` 做少量修改。

**2.2 过滤与预处理**
```bash
bash data/amazon18_data_process.sh \
     --dataset  your_dataset_type \   # 如 Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```

**2.3 将物品文本编码为嵌入**
```bash
bash rq/amazon_text2emb.sh \
     --dataset your_dataset_type \   # 如 Industrial
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID 构建

在以下 4 种方法中选择一种（3.1.1 / 3.1.2 / 3.1.3 / 3.1.4）：

**3.1.1 在嵌入上训练 RQ-VAE**
```bash
bash rq/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

**3.1.2 在嵌入上训练 RQ-Kmeans**
```bash
conda install faiss-gpu
python rqkmeans_faiss.py --dataset Industrial_and_Scientific
# 注意：基于语义嵌入的 RQ-Kmeans 方法碰撞率相对较高
```

**3.1.3 在嵌入上训练 Constrained RQ-Kmeans**
对冲突物品添加额外层进行去重；同时使用均衡约束保证 SID 分布均匀。
```bash
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
```

**3.1.4 在嵌入上训练 RQ-Kmeans+**
```bash
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
bash rqkmeans_plus.sh
```

**3.2 生成索引（仅 RQ-VAE 和 RQ-Kmeans+ 需要此步骤）**
```bash
python rq/generate_indices.py
# 或
bash rq/generate_indices_plus.sh
```

**3.3 转换数据集格式**
```bash
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/output_dir
```

### 4. 监督微调（SFT）

```bash
bash sft.sh \
     --base_model your_model_path \
     --output_dir your_output_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

### 5. 面向推荐的强化学习（RL）

> （可选）对于生产级大规模数据集，综合考虑强化学习成本和边际收益递减，可仅使用数万量级的子集执行 RL 阶段。

```bash
bash rl.sh \
     --model_path your_model_path \
     --output_dir output_dir
```

### 6. 离线评估

```bash
bash evaluate.sh \
     --exp_name your_model_path
```

低显存 GPU 推荐配置：
```bash
export EVAL_BATCH_SIZE=1
export EVAL_NUM_BEAMS=10
export EVAL_MAX_NEW_TOKENS=96
bash evaluate.sh
```

### 7. 训练指标可视化

从训练目录或指定的 `trainer_state.json` 生成本地训练曲线（`loss`、`grad_norm`、`learning_rate`）：

```bash
python visualize_training_metrics.py --path output_dir/xxx
```

输出文件：
- `training_analysis/training_metrics.png`
- `training_analysis/training_metrics.csv`

---

## 🩺 常见问题与解决方案

**RL 阶段 `loss` 显示为 `0.0`**
- 检查日志中的 `loss_raw` / `loss_abs`。在 GRPO 中这通常是显示精度问题，非训练停滞。

**大量 `No valid tokens found` 警告**
- 来自约束解码——束搜索偏离了有效前缀路径。当前代码已包含警告限速和回退行为。

**评估时出现 `ModuleNotFoundError: transformers`**
- 确保使用 `MiniOneRec` 环境。`evaluate.sh` 现在通过 `ENV_PYTHON` 解析 Python（默认指向 `.../envs/MiniOneRec/bin/python`）。

**评估时 CUDA OOM**
- 降低 `EVAL_BATCH_SIZE`、`EVAL_NUM_BEAMS` 和/或 `EVAL_MAX_NEW_TOKENS`。
- 使用上述低显存预设配置。

---

## 🤖 支持的 LLM 提供商

MiniOneRec 支持多种 LLM 提供商用于文本增强任务（如用户偏好和物品特征提取），在 `api_info` 字典中配置：

| 提供商 | `provider` 值 | 默认 Base URL | 示例模型 |
| --- | --- | --- | --- |
| OpenAI | `"openai"` | — | `text-davinci-003` |
| DeepSeek | `"deepseek"` | `https://api.deepseek.com` | `deepseek-chat` |
| [MiniMax](https://www.minimaxi.com) | `"minimax"` | `https://api.minimax.io/v1` | `MiniMax-M2.7`、`MiniMax-M2.5` |

**示例 — 使用 MiniMax：**

```python
api_info = {
    "provider": "minimax",
    "api_key_list": ["your-minimax-api-key"],
    "base_url": "https://api.minimax.io/v1",  # 可选，这是默认值
}
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=api_info)
```

**`.env` 自动加载（可选）**

`rq/text2emb/utils.py` 现支持从项目根目录的 `.env` 文件自动加载配置。

```bash
cp .env.example .env
```

编辑 `.env`（示例）：
```env
LLM_PROVIDER=minimax
API_KEY_LIST=
MINIMAX_API_KEY=your-minimax-api-key
MINIMAX_BASE_URL=https://api.minimax.io/v1
TEMPERATURE=0.4
```

调用方式：
```python
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=None)
```

优先级：显式 `api_info` > 进程环境变量 > `.env` 文件。

---

## 📝 路线图

我们正在积极扩展 MiniOneRec 的能力，以下特性已列入规划：

- ⏱️ **更多 SID 构建算法**：即将支持 R-VQ、RQ-Kmeans、RQ-OPQ 及 RQ-VAE-v2（PLUM）。
- ⚙️ **MiniOneRec-Think**：无缝整合对话、推理与个性化推荐的模块，为复杂交互场景提供一体化解决方案。
- 🔍 **更广泛的数据集支持**：新增 Yelp 等主流公开数据集，进一步验证算法的泛化能力。

---

---

# 第二部分：推荐系统核心知识 · 技术原理与行业实践

---

## 一、推荐系统概述

推荐系统（Recommender System）是信息过滤系统的核心组成，旨在从海量候选物品中预测用户偏好，并返回个性化结果。其核心目标是解决**信息过载**问题，提升用户发现内容的效率。工业界主流推荐系统普遍采用**召回 → 粗排 → 精排 → 重排**的多级级联架构，在精度与效率之间取得平衡。

### 推荐范式演进

| 阶段 | 代表方法 | 核心思路 |
| --- | --- | --- |
| 协同过滤（CF）时代 | UserCF、ItemCF、矩阵分解（MF） | 基于用户/物品共现构建相似度 |
| 深度学习时代 | DIN、DIEN、SIM、DCN | 特征交叉 + 序列建模 + 用户兴趣抽取 |
| 预训练模型时代 | BERT4Rec、SASRec、UniSRec | 自监督预训练 + 下游微调 |
| 生成式推荐时代 | P5、GenRec、MiniOneRec | 将推荐视为生成任务，端到端 LLM 建模 |

---

## 二、语义 ID（Semantic ID / SID）技术

SID 是生成式推荐的核心创新，将离散物品空间映射为 LLM 可理解的 token 序列，使推荐任务等价于 **token 生成任务**。

### 2.1 为何需要 SID？

传统推荐系统用整数 ID 表示物品（如物品编号 42857），但：
- 整数 ID 对 LLM 无语义，无法利用预训练知识；
- 直接将物品标题作为 token 序列会导致生成空间过大，且无法保证生成的物品真实存在；
- SID 通过量化语义嵌入，将每个物品压缩为少量（通常 3–4 个）码字 token，既保留语义又约束生成空间。

### 2.2 残差量化（RQ-VAE）

**残差量化变分自编码器（RQ-VAE）** 是构建 SID 的主流方法之一：

```
物品文本（标题 + 描述）
        ↓  文本编码器（冻结）
    连续嵌入向量 e
        ↓  第 1 层量化
    码字 c₁ → 残差 r₁ = e - c₁
        ↓  第 2 层量化
    码字 c₂ → 残差 r₂ = r₁ - c₂
        ↓  第 3 层量化
    码字 c₃
        ↓
    SID = [c₁, c₂, c₃]
```

**关键特性：**
- 分层量化，逐层逼近，语义保真度高；
- 通过码本（Codebook）构建有限词表，保证生成合法性；
- 层级结构天然支持层次化约束解码。

### 2.3 RQ-Kmeans 与 Constrained RQ-Kmeans

RQ-Kmeans 以 K-means 聚类替代神经量化，无需训练神经网络，速度更快：

- **RQ-Kmeans**：迭代残差聚类，碰撞率较高（不同物品可能获得相同 SID）；
- **Constrained RQ-Kmeans**：通过均衡约束（balanced constraint）确保每个码字被大致相同数量的物品使用，并在冲突物品上额外增加一层去重，消除碰撞；
- **RQ-Kmeans+**（首次开源，源自 GPR）：进一步提升 SID 质量的增强版本。

---

## 三、生成式推荐的训练范式

MiniOneRec 采用 **SFT → RL** 的两阶段训练策略，是当前生成式推荐的最佳实践。

### 3.1 监督微调（SFT）

SFT 阶段将推荐建模为**序列到序列**的 next-token 预测问题：

```
输入：用户历史交互序列的 SID token
    [SID(item₁), SID(item₂), ..., SID(itemₙ₋₁)]
输出：下一个物品的 SID
    [SID(itemₙ)]
```

**语言对齐目标（Language Alignment）：**

SFT 阶段同时引入多个辅助任务，使模型在语言空间与 SID 空间之间双向对齐：
- 物品描述 → SID（文本理解到编码）
- SID → 物品描述（编码到文本解释）
- 用户历史文本 → 推荐物品 SID（自然语言推荐）

这使推荐器能继承 LLM 的世界知识，同时将该知识锚定到离散物品码上。

**价值感知微调（VAFT，来自 GPR）：**

`sft_gpr.py` 实现了加权损失——根据模拟的物品价值对不同样本赋予差异化权重，使模型更关注高价值推荐。

### 3.2 强化学习（RL）—— GRPO

RL 阶段基于 **GRPO（Group Relative Policy Optimization，群体相对策略优化）** 进行策略精调。

#### GRPO 核心机制

```
对每个用户 prompt：
  1. 采样 G 个候选推荐（num_generations = G）
  2. 计算每个候选的奖励 rᵢ
  3. 组内归一化：Aᵢ = (rᵢ - mean(r)) / std(r)
  4. 策略梯度更新 + KL 惩罚
```

**优势（Advantage）归一化** 解决了奖励量级不稳定的问题，无需学习独立的价值函数（Critic），大幅降低训练复杂度。

#### 奖励函数设计

MiniOneRec 的奖励信号融合了两个维度：

```
R(item) = λ · R_correctness + (1-λ) · R_rank
```

- **R_correctness**（正确性奖励）：二元奖励，命中目标物品得 1 分，否则 0 分；
- **R_rank**（排名感知奖励）：对排名靠前但错误的物品给予更重的惩罚——模型对某物品越自信却推错，惩罚越大；
- **可选扩展**：融合协同过滤（CF）分数作为奖励的附加项，引入群体偏好信号。

#### 层次增强策略优化（HEPO，来自 GPR）

`rl_gpr.py` 实现了 HEPO：在 SID 的不同层级（码字位）施加差异化的策略梯度权重，使模型对高层语义码字（更具辨别力的位）给予更多关注。

---

## 四、约束解码（Constrained Decoding）

约束解码是生成式推荐中保证输出合法性的关键技术。

### 4.1 为何需要约束解码？

LLM 在自由生成模式下可能产生不存在于物品库中的 SID token 序列（"幻觉物品"），导致推荐结果无效。约束解码通过在每一步生成中将候选 logits 限制在有效前缀路径上，从根本上保证生成的 SID 一定对应真实物品。

### 4.2 前缀树（Trie）约束

实现方式通常基于前缀树（Trie）：

```
所有合法 SID：
  [c₁=A, c₂=B, c₃=1]
  [c₁=A, c₂=B, c₃=2]
  [c₁=A, c₂=C, c₃=3]
  [c₁=B, c₂=D, c₃=4]

生成第 1 个 token 时，只有 {A, B} 合法
生成第 2 个 token（已生成 A）时，只有 {B, C} 合法
......
```

`LogitProcessor.py` 实现了上述逻辑，在 beam search 的每一步屏蔽非法 token 的 logit。

### 4.3 约束束搜索（Constrained Beam Search）

在约束解码基础上，MiniOneRec 使用约束束搜索：
- 每条束（beam）始终处于有效前缀路径上；
- 通过强制去重保证 top-K 推荐互不重复；
- 显著提升采样效率与结果多样性。

---

## 五、推荐系统评估指标体系

工业界标准评估框架通常涵盖以下维度：

### 5.1 排名质量指标

| 指标 | 全称 | 公式要点 | 关注维度 |
| --- | --- | --- | --- |
| **HR@K** | Hit Rate at K | 测试集中命中率（top-K 中是否含有目标物品） | 覆盖率 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | 对命中位置进行对数折扣，命中越靠前得分越高 | 排名质量 |
| **MRR** | Mean Reciprocal Rank | 首次命中位置倒数的均值 | 首位精度 |
| **MAP@K** | Mean Average Precision | 多相关结果场景下的平均精度均值 | 综合精度 |
| **Precision@K / Recall@K** | — | 精确率与召回率 | 基础精度 |

### 5.2 多样性与覆盖率指标

| 指标 | 说明 |
| --- | --- |
| **Coverage** | 系统推荐物品占总物品库的比例，衡量长尾覆盖能力 |
| **Entropy / Gini** | 推荐分布的熵与基尼系数，衡量推荐集中度与偏差 |
| **Novelty** | 推荐物品的新颖程度，通常用物品流行度的对数倒数衡量 |
| **Popularity Bias** | 系统是否过度推荐热门物品（马太效应） |

### 5.3 分布与公平性诊断

| 指标 | 说明 |
| --- | --- |
| **Long-tail Distribution** | 各物品被推荐频次的长尾分布，评估长尾推荐能力 |
| **Lorenz Curve** | 推荐曝光的洛伦兹曲线，直观展示曝光不均衡程度 |
| **First-Hit Rank Distribution** | 首次命中位置的分布，反映模型把好结果排到前面的能力 |

---

## 六、序列推荐建模

MiniOneRec 以用户历史行为序列作为输入，是**序列推荐（Sequential Recommendation）**的典型范式。

### 6.1 序列建模发展脉络

```
早期        GRU4Rec（RNN）→ Caser（CNN）
           ↓
Attention  SASRec（单向 Transformer）→ BERT4Rec（双向 Transformer）
           ↓
预训练     UniSRec（迁移学习）→ S3-Rec（自监督）
           ↓
生成式     P5（T5 统一框架）→ GPT4Rec → MiniOneRec（SFT+RL）
```

### 6.2 关键建模技术

**位置编码（Positional Encoding）：**
- 可学习位置编码：直接学习每个时序位置的表示；
- 相对位置编码（RoPE）：Qwen/Llama 等现代 LLM 使用，更好地泛化到不同序列长度。

**注意力机制变体：**
- **因果自注意力（Causal Self-Attention）**：生成式推荐使用单向注意力，确保生成时不泄露未来信息；
- **Flash Attention**：硬件级优化，大幅降低长序列下的显存占用和计算延迟。

**序列截断与采样：**
- 用户历史过长时，常采用滑动窗口截断（如最近 50 个交互）；
- 时序加权：近期行为权重高于远期行为。

---

## 七、知识蒸馏与模型压缩

MiniOneRec 定位为**轻量化**生成式推荐框架，在效果与效率间取得平衡。

### 7.1 参数规模权衡

| 模型规模 | 优势 | 劣势 |
| --- | --- | --- |
| 1B–3B（如 Qwen2.5-1.5B） | 推理快、显存低、易部署 | 表达能力有上限 |
| 7B–13B | 效果与效率均衡 | 需要多卡推理 |
| 70B+ | 最强表达能力 | 推理成本极高 |

### 7.2 冻结策略

`freeze_LLM=True` 时，只训练新增 SID token 的 Embedding 层，LLM 主干参数不更新：
- 优点：训练参数量大幅减少，防止遗忘预训练知识，训练速度快；
- 缺点：LLM 无法充分适配推荐任务分布，效果上限可能低于全量微调。

---

## 八、工业级推荐系统架构参考

### 8.1 召回 → 精排 多级架构

```
用户请求
    ↓
【召回层】向量检索（FAISS / ANN）+ 倒排索引 + 规则召回
    ↓ ~1000 候选
【粗排层】轻量模型快速打分（如双塔模型）
    ↓ ~200 候选
【精排层】大模型精确打分（如深度兴趣网络）
    ↓ ~50 候选
【重排层】多样性调整 + 业务规则 + 曝光去重
    ↓ 最终推荐列表（10–20 个）
```

生成式推荐（如 MiniOneRec）可直接承担**召回层**和**精排层**的职责，端到端生成 top-K 候选。

### 8.2 近似最近邻检索（ANN）

SID 构建完成后，在推理阶段可利用 FAISS 等 ANN 库：
- 将 SID 嵌入建立向量索引；
- 推理时根据生成的 SID token 快速检索最相似的真实物品；
- 支持十亿量级物品的毫秒级检索。

### 8.3 实时与离线协同

| 场景 | 策略 |
| --- | --- |
| 离线评估 | 全量测试集，计算 HR@K / NDCG@K，注重指标精度 |
| 在线 A/B 测试 | 小流量实验，关注 CTR、时长、转化率等业务指标 |
| 近实时更新 | 增量训练或 LoRA 微调，快速响应用户兴趣漂移 |

---

## 🏫 研究机构

本项目由以下机构联合开发：

- <img src="assets/lds.png" width="28px"> [LDS（中科大数据科学学院）](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab（中科大）](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT Research Centre](https://www.nextcenter.org/)

---

## 🧩 参与贡献

欢迎并感谢一切形式的贡献！如有改进 MiniOneRec 的想法，欢迎提交 Pull Request。

---

## 🙏 致谢

本仓库复用或改编了以下开源项目的部分代码，在此对作者及贡献者致以诚挚感谢：

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 引用

如果我们的代码 / 论文 / 模型对您有所帮助，欢迎引用相关论文 📝 并给我们 Star ⭐️！

```bibtex
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}
```

---

<div align="center">
欢迎社区贡献！🤝
</div>
