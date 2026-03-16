# 英国电力需求预测 —— 基于 Transformer 模型

## 项目概述

本项目（`uk-electricity-transformer.ipynb`）使用 **基于 Transformer 的时序预测模型** 预测英国电力需求（TSD，单位 MW）。它在原始参考 notebook（`uk-electricity-consumption-prediction-time-series.ipynb`）的基础上进行了多项改进。

数据集覆盖 **2009–2024 年** 英国国家电网的半小时电力需求数据，每天 48 个采样点。

---

## Notebook 结构

| 模块 | 说明 |
|------|------|
| **超参数配置** | 所有可调参数集中在顶部，方便统一修改 |
| **数据加载与清洗** | 读取 CSV，删除缺失列，过滤异常值 |
| **特征工程** | 提取时间特征（日、月、季度等）和滞后特征（1–3 年前的需求值） |
| **EDA 与可视化** | 散点图、柱状图、时序总览 |
| **Train / Val / Test 划分** | 按时间顺序划分，无数据泄露 |
| **归一化** | 仅对数值范围大的列做 MinMax 归一化 |
| **Transformer 模型** | PyTorch Transformer Encoder + 概率输出 |
| **训练循环** | GaussianNLL 损失、早停机制、学习率调度 |
| **预测与评估** | 反归一化预测结果，计算 MAPE 和 RMSE |
| **结果可视化** | 预测值 + 95% 置信区间 |

---

## 相比原 Notebook 的关键改动

### 1. 数据划分：train/test/hold_out → train/val/test

| | 原 notebook | 现在 |
|---|------------|------|
| 训练集 (Train) | < 2019-06-01 | < 2019-06-01 |
| 验证集 (Validation) | — | 2019-06-01 ~ 2021-06-01 |
| 测试集 (Test) | 2019-06-01 ~ 2021-06-01（同时用于早停） | >= 2021-06-01 |
| 保留集 (Hold-out) | >= 2021-06-01（定义了但**从未使用**） | — |

**为什么改：** 原 notebook 的"测试集"同时用于早停/学习率调度**和**最终评估，这导致了**信息泄露** —— 模型在训练过程中间接"看到"了测试数据，使得评估指标偏乐观。现在验证集负责模型选择，测试集仅在最终评估时使用一次。

### 2. 特征：加入 lag 特征，移除 year

| | 原 notebook | 现在 |
|---|------------|------|
| 时间特征 | settlement_period, day_of_month, day_of_week, day_of_year, quarter, month, **year**, week_of_year, is_holiday | settlement_period, day_of_month, day_of_week, day_of_year, quarter, month, week_of_year, is_holiday |
| 滞后特征 | 通过 `add_lags()` 创建了，但**未加入模型** | **lag1, lag2, lag3** 加入特征列表 |
| 特征总数 | 9 | 11 |

**为什么改：**
- **滞后特征（lag）** 提供了历史同期的需求绝对值（1、2、3 年前）。没有它们，模型只有周期性时间特征，无法捕捉需求的绝对水平，导致预测系统性偏低。
- **移除 year** 是因为 MinMax 归一化在 year 上会造成外推问题：训练集是 2012–2019 年，但测试集是 2021–2024 年，归一化后的值远超 [0, 1] 范围。lag 特征已经能捕捉长期趋势，year 变得多余。

### 3. 归一化：全部 MinMax → 选择性 MinMax

| | 原 notebook | 现在 |
|---|------------|------|
| 归一化的列 | 所有特征 + 目标（10 列） | 仅 lag1, lag2, lag3, tsd（4 列） |

**为什么改：** 时间特征的范围小且固定（如 settlement_period: 1–48，month: 1–12），在 train/val/test 中完全一致，归一化没有意义。只有 lag 和 tsd 列的数值范围大（约 16,000–60,000 MW），不归一化会主导梯度更新。

### 4. 模型输出：点预测 → 概率预测

| | 原 notebook | 现在 |
|---|------------|------|
| 输出 | 单个标量（预测需求值） | 均值 (μ) + 方差 (σ²) |
| 输出头 | `head`（Linear → ReLU → Linear） | `mu_head` + `logvar_head` |
| 损失函数 | `RMSELoss`（自定义 `sqrt(MSE)`） | `GaussianNLLLoss`（高斯负对数似然） |

**为什么改：** 概率模型不仅预测期望需求值，还量化**预测的不确定性**：
- 可视化时可以绘制 **95% 置信区间**
- 帮助电网调度人员了解预测的可信程度，做出更好的决策
- 注意：GaussianNLL 损失可以为负值，这在数学上是正常的（与 RMSE 恒正不同）

### 5. 超参数调整

| 参数 | 原 notebook | 现在 | 原因 |
|------|------------|------|------|
| LEARNING_RATE | 1e-3 | 5e-4 | 减少训练后期的震荡 |
| PATIENCE | 8 | 12 | 给模型更多训练时间 |
| LR_FACTOR | 0.1 | 0.5 | 更温和的学习率衰减（减半，而非缩小到 1/10） |

**为什么改：** 原始设置导致学习率衰减过于激进，随后早停过早触发。从损失曲线观察，模型在被终止时仍有改善空间。

### 6. 可视化：中文 → 英文

所有图表的标签、标题和图例从中文改为英文，解决了在部分系统上中文字体缺失导致的乱码问题。

---

## 模型架构

### 输入参数含义

模型输入形状为 `(batch, 48, 11)`，三个维度分别表示：

| 维度 | 含义 | 说明 |
|------|------|------|
| **batch** | 批次大小 | 每次送入模型的样本数（默认 144） |
| **48** | 序列长度 (SEQ_LEN) | 滑动窗口包含 48 个时间步，即一天的半小时采样数 |
| **11** | 特征数 | 每个时间步包含 11 个输入特征 |

11 个输入特征如下：

| # | 特征名 | 含义 | 值范围 | 是否归一化 |
|---|--------|------|--------|-----------|
| 1 | is_holiday | 是否为公共假日 | 0 或 1 | 否 |
| 2 | settlement_period | 每日第几个半小时 | 1–48 | 否 |
| 3 | day_of_month | 当月第几天 | 1–31 | 否 |
| 4 | day_of_week | 星期几 | 0（周一）–6（周日） | 否 |
| 5 | day_of_year | 当年第几天 | 1–366 | 否 |
| 6 | quarter | 季度 | 1–4 | 否 |
| 7 | month | 月份 | 1–12 | 否 |
| 8 | week_of_year | 当年第几周 | 1–53 | 否 |
| 9 | lag1 | 364 天前（约 1 年前）的电力需求 | ~16000–60000 MW | 是 |
| 10 | lag2 | 728 天前（约 2 年前）的电力需求 | ~16000–60000 MW | 是 |
| 11 | lag3 | 1092 天前（约 3 年前）的电力需求 | ~16000–60000 MW | 是 |

预测目标为 **tsd**（输电系统需求，单位 MW），同样做了 MinMax 归一化。

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    输入层                                │
│             (batch, 48, 11)                              │
│   48 个时间步 × 11 个特征（8 个时间特征 + 3 个 lag）      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               线性投影 (input_proj)                      │
│          Linear(11 → 64)                                │
│    将 11 维特征映射到 d_model=64 维                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              正弦位置编码 (pos_enc)                       │
│         PE(pos, 2i)   = sin(pos / 10000^(2i/d))        │
│         PE(pos, 2i+1) = cos(pos / 10000^(2i/d))        │
│    为每个时间步注入位置信息，让模型区分不同时刻            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Transformer Encoder × 2 层                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Multi-Head Self-Attention（4 头，每头 16 维）     │  │
│  │  → Add & LayerNorm                                │  │
│  │  → Feed-Forward Network（64 → 128 → 64）          │  │
│  │  → Add & LayerNorm                                │  │
│  │  → Dropout (0.5)                                  │  │
│  └───────────────────────────────────────────────────┘  │
│                     × 2 层堆叠                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              取最后一个时间步                             │
│         x[:, -1, :]  →  (batch, 64)                     │
│    用最后时间步的隐藏状态代表整个序列                      │
└────────────────────┬────────────────────────────────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
┌──────────────────┐ ┌──────────────────┐
│    mu_head       │ │   logvar_head    │
│ Linear(64→32)    │ │ Linear(64→32)    │
│ ReLU             │ │ ReLU             │
│ Dropout(0.5)     │ │ Dropout(0.5)     │
│ Linear(32→1)     │ │ Linear(32→1)     │
│                  │ │                  │
│   输出: μ        │ │ 输出: log(σ²)    │
│  （预测均值）     │ │ → σ² = exp(·)   │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                 最终输出: (μ, σ²)                        │
│                                                         │
│   μ  = 预测的电力需求均值                                │
│   σ² = 预测的方差（不确定性）                             │
│   σ  = 标准差，用于绘制 95% 置信区间 [μ-1.96σ, μ+1.96σ] │
└─────────────────────────────────────────────────────────┘
```

### 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 64 | 模型内部维度 |
| nhead | 4 | 注意力头数（每头 64/4=16 维） |
| num_layers | 2 | Transformer Encoder 堆叠层数 |
| dim_feedforward | 128 | 前馈网络隐藏层维度 |
| dropout | 0.5 | Dropout 比率 |
| 可训练参数量 | 71,938 | — |

---

## 性能指标

在**测试集**（2021-06-01 之后，模型在训练和模型选择过程中从未见过的数据）上评估：

| 指标 | 数值 |
|------|------|
| MAPE | 7.09% |
| RMSE | 2567.61 MW |

---

## 环境配置与运行

### 依赖

- Python >= 3.10
- PyTorch >= 2.0（建议安装 CUDA 版本以启用 GPU 加速）
- NumPy
- Pandas
- Matplotlib

### 安装

使用 conda / mamba 创建环境：

```bash
conda create -n comp0197 python=3.12 -y
conda activate comp0197
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy pandas matplotlib -y
```

或使用 pip：

```bash
python -m venv comp0197
source comp0197/bin/activate  # Linux/Mac
# comp0197\Scripts\activate   # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib
```

> 如果没有 NVIDIA GPU，安装 CPU 版 PyTorch 即可：
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
> 模型会自动回退到 CPU 运行，但训练速度会显著变慢。

### 运行

1. 激活环境（`conda activate comp0197`）
2. 在 VS Code 或 Jupyter 中打开 `uk-electricity-transformer.ipynb`
3. 按顺序运行所有 cell（Run All）
4. GPU 上训练约 2–5 分钟，CPU 上约 15–30 分钟

---

## 文件结构

```
elec_transformer/
├── data/
│   └── historic_demand_2009_2024_noNaN.csv
├── uk-electricity-consumption-prediction-time-series.ipynb  # 原始参考 notebook
├── uk-electricity-transformer.ipynb                         # Transformer notebook（本项目）
└── README.md                                                # 说明文档（本文件）
```
