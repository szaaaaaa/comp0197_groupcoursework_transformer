# 英国电力需求预测 —— 基于 Transformer 模型

## 项目概述

本项目（`uk-electricity-transformer.ipynb`）使用 **基于 Transformer 的时序预测模型** 预测英国电力需求（TSD，单位 MW）。它在原始参考 notebook（`uk-electricity-consumption-prediction-time-series.ipynb`）的基础上进行了多项改进。

数据集覆盖 **2019–2024 年** 英国国家电网的半小时电力需求数据，每天 48 个采样点。2019 年之前的数据因概念漂移（COVID-19、能源危机、可再生能源渗透率变化）而被丢弃。

---

## Notebook 结构

| 模块 | 说明 |
|------|------|
| **超参数配置** | 所有可调参数集中在顶部，方便统一修改 |
| **数据加载与清洗** | 读取 CSV，删除缺失列，过滤异常值，截取 2019–2024 数据 |
| **特征工程** | 提取 8 个时间特征（日、月、季度等） |
| **EDA 与可视化** | 散点图、柱状图、时序总览 |
| **Train / Val / Test 划分** | 按时间顺序划分，无数据泄露 |
| **归一化** | 对所有特征和目标做 MinMax 归一化（仅用训练集统计量） |
| **Transformer 模型** | PyTorch Transformer Encoder + 概率输出 |
| **训练循环** | GaussianNLL 损失、早停机制、学习率调度 |
| **预测与评估** | 反归一化预测结果，计算 MAPE 和 RMSE |
| **结果可视化** | 预测值 + 95% 置信区间 |

---

## 数据划分

```
2019-01-01 ═══ 训练集 (4年) ═══ 2023-01-01 ═══ 验证集 (1年) ═══ 2024-01-01 ═══ 测试集 (~1年) ═══ 2024-12-31
```

- **训练集**：2019-01-01 ~ 2022-12-31（用于模型训练）
- **验证集**：2023-01-01 ~ 2023-12-31（用于早停和学习率调度）
- **测试集**：2024-01-01 ~ 2024-12-31（仅用于最终评估，训练过程中从未使用）

---

## 特征

模型输入形状为 `(batch, 48, 8)`，三个维度分别表示：

| 维度 | 含义 | 说明 |
|------|------|------|
| **batch** | 批次大小 | 每次送入模型的样本数（默认 144） |
| **48** | 序列长度 (SEQ_LEN) | 滑动窗口包含 48 个时间步，即一天的半小时采样数 |
| **8** | 特征数 | 每个时间步包含 8 个输入特征 |

8 个输入特征如下：

| # | 特征名 | 含义 | 原始值范围 | 归一化 |
|---|--------|------|-----------|--------|
| 1 | is_holiday | 是否为公共假日 | 0 或 1 | 是 |
| 2 | settlement_period | 每日第几个半小时 | 1–48 | 是 |
| 3 | day_of_month | 当月第几天 | 1–31 | 是 |
| 4 | day_of_week | 星期几 | 0（周一）–6（周日） | 是 |
| 5 | day_of_year | 当年第几天 | 1–366 | 是 |
| 6 | quarter | 季度 | 1–4 | 是 |
| 7 | month | 月份 | 1–12 | 是 |
| 8 | week_of_year | 当年第几周 | 1–53 | 是 |

预测目标为 **tsd**（输电系统需求，单位 MW），同样做了 MinMax 归一化。所有归一化仅使用训练集的 min/max 计算。

---

## 模型架构

```
┌─────────────────────────────────────────────────────────┐
│                    输入层                                │
│             (batch, 48, 8)                               │
│        48 个时间步 × 8 个时间特征                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               线性投影 (input_proj)                      │
│          Linear(8 → 64)                                 │
│    将 8 维特征映射到 d_model=64 维                       │
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

---

## 环境配置与运行

### 依赖

基础环境 `comp0197-pt`（Python 3.12 + torch + torchvision + pillow），额外需要：
- pandas
- matplotlib
- pyyaml（`train.py` / `predict.py` 读取 `configs/*.yaml` 所需）

（共 3 个额外包，符合作业要求的 max 3 限制）

### 安装

使用课程指定的 micromamba 环境：

```bash
micromamba create --name comp0197-pt python=3.12 -y
micromamba activate comp0197-pt
pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib pyyaml
```

> 如果有 NVIDIA GPU，可安装 CUDA 版 PyTorch 加速训练：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> 代码会自动检测并使用 GPU，与 CPU 版本完全兼容。

### 运行方式

#### 方式一：Notebook（交互式，适合展示和调试）

1. 激活环境（`micromamba activate comp0197-pt`）
2. 在 VS Code 或 Jupyter 中打开对应 notebook
3. 按顺序运行所有 cell（Run All）

可用的 Notebook：
- `uk-electricity-transformer.ipynb` — 含特征工程的完整版
- `uk-electricity-transformer-no-fe.ipynb` — 无特征工程对比版

#### 方式二：命令行脚本（模块化代码）

```bash
cd elec_transformer

# 含特征工程训练
python train.py --config configs/default.yaml

# 无特征工程训练
python train.py --config configs/no_fe.yaml

# Mamba 模型训练
python train.py --config configs/mamba.yaml

# 使用已保存的 checkpoint 推理（将 <run_id> 替换为训练生成的时间戳）
python predict.py --config configs/default.yaml --checkpoint checkpoints/transformer_best_<run_id>.pt
```

命令行脚本与 Notebook 的逻辑完全等价，训练完成后会自动保存最优模型到 `checkpoints/`，并输出 MAPE 和 RMSE。
补充：当前代码会将 checkpoint 实际保存为 `checkpoints/<model>_best_<run_id>.pt`，并在 `logs/run_<run_id>/` 下写入 `train.log`，训练结束后同目录生成 `result.json`。

---

## 文件结构

```
comp0197/
├── pyproject.toml                                              # 项目依赖配置
├── README.md                                                   # 说明文档（本文件）
└── elec_transformer/
    ├── data/
    │   └── historic_demand_2009_2024_noNaN.csv                 # 数据集
    │
    ├── configs/
    │   ├── default.yaml                                        # 默认配置（含特征工程）
    │   ├── mamba.yaml                                          # Mamba 配置
    │   └── no_fe.yaml                                          # 无特征工程配置
    │
    ├── src/                                                    # 模块化源码
    │   ├── data/
    │   │   ├── loader.py                                       # CSV 加载、清洗、数据划分
    │   │   ├── feature.py                                      # 时间特征提取
    │   │   └── dataset.py                                      # 归一化、滑动窗口、DataLoader
    │   ├── models/
    │   │   ├── base.py                                         # 模型基类（统一接口）
    │   │   ├── mamba.py                                        # Mamba 模型
    │   │   └── transformer.py                                  # Transformer 模型
    │   ├── training/
    │   │   ├── trainer.py                                      # 训练循环、早停、checkpoint
    │   │   └── loss.py                                         # 损失函数
    │   └── evaluation/
    │       ├── metrics.py                                      # MAPE、RMSE
    │       └── visualize.py                                    # 可视化函数
    │
    ├── train.py                                                # 训练入口脚本
    ├── predict.py                                              # 推理入口脚本
    ├── checkpoints/                                            # 模型权重保存目录（实际文件名为 <model>_best_<run_id>.pt）
    ├── logs/                                                   # 训练日志目录（按 run_id 生成子目录）
    │   └── run_<run_id>/
    │       ├── train.log                                       # 每个 epoch 追加写入
    │       └── result.json                                     # 训练完成后写入指标与配置
    │
    ├── uk-electricity-transformer.ipynb                        # Notebook（含特征工程）
    ├── uk-electricity-transformer-no-fe.ipynb                  # Notebook（无特征工程）
    └── uk-electricity-consumption-prediction-time-series.ipynb # 原始参考 notebook
```

### 扩展新模型

项目使用**自动注册机制**，添加新模型**不需要修改 `train.py` 或 `predict.py`**，只需两步：

#### 第一步：在 `src/models/` 下新建模型文件

以 GRU 为例，创建 `src/models/gru.py`：

```python
import torch
import torch.nn as nn

from .base import BaseModel
from . import register_model


@register_model("gru")                            # ← 注册名，yaml 中用这个名字引用
class TimeSeriesGRU(BaseModel):
    """基于 GRU 的时序预测模型。"""

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)                      # (batch, seq_len, hidden)
        out = out[:, -1, :]                        # 取最后时间步
        mu = self.mu_head(out).squeeze(-1)
        log_var = self.logvar_head(out).squeeze(-1)
        var = torch.exp(log_var)
        return mu, var                             # ← 必须返回 (mu, var) 元组

    @classmethod
    def from_config(cls, model_cfg, n_features):   # ← 从 yaml 读取参数构造实例
        return cls(
            n_features=n_features,
            hidden_size=model_cfg.get("d_model", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.5),
        )
```

**要点**：
- 继承 `BaseModel`，加 `@register_model("名字")` 装饰器
- `forward()` 返回 `(mu, var)` 元组（概率预测接口）
- `from_config()` 从 yaml 的 `model` 字段读取参数

#### 第二步：创建或修改 yaml 配置

复制一份已有配置，只改 `model.type`：

```yaml
# configs/gru.yaml
model:
  type: gru          # ← 对应 @register_model("gru")
  d_model: 64        # GRU 中作为 hidden_size
  num_layers: 2
  dropout: 0.5
  # 注意：nhead 和 dim_feedforward 是 Transformer 专用参数，GRU 不需要

# 其余字段（data、features、training）与 default.yaml 相同
```

#### 运行

```bash
python train.py --config configs/gru.yaml
```

完成。`src/models/__init__.py` 会自动扫描目录下所有模型文件，找到带 `@register_model` 的类并注册。

#### 已内置的模型

| 注册名 | 文件 | 说明 |
|--------|------|------|
| `mamba` | `src/models/mamba.py` | Selective SSM / Mamba 概率输出模型 |
| `transformer` | `src/models/transformer.py` | Transformer Encoder + 概率输出 |
