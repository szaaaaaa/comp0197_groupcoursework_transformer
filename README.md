# 英国电力需求预测 —— 概率时序预测框架

## 项目概述

本项目使用多种模型对英国电力需求（TSD，单位 MW）进行**概率预测**，所有模型统一输出均值和标准差，支持不确定性量化。

数据集覆盖 **2019–2024 年** 英国国家电网的半小时电力需求数据，每天 48 个采样点。

---

## 支持的模型

| 模型 | 类型 | 配置文件 | 保存格式 |
|------|------|---------|---------|
| Transformer | 深度学习 | `configs/default.yaml` | `.pt` |
| LSTM | 深度学习 | `configs/lstm.yaml` | `.pt` |
| CNN | 深度学习 | `configs/cnn.yaml` | `.pt` |
| SARIMA | 传统统计 | `configs/sarima.yaml` | `.pkl` |

所有深度学习模型通过 `@register_model` 装饰器自动注册；SARIMA 因不兼容 PyTorch，在 `train.py` / `test.py` 中通过 if-else 分支调用。

---

## 数据划分

```
2019-01-01 ═══ 训练集 (4年) ═══ 2023-01-01 ═══ 验证集 (1年) ═══ 2024-01-01 ═══ 测试集 ═══ 2024-12-05
```

- **训练集**：2019-01-01 ~ 2022-12-31
- **验证集**：2023-01-01 ~ 2023-12-31（深度学习用于早停；SARIMA 合并入训练集，内部 rolling validation 选参）
- **测试集**：2024-01-01 ~ 2024-12-05（CSV 数据截止日期）

---

## 特征工程

| 模型 | 输入特征 | 说明 |
|------|---------|------|
| Transformer / LSTM / CNN | `is_day_off` + 历史 `tsd` | `is_day_off` 做 `shift(-1)` 对齐到预测目标时刻，仅对 `tsd` 做 MinMax 归一化 |
| SARIMA | 仅历史 `tsd` | 纯单变量自回归，无外部特征，无归一化（通过差分处理非平稳性） |

`is_day_off`：周末或公共假日标记为 1，否则为 0。`seq_len=48` 覆盖一整天，模型从历史波形学习日内周期和季节信息，`is_day_off` 提供日历跳变信息。

---

## 统一评估指标

所有模型在测试集上使用相同的 6 个指标：

| 指标 | 含义 |
|------|------|
| MAE | 平均绝对误差 |
| MAPE | 平均绝对百分比误差 (%) |
| RMSE | 均方根误差 |
| Gaussian NLL | 高斯负对数似然（评估概率校准质量） |
| PICP | 预测区间覆盖概率 (%, 95% 区间) |
| MPIW | 平均预测区间宽度 |

---

## 统一实验设置

深度学习模型使用相同的超参数以保证公平对比：

| 参数 | 值 |
|------|-----|
| seed | 221 |
| seq_len | 48 |
| batch_size | 144 |
| learning_rate | 5e-4 |
| dropout | 0.5 |
| patience | 12 |
| num_epochs | 50 |

---

## 模型架构

### Transformer

线性投影 → 正弦位置编码 → Transformer Encoder (2层, 4头, d_model=64, ff=128) → 取最后时间步 → 双头输出 (mu, var)

### LSTM

线性投影 → LayerNorm → LSTM (2层, d_model=64) → 取最后时间步 → 双头输出 (mu, var)

### CNN

Conv1d (3层, hidden=128, BN+ReLU) → AdaptiveAvgPool1d → FC → 双头输出 (mu, var)

### SARIMA

在候选参数网格中通过 rolling daily validation 选择最优 (p,d,q)(P,D,Q,48) → 全训练集拟合 → rolling day-ahead 预测（每天预测 48 步，用真实值更新状态，固定参数）

---

## 环境配置与运行

### 依赖

基础环境 `comp0197-pt`（Python 3.12 + torch + torchvision + pillow），额外需要：
- pandas
- matplotlib
- pyyaml

SARIMA 额外需要：
- statsmodels

### 安装

```bash
micromamba create --name comp0197-pt python=3.12 -y
micromamba activate comp0197-pt
pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib pyyaml statsmodels
```

### 训练

```bash
cd elec_transformer

# Transformer
python train.py --config configs/default.yaml

# LSTM
python train.py --config configs/lstm.yaml

# CNN
python train.py --config configs/cnn.yaml

# SARIMA
python train.py --config configs/sarima.yaml
```

### 推理

```bash
# 深度学习模型（将 <run_id> 替换为训练生成的时间戳）
python test.py --config configs/default.yaml --checkpoint checkpoints/transformer_best_<run_id>.pt
python test.py --config configs/lstm.yaml --checkpoint checkpoints/lstm_best_<run_id>.pt
python test.py --config configs/cnn.yaml --checkpoint checkpoints/cnn_best_<run_id>.pt

# SARIMA
python test.py --config configs/sarima.yaml --checkpoint checkpoints/sarima_best.pkl
```

---

## 文件结构

```
comp0197/
├── README.md
├── CHANGELOG.md
└── elec_transformer/
    ├── data/
    │   └── historic_demand_2009_2024_noNaN.csv
    │
    ├── configs/
    │   ├── default.yaml          # Transformer
    │   ├── lstm.yaml             # LSTM
    │   ├── cnn.yaml              # CNN
    │   └── sarima.yaml           # SARIMA
    │
    ├── src/
    │   ├── data/
    │   │   ├── loader.py         # CSV 加载、清洗、数据划分
    │   │   ├── feature.py        # is_day_off 特征（shift(-1) 对齐）
    │   │   └── dataset.py        # 归一化、滑动窗口、DataLoader
    │   ├── models/
    │   │   ├── __init__.py       # 模型注册表 + build_model
    │   │   ├── base.py           # BaseModel(nn.Module) 基类
    │   │   ├── transformer.py    # Transformer 模型
    │   │   ├── lstm.py           # LSTM 模型
    │   │   ├── cnn.py            # CNN 模型
    │   │   └── sarima.py         # SARIMA 模型（不继承 BaseModel）
    │   ├── training/
    │   │   ├── trainer.py        # 训练循环、早停、checkpoint
    │   │   └── loss.py           # 损失函数
    │   └── evaluation/
    │       ├── metrics.py        # MAE、MAPE、RMSE、Gaussian NLL、PICP、MPIW
    │       └── visualize.py      # 可视化函数
    │
    ├── train.py                  # 训练入口（深度学习 + SARIMA 双路径）
    ├── test.py                   # 推理入口（深度学习 + SARIMA 双路径）
    ├── checkpoints/              # 模型保存目录
    └── logs/                     # 训练日志目录
```

### 扩展新的深度学习模型

只需两步，不需要修改 `train.py` 或 `test.py`：

1. 在 `src/models/` 下新建模型文件，继承 `BaseModel`，加 `@register_model("名字")` 装饰器，实现 `forward(x) -> (mu, var)` 和 `from_config(cls, model_cfg, n_features)`
2. 创建对应的 `configs/xxx.yaml`，设置 `model.type` 为注册名
