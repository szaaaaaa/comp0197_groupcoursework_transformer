# 更新日志

本日志根据仓库中的关键 Git 提交整理，默认忽略纯合并提交；提交哈希仅用于追溯。

## 2026-03-18

### 训练图像归档

- 调整 `train.py` 的可视化收尾逻辑，在保留 `plt.show()` 的同时，将当前运行生成的图像一并保存到对应的 `logs/run_<run_id>/` 子目录。
- 当前每次训练运行会额外保留 `split.png`、`loss_curve.png`、`predictions.png` 和 `detail_2024-08-01_2024-08-14.png`，便于离线查看和结果留档。

对应提交：
- 本次提交

### 文档补充与说明对齐

- 补充 `README` 中缺失的 `pyyaml` 依赖说明，并将额外依赖数量更新为 3 个。
- 补充训练产物说明，明确 checkpoint 实际保存为 `checkpoints/<model>_best_<run_id>.pt`。
- 补充 `logs/run_<run_id>/train.log` 与 `result.json` 的输出位置。
- 在文档中补充 `mamba` 模型与 `configs/mamba.yaml` 的使用说明。

对应提交：
- `f61fb6f` `docs: 补充README依赖与训练产物说明`

### 训练与推理流程模块化

- 将原先主要依赖 Notebook 的流程拆分为模块化代码结构，新增 `src/data`、`src/models`、`src/training`、`src/evaluation`。
- 新增 `train.py` 与 `predict.py`，支持通过 YAML 配置进行训练和推理。
- 新增 `default.yaml`、`no_fe.yaml`、`mamba.yaml`，支持不同实验配置。
- 引入模型自动注册机制，支持按 `model.type` 动态构建模型。
- 新增 `mamba` 模型实现，并补充无特征工程版本 Notebook。
- 新增按运行时间戳写入的训练日志与 checkpoint 保存逻辑。
- 更新根目录 `README`，补充命令行运行方式、文件结构和扩展模型说明。

对应提交：
- `4c694ae` `feat: 完善电力预测训练推理流程`

## 2026-03-17

### 数据范围与特征方案调整

- 将建模数据范围固定为 2019-2024，以降低 COVID-19、能源危机等概念漂移带来的影响。
- 将训练/验证/测试划分明确为 2019-2022、2023、2024。
- 删除 `lag1/lag2/lag3` 特征，避免在仅保留 2019+ 数据时引入不稳定依赖。
- 将归一化范围扩展为全部 8 个输入特征加目标变量。
- 更新 Notebook 与文档说明，并记录当时结果为 `MAPE 6.66%`、`RMSE 2400.05 MW`。

对应提交：
- `7bebd71` `修改调用数据年限（2019-2024）与训练集划分（2019-2022；2023；2024）`
- `1ea004d` `Remove lag features, use 2019+ data only, normalize all features`

### 仓库整理

- 将 `cw1` 加入 `.gitignore`，避免课程作业目录继续进入版本控制。

对应提交：
- `60110f6` `add cw1 to gitignore`

## 2026-03-16

### Transformer Notebook 首轮重构

- 将数据划分从原先存在信息泄漏风险的方式调整为 `train / val / test`。
- 引入滞后特征方案，并移除 `year` 特征。
- 将模型输出改为概率预测形式，使用均值与方差建模。
- 使用 `GaussianNLL` 作为训练目标，并将图表标签切换为英文。
- 为项目新增说明文档，开始形成较完整的实验说明。

对应提交：
- `628d878` `Improve Transformer notebook: train/val/test split, lag features, probabilistic output`

### 信息泄漏修复与概率预测完善

- 修复滞后特征构造中的数据泄漏问题，确保仅使用训练集 `tsd` 值生成相关特征。
- 将确定性输出进一步替换为高斯分布输出 `(mu, var)`。
- 增加 95% 置信区间可视化。
- 新增 `pyproject.toml`，开始整理项目依赖。

对应提交：
- `7626f5d` `Fix lag feature data leakage and add probabilistic output`

### 其他基础整理

- 新增 `.gitignore`，忽略 `.claude/`。
- 调整 README 位置，并将安装说明改为更通用的表述。

对应提交：
- `5f2700a` `Add .gitignore to exclude .claude directory`
- `d1d7085` `move readme`
- `b2fef5b` `Update README: make setup instructions generic for all users`
