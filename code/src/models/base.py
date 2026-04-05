import torch.nn as nn


class BaseModel(nn.Module):
    """所有时序预测模型的基类。

    子类需实现：
    - forward(x) -> (mu, var)
    - from_config(cls, model_cfg, n_features) -> model instance
    """

    def forward(self, x):
        raise NotImplementedError

    @classmethod
    def from_config(cls, model_cfg: dict, n_features: int):
        """从 config 字典构造模型实例，子类需重写。"""
        raise NotImplementedError
