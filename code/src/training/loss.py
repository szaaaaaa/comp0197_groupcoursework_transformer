import torch.nn as nn


def get_criterion(name: str = "gaussian_nll") -> nn.Module:
    """根据名称返回损失函数。"""
    if name == "gaussian_nll":
        return nn.GaussianNLLLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}")
