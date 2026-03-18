import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from . import register_model


class SelectiveSSM(nn.Module):
    """选择性状态空间模型（Mamba 的核心）。

    与传统 SSM 的区别：A, B, C, delta 都是输入相关的（selective），
    让模型能根据输入内容动态决定"记住什么、忘记什么"。
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state

        # 从输入生成 B, C, delta（输入相关，这是"选择性"的关键）
        self.proj_b = nn.Linear(d_model, d_state, bias=False)
        self.proj_c = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=True)

        # A 用 log 空间参数化，保证离散化后的衰减系数在 (0, 1)
        self.log_A = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).unsqueeze(0).expand(d_model, -1)
        )

        # 输出投影
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # 生成输入相关的参数
        B = self.proj_b(x)                              # (batch, seq_len, d_state)
        C = self.proj_c(x)                              # (batch, seq_len, d_state)
        delta = F.softplus(self.proj_delta(x))          # (batch, seq_len, d_model)，正值

        # 离散化 A：A_bar = exp(-delta * exp(log_A))
        A = -torch.exp(self.log_A)                      # (d_model, d_state)，负值
        A_bar = torch.exp(delta.unsqueeze(-1) * A)      # (batch, seq_len, d_model, d_state)

        # 离散化 B：B_bar = delta * B
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)    # (batch, seq_len, d_model, d_state)

        # 选择性扫描（循环计算）
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)   # (batch, d_model)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                 # (batch, seq_len, d_model)
        y = y + x * self.D                              # skip connection
        return y


class MambaBlock(nn.Module):
    """单个 Mamba 块。

    数据流: x → [分支1: Conv1d → SiLU → SSM] × [分支2: SiLU] → 输出投影
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.5):
        super().__init__()
        d_inner = d_model * expand

        # 输入投影（扩展维度）
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 因果 1D 卷积（局部上下文）
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )

        # 选择性 SSM
        self.ssm = SelectiveSSM(d_inner, d_state)

        # 输出投影（压缩回原维度）
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        residual = x
        x = self.norm(x)

        # 双分支投影
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # 分支1: Conv1d → SiLU → SSM
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :x.size(1)]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)
        x_branch = self.ssm(x_branch)

        # 分支2: SiLU 门控
        z = F.silu(z)

        # 合并
        out = self.out_proj(x_branch * z)
        out = self.dropout(out)
        return out + residual


@register_model("mamba")
class TimeSeriesMamba(BaseModel):
    """基于 Mamba (Selective SSM) 的时序预测模型。

    数据流: 输入(batch, seq_len, n_features)
        -> Linear 投影 -> MambaBlock x N -> 取最后时间步
        -> mu_head / logvar_head -> (mu, var)
    """

    def __init__(self, n_features, d_model=64, d_state=16, d_conv=4,
                 expand=2, num_layers=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.mu_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x[:, -1, :]
        mu = self.mu_head(x).squeeze(-1)
        log_var = self.logvar_head(x).squeeze(-1)
        var = torch.exp(log_var)
        return mu, var

    @classmethod
    def from_config(cls, model_cfg, n_features):
        return cls(
            n_features=n_features,
            d_model=model_cfg.get("d_model", 64),
            d_state=model_cfg.get("d_state", 16),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.5),
        )
