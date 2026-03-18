import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel
from . import register_model


class PositionalEncoding(nn.Module):
    """正弦-余弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


@register_model("transformer")
class TimeSeriesTransformer(BaseModel):
    """基于 Transformer Encoder 的时序预测模型。

    数据流: 输入(batch, seq_len, n_features)
        -> Linear 投影 -> 位置编码 -> Transformer Encoder x N
        -> 取最后时间步 -> mu_head / logvar_head -> (mu, var)
    """

    def __init__(self, n_features, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        mu = self.mu_head(x).squeeze(-1)
        log_var = self.logvar_head(x).squeeze(-1)
        var = torch.exp(log_var)
        return mu, var

    @classmethod
    def from_config(cls, model_cfg, n_features):
        return cls(
            n_features=n_features,
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_layers=model_cfg["num_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
        )
