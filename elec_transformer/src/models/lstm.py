import torch
import torch.nn as nn

from .base import BaseModel
from . import register_model


@register_model("lstm")
class TimeSeriesLSTM(BaseModel):
    """基于 LSTM 的时序预测模型。

    数据流: 输入(batch, seq_len, n_features)
        -> Linear 投影 -> LSTM x num_layers -> 取最后时间步
        -> mu_head / logvar_head -> (mu, var)

    特点：
    - 可选的输入投影层（当 d_model != n_features 时自动启用）
    - 多层 LSTM，支持 dropout（层间）
    - 双头输出：mu_head 预测均值，logvar_head 预测 log 方差
    """

    def __init__(self, n_features, d_model=64, num_layers=2, dropout=0.5,
                 bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.num_directions = 2 if bidirectional else 1

        # 输入投影: 将原始特征维度映射到 d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # LSTM 主体
        # dropout 仅在 num_layers > 1 时生效（作用于层间）
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(d_model * self.num_directions)

        # 输出头: 均值和 log 方差
        head_input_dim = d_model * self.num_directions
        self.mu_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        返回: (mu, var)，各 shape 为 (batch,)
        """
        # 投影到 d_model 维
        x = self.input_proj(x)              # (batch, seq_len, d_model)

        # LSTM 编码
        output, (h_n, _) = self.lstm(x)     # output: (batch, seq_len, d_model * num_directions)

        # 取最后时间步的输出
        x = output[:, -1, :]               # (batch, d_model * num_directions)
        x = self.norm(x)

        # 双头输出
        mu = self.mu_head(x).squeeze(-1)            # (batch,)
        log_var = self.logvar_head(x).squeeze(-1)   # (batch,)
        var = torch.exp(log_var)

        return mu, var

    @classmethod
    def from_config(cls, model_cfg, n_features):
        return cls(
            n_features=n_features,
            d_model=model_cfg.get("d_model", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.5),
            bidirectional=model_cfg.get("bidirectional", False),
        )
