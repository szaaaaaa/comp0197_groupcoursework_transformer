import torch
import torch.nn as nn

from .base import BaseModel
from . import register_model


@register_model("lstm")
class TimeSeriesLSTM(BaseModel):
    """LSTM-based time series forecasting model.

    Data flow::

        input(batch, seq_len, n_features)
        -> Linear projection -> LSTM x num_layers -> last time step
        -> mu_head / logvar_head -> (mu, var)

    Parameters
    ----------
    n_features : int
        Number of input features.
    d_model : int
        Hidden dimension for LSTM and projection.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate (applied between LSTM layers and in output heads).
    bidirectional : bool
        Whether to use bidirectional LSTM.
    """

    def __init__(self, n_features, d_model=64, num_layers=2, dropout=0.5,
                 bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_model = d_model
        self.num_directions = 2 if bidirectional else 1

        # Input projection: map raw feature dim to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # LSTM backbone
        # dropout only applies between layers when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(d_model * self.num_directions)

        # Output heads: mean and log variance
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
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, n_features)
            Input time series.

        Returns
        -------
        mu : torch.Tensor, shape (batch,)
            Predicted mean.
        var : torch.Tensor, shape (batch,)
            Predicted variance (exp of log-variance).
        """
        # Project to d_model dim
        x = self.input_proj(x)              # (batch, seq_len, d_model)

        # LSTM encoding
        output, (h_n, _) = self.lstm(x)     # output: (batch, seq_len, d_model * num_directions)

        # Take last time step output
        x = output[:, -1, :]               # (batch, d_model * num_directions)
        x = self.norm(x)

        # Dual-head output
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
