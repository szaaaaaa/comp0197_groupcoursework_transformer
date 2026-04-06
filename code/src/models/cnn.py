import torch
import torch.nn as nn

from .base import BaseModel
from . import register_model


@register_model("cnn")
class TimeSeriesCNN(BaseModel):
    """1D-CNN based time series forecasting model.

    Data flow::

        input(batch, seq_len, n_features)
        -> permute -> Conv1d x 3 -> AdaptiveAvgPool1d
        -> FC -> mu_head / logvar_head -> (mu, var)

    Parameters
    ----------
    n_features : int
        Number of input features (channels for Conv1d).
    hidden_dim : int
        Number of filters in the first two convolutional layers.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_features, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

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
            Predicted variance.
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))

        x = self.pool(x).squeeze(-1)   # (batch, hidden_dim * 2)

        x = self.fc(x)                 # (batch, hidden_dim)

        mu = self.mu_head(x).squeeze(-1)            # (batch,)
        log_var = self.logvar_head(x).squeeze(-1)   # (batch,)
        var = torch.exp(log_var)

        return mu, var

    @classmethod
    def from_config(cls, model_cfg, n_features):
        return cls(
            n_features=n_features,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            dropout=model_cfg.get("dropout", 0.2),
        )
