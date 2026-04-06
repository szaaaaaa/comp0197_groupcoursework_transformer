import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from . import register_model


class SelectiveSSM(nn.Module):
    """Selective State Space Model (core of Mamba).

    Unlike traditional SSMs, A, B, C, delta are all input-dependent
    (selective), allowing the model to dynamically decide what to
    remember and what to forget.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        State dimension of the SSM.
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state

        # Generate B, C, delta from input (input-dependent, key to "selectivity")
        self.proj_b = nn.Linear(d_model, d_state, bias=False)
        self.proj_c = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=True)

        # A parameterized in log space to ensure discretized decay coefficients in (0, 1)
        self.log_A = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).unsqueeze(0).expand(d_model, -1)
        )

        # Output projection
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """Forward pass through the selective SSM.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, d_model)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, d_model)
            Output tensor with skip connection.
        """
        batch, seq_len, d_model = x.shape

        # Generate input-dependent parameters
        B = self.proj_b(x)                              # (batch, seq_len, d_state)
        C = self.proj_c(x)                              # (batch, seq_len, d_state)
        delta = F.softplus(self.proj_delta(x))          # (batch, seq_len, d_model), positive

        # Discretize A: A_bar = exp(-delta * exp(log_A))
        A = -torch.exp(self.log_A)                      # (d_model, d_state), negative
        A_bar = torch.exp(delta.unsqueeze(-1) * A)      # (batch, seq_len, d_model, d_state)

        # Discretize B: B_bar = delta * B
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)    # (batch, seq_len, d_model, d_state)

        # Selective scan (recurrent computation)
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
    """Single Mamba block.

    Data flow::

        x -> [branch1: Conv1d -> SiLU -> SSM] x [branch2: SiLU] -> output projection

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        SSM state dimension.
    d_conv : int
        Kernel size for the causal 1-D convolution.
    expand : int
        Expansion factor for the inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.5):
        super().__init__()
        d_inner = d_model * expand

        # Input projection (expand dimensions)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal 1D convolution (local context)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner, d_state)

        # Output projection (compress back to original dim)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, d_model)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, d_model)
            Output tensor with residual connection.
        """
        residual = x
        x = self.norm(x)

        # Dual-branch projection
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Branch 1: Conv1d -> SiLU -> SSM
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :x.size(1)]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)
        x_branch = self.ssm(x_branch)

        # Branch 2: SiLU gating
        z = F.silu(z)

        # Merge
        out = self.out_proj(x_branch * z)
        out = self.dropout(out)
        return out + residual


@register_model("mamba")
class TimeSeriesMamba(BaseModel):
    """Mamba (Selective SSM) based time series forecasting model.

    Data flow::

        input(batch, seq_len, n_features)
        -> Linear projection -> MambaBlock x N -> last time step
        -> mu_head / logvar_head -> (mu, var)

    Parameters
    ----------
    n_features : int
        Number of input features.
    d_model : int
        Model dimension.
    d_state : int
        SSM state dimension.
    d_conv : int
        Convolution kernel size.
    expand : int
        Expansion factor.
    num_layers : int
        Number of Mamba blocks.
    dropout : float
        Dropout rate.
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
