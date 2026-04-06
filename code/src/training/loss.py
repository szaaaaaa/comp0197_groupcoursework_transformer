import torch.nn as nn


class MSEWrapper(nn.Module):
    """Wrap ``nn.MSELoss`` to match the ``GaussianNLLLoss`` call signature.

    The third positional argument (*var*) is accepted but ignored so that
    the training loop can call ``criterion(mu, target, var)`` uniformly.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, mu, target, var=None):
        return self.mse(mu, target)


def get_criterion(name: str = "gaussian_nll") -> nn.Module:
    """Return a loss function by name.

    Parameters
    ----------
    name : str
        ``"gaussian_nll"`` or ``"mse"``.

    Returns
    -------
    nn.Module
        Loss module.

    Raises
    ------
    ValueError
        If *name* is not recognized.
    """
    if name == "gaussian_nll":
        return nn.GaussianNLLLoss()
    elif name == "mse":
        return MSEWrapper()
    else:
        raise ValueError(f"Unknown loss: {name}")
