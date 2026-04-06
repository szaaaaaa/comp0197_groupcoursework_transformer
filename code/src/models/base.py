import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all time series forecasting models.

    Subclasses must implement:

    * ``forward(x)`` -> ``(mu, var)``
    * ``from_config(cls, model_cfg, n_features)`` -> model instance
    """

    def forward(self, x):
        raise NotImplementedError

    @classmethod
    def from_config(cls, model_cfg: dict, n_features: int):
        """Construct a model instance from a config dict.

        Parameters
        ----------
        model_cfg : dict
            Model-specific configuration entries.
        n_features : int
            Number of input features.

        Returns
        -------
        BaseModel
            Constructed model instance.
        """
        raise NotImplementedError
