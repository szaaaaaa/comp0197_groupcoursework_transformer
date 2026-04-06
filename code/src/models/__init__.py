MODEL_REGISTRY = {}


def register_model(name):
    """Register a model class into the global registry.

    Parameters
    ----------
    name : str
        Key used to look up the model class.

    Returns
    -------
    callable
        Class decorator.
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(cfg, n_features, device):
    """Auto-discover and build a model from config.

    All modules under ``src/models/`` are imported to trigger
    ``@register_model`` decorators before looking up the requested type.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration (must contain ``cfg["model"]["type"]``).
    n_features : int
        Number of input features.
    device : torch.device
        Device to place the model on.

    Returns
    -------
    nn.Module
        Constructed model on *device*.

    Raises
    ------
    ValueError
        If the requested model type is not registered.
    """
    import torch
    # Auto-import all modules under src/models/ to trigger @register_model
    import importlib
    import pkgutil
    import src.models as pkg
    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"src.models.{module_name}")

    model_type = cfg["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls.from_config(cfg["model"], n_features)
    return model.to(device)
