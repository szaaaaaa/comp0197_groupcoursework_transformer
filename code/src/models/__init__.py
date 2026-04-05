MODEL_REGISTRY = {}


def register_model(name):
    """装饰器：将模型类注册到全局字典。"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(cfg, n_features, device):
    """根据 config 自动查找并构建模型，无需硬编码。"""
    import torch
    # 自动导入 src/models/ 下所有模块，触发 @register_model
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
