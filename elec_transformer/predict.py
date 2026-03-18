"""
推理入口：加载 checkpoint 对测试集进行预测。

用法:
    python predict.py --config configs/default.yaml --checkpoint checkpoints/transformer_best.pt
"""
import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loader import load_and_clean, split_data
from src.data.feature import create_features
from src.data.dataset import TimeSeriesDataset
from src.models import build_model
from src.evaluation.metrics import mape, rmse
from src.evaluation.visualize import setup_matplotlib, plot_predictions, plot_detail


def main(config_path: str, checkpoint_path: str):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    df = load_and_clean(cfg["data"]["csv_path"], cfg["data"]["date_start"], cfg["data"]["date_end"])

    feature_cfg = cfg["features"]
    if feature_cfg["enabled"]:
        df = create_features(df)
        features = feature_cfg["columns"]
        cols_to_scale = feature_cfg["cols_to_scale"]
        if cols_to_scale == "all":
            cols_to_scale = features + [cfg["data"]["target"]]
    else:
        features = []
        cols_to_scale = "all"

    train_df, val_df, test_df = split_data(
        df, cfg["data"]["threshold_date_1"], cfg["data"]["threshold_date_2"]
    )

    dataset = TimeSeriesDataset(
        train_df, val_df, test_df,
        features=features,
        target=cfg["data"]["target"],
        cols_to_scale=cols_to_scale,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
    )
    _, _, test_loader = dataset.get_loaders()

    # 模型
    model = build_model(cfg, dataset.n_features, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 预测
    mu_list, std_list = [], []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            mu, var = model(X_batch)
            mu_list.append(mu.cpu().numpy())
            std_list.append(torch.sqrt(var).cpu().numpy())

    pred_scaled = np.concatenate(mu_list)
    std_scaled = np.concatenate(std_list)
    pred, std = dataset.inverse_transform(pred_scaled, std_scaled)

    result = pd.DataFrame({
        "tsd": dataset.test_actual,
        "pred": pred,
        "std": std,
    }, index=dataset.test_index)

    print(f"MAPE: {mape(result['tsd'], result['pred']):.2f}%")
    print(f"RMSE: {rmse(result['tsd'], result['pred']):.2f} MW")

    # 可视化
    setup_matplotlib()
    plot_predictions(result)
    plot_detail(result, "08-01-2024", "08-14-2024")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_best.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
