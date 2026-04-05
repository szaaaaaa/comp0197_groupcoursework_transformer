"""
推理入口：加载 checkpoint 对测试集进行预测。

用法:
    python test.py --config configs/default.yaml --checkpoint checkpoints/transformer_best.pt
    python test.py --config configs/sarima.yaml --checkpoint checkpoints/sarima_best.pkl
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
from src.evaluation.metrics import mae, mape, rmse, gaussian_nll, picp, mpiw
from src.evaluation.visualize import setup_matplotlib, plot_predictions, plot_detail


def main(config_path: str, checkpoint_path: str):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])

    model_type = cfg["model"]["type"]

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

    target = cfg["data"]["target"]

    if model_type == "sarima":
        # ============ SARIMA 路径 ============
        from src.models.sarima import load_model, predict_sarima

        spec, params = load_model(checkpoint_path)
        full_train = pd.concat([train_df[target], val_df[target]])
        test_series = test_df[target]

        forecast_df = predict_sarima(params, spec, full_train, test_series)

        result = pd.DataFrame({
            "tsd": forecast_df["actual"].values,
            "pred": forecast_df["prediction"].values,
            "std": forecast_df["prediction_std"].values,
        }, index=forecast_df.index)

    else:
        # ============ 深度学习路径 ============
        torch.manual_seed(cfg["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = TimeSeriesDataset(
            train_df, val_df, test_df,
            features=features,
            target=target,
            cols_to_scale=cols_to_scale,
            seq_len=cfg["seq_len"],
            batch_size=cfg["batch_size"],
        )
        _, _, test_loader = dataset.get_loaders()

        model = build_model(cfg, dataset.n_features, device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

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

    # ---- 统一评估 ----
    print(f"MAE:  {mae(result['tsd'], result['pred']):.2f} MW")
    print(f"MAPE: {mape(result['tsd'], result['pred']):.2f}%")
    print(f"RMSE: {rmse(result['tsd'], result['pred']):.2f} MW")
    print(f"NLL:  {gaussian_nll(result['tsd'], result['pred'], result['std']):.4f}")
    print(f"PICP: {picp(result['tsd'], result['pred'], result['std']):.2f}%")
    print(f"MPIW: {mpiw(result['std']):.2f} MW")

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
