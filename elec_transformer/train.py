"""
入口脚本：加载 config → 数据 → 模型 → 训练 → 评估

用法:
    python train.py                          # 使用默认配置（含特征工程）
    python train.py --config configs/no_fe.yaml  # 使用指定配置
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loader import load_and_clean, split_data
from src.data.feature import create_features
from src.data.dataset import TimeSeriesDataset
from src.models import build_model
from src.training.loss import get_criterion
from src.training.trainer import Trainer
from src.evaluation.metrics import mape, rmse
from src.evaluation.visualize import (
    setup_matplotlib, plot_split, plot_loss_curve, plot_predictions, plot_detail,
)


def main(config_path: str):
    # 加载配置
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 随机种子
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 数据加载与清洗 ----
    df = load_and_clean(
        cfg["data"]["csv_path"],
        cfg["data"]["date_start"],
        cfg["data"]["date_end"],
    )

    # ---- 特征工程（可选） ----
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

    # ---- 数据划分 ----
    train_df, val_df, test_df = split_data(
        df, cfg["data"]["threshold_date_1"], cfg["data"]["threshold_date_2"]
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ---- 构建 Dataset 与 DataLoader ----
    dataset = TimeSeriesDataset(
        train_df, val_df, test_df,
        features=features,
        target=cfg["data"]["target"],
        cols_to_scale=cols_to_scale,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
    )
    train_loader, val_loader, test_loader = dataset.get_loaders()
    print(f"Features: {dataset.n_features}, Seq len: {cfg['seq_len']}")

    # ---- 构建模型 ----
    model = build_model(cfg, dataset.n_features, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- 训练 ----
    tcfg = cfg["training"]
    criterion = get_criterion("gaussian_nll")
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg["lr_factor"], patience=tcfg["lr_patience"]
    )

    ckpt_path = os.path.join("checkpoints", f"{cfg['model']['type']}_best.pt")
    trainer = Trainer(
        model, criterion, optimizer, scheduler, device,
        patience=tcfg["patience"],
        checkpoint_path=ckpt_path,
    )
    trainer.train(train_loader, val_loader, tcfg["num_epochs"])

    # ---- 预测与评估 ----
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

    # 构建结果 DataFrame
    result = pd.DataFrame({
        "tsd": dataset.test_actual,
        "pred": pred,
        "std": std,
    }, index=dataset.test_index)

    mape_val = mape(result["tsd"], result["pred"])
    rmse_val = rmse(result["tsd"], result["pred"])
    print(f"\nMAPE: {mape_val:.2f}%")
    print(f"RMSE: {rmse_val:.2f} MW")

    # ---- 保存日志 ----
    trainer.save_log(
        metrics={"mape": round(mape_val, 2), "rmse": round(rmse_val, 2)},
        config=cfg,
    )

    # ---- 可视化 ----
    setup_matplotlib()
    figures = {
        "split.png": plot_split(
            train_df,
            val_df,
            test_df,
            cfg["data"]["threshold_date_1"],
            cfg["data"]["threshold_date_2"],
        ),
        "loss_curve.png": plot_loss_curve(trainer.train_losses, trainer.val_losses),
        "predictions.png": plot_predictions(result),
        "detail_2024-08-01_2024-08-14.png": plot_detail(result, "08-01-2024", "08-14-2024"),
    }

    for filename, fig in figures.items():
        fig.savefig(os.path.join(trainer.run_dir, filename), dpi=200, bbox_inches="tight")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
