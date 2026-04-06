"""
Training entry point.

Usage:
    python train.py --config configs/lstm.yaml       # Train a single model
    python train.py --all                            # Train all models + inference + summary
"""
import argparse
import glob
import json
import os
import subprocess
import sys

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

# All experiment configurations
ALL_CONFIGS = [
    "configs/transformer.yaml",
    "configs/lstm.yaml",
    "configs/cnn.yaml",
    "configs/sarima.yaml",
    "configs/lstm_mse.yaml",
    "configs/lstm_no_fe.yaml",
]


def train_single(config_path: str):
    """Train a single model and save its checkpoint.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    np.random.seed(seed)
    model_type = cfg["model"]["type"]

    # ---- Data ----
    df = load_and_clean(
        cfg["data"]["csv_path"],
        cfg["data"]["date_start"],
        cfg["data"]["date_end"],
    )

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
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    target = cfg["data"]["target"]

    if model_type == "sarima":
        # ============ SARIMA path ============
        from src.models.sarima import train_sarima
        train_sarima(train_df[target], val_df[target], cfg)

    else:
        # ============ Deep learning path ============
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        dataset = TimeSeriesDataset(
            train_df, val_df, test_df,
            features=features,
            target=target,
            cols_to_scale=cols_to_scale,
            seq_len=cfg["seq_len"],
            batch_size=cfg["batch_size"],
        )
        train_loader, val_loader, _ = dataset.get_loaders()
        print(f"Features: {dataset.n_features}, Seq len: {cfg['seq_len']}")

        model = build_model(cfg, dataset.n_features, device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        tcfg = cfg["training"]
        loss_name = tcfg.get("loss", "gaussian_nll")
        criterion = get_criterion(loss_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=tcfg["lr_factor"], patience=tcfg["lr_patience"]
        )

        ckpt_suffix = f"_{loss_name}" if loss_name != "gaussian_nll" else ""
        fe_suffix = "" if feature_cfg["enabled"] else "_no_fe"
        ckpt_path = os.path.join("checkpoints", f"{model_type}{ckpt_suffix}{fe_suffix}_best.pt")
        trainer = Trainer(
            model, criterion, optimizer, scheduler, device,
            patience=tcfg["patience"],
            checkpoint_path=ckpt_path,
            loss_name=loss_name.upper(),
        )
        trainer.train(train_loader, val_loader, tcfg["num_epochs"])

        # Save training log
        trainer.save_log(config=cfg)

    print(f"\nTraining complete: {config_path}")


# ---- --all mode ----

CHECKPOINT_PATTERNS = {
    "configs/transformer.yaml":    "checkpoints/transformer_best_*.pt",
    "configs/lstm.yaml":           "checkpoints/lstm_best_*.pt",
    "configs/cnn.yaml":            "checkpoints/cnn_best_*.pt",
    "configs/sarima.yaml":         "checkpoints/sarima_best.pkl",
    "configs/lstm_mse.yaml":       "checkpoints/lstm_mse_best_*.pt",
    "configs/lstm_no_fe.yaml":     "checkpoints/lstm_no_fe_best_*.pt",
}


def find_latest_checkpoint(pattern):
    """Return the most recently modified file matching *pattern*, or None.

    Parameters
    ----------
    pattern : str
        Glob pattern for checkpoint files.

    Returns
    -------
    str or None
        Path to the latest checkpoint, or ``None`` if no match.
    """
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def run_all():
    """Train all models, run inference, and summarize results."""
    valid_configs = [c for c in ALL_CONFIGS if os.path.exists(c)]
    total = len(valid_configs)

    # 1. Train
    for i, config in enumerate(valid_configs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] TRAINING: {config}")
        print(f"{'='*60}")
        train_single(config)

    # 2. Inference
    for i, config in enumerate(valid_configs, 1):
        pattern = CHECKPOINT_PATTERNS.get(config)
        if not pattern:
            continue
        ckpt = find_latest_checkpoint(pattern)
        if ckpt is None:
            print(f"SKIP test: no checkpoint for {config}")
            continue
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] TESTING: {config}")
        print(f"{'='*60}")
        subprocess.run([sys.executable, "test.py", "--config", config, "--checkpoint", ckpt])

    # 3. Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    results_files = sorted(glob.glob("results/*/metrics.json"))
    if not results_files:
        print("No results found.")
        return

    rows = []
    for path in results_files:
        exp_name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            m = json.load(f)
        m["Experiment"] = exp_name
        rows.append(m)

    import pandas as pd
    df = pd.DataFrame(rows).set_index("Experiment")
    df = df[["MAE", "MAPE", "RMSE", "NLL", "PICP", "MPIW"]]

    print(df.to_string())

    summary_path = os.path.join("results", "summary.csv")
    df.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml", help="Single model config file")
    parser.add_argument("--all", action="store_true", help="Train all models + inference + summary")
    args = parser.parse_args()

    if args.all:
        run_all()
    else:
        train_single(args.config)
