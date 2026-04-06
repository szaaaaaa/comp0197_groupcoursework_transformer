"""
Inference entry point: load checkpoint and predict on the test set.

Usage:
    python test.py --config configs/lstm.yaml --checkpoint checkpoints/lstm_best_xxx.pt
    python test.py --all                     # Test all models + summarize
"""
import argparse
import glob
import json
import os

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


# All experiment configurations
ALL_CONFIGS = [
    "configs/transformer.yaml",
    "configs/lstm.yaml",
    "configs/cnn.yaml",
    "configs/sarima.yaml",
    "configs/lstm_mse.yaml",
    "configs/lstm_no_fe.yaml",
    "configs/transformer_mse.yaml",
    "configs/transformer_no_fe.yaml",
]

CHECKPOINT_PATTERNS = {
    "configs/transformer.yaml":        "checkpoints/transformer_best_*.pt",
    "configs/lstm.yaml":               "checkpoints/lstm_best_*.pt",
    "configs/cnn.yaml":                "checkpoints/cnn_best_*.pt",
    "configs/sarima.yaml":             "checkpoints/sarima_best.pkl",
    "configs/lstm_mse.yaml":           "checkpoints/lstm_mse_best_*.pt",
    "configs/lstm_no_fe.yaml":         "checkpoints/lstm_no_fe_best_*.pt",
    "configs/transformer_mse.yaml":    "checkpoints/transformer_mse_best_*.pt",
    "configs/transformer_no_fe.yaml":  "checkpoints/transformer_no_fe_best_*.pt",
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


def get_experiment_name(cfg):
    """Generate experiment name from config for the output directory.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration.

    Returns
    -------
    str
        Experiment name (e.g. ``"lstm"``, ``"lstm_mse"``, ``"lstm_no_fe"``).
    """
    model_type = cfg["model"]["type"]
    loss_name = cfg.get("training", {}).get("loss", "gaussian_nll")
    fe = cfg["features"]["enabled"]

    name = model_type
    if loss_name != "gaussian_nll":
        name += f"_{loss_name}"
    if not fe and model_type != "sarima":
        name += "_no_fe"
    return name


def test_single(config_path: str, checkpoint_path: str, out_root: str = "results"):
    """Run inference on the test set, compute metrics, and save figures.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    checkpoint_path : str
        Path to the saved model checkpoint.
    out_root : str
        Root directory for saving results (e.g. ``results/run_20260406``).
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])

    model_type = cfg["model"]["type"]
    exp_name = get_experiment_name(cfg)

    # Output directory
    out_dir = os.path.join(out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Data
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
        # ============ SARIMA path ============
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
        # ============ Deep learning path ============
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

    # ---- Unified evaluation ----
    metrics = {
        "MAE": round(float(mae(result["tsd"], result["pred"])), 2),
        "MAPE": round(float(mape(result["tsd"], result["pred"])), 2),
        "RMSE": round(float(rmse(result["tsd"], result["pred"])), 2),
        "NLL": round(float(gaussian_nll(result["tsd"], result["pred"], result["std"])), 4),
        "PICP": round(float(picp(result["tsd"], result["pred"], result["std"])), 2),
        "MPIW": round(float(mpiw(result["std"])), 2),
    }

    print(f"\n[{exp_name}]")
    print(f"MAE:  {metrics['MAE']:.2f} MW")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: {metrics['RMSE']:.2f} MW")
    print(f"NLL:  {metrics['NLL']:.4f}")
    print(f"PICP: {metrics['PICP']:.2f}%")
    print(f"MPIW: {metrics['MPIW']:.2f} MW")

    # Save metrics to JSON
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_dir}/metrics.json")

    # ---- Visualization (save to files) ----
    setup_matplotlib()
    loss_name = cfg.get("training", {}).get("loss", "gaussian_nll")
    show_ci = loss_name != "mse"

    fig1 = plot_predictions(result, title=f"{exp_name} — Test Set Predictions", show_ci=show_ci)
    fig1.savefig(os.path.join(out_dir, "predictions.png"), dpi=200, bbox_inches="tight")

    fig2 = plot_detail(result, "08-01-2024", "08-14-2024", show_ci=show_ci)
    fig2.savefig(os.path.join(out_dir, "detail.png"), dpi=200, bbox_inches="tight")

    print(f"Figures saved to {out_dir}/")

    import matplotlib.pyplot as plt
    plt.close("all")


def summarize(run_dir):
    """Read all metrics.json under *run_dir* and print + save summary table.

    Parameters
    ----------
    run_dir : str
        Path to the run directory (e.g. ``results/run_20260406``).
    """
    results_files = sorted(glob.glob(os.path.join(run_dir, "*/metrics.json")))
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

    df = pd.DataFrame(rows).set_index("Experiment")
    df = df[["MAE", "MAPE", "RMSE", "NLL", "PICP", "MPIW"]]

    print(df.to_string())

    summary_path = os.path.join(run_dir, "summary.csv")
    df.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")


def run_all():
    """Test all models with available checkpoints, then summarize.

    Creates a timestamped run directory under ``results/``.
    """
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    valid_configs = [c for c in ALL_CONFIGS if os.path.exists(c)]
    total = len(valid_configs)

    for i, config in enumerate(valid_configs, 1):
        pattern = CHECKPOINT_PATTERNS.get(config)
        if not pattern:
            continue
        ckpt = find_latest_checkpoint(pattern)
        if ckpt is None:
            print(f"SKIP: no checkpoint for {config}")
            continue
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] TESTING: {config}")
        print(f"{'='*60}")
        test_single(config, ckpt, out_root=run_dir)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summarize(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--all", action="store_true", help="Test all models + summarize")
    args = parser.parse_args()

    if args.all:
        run_all()
    else:
        # Auto-find checkpoint if not specified
        if args.checkpoint is None:
            pattern = CHECKPOINT_PATTERNS.get(args.config)
            if pattern:
                args.checkpoint = find_latest_checkpoint(pattern)
            if args.checkpoint is None:
                parser.error(f"No checkpoint found for {args.config}. Specify --checkpoint.")
        test_single(args.config, args.checkpoint)
