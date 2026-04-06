# UK Electricity Demand Forecasting — Probabilistic Time Series Framework

## Overview

This project applies multiple models to probabilistic forecasting of UK electricity demand (TSD, in MW). All models output a mean and standard deviation, enabling uncertainty quantification.

The dataset covers **2019–2024** half-hourly electricity demand data from the UK National Grid, with 48 data points per day.

---

## Supported Models

| Model | Type | Config | Checkpoint |
|-------|------|--------|------------|
| Transformer | Deep learning | `configs/transformer.yaml` | `.pt` |
| LSTM | Deep learning | `configs/lstm.yaml` | `.pt` |
| CNN | Deep learning | `configs/cnn.yaml` | `.pt` |
| SARIMA | Statistical | `configs/sarima.yaml` | `.pkl` |
| LSTM (MSE) | Deterministic baseline | `configs/lstm_mse.yaml` | `.pt` |
| LSTM (no FE) | Ablation study | `configs/lstm_no_fe.yaml` | `.pt` |

All deep learning models are auto-registered via the `@register_model` decorator. SARIMA, being incompatible with PyTorch, is invoked through if-else branching in `train.py` / `test.py`.

---

## Data Split

```
2019-01-01 ═══ Train (4 yrs) ═══ 2023-01-01 ═══ Val (1 yr) ═══ 2024-01-01 ═══ Test ═══ 2024-12-05
```

- **Train**: 2019-01-01 ~ 2022-12-31
- **Validation**: 2023-01-01 ~ 2023-12-31 (early stopping for deep learning; merged into training set for SARIMA with internal rolling validation)
- **Test**: 2024-01-01 ~ 2024-12-05 (CSV data end date)

---

## Feature Engineering

| Model | Input Features | Notes |
|-------|---------------|-------|
| Transformer / LSTM / CNN | `is_day_off` + historical `tsd` | `is_day_off` shifted by -1 to align with the prediction target; MinMax normalization applied to `tsd` only |
| SARIMA | Historical `tsd` only | Univariate autoregressive, no external features, no normalization (handles non-stationarity via differencing) |

`is_day_off`: weekend or public holiday marked as 1, otherwise 0. With `seq_len=48` covering a full day, the model learns intra-day cycles and seasonal patterns from historical waveforms; `is_day_off` provides calendar shift information.

---

## Evaluation Metrics

All models are evaluated on the test set using the same 6 metrics:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error (%) |
| RMSE | Root Mean Square Error |
| Gaussian NLL | Gaussian Negative Log-Likelihood (probabilistic calibration) |
| PICP | Prediction Interval Coverage Probability (%, 95% interval) |
| MPIW | Mean Prediction Interval Width |

---

## Unified Experiment Settings

Deep learning models share the same hyperparameters for fair comparison:

| Parameter | Value |
|-----------|-------|
| seed | 221 |
| seq_len | 48 |
| batch_size | 144 |
| learning_rate | 5e-4 |
| dropout | 0.5 |
| patience | 12 |
| num_epochs | 50 |

---

## Model Architectures

### Transformer

Linear projection → Sinusoidal positional encoding → Transformer Encoder (2 layers, 4 heads, d_model=64, ff=128) → Last time step → Dual-head output (mu, var)

### LSTM

Linear projection → LSTM (2 layers, d_model=64) → LayerNorm → Last time step → Dual-head output (mu, var)

### CNN

Conv1d (3 layers, hidden=128, BN+ReLU) → AdaptiveAvgPool1d → FC → Dual-head output (mu, var)

### SARIMA

Grid search over candidate parameter specs via rolling daily validation → Select best (p,d,q)(P,D,Q,48) → Fit on full training set → Rolling day-ahead forecast (48 steps per day, update state with actuals, fixed parameters)

---

## Environment Setup and Running

### Dependencies

Base environment `comp0197-pt`, plus one additional package:

```bash
pip install statsmodels
```

> `pyyaml` and `tqdm` are also used but are typically pre-installed in the `comp0197-pt` environment.

### Running All Experiments

```bash
cd code/
python train.py --all
```

This trains all 6 models, runs inference, and saves results (metrics + figures) to `results/`.

### Training a Single Model

```bash
cd code/

# Example: LSTM
python train.py --config configs/lstm.yaml

# Example: SARIMA
python train.py --config configs/sarima.yaml
```

### Inference

```bash
# Deep learning (replace <timestamp> with actual run ID)
python test.py --config configs/lstm.yaml --checkpoint checkpoints/lstm_best_<timestamp>.pt

# SARIMA
python test.py --config configs/sarima.yaml --checkpoint checkpoints/sarima_best.pkl
```

---

## Project Structure

```
comp0197/
├── README.md
├── CHANGELOG.md
└── code/
    ├── train.py                  # Training entry point (--all for all models)
    ├── test.py                   # Inference entry point
    ├── instruction.pdf           # Reproduction instructions
    │
    ├── data/
    │   └── historic_demand_2009_2024_noNaN.csv
    │
    ├── configs/
    │   ├── transformer.yaml      # Transformer (probabilistic)
    │   ├── lstm.yaml             # LSTM (probabilistic, primary model)
    │   ├── cnn.yaml              # CNN (probabilistic)
    │   ├── sarima.yaml           # SARIMA (statistical baseline)
    │   ├── lstm_mse.yaml         # LSTM + MSE (deterministic baseline)
    │   └── lstm_no_fe.yaml       # LSTM w/o feature engineering (ablation)
    │
    ├── src/
    │   ├── data/
    │   │   ├── loader.py         # CSV loading, cleaning, data splitting
    │   │   ├── feature.py        # is_day_off feature (shift(-1) aligned)
    │   │   └── dataset.py        # Normalization, sliding window, DataLoader
    │   ├── models/
    │   │   ├── __init__.py       # Model registry + build_model
    │   │   ├── base.py           # BaseModel(nn.Module) base class
    │   │   ├── transformer.py    # Transformer model
    │   │   ├── lstm.py           # LSTM model
    │   │   ├── cnn.py            # CNN model
    │   │   ├── mamba.py          # Mamba (Selective SSM) model
    │   │   └── sarima.py         # SARIMA model (does not inherit BaseModel)
    │   ├── training/
    │   │   ├── trainer.py        # Training loop, early stopping, checkpoint
    │   │   └── loss.py           # Loss functions (GaussianNLL, MSEWrapper)
    │   └── evaluation/
    │       ├── metrics.py        # MAE, MAPE, RMSE, Gaussian NLL, PICP, MPIW
    │       └── visualize.py      # Visualization functions
    │
    ├── checkpoints/              # Saved model weights
    ├── results/                  # Output metrics and figures
    └── logs/                     # Training logs
```

### Adding a New Deep Learning Model

Two steps only — no changes to `train.py` or `test.py` needed:

1. Create a new model file under `src/models/`, inherit from `BaseModel`, add the `@register_model("name")` decorator, and implement `forward(x) -> (mu, var)` and `from_config(cls, model_cfg, n_features)`.
2. Create a corresponding `configs/xxx.yaml` with `model.type` set to the registered name.
