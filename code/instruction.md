# Instruction: Reproducing All Reported Results

## 1. Additional Packages

The following **3 additional packages** must be installed on top of the `comp0197-pt` micromamba environment:

| # | Package       | Version  | Purpose                              |
|---|---------------|----------|--------------------------------------|
| 1 | `pyyaml`      | >= 6.0   | Loading YAML configuration files     |
| 2 | `statsmodels` | >= 0.14  | SARIMA baseline model                |
| 3 | `tqdm`        | >= 4.60  | Training progress bars               |

**Installation command:**

```bash
micromamba activate comp0197-pt
pip install pyyaml statsmodels tqdm
```

## 2. Project Structure

```
code/
├── train.py                  # Training entry point
├── test.py                   # Inference and evaluation entry point
├── configs/                  # Model configurations (YAML)
│   ├── transformer.yaml      # Transformer (probabilistic)
│   ├── lstm.yaml             # LSTM (probabilistic, primary model)
│   ├── cnn.yaml              # CNN (probabilistic)
│   ├── sarima.yaml           # SARIMA (statistical baseline)
│   ├── lstm_mse.yaml         # LSTM with MSE loss (deterministic baseline)
│   └── lstm_no_fe.yaml       # LSTM without feature engineering (ablation)
├── data/
│   └── historic_demand_2009_2024_noNaN.csv
├── checkpoints/              # Saved model weights (included)
├── results/                  # Output metrics and figures
│   ├── <experiment>/metrics.json
│   ├── <experiment>/predictions.png
│   ├── <experiment>/detail.png
│   └── summary.csv
├── src/
│   ├── data/                 # Data loading, feature engineering, dataset
│   ├── models/               # Model definitions (Transformer, LSTM, CNN, SARIMA)
│   ├── training/             # Loss functions, training loop
│   └── evaluation/           # Metrics and visualization
└── logs/                     # Training logs
```

## 3. Steps to Reproduce All Results

### Step 1: Activate the environment

```bash
micromamba activate comp0197-pt
pip install pyyaml statsmodels tqdm
```

### Step 2: Navigate to the code directory

```bash
cd code/
```

### Step 3: Train all models and generate all results (one command)

```bash
python train.py --all
```

This single command performs the following in sequence:

1. **Trains all 6 models** using their respective configuration files:
   - Transformer (probabilistic, Gaussian NLL loss)
   - LSTM (probabilistic, Gaussian NLL loss)
   - CNN (probabilistic, Gaussian NLL loss)
   - SARIMA (statistical baseline with grid search)
   - LSTM with MSE loss (deterministic baseline)
   - LSTM without feature engineering (ablation study)

2. **Runs inference** (`test.py`) for each trained model on the test set.

3. **Saves results** for each model:
   - `results/<model>/metrics.json` — MAE, MAPE, RMSE, NLL, PICP, MPIW
   - `results/<model>/predictions.png` — full test set prediction plot with confidence intervals
   - `results/<model>/detail.png` — zoomed-in 2-week detail view

4. **Prints and saves a summary table** to `results/summary.csv`.

### Alternative: Train and test a single model

```bash
# Train
python train.py --config configs/lstm.yaml

# Test (replace checkpoint path with the actual saved file)
python test.py --config configs/lstm.yaml --checkpoint checkpoints/lstm_best_<timestamp>.pt
```

## 4. Using Pre-trained Checkpoints

If pre-trained checkpoints are included in the `checkpoints/` directory, you can skip training and run inference directly:

```bash
python test.py --config configs/lstm.yaml --checkpoint checkpoints/lstm_best_<timestamp>.pt
python test.py --config configs/sarima.yaml --checkpoint checkpoints/sarima_best.pkl
```

## 5. Expected Output

After running `python train.py --all`, the `results/` directory will contain:

- `results/transformer/` — Transformer model results
- `results/lstm/` — LSTM model results
- `results/cnn/` — CNN model results
- `results/sarima/` — SARIMA baseline results
- `results/lstm_mse/` — Deterministic LSTM baseline results
- `results/lstm_no_fe/` — LSTM ablation (no feature engineering) results
- `results/summary.csv` — Consolidated metrics table for all models

Each experiment directory contains `metrics.json`, `predictions.png`, and `detail.png`.

## 6. Hardware and Runtime

- **Framework:** PyTorch (`comp0197-pt` environment)
- **GPU:** Training automatically uses CUDA if available; CPU is also supported.
- **Estimated runtime:** Approximately 30-60 minutes for all 6 models on CPU (faster with GPU). SARIMA grid search is the most time-consuming step.
