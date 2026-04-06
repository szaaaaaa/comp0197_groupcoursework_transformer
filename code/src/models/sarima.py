"""
SARIMA model wrapper.

Does not inherit BaseModel(nn.Module) since SARIMA is a traditional
statistical model without PyTorch.  Invoked via if-else branching in
train.py / test.py.
"""

import os
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


# ---- Constants ----
SEASONAL_PERIOD = 48
FIT_METHOD = "statespace"
FIT_COV_TYPE = "opg"
SAVED_FIT_COV_TYPE = "none"
Z_SCORE_95 = 1.959963984540054


@dataclass(frozen=True)
class SarimaSpec:
    order: tuple
    seasonal_order: tuple

    @property
    def label(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return f"SARIMA({p},{d},{q})x({P},{D},{Q},{s})"


def parse_candidate_specs(specs_cfg, seasonal_period):
    """Parse candidate SARIMA parameter specs from YAML config.

    Parameters
    ----------
    specs_cfg : list of dict
        Each dict must contain ``order`` and ``seasonal_order`` keys.
    seasonal_period : int
        Seasonal period appended to ``seasonal_order``.

    Returns
    -------
    list of SarimaSpec
        Parsed candidate specifications.
    """
    return [
        SarimaSpec(
            order=tuple(s["order"]),
            seasonal_order=tuple(s["seasonal_order"]) + (seasonal_period,),
        )
        for s in specs_cfg
    ]


# ---- Utility functions ----

def _configure_warnings():
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Provided `endog` series has been differenced")
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)


def _as_sequential(series, start=0):
    """Convert datetime index to sequential integer index.

    Avoids DST gap issues that statsmodels cannot handle.

    Parameters
    ----------
    series : pd.Series
        Series with datetime index.
    start : int
        Starting integer index.

    Returns
    -------
    pd.Series
        Same values with a ``RangeIndex``.
    """
    return pd.Series(
        series.to_numpy(copy=False),
        index=pd.RangeIndex(start=start, stop=start + len(series), step=1),
        name=series.name,
    )


def _calendar_day_blocks(series):
    """Group a series into blocks by calendar day.

    Parameters
    ----------
    series : pd.Series
        Series with datetime index.

    Returns
    -------
    list of pd.Series
        One entry per calendar day, sorted chronologically.
    """
    return [block.copy() for _, block in series.groupby(series.index.normalize(), sort=True)]


# ---- Model fitting ----

def _build_arima(series, spec):
    return ARIMA(
        series,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def _fit(series, spec, cov_type=FIT_COV_TYPE, low_memory=False):
    seq = _as_sequential(series)
    model = _build_arima(seq, spec)
    return model.fit(method=FIT_METHOD, low_memory=low_memory, cov_type=cov_type)


def _rebuild_from_params(series, spec, params):
    seq = _as_sequential(series)
    return _build_arima(seq, spec).filter(np.asarray(params, dtype=float))


def _fit_converged(result):
    fit_details = getattr(result, "fit_details", None)
    minimize_results = getattr(fit_details, "minimize_results", None)
    if minimize_results is not None and hasattr(minimize_results, "success"):
        return bool(minimize_results.success)
    mle_retvals = getattr(result, "mle_retvals", None)
    if isinstance(mle_retvals, dict) and "converged" in mle_retvals:
        return bool(mle_retvals["converged"])
    return None


# ---- Hyperparameter selection ----

def _build_selection_split(train, window_steps, val_day_count):
    """Split within training set for internal validation.

    Parameters
    ----------
    train : pd.Series
        Full training series.
    window_steps : int
        Maximum number of recent steps to keep as fitting history.
    val_day_count : int
        Number of trailing calendar days used for validation.

    Returns
    -------
    history : pd.Series
        Fitting history (most recent ``window_steps`` excluding validation).
    validation : pd.Series
        Validation portion (last ``val_day_count`` calendar days).
    """
    unique_days = pd.Index(train.index.normalize().unique()).sort_values()
    val_days = unique_days[-val_day_count:]
    val_mask = train.index.normalize().isin(val_days)
    validation = train.loc[val_mask].copy()
    history = train.loc[~val_mask].copy().iloc[-min(window_steps, (~val_mask).sum()):]
    return history, validation


def _evaluate_candidate(history, validation, spec, index):
    """Evaluate a single candidate spec via fit + rolling validation.

    Parameters
    ----------
    history : pd.Series
        Training history for fitting.
    validation : pd.Series
        Validation series for rolling evaluation.
    spec : SarimaSpec
        Candidate SARIMA specification.
    index : int
        Candidate index (for reporting).

    Returns
    -------
    dict
        Evaluation results including ``validation_rmse`` and ``aic``.
    """
    row = {
        "index": index,
        "label": spec.label,
        "order": list(spec.order),
        "seasonal_order": list(spec.seasonal_order),
        "status": "ok",
        "converged": False,
        "validation_rmse": np.nan,
        "aic": np.nan,
    }
    try:
        fitted = _fit(history, spec)
        row["aic"] = float(fitted.aic)
        row["converged"] = _fit_converged(fitted)
        forecast_df = rolling_day_ahead_forecast(fitted, history, validation)
        residual = forecast_df["actual"].values - forecast_df["prediction"].values
        row["validation_rmse"] = float(np.sqrt(np.mean(residual ** 2)))
    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)
    return row


def _eval_task(args):
    return _evaluate_candidate(*args)


def select_best_spec(train, candidate_specs, window_steps, val_day_count):
    """Grid search to select the best SARIMA parameters.

    Parameters
    ----------
    train : pd.Series
        Full training series (train + val merged).
    candidate_specs : list of SarimaSpec
        Candidate specifications to evaluate.
    window_steps : int
        Fitting history window size.
    val_day_count : int
        Number of validation days.

    Returns
    -------
    best_spec : SarimaSpec
        Specification with the lowest validation RMSE.
    selection_df : pd.DataFrame
        Full evaluation results for all candidates.
    """
    _configure_warnings()
    history, validation = _build_selection_split(train, window_steps, val_day_count)

    tasks = [(history, validation, spec, i) for i, spec in enumerate(candidate_specs, 1)]
    max_workers = min(len(tasks), max(1, min(4, os.cpu_count() or 1)))

    if max_workers <= 1:
        records = [_eval_task(t) for t in tqdm(tasks, desc="SARIMA grid search", unit="spec")]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            records = list(tqdm(pool.map(_eval_task, tasks), total=len(tasks),
                                desc="SARIMA grid search", unit="spec"))

    df = pd.DataFrame(records)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("All SARIMA candidates failed.")

    ok = ok.sort_values(["validation_rmse", "aic"])
    best = ok.iloc[0]
    best_spec = SarimaSpec(
        order=tuple(best["order"]),
        seasonal_order=tuple(best["seasonal_order"]),
    )
    print(f"Selected: {best_spec.label}  (val RMSE={best['validation_rmse']:.2f})")
    return best_spec, df


# ---- Rolling day-ahead forecast ----

def _extract_uncertainty(forecast, predicted_mean, block_size):
    """Extract std and confidence intervals from a statsmodels forecast.

    Parameters
    ----------
    forecast : statsmodels forecast object
        Result of ``get_forecast()``.
    predicted_mean : np.ndarray
        Point predictions for the block.
    block_size : int
        Number of steps in the current block.

    Returns
    -------
    std : np.ndarray
        Standard deviations.
    lower : np.ndarray
        Lower 95 % confidence bound.
    upper : np.ndarray
        Upper 95 % confidence bound.
    """
    # std
    se = getattr(forecast, "se_mean", None)
    if se is not None:
        std = np.maximum(np.asarray(se, dtype=float)[:block_size], 0.0)
    else:
        std = np.full(block_size, np.nan)

    # conf_int
    try:
        ci = forecast.conf_int(alpha=0.05)
        if isinstance(ci, pd.DataFrame):
            lower = ci.iloc[:block_size, 0].values.astype(float)
            upper = ci.iloc[:block_size, 1].values.astype(float)
        else:
            arr = np.asarray(ci, dtype=float)
            lower, upper = arr[:block_size, 0], arr[:block_size, 1]
    except Exception:
        lower = predicted_mean - Z_SCORE_95 * std
        upper = predicted_mean + Z_SCORE_95 * std

    # Fill NaN in std using conf_int
    derived_std = (upper - lower) / (2.0 * Z_SCORE_95)
    std = np.where(np.isfinite(std), std, np.maximum(derived_std, 0.0))

    lower = np.minimum(lower, predicted_mean)
    upper = np.maximum(upper, predicted_mean)

    return std, lower, upper


def rolling_day_ahead_forecast(fitted_result, history, evaluation):
    """Rolling day-ahead forecast.

    Predict one day at a time, then update the model state with actual
    values (parameters stay fixed).

    Parameters
    ----------
    fitted_result : statsmodels result
        Fitted ARIMA result object.
    history : pd.Series
        Training history used for fitting.
    evaluation : pd.Series
        Evaluation series to forecast over.

    Returns
    -------
    pd.DataFrame
        Columns: ``actual``, ``prediction``, ``prediction_std``,
        ``lower_95``, ``upper_95``.
    """
    current = fitted_result
    next_pos = len(history)
    records = []
    day_blocks = _calendar_day_blocks(evaluation)

    for actual_block in tqdm(day_blocks, desc="Rolling forecast", unit="day"):
        n = len(actual_block)
        forecast = current.get_forecast(steps=n)
        predicted_mean = np.asarray(forecast.predicted_mean, dtype=float)[:n]
        std, lower, upper = _extract_uncertainty(forecast, predicted_mean, n)

        block_df = pd.DataFrame({
            "actual": actual_block.values.astype(float),
            "prediction": predicted_mean,
            "prediction_std": std,
            "lower_95": lower,
            "upper_95": upper,
        }, index=actual_block.index)
        records.append(block_df)

        # Update state with actual values (fixed parameters, state update only)
        new_obs = _as_sequential(actual_block, start=next_pos)
        current = current.extend(new_obs)
        next_pos += n

    return pd.concat(records)


# ---- Save / Load ----

def save_model(fitted_result, spec, path):
    """Save SARIMA model parameters to a pickle file.

    Parameters
    ----------
    fitted_result : statsmodels result
        Fitted ARIMA result whose ``.params`` will be saved.
    spec : SarimaSpec
        Model specification (order and seasonal_order).
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "order": list(spec.order),
        "seasonal_order": list(spec.seasonal_order),
        "params": np.asarray(fitted_result.params, dtype=float).tolist(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {path}")


def load_model(path):
    """Load saved SARIMA model parameters.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    spec : SarimaSpec
        Model specification.
    params : list of float
        Fitted parameter values.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    spec = SarimaSpec(
        order=tuple(payload["order"]),
        seasonal_order=tuple(payload["seasonal_order"]),
    )
    return spec, payload["params"]


# ---- Main entry points (called by train.py / test.py) ----

def train_sarima(train_series, val_series, cfg):
    """Train SARIMA: parameter selection, fit, and checkpoint saving.

    SARIMA merges train + val for training (no early-stopping validation
    set needed).  Parameter selection uses rolling validation within the
    merged training set.

    Parameters
    ----------
    train_series : pd.Series
        Training target series.
    val_series : pd.Series
        Validation target series (merged with train for fitting).
    cfg : dict
        Full experiment configuration.

    Returns
    -------
    fitted : statsmodels result
        Fitted model on the full training set.
    best_spec : SarimaSpec
        Selected specification.
    full_train : pd.Series
        Merged train + val series.
    ckpt_path : str
        Path to the saved checkpoint.
    """
    _configure_warnings()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    seasonal_period = model_cfg.get("seasonal_period", SEASONAL_PERIOD)

    # Merge train + val as SARIMA's full training set
    full_train = pd.concat([train_series, val_series])

    # Parse candidate specs
    specs = parse_candidate_specs(model_cfg["candidate_specs"], seasonal_period)

    # Select best parameters
    window_steps = train_cfg.get("selection_window_steps", SEASONAL_PERIOD * 120)
    val_days = train_cfg.get("validation_day_count", 14)
    best_spec, selection_df = select_best_spec(full_train, specs, window_steps, val_days)

    # Fit on full training set
    print("Fitting on full training set...")
    started = time.perf_counter()
    fitted = _fit(full_train, best_spec, cov_type=SAVED_FIT_COV_TYPE, low_memory=True)
    fit_time = time.perf_counter() - started
    print(f"Fit completed in {fit_time:.1f}s")

    # Save model
    ckpt_path = os.path.join("checkpoints", "sarima_best.pkl")
    save_model(fitted, best_spec, ckpt_path)

    return fitted, best_spec, full_train, ckpt_path


def predict_sarima(fitted_or_params, spec, train_series, test_series):
    """Make rolling predictions using a fitted model or loaded parameters.

    Parameters
    ----------
    fitted_or_params : statsmodels result or list of float
        Either a fitted result object or saved parameter values.
    spec : SarimaSpec
        Model specification.
    train_series : pd.Series
        Training series used for state initialization.
    test_series : pd.Series
        Test series to forecast.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with ``actual``, ``prediction``,
        ``prediction_std``, ``lower_95``, ``upper_95``.
    """
    _configure_warnings()

    if isinstance(fitted_or_params, list):
        # Rebuild from saved parameters
        result = _rebuild_from_params(train_series, spec, fitted_or_params)
    else:
        # Use rebuild to ensure clean state
        result = _rebuild_from_params(train_series, spec, fitted_or_params.params)

    forecast_df = rolling_day_ahead_forecast(result, train_series, test_series)
    return forecast_df
