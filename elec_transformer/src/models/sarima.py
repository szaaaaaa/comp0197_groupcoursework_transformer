"""
SARIMA 模型封装。

不继承 BaseModel(nn.Module)，因为 SARIMA 是传统统计模型，不使用 PyTorch。
通过 train.py / test.py 中的 if-else 分支调用。
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


# ---- 常量 ----
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
    """从 yaml 配置解析候选参数列表。"""
    return [
        SarimaSpec(
            order=tuple(s["order"]),
            seasonal_order=tuple(s["seasonal_order"]) + (seasonal_period,),
        )
        for s in specs_cfg
    ]


# ---- 工具函数 ----

def _configure_warnings():
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Provided `endog` series has been differenced")
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)


def _as_sequential(series, start=0):
    """将 datetime 索引转为连续整数索引（避免夏令时间隔问题）。"""
    return pd.Series(
        series.to_numpy(copy=False),
        index=pd.RangeIndex(start=start, stop=start + len(series), step=1),
        name=series.name,
    )


def _calendar_day_blocks(series):
    """按日历天分组。"""
    return [block.copy() for _, block in series.groupby(series.index.normalize(), sort=True)]


# ---- 模型拟合 ----

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


# ---- 超参选择 ----

def _build_selection_split(train, window_steps, val_day_count):
    """在训练集内部划分：最近 window_steps 作为 history，最后 val_day_count 天做验证。"""
    unique_days = pd.Index(train.index.normalize().unique()).sort_values()
    val_days = unique_days[-val_day_count:]
    val_mask = train.index.normalize().isin(val_days)
    validation = train.loc[val_mask].copy()
    history = train.loc[~val_mask].copy().iloc[-min(window_steps, (~val_mask).sum()):]
    return history, validation


def _evaluate_candidate(history, validation, spec, index):
    """评估单个候选参数：拟合 + rolling 验证。"""
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
    """网格搜索选择最优 SARIMA 参数。"""
    _configure_warnings()
    history, validation = _build_selection_split(train, window_steps, val_day_count)

    tasks = [(history, validation, spec, i) for i, spec in enumerate(candidate_specs, 1)]
    max_workers = min(len(tasks), max(1, min(4, os.cpu_count() or 1)))

    if max_workers <= 1:
        records = [_eval_task(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            records = list(pool.map(_eval_task, tasks))

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


# ---- Rolling day-ahead 预测 ----

def _extract_uncertainty(forecast, predicted_mean, block_size):
    """从 statsmodels forecast 对象提取 std 和置信区间。"""
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

    # 用 conf_int 补充 std 中的 nan
    derived_std = (upper - lower) / (2.0 * Z_SCORE_95)
    std = np.where(np.isfinite(std), std, np.maximum(derived_std, 0.0))

    lower = np.minimum(lower, predicted_mean)
    upper = np.maximum(upper, predicted_mean)

    return std, lower, upper


def rolling_day_ahead_forecast(fitted_result, history, evaluation):
    """Rolling day-ahead 预测：每天预测一天，然后用真实值更新状态。"""
    current = fitted_result
    next_pos = len(history)
    records = []

    for actual_block in _calendar_day_blocks(evaluation):
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

        # 用真实值更新状态（固定参数，只更新状态）
        new_obs = _as_sequential(actual_block, start=next_pos)
        current = current.extend(new_obs)
        next_pos += n

    return pd.concat(records)


# ---- 保存 / 加载 ----

def save_model(fitted_result, spec, path):
    """保存模型参数到 pickle。"""
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
    """加载保存的模型参数。"""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    spec = SarimaSpec(
        order=tuple(payload["order"]),
        seasonal_order=tuple(payload["seasonal_order"]),
    )
    return spec, payload["params"]


# ---- 主入口（供 train.py / test.py 调用） ----

def train_sarima(train_series, val_series, cfg):
    """训练 SARIMA：选参 → 拟合 → rolling 预测测试集。

    SARIMA 用 train+val 合并训练（因为不需要 early stopping 的独立验证集）。
    参数选择在合并后的训练集内部做 rolling validation。
    """
    _configure_warnings()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    seasonal_period = model_cfg.get("seasonal_period", SEASONAL_PERIOD)

    # 合并 train + val 作为 SARIMA 的完整训练集
    full_train = pd.concat([train_series, val_series])

    # 解析候选参数
    specs = parse_candidate_specs(model_cfg["candidate_specs"], seasonal_period)

    # 选择最优参数
    window_steps = train_cfg.get("selection_window_steps", SEASONAL_PERIOD * 120)
    val_days = train_cfg.get("validation_day_count", 14)
    best_spec, selection_df = select_best_spec(full_train, specs, window_steps, val_days)

    # 在完整训练集上拟合
    print("Fitting on full training set...")
    started = time.perf_counter()
    fitted = _fit(full_train, best_spec, cov_type=SAVED_FIT_COV_TYPE, low_memory=True)
    fit_time = time.perf_counter() - started
    print(f"Fit completed in {fit_time:.1f}s")

    # 保存模型
    ckpt_path = os.path.join("checkpoints", "sarima_best.pkl")
    save_model(fitted, best_spec, ckpt_path)

    return fitted, best_spec, full_train, ckpt_path


def predict_sarima(fitted_or_params, spec, train_series, test_series):
    """用已拟合模型（或加载的参数）做 rolling 预测。"""
    _configure_warnings()

    if isinstance(fitted_or_params, list):
        # 从保存的参数重建
        result = _rebuild_from_params(train_series, spec, fitted_or_params)
    else:
        # 直接用 rebuild 确保干净状态
        result = _rebuild_from_params(train_series, spec, fitted_or_params.params)

    forecast_df = rolling_day_ahead_forecast(result, train_series, test_series)
    return forecast_df
