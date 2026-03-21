import numpy as np


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    """Root Mean Square Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def picp(y_true, y_pred, std, z=1.96):
    """Prediction Interval Coverage Probability"""
    y_true, y_pred, std = np.array(y_true), np.array(y_pred), np.array(std)
    lower, upper = y_pred - z * std, y_pred + z * std
    return np.mean((y_true >= lower) & (y_true <= upper)) * 100


def mpiw(std, z=1.96):
    """Mean Prediction Interval Width"""
    return np.mean(2 * z * np.array(std))
