import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib():
    """Set global matplotlib font and font sizes."""
    small, medium, large = 12, 14, 16
    matplotlib.rc("font", size=small)
    matplotlib.rc("axes", titlesize=large, labelsize=medium)
    matplotlib.rc("xtick", labelsize=small)
    matplotlib.rc("ytick", labelsize=small)
    matplotlib.rc("legend", fontsize=small)


def plot_time_series(df, target="tsd"):
    """Plot a time series overview.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index.
    target : str
        Column to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df[target], ".", markersize=0.5, alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity Demand (MW)")
    ax.set_title("UK Electricity Demand Time Series Overview")
    return fig


def plot_split(train_df, val_df, test_df, threshold_1, threshold_2, target="tsd"):
    """Visualize the train / val / test data split.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames.
    threshold_1, threshold_2 : str
        Date boundaries shown as vertical lines.
    target : str
        Column to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    train_df[target].plot(ax=ax, label="Train", style=".", markersize=0.5)
    val_df[target].plot(ax=ax, label="Validation", style=".", markersize=0.5)
    test_df[target].plot(ax=ax, label="Test", style=".", markersize=0.5)
    ax.axvline(threshold_1, color="k", ls="--")
    ax.axvline(threshold_2, color="k", ls=":")
    ax.set_title("Train / Validation / Test Split")
    ax.set_ylabel("Electricity Demand (MW)")
    ax.legend()
    return fig


def plot_loss_curve(train_losses, val_losses):
    """Plot training and validation loss curves.

    Parameters
    ----------
    train_losses, val_losses : list of float
        Per-epoch loss values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gaussian NLL Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(loc="best")
    return fig


def plot_predictions(result_frame, title="Test Set Predictions with Uncertainty", show_ci=True):
    """Plot test-set predictions, optionally with 95 % confidence interval.

    Parameters
    ----------
    result_frame : pd.DataFrame
        Must contain columns ``tsd``, ``pred``, ``std``.
    title : str
        Figure title.
    show_ci : bool
        Whether to draw the confidence interval shading.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(result_frame.index, result_frame["tsd"], "o", markersize=1, alpha=0.3, label="Actual")
    ax.plot(result_frame.index, result_frame["pred"], "o", markersize=1, alpha=0.3, label="Prediction (μ)")
    if show_ci:
        ax.fill_between(
            result_frame.index,
            result_frame["pred"] - 1.96 * result_frame["std"],
            result_frame["pred"] + 1.96 * result_frame["std"],
            alpha=0.15, color="orange", label="95% Confidence Interval",
        )
    ax.legend(loc="center", bbox_to_anchor=(1.15, 0.5))
    ax.set_title(title)
    ax.set_ylabel("Electricity Demand (MW)")
    ax.set_xlabel("Date")
    return fig


def plot_detail(result_frame, begin, end, show_ci=True):
    """Plot a zoomed-in two-week detail comparison.

    Parameters
    ----------
    result_frame : pd.DataFrame
        Must contain columns ``tsd``, ``pred``, ``std``.
    begin, end : str
        Date strings defining the detail window.
    show_ci : bool
        Whether to draw the confidence interval shading.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mask = (result_frame.index > begin) & (result_frame.index < end)
    sub = result_frame.loc[mask]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(sub.index, sub["tsd"], "-o", label="Actual")
    ax.plot(sub.index, sub["pred"], "-s", label="Prediction (μ)")
    if show_ci:
        ax.fill_between(
            sub.index,
            sub["pred"] - 1.96 * sub["std"],
            sub["pred"] + 1.96 * sub["std"],
            alpha=0.25, color="orange", label="95% CI",
        )
    ax.legend(loc="center", bbox_to_anchor=(1.15, 0.5))
    ax.set_title("Test Set Predictions — Two-Week Detail")
    ax.set_ylabel("Electricity Demand (MW)")
    ax.set_xlabel("Date")
    return fig
