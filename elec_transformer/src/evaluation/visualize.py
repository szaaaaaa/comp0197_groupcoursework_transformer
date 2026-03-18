import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib():
    """设置全局字体和字号。"""
    small, medium, large = 12, 14, 16
    matplotlib.rc("font", size=small)
    matplotlib.rc("axes", titlesize=large, labelsize=medium)
    matplotlib.rc("xtick", labelsize=small)
    matplotlib.rc("ytick", labelsize=small)
    matplotlib.rc("legend", fontsize=small)


def plot_time_series(df, target="tsd"):
    """时间序列总览。"""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df[target], ".", markersize=0.5, alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity Demand (MW)")
    ax.set_title("UK Electricity Demand Time Series Overview")
    return fig


def plot_split(train_df, val_df, test_df, threshold_1, threshold_2, target="tsd"):
    """Train / Val / Test 数据划分可视化。"""
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
    """训练和验证损失曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gaussian NLL Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(loc="best")
    return fig


def plot_predictions(result_frame, title="Test Set Predictions with Uncertainty"):
    """测试集预测对比（含 95% 置信区间）。"""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(result_frame.index, result_frame["tsd"], "o", markersize=1, alpha=0.3, label="Actual")
    ax.plot(result_frame.index, result_frame["pred"], "o", markersize=1, alpha=0.3, label="Prediction (μ)")
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


def plot_detail(result_frame, begin, end):
    """两周细节对比。"""
    mask = (result_frame.index > begin) & (result_frame.index < end)
    sub = result_frame.loc[mask]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(sub.index, sub["tsd"], "-o", label="Actual")
    ax.plot(sub.index, sub["pred"], "-s", label="Prediction (μ)")
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
