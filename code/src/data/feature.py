import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the is_day_off feature, shifted by -1 to align with the prediction target.

    Since seq_len=48 covers a full day, the model can learn intra-day and seasonal
    patterns from history alone. The only information not inferable from history is
    calendar shifts (e.g., a sudden holiday tomorrow). shift(-1) ensures the last
    step of the input window carries the calendar info of the prediction target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with ``is_holiday`` column and datetime index.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``is_day_off`` column (float32).
    """
    df = df.copy()

    is_holiday = df["is_holiday"].astype(str).str.strip() == "1"
    df["is_day_off"] = ((df.index.dayofweek >= 5) | is_holiday).astype(np.float32)

    # Align to prediction target time t+1
    df["is_day_off"] = df["is_day_off"].shift(-1)
    df.loc[df.index[-1], "is_day_off"] = df.loc[df.index[-2], "is_day_off"]

    return df
