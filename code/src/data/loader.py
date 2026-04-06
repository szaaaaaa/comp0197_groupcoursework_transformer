import pandas as pd


def load_and_clean(csv_path: str, date_start: str, date_end: str) -> pd.DataFrame:
    """Load CSV, clean data, set date index, and filter date range.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.
    date_start : str
        Start date for filtering (inclusive).
    date_end : str
        End date for filtering (inclusive).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with datetime index.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    # Sort
    df.sort_values(by=["settlement_date", "settlement_period"], inplace=True, ignore_index=True)

    # Drop columns with many missing values
    for col in ["nsl_flow", "eleclink_flow"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Remove anomalous settlement_period (> 48)
    df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove full days where tsd is 0
    null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()
    null_days_index = []
    for day in null_days:
        null_days_index.append(df[df["settlement_date"] == day].index.tolist())
    null_days_index = [item for sublist in null_days_index for item in sublist]
    if null_days_index:
        df.drop(index=null_days_index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Set date index
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df.set_index("settlement_date", inplace=True)
    df.sort_index(inplace=True)

    # Filter date range
    df = df[date_start:date_end]

    return df


def split_data(df: pd.DataFrame, threshold_1: str, threshold_2: str):
    """Split data into train / val / test by date.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with datetime index.
    threshold_1 : str
        Date string for train/val split boundary.
    threshold_2 : str
        Date string for val/test split boundary.

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, val_df, test_df).
    """
    train = df.loc[df.index < threshold_1].copy()
    val = df.loc[(df.index >= threshold_1) & (df.index < threshold_2)].copy()
    test = df.loc[df.index >= threshold_2].copy()
    return train, val, test
