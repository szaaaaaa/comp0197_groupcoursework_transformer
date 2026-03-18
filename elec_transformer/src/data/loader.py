import pandas as pd


def load_and_clean(csv_path: str, date_start: str, date_end: str) -> pd.DataFrame:
    """加载 CSV，清洗数据，设置日期索引，过滤日期范围。"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    # 排序
    df.sort_values(by=["settlement_date", "settlement_period"], inplace=True, ignore_index=True)

    # 删除含大量缺失值的列
    for col in ["nsl_flow", "eleclink_flow"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 删除异常 settlement_period (> 48)
    df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 删除 tsd 为 0 的整天数据
    null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()
    null_days_index = []
    for day in null_days:
        null_days_index.append(df[df["settlement_date"] == day].index.tolist())
    null_days_index = [item for sublist in null_days_index for item in sublist]
    if null_days_index:
        df.drop(index=null_days_index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 设置日期索引
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df.set_index("settlement_date", inplace=True)
    df.sort_index(inplace=True)

    # 过滤日期范围
    df = df[date_start:date_end]

    return df


def split_data(df: pd.DataFrame, threshold_1: str, threshold_2: str):
    """按日期划分 train / val / test。"""
    train = df.loc[df.index < threshold_1].copy()
    val = df.loc[(df.index >= threshold_1) & (df.index < threshold_2)].copy()
    test = df.loc[df.index >= threshold_2].copy()
    return train, val, test
