import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程：仅生成 is_day_off（周末或节假日），并 shift(-1) 对齐到预测目标时刻。

    设计理由：
    - seq_len=48 已覆盖一整天，模型从历史波形即可学到日内周期和季节信息
    - 唯一无法从历史推断的是日历跳变（明天是否突然放假）
    - shift(-1) 使输入窗口最后一步携带的是预测目标时刻的日历信息
    """
    df = df.copy()

    is_holiday = df["is_holiday"].astype(str).str.strip() == "1"
    df["is_day_off"] = ((df.index.dayofweek >= 5) | is_holiday).astype(np.float32)

    # 对齐到预测目标时刻 t+1
    df["is_day_off"] = df["is_day_off"].shift(-1)
    df.loc[df.index[-1], "is_day_off"] = df.loc[df.index[-2], "is_day_off"]

    return df
