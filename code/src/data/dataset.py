import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TimeSeriesDataset:
    """处理归一化、滑动窗口、DataLoader 构建。

    支持两种模式：
    - 有特征工程：输入 (seq_len, n_features)，目标为最后一列 tsd
    - 无特征工程：输入 (seq_len, 1)，仅用历史 tsd
    """

    def __init__(self, train_df, val_df, test_df, features, target, cols_to_scale,
                 seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target = target
        self.features = features
        self.use_features = len(features) > 0

        if self.use_features:
            self._prepare_with_features(train_df, val_df, test_df, features, target, cols_to_scale)
        else:
            self._prepare_no_features(train_df, val_df, test_df, target)

    def _prepare_with_features(self, train_df, val_df, test_df, features, target, cols_to_scale):
        """有特征工程模式：多维输入。"""
        columns = features + [target]
        train_np = train_df[columns].values.astype(np.float32)
        val_np = val_df[columns].values.astype(np.float32)
        test_np = test_df[columns].values.astype(np.float32)

        # 确定需要归一化的列索引
        if cols_to_scale == "all":
            scale_cols = columns
        else:
            scale_cols = cols_to_scale
        scale_indices = [columns.index(c) for c in scale_cols]

        # 在训练集上计算 min / max
        data_min = train_np[:, scale_indices].min(axis=0)
        data_max = train_np[:, scale_indices].max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        # 归一化
        train_scaled = train_np.copy()
        val_scaled = val_np.copy()
        test_scaled = test_np.copy()
        train_scaled[:, scale_indices] = (train_np[:, scale_indices] - data_min) / data_range
        val_scaled[:, scale_indices] = (val_np[:, scale_indices] - data_min) / data_range
        test_scaled[:, scale_indices] = (test_np[:, scale_indices] - data_min) / data_range

        # tsd 的反归一化参数
        tsd_scale_idx = scale_cols.index(target)
        self.tsd_min = data_min[tsd_scale_idx]
        self.tsd_range = data_range[tsd_scale_idx]

        # 滑动窗口
        self.X_train, self.y_train = self._create_sequences_multi(train_scaled)
        self.X_val, self.y_val = self._create_sequences_multi(val_scaled)
        self.X_test, self.y_test = self._create_sequences_multi(test_scaled)

        self.n_features = len(features) + 1  # features + target(tsd)

        # 保存原始测试集用于结果构建
        self.test_index = test_df.index[self.seq_len:]
        self.test_actual = test_df[target].iloc[self.seq_len:].values

    def _prepare_no_features(self, train_df, val_df, test_df, target):
        """无特征工程模式：仅 tsd，1 维输入。"""
        train_np = train_df[target].values.astype(np.float32)
        val_np = val_df[target].values.astype(np.float32)
        test_np = test_df[target].values.astype(np.float32)

        # MinMax 归一化
        self.tsd_min = train_np.min()
        tsd_max = train_np.max()
        self.tsd_range = tsd_max - self.tsd_min

        train_scaled = (train_np - self.tsd_min) / self.tsd_range
        val_scaled = (val_np - self.tsd_min) / self.tsd_range
        test_scaled = (test_np - self.tsd_min) / self.tsd_range

        # 滑动窗口
        self.X_train, self.y_train = self._create_sequences_single(train_scaled)
        self.X_val, self.y_val = self._create_sequences_single(val_scaled)
        self.X_test, self.y_test = self._create_sequences_single(test_scaled)

        self.n_features = 1

        # 保存原始测试集用于结果构建
        self.test_index = test_df.index[self.seq_len:]
        self.test_actual = test_df[target].iloc[self.seq_len:].values

    def _create_sequences_multi(self, data):
        """多维滑动窗口：X = data[i:i+seq_len, :] (含 tsd), y = data[i+seq_len, -1]"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])          # 所有列（含历史 tsd）
            y.append(data[i + self.seq_len, -1])        # 目标：下一步 tsd
        return np.array(X), np.array(y)

    def _create_sequences_single(self, data):
        """单维滑动窗口：X = data[i:i+seq_len], y = data[i+seq_len]"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len])
        X = np.array(X)[:, :, np.newaxis]  # (samples, seq_len, 1)
        return X, np.array(y)

    def get_loaders(self):
        """返回 train / val / test DataLoader。"""
        train_ds = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(self.X_val, dtype=torch.float32),
            torch.tensor(self.y_val, dtype=torch.float32),
        )
        test_ds = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def inverse_transform(self, pred_scaled, std_scaled=None):
        """将归一化的预测值还原到原始尺度。"""
        pred = pred_scaled * self.tsd_range + self.tsd_min
        if std_scaled is not None:
            std = std_scaled * self.tsd_range
            return pred, std
        return pred
