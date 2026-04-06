import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TimeSeriesDataset:
    """Normalization, sliding window, and DataLoader construction.

    Supports two modes:

    * With feature engineering: input ``(seq_len, n_features)``, target is
      the last column (tsd).
    * Without feature engineering: input ``(seq_len, 1)``, using only
      historical tsd.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames with datetime index.
    features : list of str
        Feature column names (empty list for no-feature mode).
    target : str
        Target column name.
    cols_to_scale : list of str or ``"all"``
        Columns to apply MinMax normalization.
    seq_len : int
        Input sequence length.
    batch_size : int
        Batch size for DataLoaders.

    Attributes
    ----------
    n_features : int
        Number of input features (including target).
    test_index : pd.DatetimeIndex
        Datetime index of the test predictions.
    test_actual : np.ndarray
        Ground-truth target values for the test set.
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
        """Prepare multi-dimensional input with feature engineering."""
        columns = features + [target]
        train_np = train_df[columns].values.astype(np.float32)
        val_np = val_df[columns].values.astype(np.float32)
        test_np = test_df[columns].values.astype(np.float32)

        # Determine column indices to normalize
        if cols_to_scale == "all":
            scale_cols = columns
        else:
            scale_cols = cols_to_scale
        scale_indices = [columns.index(c) for c in scale_cols]

        # Compute min / max on training set
        data_min = train_np[:, scale_indices].min(axis=0)
        data_max = train_np[:, scale_indices].max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        # Normalize
        train_scaled = train_np.copy()
        val_scaled = val_np.copy()
        test_scaled = test_np.copy()
        train_scaled[:, scale_indices] = (train_np[:, scale_indices] - data_min) / data_range
        val_scaled[:, scale_indices] = (val_np[:, scale_indices] - data_min) / data_range
        test_scaled[:, scale_indices] = (test_np[:, scale_indices] - data_min) / data_range

        # Inverse normalization parameters for tsd
        tsd_scale_idx = scale_cols.index(target)
        self.tsd_min = data_min[tsd_scale_idx]
        self.tsd_range = data_range[tsd_scale_idx]

        # Sliding window
        self.X_train, self.y_train = self._create_sequences_multi(train_scaled)
        self.X_val, self.y_val = self._create_sequences_multi(val_scaled)
        self.X_test, self.y_test = self._create_sequences_multi(test_scaled)

        self.n_features = len(features) + 1  # features + target(tsd)

        # Save original test set for result construction
        self.test_index = test_df.index[self.seq_len:]
        self.test_actual = test_df[target].iloc[self.seq_len:].values

    def _prepare_no_features(self, train_df, val_df, test_df, target):
        """Prepare 1-dimensional input using only tsd."""
        train_np = train_df[target].values.astype(np.float32)
        val_np = val_df[target].values.astype(np.float32)
        test_np = test_df[target].values.astype(np.float32)

        # MinMax normalization
        self.tsd_min = train_np.min()
        tsd_max = train_np.max()
        self.tsd_range = tsd_max - self.tsd_min

        train_scaled = (train_np - self.tsd_min) / self.tsd_range
        val_scaled = (val_np - self.tsd_min) / self.tsd_range
        test_scaled = (test_np - self.tsd_min) / self.tsd_range

        # Sliding window
        self.X_train, self.y_train = self._create_sequences_single(train_scaled)
        self.X_val, self.y_val = self._create_sequences_single(val_scaled)
        self.X_test, self.y_test = self._create_sequences_single(test_scaled)

        self.n_features = 1

        # Save original test set for result construction
        self.test_index = test_df.index[self.seq_len:]
        self.test_actual = test_df[target].iloc[self.seq_len:].values

    def _create_sequences_multi(self, data):
        """Create multi-dimensional sliding windows.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_cols)
            Scaled data array.

        Returns
        -------
        X : np.ndarray, shape (n_windows, seq_len, n_cols)
            Input sequences.
        y : np.ndarray, shape (n_windows,)
            Target values (last column at ``i + seq_len``).
        """
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])          # All columns (incl. historical tsd)
            y.append(data[i + self.seq_len, -1])        # Target: next-step tsd
        return np.array(X), np.array(y)

    def _create_sequences_single(self, data):
        """Create single-dimensional sliding windows.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples,)
            Scaled 1-D data array.

        Returns
        -------
        X : np.ndarray, shape (n_windows, seq_len, 1)
            Input sequences.
        y : np.ndarray, shape (n_windows,)
            Target values at ``i + seq_len``.
        """
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len])
        X = np.array(X)[:, :, np.newaxis]  # (samples, seq_len, 1)
        return X, np.array(y)

    def get_loaders(self):
        """Build and return train / val / test DataLoaders.

        Returns
        -------
        tuple of DataLoader
            (train_loader, val_loader, test_loader).
        """
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
        """Inverse-transform normalized predictions back to original scale.

        Parameters
        ----------
        pred_scaled : np.ndarray
            Normalized predicted means.
        std_scaled : np.ndarray, optional
            Normalized predicted standard deviations.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            Rescaled predictions, and optionally rescaled std.
        """
        pred = pred_scaled * self.tsd_range + self.tsd_min
        if std_scaled is not None:
            std = std_scaled * self.tsd_range
            return pred, std
        return pred
