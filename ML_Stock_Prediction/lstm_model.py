import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # hide TensorFlow's start-up chatter
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def make_sequences(F, y, L, idx):
    """For each feature-date index i in idx, take F[i-L+1 .. i] for every ticker.
    Returns X (len(idx)*T, L, K) and yy (len(idx)*T,), ordered date-major, ticker-minor."""
    idx = np.asarray(idx)
    if idx.min() < L - 1:
        raise ValueError("not enough history before the first index for the window length")
    X = np.stack([F[i - L + 1: i + 1].transpose(1, 0, 2) for i in idx])      # (n, T, L, K)
    K = F.shape[2]
    return X.reshape(-1, L, K), y[idx].reshape(-1)


class Scaler:
    def __init__(self, F_train, y_train):
        K = F_train.shape[-1]
        flat = F_train.reshape(-1, K)
        self.mu, self.sd = flat.mean(axis=0), flat.std(axis=0) + 1e-12
        self.ysd = float(np.std(y_train)) + 1e-12

    def x(self, X):
        return (X - self.mu) / self.sd

    def y(self, v):
        return v / self.ysd


def prepare(F, y, train_idx, val_idx, test_idx, L, feature_cols=None):
    """Scale and window the data. The scaler is fitted on TRAINING dates only."""
    if feature_cols is not None:
        F = F[:, :, feature_cols]
    sc = Scaler(F[train_idx], y[train_idx])
    Fs = sc.x(F)
    Xtr, ytr = make_sequences(Fs, y, L, train_idx)
    Xva, yva = make_sequences(Fs, y, L, val_idx)
    Xte, _ = make_sequences(Fs, y, L, test_idx)
    return Xtr, sc.y(ytr), Xva, sc.y(yva), Xte, sc


def _build(L, K, units, dropout, lr):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(L, K)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return m


def train_one(Xtr, ytr, Xva, yva, units, seed, dropout=0.2, lr=1e-3, epochs=60, patience=6, batch=256):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(seed)
    m = _build(Xtr.shape[1], Xtr.shape[2], units, dropout, lr)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, verbose=0, callbacks=[es],
              shuffle=True)
    return m, float(min(h.history["val_loss"])), len(h.history["val_loss"])


def lstm_forecasts(F, y, train_idx, val_idx, test_idx, L=20, feature_cols=None, units_grid=(16, 32, 64),
                   n_seeds=5, verbose=True):
    """Train on train_idx dates, early-stop/select on val_idx dates, predict test_idx dates.
    feature_cols: optional list of column positions to use (e.g. only the return column)."""
    Xtr, ytr, Xva, yva, Xte, sc = prepare(F, y, train_idx, val_idx, test_idx, L, feature_cols)
    T = F.shape[1]

    scores = {}
    for u in units_grid:
        _, vl, ep = train_one(Xtr, ytr, Xva, yva, u, seed=0)
        scores[u] = vl
        if verbose:
            print(f"    units={u:3d}  best val loss {vl:.4f}  ({ep} epochs)")
    best_u = min(scores, key=scores.get)
    per_seed = []
    for s in range(n_seeds):
        m, vl, ep = train_one(Xtr, ytr, Xva, yva, best_u, seed=100 + s)
        per_seed.append(m.predict(Xte, batch_size=1024, verbose=0).reshape(-1) * sc.ysd)
        if verbose:
            print(f"    seed {s}: val loss {vl:.4f} ({ep} epochs)")
    per_seed = np.array(per_seed)                                            # (n_seeds, n*T)
    ens = per_seed.mean(axis=0).reshape(len(test_idx), T)
    singles = per_seed.reshape(n_seeds, len(test_idx), T)
    return ens, singles, best_u