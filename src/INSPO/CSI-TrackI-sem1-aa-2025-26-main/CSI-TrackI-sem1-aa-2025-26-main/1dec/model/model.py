"""Model utilities and a small neural-network wrapper used by clients.

This file is a centralized copy of the per-client `model.py` used by
clients. It provides a `Model` class that the single `client_flwr.py`
imports and uses.
"""

import numpy as np
import pandas as pd
import hashlib
from typing import Optional, Sequence

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras import regularizers

set_random_seed(42)


def _sanitize_X(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _to_label_vector_and_remap(y):
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    is_one_hot = (
        y_arr.ndim == 2
        and set(np.unique(y_arr)).issubset({0, 1})
        and (y_arr.sum(axis=1) == 1).all()
    )
    if is_one_hot:
        y_arr = y_arr.argmax(axis=1)
    if y_arr.ndim != 1:
        raise ValueError(f"Formato y non supportato: shape={y_arr.shape}")
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(int)
    classes = np.unique(y_arr)
    remap = {int(c): i for i, c in enumerate(sorted(map(int, classes)))}
    y_mapped = np.vectorize(lambda t: remap[int(t)])(y_arr)
    return y_mapped, remap


def _row_hashes(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return np.array([hashlib.sha1(row.tobytes()).hexdigest() for row in X])


def _find_leaky_columns(X_tr: np.ndarray, y_tr: np.ndarray, threshold: float = 0.999) -> Sequence[int]:
    # permetti di disabilitare la leak-check
    if threshold is None or threshold <= 0:
        return []

    df = pd.DataFrame(X_tr)
    yv = np.asarray(y_tr).ravel()
    to_drop = []
    for col in df.columns:
        v = df[col].values
        if np.array_equal(v, yv):
            to_drop.append(col)
            continue
        if np.std(v) > 0:
            corr = np.corrcoef(v, yv)[0, 1]
            if np.isfinite(corr) and abs(corr) >= threshold:
                to_drop.append(col)
    if to_drop:
        print(f"[leak-guard] Drop {len(to_drop)} suspicious columns (corr>= {threshold} or identical to y):", to_drop[:10], "...")
    return to_drop


class WeightChangeLogger(Callback):
    def on_train_begin(self, logs=None):
        self.prev_weights = [w.numpy().copy() for w in self.model.weights]

    def on_epoch_end(self, epoch, logs=None):
        deltas = []
        for w, prev in zip(self.model.weights, self.prev_weights):
            arr = w.numpy()
            deltas.append(np.linalg.norm(arr - prev))
        self.prev_weights = [w.numpy().copy() for w in self.model.weights]
        tot_delta = float(np.sum(deltas))
        print(f"[diag] L2 delta pesi epoca {epoch+1}: {tot_delta:.8f}")


class AccuracyLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        print(f"Epoch {epoch + 1}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")


class Model:
    def __init__(self, input_size, n_classes=3):
        self.input_size = input_size
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self._scaler_fitted = False

        self.model = Sequential([
            Dense(24, activation='relu', input_shape=(input_size,), kernel_regularizer=regularizers.l2(1e-3)),
            Dropout(0.4),
            Dense(12, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=2e-3, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False,  # molto più veloce
        )

    def fit_scaler(self, X):
        X = _sanitize_X(X)
        self.scaler.fit(X)
        self._scaler_fitted = True

    def _ensure_scaler(self, X):
        if not self._scaler_fitted:
            self.fit_scaler(X)

    def fit(self, X, y, epochs=30, batch_size=8, validation_split=0.2, use_class_weight=True, groups: Optional[np.ndarray] = None, leak_corr_threshold: float = 0.999):
        X = _sanitize_X(X)
        y, _ = _to_label_vector_and_remap(y)

        if groups is None:
            # prima usavi SHA1 riga per riga: molto lento
            # qui basta un gruppo per riga (indice) per usare GroupShuffleSplit
            groups = np.arange(X.shape[0])
        else:
            groups = np.asarray(groups)
            if groups.shape[0] != X.shape[0]:
                raise ValueError("La lunghezza di 'groups' deve coincidere con il numero di righe di X.")


        val_size = 0.2 if validation_split is None else validation_split
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        drop_cols = _find_leaky_columns(X_tr, y_tr, threshold=leak_corr_threshold)
        if drop_cols:
            keep_mask = np.ones(X_tr.shape[1], dtype=bool)
            keep_mask[np.array(drop_cols, dtype=int)] = False
            X_tr = X_tr[:, keep_mask]
            X_va = X_va[:, keep_mask]
            print(f"[leak-guard] Rimangono {X_tr.shape[1]} feature dopo il drop.")


        X_tr_scaled = self.scaler.fit_transform(X_tr).astype(np.float32)
        X_va_scaled = self.scaler.transform(X_va).astype(np.float32)
        self._scaler_fitted = True

        class_weight = None
        if use_class_weight and np.unique(y_tr).size > 1:
            present = np.unique(y_tr)
            cw = compute_class_weight(class_weight='balanced', classes=present, y=y_tr)
            class_weight = {int(c): float(w) for c, w in zip(present, cw)}
            for c in range(self.n_classes):
                class_weight.setdefault(int(c), 1.0)
            print("[info] class_weight (train):", class_weight)

        early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1)
        csv_logger = CSVLogger("training_log.csv", append=False)
        wlogger = WeightChangeLogger()
        acc_logger = AccuracyLogger()

        history = self.model.fit(
            X_tr_scaled, y_tr,
            validation_data=(X_va_scaled, y_va),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stop, reduce_lr, csv_logger, wlogger, acc_logger],
            class_weight=class_weight,
            verbose=1
        )

        for k in ("loss", "accuracy", "val_loss", "val_accuracy"):
            if k in history.history:
                v = history.history[k]
                print(f"{k}: {v[0]:.8f} → {v[-1]:.8f} ({len(v)} epoche)")

    def evaluate(self, X, y):
        X = _sanitize_X(X)
        y, _ = _to_label_vector_and_remap(y)
        self._ensure_scaler(X)
        X_scaled = self.scaler.transform(X).astype(np.float32)

        proba = self.model.predict(X_scaled, verbose=0)
        y_pred = np.argmax(proba, axis=1)

        errors_mask = (y_pred != y)
        errors = int(errors_mask.sum())
        n = len(y)
        acc = 1.0 - errors / n

        print(f"[eval] errors: {errors}/{n}  (acc={acc:.6f})")
        if errors > 0:
            err_idx = np.where(errors_mask)[0].tolist()
            print(f"[eval] misclassified idx: {err_idx[:20]}{'...' if len(err_idx) > 20 else ''}")

        print("\nConfusion Matrix:")
        print(pd.DataFrame(confusion_matrix(y, y_pred)))
        print("\nClassification Report:")
        print(pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose())

        loss, _ = self.model.evaluate(X_scaled, y, verbose=0)
        return loss, float(acc), errors

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
