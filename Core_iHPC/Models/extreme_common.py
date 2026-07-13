"""Leakage-safe utilities for extreme-aware multi-station PM2.5 models."""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import callbacks

from Core_iHPC.Models.extreme_distributions import (
    gev_negative_log_likelihood,
    gpd_negative_log_likelihood,
)

EPS = tf.keras.backend.epsilon()


def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def config_value(configuration, name, default):
    value = getattr(configuration, name, None)
    if value is None:
        value = (getattr(configuration, "model_parameters_dict", {}) or {}).get(name)
    return default if value is None else value


def parse_quantiles(value):
    if value is None:
        return (0.50, 0.75, 0.90, 0.95, 0.99)
    if isinstance(value, str):
        value = value.split(",")
    result = tuple(sorted({float(item) for item in value}))
    if not result or any(q <= 0 or q >= 1 for q in result):
        raise ValueError("All quantiles must be between zero and one.")
    return result


@keras.utils.register_keras_serializable(package="extreme_pm25")
class QuantileLoss(keras.losses.Loss):
    def __init__(self, quantiles: Sequence[float], name="quantile_loss"):
        super().__init__(name=name)
        self.quantiles = tuple(float(q) for q in quantiles)

    def call(self, y_true, y_pred):
        error = tf.expand_dims(tf.cast(y_true, y_pred.dtype), -1) - y_pred
        q = tf.constant(self.quantiles, dtype=y_pred.dtype)
        return tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error), axis=[1, 2, 3])

    def get_config(self):
        return {**super().get_config(), "quantiles": list(self.quantiles)}


def focal_binary_crossentropy(y_true, y_prob, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, y_prob.dtype)
    y_prob = tf.clip_by_value(y_prob, EPS, 1.0 - EPS)
    pt = tf.where(tf.equal(y_true, 1.0), y_prob, 1.0 - y_prob)
    alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
    return -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)


def inverse_scale_3d(scaler, values):
    values = np.asarray(values)
    if values.ndim == 2:
        return scaler.inverse_transform(values)
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, shape[-1])).reshape(shape)


def scale_thresholds(scaler, thresholds):
    return scaler.transform(np.asarray(thresholds).reshape(1, -1)).reshape(-1)


def choose_target_column(frame, target):
    lookup = {str(col).upper().replace(" ", ""): col for col in frame.columns}
    for candidate in ("Original", target, target.replace(".", ""), "Residual", "Reconstructed"):
        key = str(candidate).upper().replace(" ", "")
        if key in lookup:
            return lookup[key]
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric target column found.")
    return numeric[0]


def extract_station_matrix(station_frames: Mapping[str, pd.DataFrame], target):
    series, names = [], []
    for station, frame in station_frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        values = pd.to_numeric(frame[choose_target_column(frame, target)], errors="coerce")
        values.index = pd.to_datetime(frame.index, errors="coerce")
        series.append(values.rename(str(station)))
        names.append(str(station))
    if not series:
        raise ValueError("No usable station data found.")
    joined = pd.concat(series, axis=1, join="inner").sort_index()
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise ValueError("Station series have no common finite timestamps.")
    return pd.DatetimeIndex(joined.index), names, joined.to_numpy(np.float32)


def make_sequences(data, lookback, horizon, target_mode="trajectory"):
    X, y = [], []
    for start in range(max(len(data) - lookback - horizon + 1, 0)):
        future = data[start + lookback:start + lookback + horizon]
        X.append(data[start:start + lookback])
        y.append(np.max(future, axis=0) if target_mode == "block_max" else future)
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


@dataclass
class ChronologicalData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    test_timestamps: pd.DatetimeIndex


def chronological_windows(timestamps, raw, lookback, horizon, scaler, train_fraction, val_fraction, target_mode):
    n = len(raw)
    train_end = max(lookback + horizon + 1, int(n * train_fraction))
    val_end = min(max(train_end + horizon + 1, int(n * (train_fraction + val_fraction))), n - horizon)
    if val_end <= train_end or n - val_end < horizon + 1:
        raise ValueError("Insufficient data for chronological train/validation/test splits.")
    scaler.fit(raw[:train_end])
    scaled = scaler.transform(raw).astype(np.float32)

    def section(start, end):
        context = max(0, start - lookback)
        X, y = make_sequences(scaled[context:end], lookback, horizon, target_mode)
        _, y_raw = make_sequences(raw[context:end], lookback, horizon, target_mode)
        return X, y, y_raw

    X_train, y_train, yr_train = section(0, train_end)
    X_val, y_val, yr_val = section(train_end, val_end)
    X_test, y_test, yr_test = section(val_end, n)
    timestamp_offset = horizon - 1 if target_mode == "block_max" else 0
    test_ts = timestamps[val_end + timestamp_offset:val_end + timestamp_offset + len(y_test)]
    return ChronologicalData(X_train, y_train, X_val, y_val, X_test, y_test,
                             yr_train, yr_val, yr_test, pd.DatetimeIndex(test_ts))


def event_sample_weights(y_raw, thresholds, boost=4.0):
    threshold = thresholds[None, :] if y_raw.ndim == 2 else thresholds[None, None, :]
    axes = (1,) if y_raw.ndim == 2 else (1, 2)
    event = np.any(y_raw >= threshold, axis=axes)
    return np.where(event, 1.0 + float(boost), 1.0).astype(np.float32)


def compute_extreme_metrics(y_true, y_pred, thresholds):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    error = y_pred - y_true
    threshold = thresholds[None, :] if y_true.ndim == 2 else thresholds[None, None, :]
    extreme = y_true >= threshold
    predicted = y_pred >= threshold
    tp, fp, fn = np.sum(predicted & extreme), np.sum(predicted & ~extreme), np.sum(~predicted & extreme)
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else np.nan
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "TAIL_RMSE": float(np.sqrt(np.mean(error[extreme] ** 2))) if np.any(extreme) else np.nan,
        "TAIL_MAE": float(np.mean(np.abs(error[extreme]))) if np.any(extreme) else np.nan,
        "EVENT_PRECISION": float(precision),
        "EVENT_RECALL": float(recall),
        "EVENT_F1": float(f1),
    }


def station_metrics_frame(y_true, y_pred, thresholds, station_names):
    rows = []
    for index, station in enumerate(station_names):
        true_station = y_true[..., index]
        pred_station = y_pred[..., index]
        metrics = compute_extreme_metrics(true_station[..., None], pred_station[..., None], np.asarray([thresholds[index]]))
        rows.append({"Stations": station, **metrics})
    rows.append({"Stations": "ALL", **compute_extreme_metrics(y_true, y_pred, thresholds)})
    return pd.DataFrame(rows)


class BaseExtremeForecaster:
    MODEL_DISPLAY_NAME = "BaseExtremeForecaster"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "extreme_pm25"
    TARGET_MODE = "trajectory"

    def __init__(self, Configuration):
        self.Configuration = Configuration
        self.lookback = int(config_value(Configuration, "n_steps_in", 72))
        self.horizon = int(config_value(Configuration, "n_steps_out", 24))
        self.n_epochs = int(config_value(Configuration, "n_epochs", 200))
        self.batch_size = int(config_value(Configuration, "batch_size", 32))
        self.learning_rate = float(config_value(Configuration, "learning_rate", 1e-3))
        self.quantiles = parse_quantiles(config_value(Configuration, "extreme_quantiles", None))
        self.extreme_percentile = float(config_value(Configuration, "extreme_percentile", 0.95))
        self.extreme_weight = float(config_value(Configuration, "extreme_weight", 4.0))
        self.train_fraction = float(config_value(Configuration, "train_fraction", 0.70))
        self.val_fraction = float(config_value(Configuration, "val_fraction", 0.15))
        self.target = str((getattr(Configuration, "var_to_predict", None) or ["PM2.5"])[0])
        base = getattr(Configuration, "model_base_path", "AI_Runs/Model_weights")
        configured_name = str(getattr(Configuration, "model_name", None) or "")
        if not configured_name or (configured_name.lower().startswith("sparse_lstm") and not self.DEFAULT_MODEL_NAME.startswith("sparse_lstm")):
            configured_name = self.DEFAULT_MODEL_NAME
        version = str(getattr(Configuration, "model_version", "v3.0"))
        self.model_root = os.path.join(base, f"{configured_name}_{version}")
        self.weights_path = os.path.join(self.model_root, "model.weights.h5")
        self.scaler_path = os.path.join(self.model_root, "scaler.pkl")
        self.metadata_path = os.path.join(self.model_root, "metadata.json")
        self.metrics_path = os.path.join(self.model_root, "metrics.csv")
        self.scaler, self.model = RobustScaler(quantile_range=(5, 95)), None
        self.station_names: List[str] = []
        self.thresholds: Optional[np.ndarray] = None
        self.latest_training_plot_data = None
        self.latest_dashboard_metrics_pd = None

    def build_model(self, num_sites):
        raise NotImplementedError

    def loss(self):
        return QuantileLoss(self.quantiles)

    def compile_model(self):
        self.model.compile(keras.optimizers.Adam(self.learning_rate), loss=self.loss())

    def prepare_model_targets(self, y_scaled, y_raw):
        return y_scaled

    def decode_model_output(self, output):
        median = int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))
        return inverse_scale_3d(self.scaler, np.asarray(output)[..., median])

    def _callbacks(self):
        return [callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)]

    def _save(self):
        os.makedirs(self.model_root, exist_ok=True)
        self.model.save_weights(self.weights_path)
        with open(self.scaler_path, "wb") as handle:
            pickle.dump(self.scaler, handle)
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump({"model": self.MODEL_DISPLAY_NAME, "station_names": self.station_names,
                       "thresholds": self.thresholds.tolist(), "quantiles": list(self.quantiles)}, handle, indent=2)
        if self.latest_dashboard_metrics_pd is not None:
            self.latest_dashboard_metrics_pd.to_csv(self.metrics_path, index=False)

    def _load_expected_stations(self, station_frames):
        if not os.path.exists(self.metadata_path):
            return station_frames
        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            expected = json.load(handle).get("station_names") or []
        missing = [station for station in expected if station not in station_frames]
        if missing:
            raise ValueError("Missing station(s) required by saved model: " + ", ".join(missing))
        return {station: station_frames[station] for station in expected}

    def _load(self, num_sites):
        for path in (self.weights_path, self.scaler_path, self.metadata_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required trained-model file not found: {path}")
        with open(self.scaler_path, "rb") as handle:
            self.scaler = pickle.load(handle)
        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.station_names = metadata["station_names"]
        self.thresholds = np.asarray(metadata["thresholds"], float)
        if os.path.exists(self.metrics_path):
            self.latest_dashboard_metrics_pd = pd.read_csv(self.metrics_path)
        self.model = self.build_model(num_sites)
        self.compile_model()
        self.model.load_weights(self.weights_path)

    def _format_output(self, values, last_timestamp, iteration):
        values = values[0] if values.ndim == 3 else values
        times = pd.date_range(pd.Timestamp(last_timestamp) + pd.Timedelta(hours=1), periods=len(values), freq="h")
        outputs = []
        for node, station in enumerate(self.station_names):
            frame = pd.DataFrame({"forecast_hours": np.arange(1, len(values) + 1),
                                  "forecast_number": iteration, "station": station,
                                  f"{self.target}_{station}": values[:, node],
                                  "model_version": str(getattr(self.Configuration, "model_version", "v3.0")),
                                  "generated_at": pd.Timestamp.now()}, index=times)
            frame.index.name = "timestamp"
            outputs.append(frame)
        return outputs

    def run_all(self, station_decomposed_dict, n_retrain=0):
        train_model = as_bool(getattr(self.Configuration, "train_model", True), True)
        station_frames = station_decomposed_dict if train_model else self._load_expected_stations(station_decomposed_dict)
        timestamps, self.station_names, raw = extract_station_matrix(station_frames, self.target)
        num_sites = len(self.station_names)
        if train_model:
            data = chronological_windows(timestamps, raw, self.lookback, self.horizon, self.scaler,
                                         self.train_fraction, self.val_fraction, self.TARGET_MODE)
            train_end = int(len(raw) * self.train_fraction)
            self.thresholds = np.quantile(raw[:train_end], self.extreme_percentile, axis=0)
            self.model = self.build_model(num_sites)
            self.compile_model()
            weights = event_sample_weights(data.y_train_raw, self.thresholds, self.extreme_weight)
            history = self.model.fit(data.X_train, self.prepare_model_targets(data.y_train, data.y_train_raw),
                                     validation_data=(data.X_val, self.prepare_model_targets(data.y_val, data.y_val_raw)),
                                     sample_weight=weights, epochs=self.n_epochs, batch_size=self.batch_size,
                                     callbacks=self._callbacks(), verbose=1)
            prediction = self.decode_model_output(self.model.predict(data.X_test, verbose=0))
            self.latest_dashboard_metrics_pd = station_metrics_frame(
                data.y_test_raw, prediction, self.thresholds, self.station_names
            )
            plot = {"station_names": self.station_names, "timestamps": data.test_timestamps,
                    "predictions": {s: prediction[:, 0, i] if prediction.ndim == 3 else prediction[:, i]
                                    for i, s in enumerate(self.station_names)},
                    "actuals": {s: data.y_test_raw[:, 0, i] if data.y_test_raw.ndim == 3 else data.y_test_raw[:, i]
                                for i, s in enumerate(self.station_names)}}
            self.latest_training_plot_data = plot
            if as_bool(getattr(self.Configuration, "save_model", True), True):
                self._save()
            return None, {"history": history.history,
                          "metrics": self.latest_dashboard_metrics_pd.to_dict(orient="records"),
                          "training_plot_data": plot}
        self._load(num_sites)
        X = self.scaler.transform(raw).astype(np.float32)[-self.lookback:][None, ...]
        prediction = self.decode_model_output(self.model.predict(X, verbose=0))
        return self._format_output(prediction, timestamps[-1], n_retrain), None
