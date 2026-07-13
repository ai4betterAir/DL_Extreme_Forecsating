"""DeepExtrema-style GEV model for maximum PM2.5 in a future block."""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import BaseExtremeForecaster, config_value, gev_negative_log_likelihood


@keras.utils.register_keras_serializable(package="extreme_pm25")
class GEVLoss(keras.losses.Loss):
    def __init__(self, name="gev_negative_log_likelihood"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(
            gev_negative_log_likelihood(y_true, y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]),
            axis=1,
        )


class DeepExtrema_GEV_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "DeepExtrema_GEV_PM25"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "deepextrema_gev_pm25"
    TARGET_MODE = "block_max"

    def loss(self):
        return GEVLoss()

    def build_model(self, num_sites):
        hidden = int(config_value(self.Configuration, "gev_hidden_size", 96))
        dropout = float(config_value(self.Configuration, "gev_dropout", 0.15))
        inputs = keras.Input((self.lookback, num_sites), name="station_history")
        x = layers.Conv1D(hidden, 5, padding="causal", activation="swish")(inputs)
        x = layers.Conv1D(hidden, 3, padding="causal", activation="swish")(x)
        x = layers.Bidirectional(layers.GRU(hidden // 2, dropout=dropout))(x)
        x = layers.Dense(hidden, activation="swish")(x)
        mu = layers.Reshape((num_sites, 1))(layers.Dense(num_sites)(x))
        sigma = layers.Reshape((num_sites, 1))(layers.Dense(num_sites, activation="softplus")(x))
        xi_raw = layers.Dense(num_sites, activation="tanh")(x)
        xi = layers.Reshape((num_sites, 1))(xi_raw * 0.45)
        output = layers.Concatenate(axis=-1, name="gev_parameters")([mu, sigma, xi])
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)

    def decode_model_output(self, raw_output):
        params = np.asarray(raw_output)
        mu = params[..., 0]
        sigma = np.maximum(params[..., 1], 1e-4)
        xi = np.clip(params[..., 2], -0.45, 0.45)
        a = -np.log(0.5)
        safe_xi = np.where(np.abs(xi) < 1e-6, 1.0, xi)
        regular = mu + sigma / safe_xi * (a ** (-xi) - 1.0)
        gumbel = mu - sigma * np.log(a)
        return self.scaler.inverse_transform(np.where(np.abs(xi) < 1e-3, gumbel, regular))

    def _format_output(self, values, last_timestamp, iteration):
        values = np.asarray(values)
        values = values[0] if values.ndim == 2 else values
        timestamp = pd.Timestamp(last_timestamp) + pd.Timedelta(hours=self.horizon)
        outputs = []
        for index, station in enumerate(self.station_names):
            frame = pd.DataFrame({
                "forecast_hours": [self.horizon],
                "forecast_number": [iteration],
                "station": [station],
                f"{self.target}_MAX_{station}": [values[index]],
                "model_version": [str(getattr(self.Configuration, "model_version", "v3.0"))],
                "generated_at": [pd.Timestamp.now()],
            }, index=pd.DatetimeIndex([timestamp], name="timestamp"))
            outputs.append(frame)
        return outputs


MODEL_CLASS = DeepExtrema_GEV_Class
