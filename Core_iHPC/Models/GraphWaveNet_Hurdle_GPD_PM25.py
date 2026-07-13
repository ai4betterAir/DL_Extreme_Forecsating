"""Graph WaveNet hurdle-GPD model for rare PM2.5 exceedances."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import (
    BaseExtremeForecaster,
    config_value,
    focal_binary_crossentropy,
    gpd_negative_log_likelihood,
    inverse_scale_3d,
    scale_thresholds,
)
from Core_iHPC.Models.GraphWaveNet_Quantile_PM25 import AdaptiveGraphMix


@keras.utils.register_keras_serializable(package="extreme_pm25")
class HurdleGPDLoss(keras.losses.Loss):
    def __init__(self, thresholds, regression_weight=0.40, event_weight=0.30,
                 tail_weight=0.30, name="hurdle_gpd_loss"):
        super().__init__(name=name)
        self.thresholds = tuple(float(v) for v in thresholds)
        self.regression_weight = float(regression_weight)
        self.event_weight = float(event_weight)
        self.tail_weight = float(tail_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        mu, probability = y_pred[..., 0], y_pred[..., 1]
        xi, beta = y_pred[..., 2], y_pred[..., 3]
        threshold = tf.constant(self.thresholds, dtype=y_pred.dtype)[None, None, :]
        event = tf.cast(y_true > threshold, y_pred.dtype)
        error = y_true - mu
        absolute = tf.abs(error)
        huber = tf.where(absolute <= 1.0, 0.5 * tf.square(error), absolute - 0.5)
        regression = tf.reduce_mean(huber, axis=[1, 2])
        classification = tf.reduce_mean(focal_binary_crossentropy(event, probability), axis=[1, 2])
        excess = tf.maximum(y_true - threshold, 0.0)
        tail_nll = gpd_negative_log_likelihood(excess, xi, beta)
        event_count = tf.maximum(tf.reduce_sum(event, axis=[1, 2]), 1.0)
        tail = tf.reduce_sum(tail_nll * event, axis=[1, 2]) / event_count
        return self.regression_weight * regression + self.event_weight * classification + self.tail_weight * tail

    def get_config(self):
        return {**super().get_config(), "thresholds": list(self.thresholds),
                "regression_weight": self.regression_weight,
                "event_weight": self.event_weight, "tail_weight": self.tail_weight}


class GraphWaveNet_Hurdle_GPD_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "GraphWaveNet_Hurdle_GPD_PM25"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "graphwavenet_hurdle_gpd_pm25"

    def loss(self):
        if self.thresholds is None:
            raise RuntimeError("Extreme thresholds must be fitted before compilation.")
        return HurdleGPDLoss(scale_thresholds(self.scaler, self.thresholds))

    def build_model(self, num_sites):
        channels = int(config_value(self.Configuration, "gwn_channels", 64))
        blocks = int(config_value(self.Configuration, "gwn_blocks", 4))
        embedding = int(config_value(self.Configuration, "gwn_embedding_dim", 16))
        dropout = float(config_value(self.Configuration, "gwn_dropout", 0.15))

        inputs = keras.Input((self.lookback, num_sites), name="station_history")
        x = AdaptiveGraphMix(num_sites, embedding_dim=embedding)(inputs)
        x = layers.Conv1D(channels, 1, padding="same")(x)
        skip = []
        for block in range(blocks):
            residual = x
            dilation = 2 ** block
            candidate = layers.Conv1D(channels, 2, padding="causal", dilation_rate=dilation, activation="tanh")(x)
            gate = layers.Conv1D(channels, 2, padding="causal", dilation_rate=dilation, activation="sigmoid")(x)
            x = layers.Multiply()([candidate, gate])
            x = layers.Dropout(dropout)(x)
            skip.append(layers.Conv1D(channels, 1)(x))
            x = layers.LayerNormalization()(layers.Add()([residual, layers.Conv1D(channels, 1)(x)]))
        x = layers.Add()(skip) if len(skip) > 1 else skip[0]
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="swish")(x)

        mu = layers.Reshape((self.horizon, num_sites, 1))(layers.Dense(self.horizon * num_sites)(x))
        probability = layers.Reshape((self.horizon, num_sites, 1))(
            layers.Dense(self.horizon * num_sites, activation="sigmoid")(x))
        xi_raw = layers.Dense(self.horizon * num_sites, activation="tanh")(x)
        xi = layers.Reshape((self.horizon, num_sites, 1))(xi_raw * 0.45)
        beta = layers.Reshape((self.horizon, num_sites, 1))(
            layers.Dense(self.horizon * num_sites, activation="softplus")(x))
        output = layers.Concatenate(axis=-1, name="hurdle_gpd_parameters")([mu, probability, xi, beta])
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)

    def decode_model_output(self, raw_output):
        raw_output = np.asarray(raw_output)
        mu, probability = raw_output[..., 0], raw_output[..., 1]
        xi = np.clip(raw_output[..., 2], -0.45, 0.45)
        beta = np.maximum(raw_output[..., 3], 1e-4)
        threshold = scale_thresholds(self.scaler, self.thresholds)[None, None, :]
        tail_mean = threshold + beta / np.maximum(1.0 - xi, 0.05)
        return inverse_scale_3d(self.scaler, (1.0 - probability) * mu + probability * tail_mean)


MODEL_CLASS = GraphWaveNet_Hurdle_GPD_Class
