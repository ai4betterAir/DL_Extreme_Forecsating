"""TFT-inspired interpretable multi-horizon PM2.5 forecaster."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import BaseExtremeForecaster, config_value


@keras.utils.register_keras_serializable(package="extreme_pm25")
class VariableSelection(layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.feature_projection = layers.Dense(hidden_size, activation="swish")
        self.gate = layers.Dense(1)

    def call(self, inputs):
        expanded = tf.expand_dims(inputs, axis=-1)
        features = self.feature_projection(expanded)
        logits = tf.squeeze(self.gate(features), axis=-1)
        weights = tf.nn.softmax(logits, axis=-1)
        return tf.reduce_sum(features * tf.expand_dims(weights, -1), axis=2)

    def get_config(self):
        return {**super().get_config(), "hidden_size": self.hidden_size}


class TFT_Extreme_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "TFT_Extreme_PM25"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "tft_extreme_pm25"

    def build_model(self, num_sites):
        hidden = int(config_value(self.Configuration, "tft_hidden_size", 64))
        heads = int(config_value(self.Configuration, "tft_attention_heads", 4))
        dropout = float(config_value(self.Configuration, "tft_dropout", 0.15))

        inputs = keras.Input((self.lookback, num_sites), name="dynamic_inputs")
        selected = VariableSelection(hidden, name="variable_selection")(inputs)
        recurrent = layers.LSTM(hidden, return_sequences=True, dropout=dropout)(selected)
        recurrent = layers.LayerNormalization()(recurrent + selected)
        attention = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=max(hidden // heads, 1),
            dropout=dropout,
            name="interpretable_attention",
        )(recurrent, recurrent, use_causal_mask=True)
        x = layers.LayerNormalization()(recurrent + attention)
        gated = layers.Dense(hidden, activation="swish")(x)
        gate = layers.Dense(hidden, activation="sigmoid")(x)
        x = layers.Multiply()([gated, gate])
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="swish")(x)
        output = layers.Dense(self.horizon * num_sites * len(self.quantiles))(x)
        output = layers.Reshape((self.horizon, num_sites, len(self.quantiles)))(output)
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)


MODEL_CLASS = TFT_Extreme_Class
