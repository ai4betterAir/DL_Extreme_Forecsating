"""Causal TCN baseline with extreme-aware quantile training."""
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import BaseExtremeForecaster, config_value


class TCN_Extreme_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "TCN_Extreme_PM25"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "tcn_extreme_pm25"

    def build_model(self, num_sites):
        channels = int(config_value(self.Configuration, "tcn_channels", 64))
        blocks = int(config_value(self.Configuration, "tcn_blocks", 6))
        kernel = int(config_value(self.Configuration, "tcn_kernel_size", 3))
        dropout = float(config_value(self.Configuration, "tcn_dropout", 0.15))

        inputs = keras.Input((self.lookback, num_sites), name="station_history")
        x = layers.Conv1D(channels, 1, padding="same")(inputs)
        for index in range(blocks):
            residual = x
            dilation = 2 ** index
            x = layers.Conv1D(channels, kernel, padding="causal", dilation_rate=dilation, activation="swish")(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(channels, kernel, padding="causal", dilation_rate=dilation)(x)
            x = layers.Add()([residual, x])
            x = layers.Activation("swish")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="swish")(x)
        output = layers.Dense(self.horizon * num_sites * len(self.quantiles))(x)
        output = layers.Reshape((self.horizon, num_sites, len(self.quantiles)))(output)
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)


MODEL_CLASS = TCN_Extreme_Class
