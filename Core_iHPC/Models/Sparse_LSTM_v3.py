"""Leakage-safe Sparse CNN-Transformer-LSTM v3 for extreme PM2.5.

Corrections relative to v2:
- uses the raw aligned station signal by default, avoiding full-series VMD leakage;
- fits scaling and extreme thresholds on the chronological training period only;
- predicts the full horizon directly instead of one isolated lag;
- uses quantile loss and event-balanced sample weights.
"""
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import BaseExtremeForecaster, config_value
from Core_iHPC.Models.layers.sparse_transformer_block import SparseTransformerBlock


class Sparse_LSTM_v3_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "Sparse_LSTM_v3"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "sparse_lstm_pm25"

    def build_model(self, num_sites):
        filters = int(config_value(self.Configuration, "sparse_lstm_cnn_filters", 64))
        kernel = int(config_value(self.Configuration, "sparse_lstm_cnn_kernel_size", 3))
        heads = int(config_value(self.Configuration, "sparse_lstm_num_heads", 4))
        window = int(config_value(self.Configuration, "sparse_lstm_window_size", min(24, self.lookback)))
        ff_dim = int(config_value(self.Configuration, "sparse_lstm_ff_dim", 256))
        lstm_units = int(config_value(self.Configuration, "sparse_lstm_units", 128))
        dropout = float(config_value(self.Configuration, "sparse_lstm_dropout", 0.15))

        inputs = keras.Input(shape=(self.lookback, num_sites), name="station_history")
        x = layers.Conv1D(filters, kernel, padding="causal", activation="swish")(inputs)
        x = layers.Conv1D(filters, kernel, padding="causal", activation="swish")(x)
        x = layers.LayerNormalization()(x)
        x = SparseTransformerBlock(
            num_heads=heads,
            window_size=min(window, self.lookback),
            ff_dim=ff_dim,
            d_model=filters,
            dropout=dropout,
        )(x)
        x = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout)(x)
        x = layers.Dense(128, activation="swish")(x)
        x = layers.Dropout(dropout)(x)
        output = layers.Dense(self.horizon * num_sites * len(self.quantiles))(x)
        output = layers.Reshape((self.horizon, num_sites, len(self.quantiles)), name="pm25_quantiles")(output)
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)


MODEL_CLASS = Sparse_LSTM_v3_Class
Sparse_LSTM_Class = Sparse_LSTM_v3_Class
