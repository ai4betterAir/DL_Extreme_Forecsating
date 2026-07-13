"""PatchTST-style long-context upper-quantile PM2.5 model."""
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import BaseExtremeForecaster, config_value


@keras.utils.register_keras_serializable(package="extreme_pm25")
class AddPositionEmbedding(layers.Layer):
    def build(self, input_shape):
        self.position = self.add_weight(
            name="position_embedding",
            shape=(1, int(input_shape[1]), int(input_shape[2])),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position


class PatchTST_TailQuantile_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "PatchTST_TailQuantile_PM25"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "patchtst_tail_quantile_pm25"

    def build_model(self, num_sites):
        patch_length = min(int(config_value(self.Configuration, "patch_length", 12)), self.lookback)
        stride = int(config_value(self.Configuration, "patch_stride", 6))
        d_model = int(config_value(self.Configuration, "patch_d_model", 96))
        heads = int(config_value(self.Configuration, "patch_attention_heads", 4))
        blocks = int(config_value(self.Configuration, "patch_transformer_blocks", 3))
        dropout = float(config_value(self.Configuration, "patch_dropout", 0.15))

        inputs = keras.Input((self.lookback, num_sites), name="station_history")
        x = layers.Conv1D(d_model, kernel_size=patch_length, strides=max(1, stride),
                          padding="valid", name="patch_embedding")(inputs)
        x = AddPositionEmbedding()(x)
        for _ in range(blocks):
            attention = layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=max(d_model // heads, 1),
                dropout=dropout,
            )(x, x)
            x = layers.LayerNormalization()(x + attention)
            feedforward = layers.Dense(d_model * 4, activation="gelu")(x)
            feedforward = layers.Dropout(dropout)(feedforward)
            feedforward = layers.Dense(d_model)(feedforward)
            x = layers.LayerNormalization()(x + feedforward)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="gelu")(x)
        output = layers.Dense(self.horizon * num_sites * len(self.quantiles))(x)
        output = layers.Reshape((self.horizon, num_sites, len(self.quantiles)))(output)
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)


MODEL_CLASS = PatchTST_TailQuantile_Class
