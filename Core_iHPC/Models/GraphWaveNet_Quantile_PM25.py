"""Graph WaveNet-style direct multi-horizon quantile model for PM2.5."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Core_iHPC.Models.extreme_common import BaseExtremeForecaster, config_value


@keras.utils.register_keras_serializable(package="extreme_pm25")
class AdaptiveGraphMix(layers.Layer):
    """Learn a directed station adjacency and mix node signals at every hour."""

    def __init__(self, num_nodes, embedding_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)

    def build(self, input_shape):
        self.source = self.add_weight(name="source_embedding", shape=(self.num_nodes, self.embedding_dim), initializer="glorot_uniform", trainable=True)
        self.target = self.add_weight(name="target_embedding", shape=(self.embedding_dim, self.num_nodes), initializer="glorot_uniform", trainable=True)
        self.projection = layers.Dense(self.num_nodes, activation="swish")
        super().build(input_shape)

    def call(self, inputs):
        adjacency = tf.nn.softmax(tf.nn.relu(tf.matmul(self.source, self.target)), axis=-1)
        propagated = tf.einsum("btn,nm->btm", inputs, adjacency)
        return self.projection(tf.concat([inputs, propagated], axis=-1))

    def get_config(self):
        return {**super().get_config(), "num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim}


class GraphWaveNet_Quantile_Class(BaseExtremeForecaster):
    MODEL_DISPLAY_NAME = "GraphWaveNet_Quantile_PM25"
    MODEL_TAG = "v3"
    DEFAULT_MODEL_NAME = "graphwavenet_quantile_pm25"

    def build_model(self, num_sites):
        channels = int(config_value(self.Configuration, "gwn_channels", 64))
        blocks = int(config_value(self.Configuration, "gwn_blocks", 4))
        kernel = int(config_value(self.Configuration, "gwn_kernel_size", 2))
        embedding = int(config_value(self.Configuration, "gwn_embedding_dim", 16))
        dropout = float(config_value(self.Configuration, "gwn_dropout", 0.15))

        inputs = keras.Input((self.lookback, num_sites), name="station_history")
        graph = AdaptiveGraphMix(num_sites, embedding_dim=embedding, name="adaptive_graph")(inputs)
        x = layers.Conv1D(channels, 1, padding="same")(graph)
        skips = []
        for block in range(blocks):
            dilation = 2 ** block
            residual = x
            tanh_path = layers.Conv1D(channels, kernel, padding="causal", dilation_rate=dilation, activation="tanh")(x)
            gate_path = layers.Conv1D(channels, kernel, padding="causal", dilation_rate=dilation, activation="sigmoid")(x)
            x = layers.Multiply()([tanh_path, gate_path])
            x = layers.Dropout(dropout)(x)
            skips.append(layers.Conv1D(channels, 1, padding="same")(x))
            x = layers.Add()([residual, layers.Conv1D(channels, 1, padding="same")(x)])
            x = layers.LayerNormalization()(x)

        x = layers.Add()(skips) if len(skips) > 1 else skips[0]
        x = layers.Activation("swish")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="swish")(x)
        output = layers.Dense(self.horizon * num_sites * len(self.quantiles))(x)
        output = layers.Reshape((self.horizon, num_sites, len(self.quantiles)))(output)
        return keras.Model(inputs, output, name=self.MODEL_DISPLAY_NAME)


MODEL_CLASS = GraphWaveNet_Quantile_Class
