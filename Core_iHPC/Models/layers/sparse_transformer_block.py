import tensorflow as tf
from tensorflow.keras import layers, Sequential
from .sparse_attention import SparseAttention

# Multi-Head Architecture: Combines multiple attention heads to capture different temporal relationships simultaneously.
# Transformer Block: Includes residual connections, layer normalization, and feed-forward networks for stable training.

class SparseTransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, window_size=32, ff_dim=256, 
                dropout=0.1, d_model=64, **kwargs):  # Add d_model parameter
        super(SparseTransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.window_size = window_size
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.d_model = d_model
        self.attn = SparseAttention(num_heads, window_size)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)  # Use parameter directly
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'd_model': self.d_model,
        })
        return config

    def build(self, input_shape):
        self.attn.build(input_shape)
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.attn(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
