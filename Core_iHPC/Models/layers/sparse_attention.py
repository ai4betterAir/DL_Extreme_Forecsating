import tensorflow as tf
from tensorflow.keras import layers

# Localized Attention: The SparseAttention layer uses a sliding window approach to focus on local temporal patterns while maintaining O(n) complexity.
class SparseAttention(layers.Layer):
    def __init__(self, num_heads=4, window_size=32, **kwargs):
        super(SparseAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.window_size = window_size

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'window_size': self.window_size,
        })
        return config

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.depth = self.d_model // self.num_heads
        
        self.query_dense = layers.Dense(self.d_model)
        self.key_dense = layers.Dense(self.d_model)
        self.value_dense = layers.Dense(self.d_model)
        self.dense = layers.Dense(self.d_model)
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        self.dense.build(input_shape)
        super().build(input_shape)

    def get_local_mask(self, seq_length):
        """Create banded attention mask for local attention"""
        range_vec = tf.range(seq_length)
        distance_matrix = tf.abs(tf.expand_dims(range_vec, 0) - tf.expand_dims(range_vec, 1))
        mask = tf.logical_and(distance_matrix >= 0, distance_matrix < self.window_size)
        return tf.cast(mask, tf.float32)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Project inputs
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        
        # Split into multiple heads
        q = tf.reshape(q, [batch_size, seq_length, self.num_heads, self.depth])
        k = tf.reshape(k, [batch_size, seq_length, self.num_heads, self.depth])
        v = tf.reshape(v, [batch_size, seq_length, self.num_heads, self.depth])
        
        # Compute attention scores
        q = tf.transpose(q, [0, 2, 1, 3])  # [batch, heads, seq, depth]
        k = tf.transpose(k, [0, 2, 3, 1])  # [batch, heads, depth, seq]
        scores = tf.matmul(q, k) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        # Apply local attention mask
        mask = self.get_local_mask(seq_length)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, 0)
        scores += (1.0 - mask) * -1e9
        
        # Apply softmax
        weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        v = tf.transpose(v, [0, 2, 1, 3])
        output = tf.matmul(weights, v)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, seq_length, self.d_model])
        return self.dense(output)
