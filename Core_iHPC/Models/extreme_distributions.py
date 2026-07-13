"""Numerically stable extreme-value distribution losses."""
import tensorflow as tf


def gpd_negative_log_likelihood(excess, xi, beta):
    excess = tf.maximum(tf.cast(excess, beta.dtype), 0.0)
    beta = tf.maximum(beta, tf.cast(1e-4, beta.dtype))
    xi = tf.clip_by_value(xi, -0.45, 0.45)
    near_zero = tf.abs(xi) < 1e-3
    safe_xi = tf.where(near_zero, tf.ones_like(xi), xi)
    z = tf.maximum(1.0 + safe_xi * excess / beta, tf.cast(1e-6, beta.dtype))
    regular = tf.math.log(beta) + (1.0 / safe_xi + 1.0) * tf.math.log(z)
    exponential = tf.math.log(beta) + excess / beta
    return tf.where(near_zero, exponential, regular)


def gev_negative_log_likelihood(y, mu, sigma, xi):
    y = tf.cast(y, mu.dtype)
    sigma = tf.maximum(sigma, tf.cast(1e-4, sigma.dtype))
    xi = tf.clip_by_value(xi, -0.45, 0.45)
    z = (y - mu) / sigma
    near_zero = tf.abs(xi) < 1e-3
    safe_xi = tf.where(near_zero, tf.ones_like(xi), xi)
    support = tf.maximum(1.0 + safe_xi * z, tf.cast(1e-6, mu.dtype))
    regular = (
        tf.math.log(sigma)
        + (1.0 + 1.0 / safe_xi) * tf.math.log(support)
        + tf.pow(support, -1.0 / safe_xi)
    )
    gumbel = tf.math.log(sigma) + z + tf.exp(-z)
    return tf.where(near_zero, gumbel, regular)
