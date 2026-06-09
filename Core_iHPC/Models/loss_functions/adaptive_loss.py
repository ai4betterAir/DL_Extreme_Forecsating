import tensorflow as tf
from ..model import create_sparse_lstm_model

# Custom adaptive loss function
def adaptive_loss(y_true, y_pred):
    # Get learned log variances from the model
    log_vars = model.layers[-1].log_vars
    
    # Calculate precision (inverse variance)
    precision = tf.exp(-log_vars)
    
    # Calculate weighted MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)
    weighted_loss = tf.reduce_sum(precision * mse + log_vars)
    
    return weighted_loss