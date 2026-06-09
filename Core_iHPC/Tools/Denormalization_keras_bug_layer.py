"""
..  module:: Denormalization_keras_bug_layer
    :platform: Unix
    :synopsis: Subclass layer to circonvent the keras 2.10 bug in reverse Normalization layer.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
   
"""  
import tensorflow as tf
import tensorflow.keras as TFK
# from keras import backend

class Denormalization_keras_bug_layer(tf.keras.layers.Layer):
    """
    https://stackoverflow.com/questions/73742308/looks-like-keras-normalization-layer-doesnt-denormalize-properly
    https://github.com/keras-team/keras/pull/17054/files
    """
    def __init__(self, N_mean_tf, N_variance_tf, name="Denormalization_keras_bug_layer", **kwargs):
        super(Denormalization_keras_bug_layer, self).__init__(name=name, **kwargs)

        self.N_mean_tf = N_mean_tf
        self.N_variance_tf = N_variance_tf
       
    def call(self, inputs):
        return inputs * tf.maximum(tf.sqrt(self.N_variance_tf), TFK.backend.epsilon()) + self.N_mean_tf
    def get_config(self):
        config = super(Denormalization_keras_bug_layer, self).get_config()
        config.update({
            "N_mean_tf" : self.N_mean_tf.numpy(),
            "N_variance_tf" : self.N_variance_tf.numpy()
            })

        return config        