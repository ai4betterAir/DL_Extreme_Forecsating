"""
..  module:: Base_model
    :platform: Unix
    :synopsis: Definition of the base model to be able to save custom functions.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
   
"""
import keras

# https://www.tensorflow.org/guide/keras/customizing_saving_and_serialization

@keras.saving.register_keras_serializable(package="my_custom_package")
class Model(keras.Model):
    # def __init__(self, input_layer, output_layer, name="default", **kwargs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.dense1 = keras.layers.Dense(8, activation="relu")
        # self.dense2 = keras.layers.Dense(4, activation="softmax")

    # def call(self, inputs):
    #     x = self.dense1(inputs)
    #     return self.dense2(x)

    def compile(self, optimizer=None, loss=None, metrics=None, run_eagerly=False):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
        self.model_optimizer = optimizer
        self.loss = loss
        self.loss_metrics = metrics

    def get_compile_config(self):
        # These parameters will be serialized at saving time.
        return {
            "model_optimizer": self.model_optimizer,
            "loss_fn": self.loss,
            "metric": self.loss_metrics,
        }

    def compile_from_config(self, config):
        # Deserializes the compile parameters (important, since many are custom)
        optimizer = keras.utils.deserialize_keras_object(config["model_optimizer"])
        loss_fn = keras.utils.deserialize_keras_object(config["loss_fn"])
        metrics = keras.utils.deserialize_keras_object(config["metric"])

        # Calls compile with the deserialized parameters
        self.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def custom_loss_function(self, *args):
        """custom loss function to be able to agregate different cost

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        # y_true, y_pred = args[0], args[1]
        return TKLoss.MeanSquaredError()(*args) + TKLoss.MeanAbsoluteError()(*args)
