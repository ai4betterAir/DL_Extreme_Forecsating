from tensorflow.keras import layers

# Custom layer to hold learnable log variances
class AdaptiveLossLayer(layers.Layer):
    def __init__(self, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_outputs': self.num_outputs,
        })
        return config
        
    def build(self, input_shape):
        self.log_vars = self.add_weight(
            name='log_vars',
            shape=(self.num_outputs,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        # Pass through layer without modifying inputs
        return inputs
        
