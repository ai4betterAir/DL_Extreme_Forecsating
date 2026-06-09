import warnings

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import serialization_lib
from keras.src.utils.numerical_utils import normalize


class DoubleLossFunctionWrapper(Loss):
    def __init__(
        self, fn1, fn2, reduction="sum_over_batch_size", name=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.fn1 = fn1
        self.fn2 = fn2
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        return self.fn1(y_true, y_pred, **self._fn_kwargs) + self.fn2(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {"fn1": serialization_lib.serialize_keras_object(self.fn1),
                  "fn2": serialization_lib.serialize_keras_object(self.fn2),}
        config.update(serialization_lib.serialize_keras_object(self._fn_kwargs))
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "fn1" in config and "fn2" in config:
            config = serialization_lib.deserialize_keras_object(config)
        return cls(**config)


class CustomLosses(DoubleLossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error"
    ):
        super().__init__(mean_squared_error, mean_absolute_error, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred), axis=-1)
    ```

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_squared_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    return ops.mean(ops.square(y_true - y_pred), axis=-1)



def mean_absolute_error(y_true, y_pred):
    """Computes the mean absolute error between labels and predictions.

    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_absolute_error(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    return ops.mean(ops.abs(y_true - y_pred), axis=-1)


    