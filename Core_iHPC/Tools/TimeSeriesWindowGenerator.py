"""
..  module:: TimeSeriesWindowGenerator
    :platform: Unix
    :synopsis: Definition of an object to window time series for AI processing.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""
"""
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        # https://stackoverflow.com/questions/48491737/understanding-keras-lstms-role-of-batch-size-and-statefulness
        # https://keras.io/api/preprocessing/timeseries/
        # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
        # https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
        # https://github.com/anencore94/SlidingWindowGenerator/blob/master/slidingwindow_generator/slidingwindow_generator.py

"""


import matplotlib 
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import tensorflow as tf
###########################################################################################
class WindowGenerator():
    """
    This class is a copy from 
    https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing
    
    Vocabulary:
    timeserie
    [t0  t1  t2  t3  t4  ...     ti  ti+1    ...     tl ...  tn]
    t0 ... ti: Input width, or history lenght of the timeserie to make the prediction
    ti+1 ... tn: Offset width, or shift, or how many timesteps forward to reach the end of target for training
    tl ... tn: Label width,  or how many timesteps the prediction is to be, this is the target for training.
    t0 ... tn is the total width, total width = input width + Offset


    **
    The main features of the input windows are:
        - The width (number of time steps) of the input and label windows.
        - The time offset between them.
        - Which features are used as inputs, labels, or both.
    
    This tutorial builds a variety of models (including Linear, DNN, CNN and RNN models), and uses them for both:
        - Single-output, and multi-output predictions.
        - Single-time-step and multi-time-step predictions.
    This section focuses on implementing the data windowing so that it can be reused for all of those models.

    Depending on the task and type of model you may want to generate a variety of data windows. Here are some examples:
    For example, to make a single prediction 24 hours into the future, given 24 hours of history, you might define a window like this:

    One prediction 24 hours into the future.

    A model that makes a prediction one hour into the future, given six hours of history, would need a window like this:

    One prediction one hour into the future.

    The rest of this section defines a WindowGenerator class. This class can:

    Handle the indexes and offsets as shown in the diagrams above.
    Split windows of features into (features, labels) pairs.
    Plot the content of the resulting windows.
    Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.
    1. Indexes and offsets
    Start by creating the WindowGenerator class. The __init__ method includes all the necessary logic for the input and label indices.

    It also takes the training, evaluation, and test DataFrames as input. These will be converted to tf.data.Datasets of windows later.    
    """
    def __init__(self, input_width, label_width, shift,
                train_df=None, val_df=None, test_df=None,
                input_columns=None, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        self.input_columns = input_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in
                                            enumerate(input_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

###########################################################################################
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

###########################################################################################
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        if self.input_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # WindowGenerator.split_window = split_window        

###########################################################################################
    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        pl.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            pl.subplot(max_n, 1, n+1)
            pl.ylabel(f'{plot_col} [normed]')
            pl.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            pl.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                pl.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                pl.legend()

        pl.xlabel('Time [h]')

    # WindowGenerator.plot = plot

###########################################################################################
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    # WindowGenerator.make_dataset = make_dataset
###########################################################################################
    def make_list_indices_train(self,):
        """
        return list of list for the train pd of all the indices splits, so it can be aplied to retrieve original timedate of the data  
        """
        list_of_list_input_indices = []
        list_of_list_label_indices = []
        for ii in range(len(self.train_df) - self.total_window_size +1):
            list_input_indices = self.input_indices + ii
            list_label_indices = self.label_indices + ii
            list_of_list_input_indices.append(list_input_indices)
            list_of_list_label_indices.append(list_label_indices)
        return list_of_list_input_indices, list_of_list_label_indices
###########################################################################################
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    # WindowGenerator.train = train
    # WindowGenerator.val = val
    # WindowGenerator.test = test
    # WindowGenerator.example = example
###########################################################################################

class SlidingWindowGenerator:
    """
    This class is a copy from 
    https://github.com/anencore94/SlidingWindowGenerator

    Vocabulary:
    timeserie
    [t0  t1  t2  t3  t4  ...     ti  ti+1    ...     tl ...  tn]
    t0 ... ti: Input width, or history lenght of the timeseries to make the prediction
    ti+1 ... tn: Offset width, or shift, or how many timesteps forward to reach the end of target for training
    tl ... tn: Label width,  or how many timesteps the prediction is to be, this is the target for training.
    t0 ... tn is the total width, total width = input width + Offset
    **
    - based on tensorflow v2.3.0
        - use timeseries_dataset_from_array function which was introduced in tf v2.3.0
    - This module converts time series data from dataframe type to sliding window type
        - to use as input in RNN based layer
    - This module was based on tensorflow official docs, just aggregate some functions and add small tuning to use it more efficiently.
        - to make it possible to control batch_size, sequence_stride_size and shuffle more freely.
    """
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 input_columns=None, label_columns=None):
        # Store raw data with dataframe type
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift  # indicates the label offset far from input

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def train(self, sequence_stride=1, shuffle=True, batch_size=32):
        """
        make train time series dataset
        :param sequence_stride: int
        :param shuffle: boolean
        :param batch_size: int
        :return: time_series data
        """
        return self.make_dataset(self.train_df,
                                 sequence_stride=sequence_stride,
                                 shuffle=shuffle, batch_size=batch_size)

    def val(self, sequence_stride=1, shuffle=False, batch_size=32):
        """
        make validation time series dataset
        :param sequence_stride: int
        :param shuffle: boolean
        :param batch_size: int
        :return: time_series data
        """
        return self.make_dataset(self.val_df, sequence_stride=sequence_stride,
                                 shuffle=shuffle, batch_size=batch_size)

    def test(self, sequence_stride=1, shuffle=False, batch_size=32):
        """
        make test time series dataset
        :param sequence_stride: int
        :param shuffle: boolean
        :param batch_size: int
        :return: time_series data
        """
        return self.make_dataset(self.test_df, sequence_stride=sequence_stride,
                                 shuffle=shuffle, batch_size=batch_size)

    def example(self, sequence_stride=1, shuffle=True, batch_size=32):
        """
        Get and cache an example batch of `inputs, labels` for checking shape
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(
                self.train(sequence_stride=sequence_stride, shuffle=shuffle,
                           batch_size=batch_size)))
            # And cache it for next time
            self._example = result
        return result

    def split_window(self, features):
        """
        input 을 input 용 column 들로 이루어진 data 와
        label 용 column 들로 이루어진 data 로 분리
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, sequence_stride=1, shuffle=True,
                     batch_size=32):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            shuffle=shuffle,
            batch_size=batch_size, )

        ds = ds.map(self.split_window)

        return ds
    def make_list_indices_train(self,):
        """
        return list of list for the train pd of all the indices splits, so it can be aplied to retrieve original timedate of the data  
        """
        list_of_list_input_indices = []
        list_of_list_label_indices = []
        for ii in range(len(self.train_df) - self.total_window_size +1):
            list_input_indices = self.input_indices + ii
            list_label_indices = self.label_indices + ii
            list_of_list_input_indices.append(list_input_indices)
            list_of_list_label_indices.append(list_label_indices)
        return list_of_list_input_indices, list_of_list_label_indices