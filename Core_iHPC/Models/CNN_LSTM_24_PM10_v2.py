"""
..  module:: CNN_LSTM_generic
    :platform: Unix
    :synopsis: Definition of the basic object class to configure and run CNN_LSTM model.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
   
"""

import sys
import os
import stat

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot  as plt
import datetime

import tensorflow.keras as TFK
import tensorflow.keras.layers as TKL 
import tensorflow.keras.models as TKM
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.callbacks as TKC
import tensorflow.keras.losses as TKLoss
import tensorflow.keras.metrics as TKMetrics
import tensorflow.keras.optimizers as TKO
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization
# import bayes_opt

# import tensorflow.keras.optimizers.experimental as TKOE
import sklearn.preprocessing as SKLP

# from ..Tools import Splitting as Split
# from ..Tools import TimeSeriesWindowGenerator as TSWG
# from ..Tools import TensorToPandas as TTP
from ..Tools import Denormalization_keras_bug_layer as DKBL
from ..Processing import Data_preparation
from ..Processing import Training_evaluations
from ..Outputs import Output_manager as OM


from . import CNN_LSTM_generic_v1 as generic

### Set ramdom seed at fix value to make fixed reproduced simulation
# tf.random.set_seed(1)
###########################################################################################
class CNN_LSTM_Class(generic.CNN_LSTM_generic_Class):
    """ 
    This class defines a CNN_LSTM_Class, that configure a CNN LSTM model, prepare its data, train the model and produce a forecast. 

    Attributes
    -----------
       
    """
    def __init__(self, Configuration,
                 ):

        """
        This method defines the trainin stopping criteria and train the model using the train and validation dataset.

        Parameters
        -----------
        Configuration : Configuration.Configuration object
            instance of a configuration object that holds all th parameter of the model.

        Attributes
        ----------        
        model_save_dir : str    
            directory where to save/load the model file.

        """
        super().__init__(Configuration)

        # set the global paramaters bounds
        self.global_optimisation_parameters_bounds = {
            "lstm_1_units" : (10, 250),
            "lstm_2_units" : (10, 250),
            "cnn_1_kernel_size": (1, self.Configuration.n_steps_in), 
            "cnn_1_filters": (1, self.Configuration.n_steps_in),
        }
        
        # self.Bayesian_global_opmimiser_params = {
        #     "init_points" : 7, 
        #     "n_iter" : 70,
        # }
        # test purpose
        self.Bayesian_global_opmimiser_params = {
            "init_points" : 2, 
            "n_iter" : 2,
        }


        return
###########################################################################################
    def define_model(self,
                    n_steps_in, 
                    n_steps_out, 
                    n_features_in, 
                    n_features_out, 
                    build_architecture,
                    ):
        """
        This is Hubert's v1 model
        Input -> CNN1 -> CNN2 -> LSTM1 -> Dropout -> LSTM2 -> Dense -> Output
        A static normalisation/denormalisation layer have been added to the data flow so the normalising coefficients are saved into the model, 
        and constant once the training is done, so every other run will be made using the same values.
        """                     
        self.model_save_filename_template = "cnn_lstm_model_v1_{vars}_{nstepin}_{nstepout}_{nepoch}.h5"

        if build_architecture:
            # #############
            # # for multi GPUs
            # # Create a MirroredStrategy.
            # strategy = tf.distribute.MirroredStrategy()
            # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            # #############

            #### 24-24 good --> with cycle pattern from hours and months are good for Ozone forecast
            # cnn_1_filters = 12
            # cnn_1_kernel_size = 3
            # cnn_2_filters = 6
            # cnn_2_kernel_size = 3
            # lstm_1_units= 128

            #################################################
            lstm_1_units = int(np.rint(self.Configuration.model_parameters_dict["lstm_1_units"]))
            cnn_1_filters = int(np.rint(self.Configuration.model_parameters_dict["cnn_1_filters"]))
            cnn_1_kernel_size = int(np.rint(self.Configuration.model_parameters_dict["cnn_1_kernel_size"]))
            lstm_2_units = int(np.rint(self.Configuration.model_parameters_dict["lstm_2_units"]))

            print("*"*100)
            print("model configuration")
            print("cnn_1_kernel_size: {}".format(cnn_1_kernel_size))
            print("cnn_1_filters: {}".format(cnn_1_filters))
            print("lstm_1_units: {}".format(lstm_1_units))
            print("*"*100)

            # lstm_2_units= 64





            dropout_rate = 0.2
            padding = 'same'
            activation = 'relu'

          
            # self.model =  TKM.Sequential()
            # self.model.add(TKL.Conv1D(filters=cnn_1_filters, kernel_size=cnn_1_kernel_size, padding = padding, activation='relu', input_shape=(n_steps_in, n_features_in)))
            # self.model.add(BatchNormalization())            
            # self.model.add(TKL.Flatten())
            # self.model.add(TKL.RepeatVector(n_steps_out))
            # ## LSTM is a decoder
            # self.model.add(TKL.Dropout(dropout_rate))
            # self.model.add(TKL.LSTM(lstm_1_units, activation=activation, return_sequences=True, bias_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.005)))
            # self.model.add(BatchNormalization())

            input_layer =  TKL.Input(
                shape=(n_steps_in, n_features_in),
                )
            
            normaliser = self.normalisation_layer(input_layer)
            
            conv = TKL.Conv1D(
                filters=cnn_1_filters, 
                kernel_size=cnn_1_kernel_size, 
                padding = padding, 
                activation='relu', 
                # input_shape=(n_steps_in, n_features_in),
                name = "conv1D",
                )(normaliser)
            
            flatten = TKL.Flatten()(conv)
            
            repeat = TKL.RepeatVector(n_steps_out)(flatten)
            
            dropout = TKL.Dropout(dropout_rate)(repeat)
            
            lstm = TKL.LSTM(
                lstm_1_units, 
                activation=activation, 
                return_sequences=True, 
                bias_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.005),
                )(dropout)
            lstm = TKL.LSTM(
                lstm_2_units, 
                activation=activation, 
                return_sequences=True, 
                bias_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.005),
                )(lstm)
            output_layer = TKL.Dense(n_features_out, activation='relu')(lstm)
            
            self.model = TKM.Model([input_layer], [output_layer], name=self.KERAS_NORMALISATION_MODEL_NAME)

            # #############
            # # for multi GPUs
            # # Open a strategy scope.
            # with strategy.scope():

            
                
            loss_function = self.custom_loss_function
            # loss_function = TKLoss.MeanSquaredError()
            # loss_function = TKLoss.RootMeanSquaredError()

            metrics_list=[
                TKMetrics.MeanAbsoluteError(),
                TKMetrics.RootMeanSquaredError(),
                ]
            
            # optimizer = TKO.SGD()
            optimizer = TKO.Adam()
            
            self.model.compile(optimizer=optimizer, 
                                loss=loss_function,
                                metrics = metrics_list,
                                run_eagerly=False)

        # print(self.model.summary())
        
        
        self.logger.info('CNN LSTM 24 PM10 model v1'.ljust(self.justif-10,'.') + 'CONFIGURED')
        return
###########################################################################################
