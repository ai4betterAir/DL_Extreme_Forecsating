"""
..  module:: LSTM_BNN
    :platform: Unix
    :synopsis: Definition of the basic object class to configure and run LSTM_BNN model.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
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
from tensorflow.keras.layers import BatchNormalization

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.callbacks as TKC
import tensorflow.keras.losses as TKLoss
import tensorflow.keras.metrics as TKMetrics
import tensorflow.keras.optimizers as TKO
from keras import regularizers
# import tensorflow.keras.optimizers.experimental as TKOE
import sklearn.preprocessing as SKLP

from ..Tools import Splitting as Split
from ..Tools import TimeSeriesWindowGenerator as TSWG
from ..Tools import TensorToPandas as TTP
from ..Tools import Denormalization_keras_bug_layer as DKBL
from ..Tools import Add_vars as AV
from ..Evaluation import Plots 
from ..Processing import Data_preparation
from ..Configuration import DPE_region_stations as CL_DPE_regions

# from .. import Calib_temp

### Set ramdom seed at fix value to make fixed reproduced simulation
tf.random.set_seed(1)
###########################################################################################
class CNN_LSTM_Class(object):
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
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Configuring the Model'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        self.model_save_dir = self.Configuration.Main_model_data_full_dir

        self.all_mean = 0
        self.all_variance = 0
        self.save_model_name = None

        return
###########################################################################################
    def MakeDir(self, ddir):
        ''' This function makes the different working directories
        '''
        if not os.path.exists(ddir):
            os.makedirs(ddir)
            mod775 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IXUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IXGRP |stat.S_IROTH |stat.S_IXOTH
            os.chmod(ddir,mod775)
            #os.chmod(self.RunDir,0775)
            self.logger.info('Directory = {msg}'.format(msg=ddir).ljust(self.justif-7,'.') + 'CREATED')
        return    
###########################################################################################
    def lstm_bnn_model (self,
                             n_steps_in, n_steps_out, n_features_in, n_features_out, build_architecture):
        """
        This is Hubert's LSTM-BNN used in IEEE-ACCESS paper
        A static normalisation/denormalisation layer have been added to the data flow so the normalising coefficients are saved into the model, 
        and constant once the training is done, so every other run will be made using the same values.
        """                     

        self.model_save_filename_template = "lstm_bnn_model_{vars}_{nstepin}_{nstepout}_{nepoch}.h5"

        if build_architecture:

            lstm_1_units=  128
            # dropout_rate = 0.2
            lstm_2_units= 64
            dropout_rate = 0.4
            activation = 'relu'

            # Define the model LSTM using good for PM2.5
            self.model = TKM.Sequential()
            self.model.add(TKL.Dropout(dropout_rate))
            self.model.add(TKL.LSTM(lstm_1_units, return_sequences=True, activation=activation, input_shape=(n_steps_in, n_features_in)))
            self.model.add(BatchNormalization())
           
            self.model.add(TKL.Dropout(dropout_rate))
            self.model.add(TKL.LSTM(lstm_2_units, activation=activation, return_sequences=True))
            self.model.add(BatchNormalization())

            self.model.add(TKL.Dropout(dropout_rate))
            self.model.add((TKL.Dense(n_features_out, activation=activation)))

            loss_function = TKLoss.MeanSquaredError()
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
        self.logger.info('LSTM-BNN model v1'.ljust(self.justif-10,'.') + 'CONFIGURED')
        return
###########################################################################################
    def fit(self,
            train_ds=None, validation_ds=None, n_epochs=None,  
            save_model=None, batch_size=None):
        """
        This method defines the trainin stopping criteria and train the model using the train and validation dataset.

        Parameters
        -----------
        train_ds : tensorflow.Dataset
            Dataset instance containing the already prepared and split training data.
        validation_ds : tensorflow.Dataset
            Dataset instance containing the already prepared and split validation data.
        n_epochs : int
            Number maximum of iteration for the training.
        save_model : boolean
            Switch to trigger the model saving when training is finished.

        """
        ### simple early stopping
        es = TKC.EarlyStopping(monitor='loss', mode='min', verbose=1, patience = 10)
        # es = TKC.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 15)

        
        ### fit model
        history = self.model.fit(train_ds, validation_data=validation_ds, epochs=n_epochs, callbacks=[es], verbose=2, batch_size=batch_size)

        if save_model:
            file = os.path.join(self.model_save_dir, self.model_filename)
            self.MakeDir(self.model_save_dir)
            self.model.save(file)
            self.logger.info(self.Configuration.forecast_method + ' save file {file}'.format(file=file).ljust(self.justif-5,'.') + 'SAVED')

        self.logger.info(self.Configuration.forecast_method + ' training'.ljust(self.justif-8,'.') + 'COMPLETE')

        return history

    



###########################################################################################
    def configure_model(self, n_steps_in, n_steps_out, n_epochs, input_columns, output_columns, train_model ):
        """
        configure the model,
        if the option of train_model is true, then it calls the model definition and compiles it. 
        If false, then it just load the already computed model and weights from the file
        """
        
        
        # configure model
        n_features_in = len(input_columns)
        n_features_out = len(output_columns)
        if train_model:
            build_architecture = True
            self.lstm_bnn_model(n_steps_in, n_steps_out, n_features_in, n_features_out, build_architecture)   

            self.model_filename = self.Configuration.Name_convention.model_file_name()

        else:
            # restore the trained model    
            build_architecture = False
            self.lstm_bnn_model(n_steps_in, n_steps_out, n_features_in, n_features_out, build_architecture)   

            self.model_filename = self.Configuration.Name_convention.model_file_name()

            file = os.path.join(self.model_save_dir, self.model_filename)
    
            from packaging import version
            if (version.parse(TFK.__version__) <= version.parse("2.10")):
                self.model = TKM.load_model(file , 
                        custom_objects={"Denormalization_keras_bug_layer": DKBL.Denormalization_keras_bug_layer}
                        )
            else:
                self.model = TKM.load_model(file)
            self.logger.info('CNN-LSTM_BNN model save file {file}'.format(file=file).ljust(self.justif-6,'.') + 'LOADED')
        
        self.logger.info('Model configuration'.ljust(self.justif-2,'.') + 'OK')
        return 

###########################################################################################
    def configure_normalisation_layer(self, data_ds, input_columns, output_columns):
        """
        Reshape and inverse scaling for input data
        https://github.com/keras-team/keras/blob/master/keras/layers/normalization/batch_normalization.py
        """ 
        # print(len(list(data_ds)))
        self.normalisation_layer = TKL.Normalization()
        # for batch in data_ds:
        #     # print(len(batch[0]))
        #     # print(len(batch[1]))
        #     # print(len(batch[0]))
        #     self.normalisation_layer.adapt(batch[0])
        #     # self.normalisation_layer.adapt(batch)

        # ##### Update mean and variance of all total data for normalisation
        # N_mean = self.normalisation_layer.mean 
        # N_variance = self.normalisation_layer.variance
       
        N_mean = self.all_mean
        N_variance = self.all_variance
        
        print("N_mean.shape")
        print(N_mean.shape)
        print(N_variance.shape)
        print(self.all_mean.shape)
        print(self.all_variance.shape)
        
        
        mask_output_from_input_columns = np.isin( input_columns, output_columns, assume_unique=True)
        print(mask_output_from_input_columns)
        print("mean =", N_mean)
        print("var =", N_variance)

        N_mean_output = tf.boolean_mask(N_mean, mask_output_from_input_columns, axis=2)
        N_variance_output = tf.boolean_mask(N_variance, mask_output_from_input_columns, axis=2)
        # print("test=", test)
        
        from packaging import version
        if (version.parse(TFK.__version__) <= version.parse("2.10")):
            from keras import backend
            self.logger.warning("There is a bug in inverse normalisation in Keras 2.10, Custom layer added instead")
            self.denormalisation_layer = DKBL.Denormalization_keras_bug_layer(N_mean_output, N_variance_output)
        else:
            self.denormalisation_layer = TKL.Normalization(
                mean=N_mean_output, variance=N_variance_output, invert=True)

        self.logger.info('Normalisation and inverse layer'.ljust(self.justif-10,'.') + 'CONFIGURED')
        return 




###########################################################################################
    def run_all(self, input_data_pd, iterative ):
        """
        run all forward with iterative retrain model to reduce data drift
        """
        print(input_data_pd.shape)
        
        ###############################################################
        # Grab the configuration

        n_steps_in = self.Configuration.n_steps_in
        n_steps_out = self.Configuration.n_steps_out
        n_epochs = self.Configuration.n_epochs
        
        forecast_datetime_utc = self.Configuration.start_date_utc
        # forecast_datetime_aest = self.Configuration.start_date_aest
        forecast_datetime_aest = input_data_pd.index[-1]
        print(forecast_datetime_aest)
        
        forecast_datetime_aedt = self.Configuration.start_date_aedt
        batch_size = self.Configuration.batch_size
        
        self.var_to_predict = self.Configuration.var_to_predict
        self.additional_var_to_select = self.Configuration.additional_var_to_select
        self.var_data_list_from_input_pd = self.Configuration.var_data_list_from_input_pd
        # self.other_var_not_in_data_input_to_keep = ["day_of_year", "day_of_week", "month", "season_number", "hour"]
        self.other_var_not_in_data_input_to_keep = ["day_of_year", "day_of_week", "month", "season_number"]

        # self.other_var_not_in_data_input_to_keep = ["season_number", "hour"]
        self.other_var_not_in_data_input_to_keep = []

        # train
        train_model = self.Configuration.train_model
        # save model weigths
        save_model = self.Configuration.save_model

        # print(input_data_pd.columns)

        input_columns = sorted(self.Configuration.input_column_names + self.other_var_not_in_data_input_to_keep)
        output_columns = sorted(self.Configuration.output_column_names)

        ###############################################################
        # debug
        # tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()
        ###############################################################
        # Training
        history = None
        stop_rolling = False

        Data_prep_class = Data_preparation.Data_Preparation_Class(self.Configuration)
        Region_class = CL_DPE_regions.DPE_region_stations()
        stations = Region_class.DPE_region_stations_dict[self.Configuration.selected_region]


        if train_model:
            ############
            
            ### Prepare whole data for training - validating - testing

            # print(input_data_pd.head())
            # print(input_data_pd.columns)

            list_output_pd = pd.DataFrame(columns=['dummy'])
            # print(len(list_output_pd.columns))
            # dsa
            

            for station in stations:
                  target_cols = input_data_pd.columns[input_data_pd.columns.str.contains(station)]
                  self.logger.info('TRAINING AND EVALUATING FOR ' + station + '_'.ljust(self.justif-2,'.') + 'OK')
                  print(target_cols)
                  
                  input_data_pd_station = input_data_pd[target_cols]
                  input_columns = input_data_pd_station.columns
                  output_columns = input_data_pd_station.columns[input_data_pd_station.columns.str.contains('OZONE')]

                  # print(input_columns)
                  # print(output_columns)
       
                  print("_____________________ROLLING #{}_______________________".format(iterative+1))
                  if stop_rolling == False:
                        dict_split_final_datasets, dict_split_final_pd, stop_rolling = Data_prep_class.prepare_data_training(input_data_pd_station, n_steps_in, n_steps_out, input_columns, output_columns, batch_size, iterative)
                         #### Update the input_columns
                        input_columns = self.Configuration.input_column_names

                        
                        prev_dict_split_final_datasets, prev_dict_split_final_pd = dict_split_final_datasets, dict_split_final_pd
                        print("Stop_rolling =  ", stop_rolling)
                        print(dict_split_final_datasets)
                  else: 
                        dict_split_final_datasets, dict_split_final_pd = prev_dict_split_final_datasets, prev_dict_split_final_pd
                        print(dict_split_final_datasets)
                  

                  ### Configure normaised layer to normalise training set, then normalise the testing set
                  # self.configure_normalisation_layer(dict_split_final_datasets["train"], input_columns, output_columns)
                  
                  ### Configure model
                  self.configure_model(n_steps_in, n_steps_out, n_epochs, input_columns, output_columns, train_model)
                  ### Fitting model with training set and validation set

                  history = self.fit(train_ds=dict_split_final_datasets["train"], validation_ds=dict_split_final_datasets["validation"], n_epochs=n_epochs,  
                  save_model = save_model, batch_size=batch_size)
                  
                  
                  
                  ### Select the set to be used for forecasting
                  # which_set = "test"
                  which_set = "validation"
                  # which_set = "train"
                              
                  list_output_pd_station = Data_prep_class.evaluate_data_BNN_single_station(dict_split_final_datasets[which_set], output_columns, n_steps_in, n_steps_out, 
                                          dict_split_final_pd["input_{set}".format(set=which_set)], dict_split_final_pd["label_{set}".format(set=which_set)], self.model, station)
                  
                  #### append the dataframe of each station to a big dataframe for saving to the file
                  if len(list_output_pd.columns) < len(list_output_pd_station.columns):
                        list_output_pd = list_output_pd_station
                        forecast_hours = list_output_pd['forecast_hours']
                        forecast_number = list_output_pd['forecast_number']
                  else:
                      list_output_pd.pop('forecast_hours')
                      list_output_pd.pop('forecast_number')
                      list_output_pd = pd.concat([list_output_pd, list_output_pd_station], axis=1)

                #   self.Configuration.forecast_method = self.Configuration.forecast_method + '_' + station
                
                  self.save_model_name = self.Configuration.forecast_method + '_' + station
                  ################################################################################
                  print("Training for {} station is DONE".format(station))
      
                  print("-----------------> <------------------")
                  print(list_output_pd.columns)
                  print("-----------------> <------------------")

            
            ### remove irrelevant columns 
            # Filter columns with the substring 'OZONE'
            ozone_columns = [col for col in list_output_pd.columns if 'OZONE' in col]
            print(ozone_columns)
            list_output_pd = list_output_pd[ozone_columns]
            list_output_pd = list_output_pd.reindex(sorted(list_output_pd.columns), axis=1)
            list_output_pd['forecast_hours'] = forecast_hours
            list_output_pd['forecast_number'] = forecast_number
            print(list_output_pd.columns)

            list_output_pd = [list_output_pd]
        ###############################################################
        # run the prediction model
        else:
             #### Prepare data for training again to synchronize the input_data_pd and input columns of training again
            dict_split_final_datasets, dict_split_final_pd, stop_rolling = Data_prep_class.prepare_data_training(input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, batch_size, iterative)
            #### Update the input_columns
            input_columns = self.Configuration.input_column_names
            #### Prepare data for forecasting
     
            datetime_filtered_input_data_pd, datetime_filtered_input_data_ds = Data_prep_class.prepare_data_forecast(
                input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, forecast_datetime_aest)

            self.configure_model(n_steps_in, n_steps_out, n_epochs, input_columns, output_columns, train_model )

            # forecast_pd = Data_prep_class.make_forecast(datetime_filtered_input_data_ds, output_columns, datetime_filtered_input_data_pd, self.model)
            forecast_pd = Data_prep_class.make_forecast_BNN(datetime_filtered_input_data_ds, output_columns, datetime_filtered_input_data_pd, self.model)

            datetime_filtered_input_data_pd = datetime_filtered_input_data_pd[output_columns]
            
            full_forecast_pd = pd.concat([datetime_filtered_input_data_pd,forecast_pd], axis = 0)
        
            # Define the range of rows from -23 to 0 (inclusive)
            fill_values = list(range(-(n_steps_in-1), 1))
            print(fill_values)
            # Define the range of rows from -n_steps_in to 0 (inclusive)
            # Fill the previous rows with the values from -n_steps_in to 0
            full_forecast_pd['forecast_hours'].iloc[:(n_steps_in)] = fill_values

            list_output_pd = [full_forecast_pd]
        
        return list_output_pd,  stop_rolling, history                    