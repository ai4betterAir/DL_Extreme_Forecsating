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
import re

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
from tensorflow.keras.layers import BatchNormalization
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
from ..Processing.decomposition import decomp_joint_preparation as VJPrep
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
    def Change_permissions(self, path):
        ''' This function makes the different working directories
        '''
        import stat
        import os
        os.umask(0)
        mod664 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IROTH  
        mod666 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IROTH |stat.S_IWOTH 
        # os.chmod(path, mod664)
        os.chmod(path, mod666)
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

            if self.Configuration.model_parameters_dict["lstm_1_units"] is not None:
                lstm_1_units = int(self.Configuration.model_parameters_dict["lstm_1_units"])
            else:
                lstm_1_units=  128

            if self.Configuration.model_parameters_dict["lstm_2_units"] is not None:
                lstm_2_units = int(self.Configuration.model_parameters_dict["lstm_2_units"])
            else:
                lstm_2_units= 64
                
            dropout_rate = 0.4
            activation = 'relu'
            activation_lstm = 'tanh'

            # Define the model LSTM using good for PM2.5
            self.model = TKM.Sequential()
            self.model.add(TKL.Dropout(dropout_rate))
            self.model.add(TKL.LSTM(lstm_1_units, return_sequences=True, activation=activation_lstm, input_shape=(n_steps_in, n_features_in)))
            self.model.add(BatchNormalization())
            self.model.add(TKL.Dropout(dropout_rate))
            self.model.add(TKL.LSTM(lstm_2_units, activation=activation_lstm, return_sequences=True))
            self.model.add(BatchNormalization())
            self.model.add((TKL.Dense(n_features_out*n_steps_out, activation=activation)))
            self.model.add(TKL.Dropout(dropout_rate))
            self.model.add((TKL.Dense(n_features_out, activation=activation)))
            # self.model.add(BatchNormalization())
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
    def _station_frame_has_vmd_columns(self, station_df):
        if not isinstance(station_df, pd.DataFrame):
            return False
        for column in station_df.columns:
            if str(column).strip().upper().startswith("IMF_") or re.match(r"^IMF[_\s-]*\d+$", str(column), re.IGNORECASE):
                return True
        return False

###########################################################################################
    def _pick_station_value_column(self, station, station_df):
        if not isinstance(station_df, pd.DataFrame) or station_df.empty:
            raise ValueError(f"No station data available for {station}")

        preferred_columns = ["Residual", "Original", "Reconstructed"]
        for column in preferred_columns:
            if column in station_df.columns:
                return column

        configured_columns = list(getattr(self.Configuration, "input_column_names", []) or [])
        station_key = str(station).strip().upper()
        for candidate in configured_columns:
            candidate_str = str(candidate)
            candidate_key = candidate_str.strip().upper()
            if candidate_key == station_key or candidate_key.endswith(f"_{station_key}"):
                if candidate in station_df.columns:
                    return candidate

        for column in station_df.columns:
            column_key = str(column).strip().upper()
            if column_key == station_key or column_key.endswith(f"_{station_key}"):
                return column

        numeric_columns = [
            column for column in station_df.columns
            if pd.api.types.is_numeric_dtype(station_df[column])
        ]
        if numeric_columns:
            return numeric_columns[0]

        raise ValueError(
            f"Unable to identify a usable series for station '{station}'. "
            f"Available columns: {list(station_df.columns)}"
        )

###########################################################################################
    def _resolve_station_column_name(self, station, fallback_column):
        configured_columns = list(getattr(self.Configuration, "input_column_names", []) or [])
        station_key = str(station).strip().upper()
        for candidate in configured_columns:
            candidate_str = str(candidate)
            candidate_key = candidate_str.strip().upper()
            if candidate_key == station_key or candidate_key.endswith(f"_{station_key}"):
                return candidate
        return fallback_column

###########################################################################################
    def _normalize_input_frame(self, input_data_pd):
        if isinstance(input_data_pd, pd.DataFrame):
            return input_data_pd.copy()
        if not isinstance(input_data_pd, dict):
            raise TypeError(f"Expected DataFrame or dict, got {type(input_data_pd)}")
        if not input_data_pd:
            raise ValueError("Empty station dictionary received")

        station_dict = input_data_pd
        if any(self._station_frame_has_vmd_columns(station_df) for station_df in station_dict.values()):
            station_dict = VJPrep.collapse_joint_components(
                station_dict,
                logger=self.logger,
                justif=self.justif,
            )

        series_list = []
        for station, station_df in station_dict.items():
            if station_df is None or station_df.empty:
                continue
            value_column = self._pick_station_value_column(station, station_df)
            series = pd.to_numeric(station_df[value_column], errors="coerce").copy()
            series.name = self._resolve_station_column_name(station, value_column)
            series_list.append(series)

        if not series_list:
            raise ValueError("No usable station series found in input dictionary")

        normalized_frame = pd.concat(series_list, axis=1)
        normalized_frame.index = pd.to_datetime(normalized_frame.index, errors="coerce")
        normalized_frame = normalized_frame[~normalized_frame.index.isna()].sort_index()
        return normalized_frame




###########################################################################################
    def run_all(self, input_data_pd, iterative ):
        """
        run all forward with iterative retrain model to reduce data drift
        """
        input_data_pd = self._normalize_input_frame(input_data_pd)
        print("Run all - input_data_pd.shape: ", input_data_pd.shape)
        
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
        # self.other_var_not_in_data_input_to_keep = ["day_of_year", "day_of_week", "month", "season_number"]
        # self.other_var_not_in_data_input_to_keep = ["season_number", "hour"]
        
        self.other_var_not_in_data_input_to_keep = []

        # train
        train_model = self.Configuration.train_model
        # save model weigths
        save_model = self.Configuration.save_model

        # input_columns = sorted(self.Configuration.input_column_names + self.other_var_not_in_data_input_to_keep)
        input_columns = sorted(self.Configuration.input_column_names)
        output_columns = sorted(self.Configuration.output_column_names)


        ########################## Training ###########################
        history = None
        stop_rolling = False

        Data_prep_class = Data_preparation.Data_Preparation_Class(self.Configuration)

        if train_model:
            ############
            ### Prepare whole data for training - validating - testing
            print("_____________________ROLLING #{}_______________________".format(iterative+1))
            if stop_rolling == False:
                
                dict_split_final_datasets, dict_split_final_pd, stop_rolling = Data_prep_class.prepare_data_training(input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, batch_size, iterative)
                #### Update the input_columns
                input_columns = self.Configuration.input_column_names

                ### Copy previous dataset for update new training data after each rolling
                prev_dict_split_final_datasets, prev_dict_split_final_pd = dict_split_final_datasets, dict_split_final_pd
                print("Stop_rolling =  ", stop_rolling)
                print(dict_split_final_datasets)
            else: 
                dict_split_final_datasets, dict_split_final_pd = prev_dict_split_final_datasets, prev_dict_split_final_pd
                print(dict_split_final_datasets)
                
            
            ### Configure normaised layer to normalise training set, then normalise the testing set
            # self.configure_normalisation_layer(dict_split_final_datasets["train"], input_columns, output_columns)
            
            ### Configure model
            self.configure_model(n_steps_in, n_steps_out, n_epochs, input_columns, output_columns,train_model)
            ### Fitting model with training set and validation set
  
            history = self.fit(train_ds=dict_split_final_datasets["train"], validation_ds=dict_split_final_datasets["validation"], n_epochs=n_epochs,  
                                save_model = save_model, batch_size=batch_size)
            
            
            ### Select the set to be used for forecasting
            # which_set = "test"
            which_set = "validation"
            # which_set = "train"
            
      
            list_output_pd = Data_prep_class.evaluate_data_BNN(dict_split_final_datasets[which_set], output_columns, n_steps_in, n_steps_out, 
                                    dict_split_final_pd["input_{set}".format(set=which_set)], dict_split_final_pd["label_{set}".format(set=which_set)], self.model)


            ################################################################################
            print("Training is DONE")

            
        ###############################################################
        # run the prediction model
        else:
            # print(n_steps_in)
            # print(self.Configuration.num_batches)
            # sads
            # dasdsa
            #### Prepare data for training again to synchronize the input_data_pd and input columns of training again
            # dict_split_final_datasets, dict_split_final_pd, stop_rolling = Data_prep_class.prepare_data_training(input_data_pd, [n_steps_in], [n_steps_out], input_columns, output_columns, batch_size, iterative)
            dict_split_final_datasets, dict_split_final_pd, stop_rolling = Data_prep_class.prepare_data_training(input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, batch_size, iterative)

            #### Update the input_columns
            input_columns = self.Configuration.input_column_names
                       
            #### Prepare data for forecasting
     
            datetime_filtered_input_data_pd, datetime_filtered_input_data_ds = Data_prep_class.prepare_data_forecast(
                            input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, forecast_datetime_aest)

            self.configure_model(n_steps_in, n_steps_out, n_epochs, input_columns, output_columns, train_model)

            ### making forecast df from step 1 - n_outputs:
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
