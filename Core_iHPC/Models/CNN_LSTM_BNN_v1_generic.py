"""
..  module:: CNN_LSTM_BNN_generic
    :platform: Unix
    :synopsis: Definition of the basic object class to configure and run CNN_LSTM model.

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
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.callbacks as TKC
import tensorflow.keras.losses as TKLoss
import tensorflow.keras.metrics as TKMetrics
import tensorflow.keras.optimizers as TKO
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization
import bayes_opt

# import tensorflow.keras.optimizers.experimental as TKOE
import sklearn.preprocessing as SKLP

# from ..Tools import Splitting as Split
# from ..Tools import TimeSeriesWindowGenerator as TSWG
# from ..Tools import TensorToPandas as TTP
from ..Tools import Denormalization_keras_bug_layer as DKBL
from ..Tools import Add_vars as AV
from ..Evaluation import Plots 
from ..Processing import Data_preparation
from ..Processing import Training_evaluations
from ..Outputs import Output_manager as OM

# from .. import Calib_temp

### Set ramdom seed at fix value to make fixed reproduced simulation
# tf.random.set_seed(1)
###########################################################################################
class CNN_LSTM_BNN_generic_Class(object):
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
        self.global_optimisation_parameters_bounds = None
        self.Bayesian_global_opmimiser_params = None
        self.all_mean = 0
        self.all_variance = 0
        self.KERAS_NORMALISATION_MODEL_NAME = "model"
        self.KERAS_NORMALISATION_LAYER_NAME = "normalisation_layer"

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
        es = TKC.EarlyStopping(
            monitor='loss', 
            mode='min', 
            verbose=1, 
            patience = 50,
            restore_best_weights=True,
            )
        
        ### fit model
        history = self.model.fit(
            train_ds, 
            validation_data  = validation_ds, 
            epochs = n_epochs, 
            callbacks = [es], 
            verbose = 2, 
            batch_size = batch_size,
            )

        if save_model:
            file = os.path.join(self.model_save_dir, self.model_filename)
            self.MakeDir(self.model_save_dir)
            
            self.model.save(file)
            self.logger.info('CNN-LSTM-BNN model save file {file}'.format(file=file).ljust(self.justif-5,'.') + 'SAVED')

        self.logger.info('CNN-LSTM-BNN model training'.ljust(self.justif-8,'.') + 'COMPLETE')

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
            self.define_model(n_steps_in, n_steps_out, n_features_in, n_features_out, build_architecture)   
            self.model_filename = self.Configuration.Name_convention.model_file_name()

        else:
            # restore the trained model    
            build_architecture = False
            self.define_model(n_steps_in, n_steps_out, n_features_in, n_features_out, build_architecture)   

            self.model_filename = self.Configuration.Name_convention.model_file_name()

            file = os.path.join(self.model_save_dir, self.model_filename)

            self.model = TKM.load_model(file)

            self.logger.info('CNN-LSTM_BNN model save file {file}'.format(file=file).ljust(self.justif-6,'.') + 'LOADED')
        
        self.logger.info('Model configuration'.ljust(self.justif-2,'.') + 'OK')
        return 

###########################################################################################
    def configure_normalisation_layer_new(self, dict_split_final_datasets, ):
        """
        Reshape and inverse scaling for input data

        """ 
        self.normalisation_layer = TKL.Normalization(
            name=self.KERAS_NORMALISATION_LAYER_NAME, 
            axis=-1)
        
        # compute mean and var of training dataset
        feature_ds = dict_split_final_datasets["train"].map(lambda x, y: x)
        self.normalisation_layer.adapt(feature_ds)
        


        self.logger.info('Normalisation and inverse layer'.ljust(self.justif-10,'.') + 'CONFIGURED')
        return 
###########################################################################################
    def global_optimise(self,
                    n_steps_in, 
                    n_steps_out, 
                    n_epochs, 
                    input_columns, 
                    output_columns,
                    train_model,
                    dict_split_final_datasets,
                    ):

        # # set the global paramaters bounds
        # parameters_bounds = {
        #     "lstm_1_units" : (10, 250),
        #     "cnn_1_kernel_size": (1, n_steps_in), 
        #     "cnn_1_filters": (1, n_steps_in),
        # }
        
        # create global variable to pass training data
        self.dict_split_final_datasets = dict_split_final_datasets

        # Create a BayesianOptimization optimizer and optimize the cost function
        optimizer = bayes_opt.BayesianOptimization(
            f = self.cost_function, 
            pbounds=self.global_optimisation_parameters_bounds, 
            verbose=2,
            )
        optimizer.maximize(
            init_points = self.Bayesian_global_opmimiser_params["init_points"], 
            n_iter = self.Bayesian_global_opmimiser_params["n_iter"],
            )

        optimal_evaluation = - optimizer.max["target"]

        # print("target = ",optimizer.space.target)
        # print("params = ",dict(zip(optimizer.space.keys, optimizer.space.params)))
        # optimisation_history_dict = dict(zip(optimizer.space.keys, optimizer.space.params))
        # optimisation_history_dict.update({"cost_function":optimizer.space.target})
        # print("optim history = ",optimisation_history_dict)
        
        update_dict = {key : int(np.rint(optimizer.max["params"][key])) for key in list(optimizer.space.keys)}
        # print(update_dict)
        self.Configuration.model_parameters_dict.update(update_dict)
        
        optimisation_history_dict = optimizer.space.res()
        target_order_np = optimizer.space.target
        # print("res = ", optimizer.space.res())
        
        # write the optimal file
        Output_Manager = OM.Output_manager_Class(self.Configuration)
        Output_Manager.output_optimal_yaml_config_file(optimisation_history_dict, target_order_np)
        self.logger.info('Bayesian hyperparameter optimisation'.ljust(self.justif-4,'.') + 'DONE')
        return

        
        
###########################################################################################
    def cost_function(self,
                    # lstm_1_units,
                    # cnn_1_kernel_size, 
                    # cnn_1_filters,
                    **dict_params
                    ):
        
        
        print(dict_params)

        update_dict = {key : int(val) for key,val in dict_params.items()}
        # print(update_dict)
        self.Configuration.model_parameters_dict.update(update_dict)
        
        n_steps_in = self.Configuration.n_steps_in
        n_steps_out = self.Configuration.n_steps_out
        n_epochs = self.Configuration.n_epochs
        input_columns = self.Configuration.input_column_names
        output_columns = self.Configuration.output_column_names
        train_model = self.Configuration.train_model
            
        ### Configure model
        self.configure_model(
            n_steps_in, 
            n_steps_out, 
            n_epochs, 
            input_columns, 
            output_columns,
            train_model
            )

        ### Fitting model with training set and validation set
        
        model_history = self.fit(
            train_ds = self.dict_split_final_datasets["train"], 
            validation_ds = self.dict_split_final_datasets["validation"], 
            n_epochs = n_epochs,  
            save_model = False, 
            batch_size=self.Configuration.batch_size
            )
        
        model_evaluation = self.model.evaluate(
            self.dict_split_final_datasets["validation"], 
            verbose = 0,
            )
        
        # print(model_evaluation)
        return  - model_evaluation[-1]
        
        

###########################################################################################
    def run_all(self, input_data_pd, iterative ):
        """
        run all forward with iterative retrain model to reduce data drift
        """
        # print(input_data_pd.shape)
        
        ###############################################################
        # Grab the configuration

        n_steps_in = self.Configuration.n_steps_in
        n_steps_out = self.Configuration.n_steps_out
        n_epochs = self.Configuration.n_epochs
        
        forecast_datetime_utc = self.Configuration.start_date_utc
        forecast_datetime_aest = input_data_pd.index[-1]
        # print(forecast_datetime_aest)
        
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

        #######################################################################################
        # Training
        #######################################################################################
        history = None
        stop_rolling = False

        Data_prep_class = Data_preparation.Data_Preparation_Class(self.Configuration)

        if train_model:
            ############
 
            #######################################################################################
            ## prepare the datasets
            #######################################################################################
            dict_split_final_datasets, dict_split_final_pd = Data_prep_class.prepare_data_training(
                input_data_pd, 
                n_steps_in, 
                n_steps_out, 
                input_columns, 
                output_columns, 
                batch_size, 
                iterative
                )
            #### Update the input_columns
            input_columns = self.Configuration.input_column_names
                
            #######################################################################################
            ## normalise the datasets
            #######################################################################################
            self.configure_normalisation_layer_new(dict_split_final_datasets, )    
            
            #######################################################################################
            ## global hyper-parameters optimisation
            #######################################################################################
            model_parameters_optimisation = True
            
            if model_parameters_optimisation:
                self.global_optimise(
                    n_steps_in, 
                    n_steps_out, 
                    n_epochs, 
                    input_columns, 
                    output_columns,
                    train_model,
                    dict_split_final_datasets,
                    )


            #######################################################################################
            ## simple model training
            #######################################################################################

            # else:    
            ### Configure normalised layer to normalise training set, then normalise the testing set
            # self.configure_normalisation_layer(dict_split_final_datasets["train"], input_columns, output_columns)
            
            ### Configure model
            self.configure_model(n_steps_in, n_steps_out, n_epochs, input_columns, output_columns,train_model)
            ### Fitting model with training set and validation set

            history = self.fit(
                train_ds=dict_split_final_datasets["train"], 
                validation_ds=dict_split_final_datasets["validation"], 
                n_epochs=n_epochs,  
                save_model = save_model, 
                batch_size=batch_size
                )
            
            
            
            ### Select the set to be used for forecasting
            which_set = "test"
            # which_set = "validation"
            # which_set = "train"
                        
            TE = Training_evaluations.Training_evaluations_Class(self.Configuration)
            
            list_output_pd = TE.evaluate_cnn_lstm_bnn_training_data(
                dict_split_final_datasets[which_set], 
                output_columns, 
                n_steps_in, 
                n_steps_out, 
                dict_split_final_pd["input_{set}".format(set=which_set)], 
                dict_split_final_pd["label_{set}".format(set=which_set)], 
                self.model,
                )

            ################################################################################
            self.logger.info('Model training'.ljust(self.justif-4,'.') + 'DONE')

        #######################################################################################
        # run the prediction model
        #######################################################################################
        else:
            #######################################################################################
            ## prepare the datasets
            #######################################################################################

            #### Update the input_columns
            input_columns = self.Configuration.input_column_names
            #### Prepare data for forecasting
            datetime_filtered_input_data_pd, datetime_filtered_input_data_ds = Data_prep_class.prepare_data_forecast_bnn(
                input_data_pd, 
                n_steps_in, 
                n_steps_out, 
                input_columns, 
                output_columns, 
                forecast_datetime_aest
                )
            
            #### Call the pretrained model
            self.configure_model(n_steps_in, n_steps_out, n_epochs, input_columns, output_columns, train_model )
           
            forecast_BNN_pd, full_forecast_BNN_pd = Data_prep_class.make_forecast_BNN(datetime_filtered_input_data_ds, output_columns, datetime_filtered_input_data_pd, self.model)

            list_output_pd = [full_forecast_BNN_pd]
        
        return list_output_pd,  stop_rolling, history           