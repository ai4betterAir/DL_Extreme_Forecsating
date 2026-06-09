"""
..  module:: Forecast
    :platform: Unix
    :synopsis: Prepare tensors to train and predict DL model.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""
import sys
import os
import stat
import numpy as np
import pandas as pd
import tensorflow as tf

from ..Tools import Splitting as Split
from ..Tools import TimeSeriesWindowGenerator as TSWG
from ..Tools import TensorToPandas as TTP
from ..Tools import Denormalization_keras_bug_layer as DKBL
from ..Tools import Add_vars as AV
from ..Evaluation import Plots 
from ..Tools import Calibration


###########################################################################################
class Data_Preparation_Class(object):

    """ 
    This class defines a Forecast_Class, that contains the capacity to forecast.
    This is the entry point for all forecast methods

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, Configuration,
                 ):

        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Preparation for data training'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        
        self.train_percent = 0.98
        self.validation_percent = 0.01
        self.test_percent = 0.01

        return
    ##############################################################################
    def prepare_data_training(self, input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, batch_size, update_number):
            """
            method to filter data per season, scale them, and split into train/test
            
            the goal of this method is to prepare the training data of timeseries, convert them into the right format.
            the different successive stages are:
            #. Inputs continuous pd timeserie 
            #. Adds the data to split on: example: season number 
            #. Splits per each asked features (season) and sub-split by individual year, to keep invividual pd timeseries.
            #. Removes the unwanted columns and order them to match data input to model inputs (including order).
            #. Splits each individual pd subsplit into a list of sliding windows of [n_steps_in hours timeseries inputs] and [n_steps_out hours timeseries targets]. 
            The format produced are multiple:
                  - tensorflow dataset for the model input
                  - pd timeseries, to keep the datetime information
            #. Concatenates the tensorflow datasets and pd timeseries of the same features (season) across all years.
            #. Splits them into train/validation/test datasets and pd
            #. Outputs all in dict format containing the data.  
            
            """
            self.input_data_pd = input_data_pd

            self.Add_var = AV.Add_vars_Class(self.logger, self.justif,
                                          self.Configuration,)
            self.input_data_pd = self.Add_var.add_hour_sine_signal(self.input_data_pd)
            # self.input_data_pd = self.Add_var.add_month_sine_signal(self.input_data_pd)

            # print(self.input_data_pd.columns)

            ### Update input column names
            self.Configuration.input_column_names = self.input_data_pd.columns
            input_columns = self.Configuration.input_column_names
            

            self.SC = Split.Splitting_Class(self.logger, self.justif,)
            ######
            # filter by season
            # it returns lists of continuous pd timeseries for each season per invidual year.
            #  list_[season]_data_pd = 1 season -> [[continous pd timeseries for year m]
            #                                       [continous pd timeseries for year n]
            #                                               ]
            self.input_data_pd, list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd = self.SC.split_season(self.input_data_pd)

            ##############################
            # build a dict  by season
            # it returns lists of continuous pd timeseries for each season per invidual year.
            #  self.splitted_input_data_dict = 
            # {       season :   list_[season]_data_pd = 1 season -> [[continous pd timeseries for year m]
            #                                       [continous pd timeseries for year n]
            #                                               ]


            self.splitted_input_data_dict = {
                  1 : list_summer_data_pd,
                  2 : list_autumn_data_pd, 
                  3 : list_winter_data_pd, 
                  4 : list_spring_data_pd,
            }

            ###############################################
            # hack to keep the right number of columns
            # iterate through a list of individual [continous pd timeseries for year m] 
            # to drop the excess columns to match the model inputs.
            # each of them is then column-sorted to ensure the proper ordering to match data sources inputs and the model variable input order.

            # list of pd with columns to drop
            list_of_list_column_drop = [list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd]
            # list_of_list_column_drop = [list_summer_data_pd, ]
            list_of_split_keys_for_column_drop = [1,2,3,4]

            for key, llist in self.splitted_input_data_dict.items():
                  for df in llist:
                    columns_to_drop = df.columns[~ df.columns.isin(input_columns)]
                    df.drop(columns=columns_to_drop, inplace=True)
                    df.sort_index(axis=1, inplace=True)
            
            #### Calculate mean and variance for later normalization
            self.all_mean  = np.mean(df)
            self.all_variance  = np.var(df)

            # This below doesn't update the original pd I think
            # list = [df.drop(columns=columns_to_drop, inplace=False) for df in list]
            ### Divide the batches 
            
            self.logger.info('removing unused variables'.ljust(self.justif-2,'.') + 'OK')

            ###############################################
            ### Splitting using a sliding window to create tensorflow datasets 
            # This below splits each [continous pd timeseries for year m] into a list of [all n_hours_in splitted pds  year m]
            # inputs:
            # list_of_list_split_input_pd = [[all pds for 1 season][all pds for another season][etc...]]
            # output:
            # list_of_list_of_list_split_input_pd = [ 
            #             [       1 season ->[all n_hours_in splitted pds year n] [all n_hours_in splitted pds  year m]] 
            #             [ another season ->[all n_hours_in splitted pds year t] [...] ]
            #                               [etc...]
            #             ]

            input_width = n_steps_in
            label_width = n_steps_out
            shift = n_steps_out
            label_columns = output_columns
            list_of_list_sliding_window = list_of_list_column_drop

            list_of_list_datasets = []
            list_of_list_of_list_split_input_pd = []
            list_of_list_of_list_split_label_pd = []
            
            self.windowed_input_data_dict = {}
            self.windowed_label_data_dict = {}

            #####################################################################################################
            ### Skip the seasonal splitting, using the whole dataset 
            train_pd = self.input_data_pd          
            validation_pd = None
            test_pd = None
            df = train_pd

          

            SWG = TSWG.SlidingWindowGenerator(input_width, label_width, shift,
                                                train_pd, validation_pd, test_pd,
                                                label_columns=label_columns)
            list_of_list_input_indices, list_of_list_label_indices = SWG.make_list_indices_train()

            #### assign empty dict to both var and dict, hence, it links between 2 dicts and the contain 2 dicts are the same
            split_dict_input = self.windowed_input_data_dict[key] = {}
            split_dict_label = self.windowed_label_data_dict[key] = {}
            total_splits_counter = 0

            ########### NOTE: CHECK AGAIN THE COLUMNS OF INPUTS and COLUMNS OF self.input_data_pd  @@@@

            for index_2, indices_input in enumerate(list_of_list_input_indices):
                        
                        # print("key=",key, "index=", index, "index_2=", index_2, "total counter=", total_splits_counter)
                        indices_label = list_of_list_label_indices[index_2]
                        # windowed_input_dict[index_2] = df.iloc[indices_input]
                        # windowed_label_dict[index_2] = df.iloc[indices_label]
                        split_dict_input[total_splits_counter] = df.iloc[indices_input].loc[:, input_columns]
                        split_dict_label[total_splits_counter] = df.iloc[indices_label].loc[:, output_columns]
                        total_splits_counter += 1

            
            self.logger.info('windowing timeseries'.ljust(self.justif-2,'.') + 'OK')


            #############################
            # splitting the final dataset into train/validation/test
            windowed_train_input_data_dict = {}
            windowed_train_label_data_dict = {}
            windowed_validation_input_data_dict = {}
            windowed_validation_label_data_dict = {}
            windowed_test_input_data_dict = {}
            windowed_test_label_data_dict = {}

            
            for key, window_input_pd_dict in self.windowed_input_data_dict.items():
                #   print(key)
                  window_label_pd_dict = self.windowed_label_data_dict[key]
                  
                  counter_list = sorted(list(window_input_pd_dict.keys()))
                  len_list = len(counter_list)

                  ###########################################################
                  # train_size = int(train_percent * len_list)
                  # val_size = int(validation_percent * len_list)
                  # test_size = int(test_percent * len_list)
                  ###########################################################
                  ### Testing with proper val-size
                  num_batches = self.Configuration.num_batches

                  ### NOTE: If the Val loss is NaN and BNN evaluation part is NaN, then check the size of validation set!!!!
                  val_size = int(batch_size*num_batches - 1)
                  test_size = int(batch_size - 1)
                  train_size = int(len_list -  val_size - test_size - 1)
            

                  ##########[HB] update values by rolling update window
                  rolling_values = update_number*batch_size

                  stop_rolling = False
                  if rolling_values < val_size:
                    train_index_list = counter_list[:train_size + rolling_values]
                    val_index_list = counter_list[train_size + rolling_values: train_size + val_size]
                    test_index_list = counter_list[-test_size:]
   
                  else:
                    train_index_list = counter_list[:train_size]
                    val_index_list = counter_list[train_size: train_size + val_size]
                    test_index_list = counter_list[-test_size:]

                    stop_rolling = True
                    
                  windowed_train_input_data_dict[key] = {ii:window_input_pd_dict[k] for ii, k in enumerate(train_index_list)}
                  windowed_train_label_data_dict[key] = {ii:window_label_pd_dict[k] for ii, k in enumerate(train_index_list)}

                  windowed_validation_input_data_dict[key] = {ii:window_input_pd_dict[k] for ii, k in enumerate(val_index_list)}
                  windowed_validation_label_data_dict[key] = {ii:window_label_pd_dict[k] for ii, k in enumerate(val_index_list)}

                  windowed_test_input_data_dict[key] = {ii:window_input_pd_dict[k] for ii, k in enumerate(test_index_list)}
                  windowed_test_label_data_dict[key] = {ii:window_label_pd_dict[k] for ii, k in enumerate(test_index_list)}   
            
            self.logger.info('train/validation/test splits'.ljust(self.justif-2,'.') + 'OK')
            ############
            # convert to datasets
            # train dataset

            train_ds_dict = self.convert_pd_dict_to_ds(windowed_train_input_data_dict, windowed_train_label_data_dict, batch_size)
      
            # validation dataset
            validation_ds_dict = self.convert_pd_dict_to_ds(windowed_validation_input_data_dict, windowed_validation_label_data_dict, batch_size)

            # test dataset
            test_ds_dict = self.convert_pd_dict_to_ds(windowed_test_input_data_dict, windowed_test_label_data_dict, batch_size)

            self.logger.info('Conversion to datasets'.ljust(self.justif-2,'.') + 'OK')

            ################################################
            # concatenate datasets
            ################################################
            ## From dict --> list --> array ??? Too clumsy?

            llist = list(train_ds_dict.values())
            
            print("Total set: ", self.input_data_pd.shape)
            ### First making an empty array to convert from list to array!!!
            input_arr = np.empty((1, n_steps_in, len(input_columns)))
            label_arr = np.empty((1, n_steps_out, len(output_columns)))

            for i in llist:     ### i[inoput, label]
                  input_arr = np.vstack((np.array(i[0]), input_arr))
                  label_arr = np.vstack((np.array(i[1]), label_arr))

            #### For Convolutional 2D layer, extend one more dimension
            # if self.forecast_method:
            #     input_arr = np.expand_dims(input_arr, axis=-1)
            #     label_arr = np.expand_dims(label_arr, axis=-1)
            print("Training set: ", input_arr.shape)
            print("Number of nan", np.isnan(input_arr).sum())
      
            ### modify for LSTM model
            input_ts = tf.data.Dataset.from_tensor_slices(input_arr[1:len(input_arr)])
            label_ts = tf.data.Dataset.from_tensor_slices(label_arr[1:len(label_arr)])

            final_train_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size)

            #################################################
            llist = list(validation_ds_dict.values())
            # print(llist)
            input_arr = np.empty((1, n_steps_in, len(input_columns)))
            label_arr = np.empty((1, n_steps_out, len(output_columns)))
            for i in llist:
                  input_arr = np.vstack((np.array(i[0]), input_arr))
                  label_arr = np.vstack((np.array(i[1]), label_arr))

            print("Validating set: ", input_arr.shape)
            if(np.any(np.isnan(input_arr))):
                # Fill NaN values with the specified fill value
                input_arr = np.nan_to_num(input_arr, nan=0)
            print("Number of nan", np.isnan(input_arr).sum())


            # if Conv2D:
            #     input_arr = np.expand_dims(input_arr, axis=-1)
            #     label_arr = np.expand_dims(label_arr, axis=-1)
            # print("Validating set: ", input_arr.shape)
      
            input_ts = tf.data.Dataset.from_tensor_slices(input_arr[1:len(input_arr)])
            label_ts = tf.data.Dataset.from_tensor_slices(label_arr[1:len(label_arr)])

            if(len(input_arr)> batch_size):
                  final_validation_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size, drop_remainder=True)
            else: 
                  final_validation_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size)

        
            
            ##################################################
            llist = list(test_ds_dict.values())
            input_arr = np.empty((1, n_steps_in, len(input_columns)))
            label_arr = np.empty((1, n_steps_out, len(output_columns)))
            for i in llist:
                  input_arr = np.vstack((np.array(i[0]), input_arr))
                  label_arr = np.vstack((np.array(i[1]), label_arr))
                  
            # if Conv2D:
            #     input_arr = np.expand_dims(input_arr, axis=-1)
            #     label_arr = np.expand_dims(label_arr, axis=-1)

            print("Testing set: ",input_arr.shape)
            if(np.any(np.isnan(input_arr))):
                # Fill NaN values with the specified fill value
                input_arr = np.nan_to_num(input_arr, nan=0)
            print("Number of nan", np.isnan(input_arr).sum())


            
            input_ts = tf.data.Dataset.from_tensor_slices(input_arr[1:len(input_arr)])
            label_ts = tf.data.Dataset.from_tensor_slices(label_arr[1:len(label_arr)])
            if(len(input_arr)> batch_size):
                  final_test_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size, drop_remainder=True)
            else: 
                  final_test_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size)

            # ##################################################
            # print("Train batches")
            # for i, (batch_input, batch_label)  in final_train_ds.enumerate():
            #     print(i,batch_input.shape, batch_label.shape)

            # print("Validate batches")
            # for i, (batch_input, batch_label)  in final_validation_ds.enumerate():
            #     print(i,batch_input.shape, batch_label.shape)

            # print("Test batches")
            # for i, (batch_input, batch_label)  in final_test_ds.enumerate():
            #     print(i,batch_input.shape, batch_label.shape)

           
            self.logger.info('Concatenation to datasets'.ljust(self.justif-2,'.') + 'OK')

            ###################################################
            # pack everything in a dict
            dict_split_final_datasets = {
                  "train" : final_train_ds,
                  "validation" : final_validation_ds,
                  "test" : final_test_ds,
            }
            dict_split_final_pd = {
                  "input_train" : [v2 for k1, v1 in windowed_train_input_data_dict.items() for k2, v2 in v1.items()],
                  "input_validation" : [v2 for k1, v1 in windowed_validation_input_data_dict.items() for k2, v2 in v1.items()],
                  "input_test" : [v2 for k1, v1 in windowed_test_input_data_dict.items() for k2, v2 in v1.items()],
                  "label_train" : [v2 for k1, v1 in windowed_train_label_data_dict.items() for k2, v2 in v1.items()],
                  "label_validation" : [v2 for k1, v1 in windowed_validation_label_data_dict.items() for k2, v2 in v1.items()],
                  "label_test" : [v2 for k1, v1 in windowed_test_label_data_dict.items() for k2, v2 in v1.items()],
            }

            self.logger.info('Data preparation for training'.ljust(self.justif-2,'.') + 'OK')
            
            return dict_split_final_datasets, dict_split_final_pd, stop_rolling

      
    ###########################################################################################
    def convert_pd_dict_to_ds(self, windowed_train_input_data_dict, windowed_train_label_data_dict, batch_size):
        """
        Convert 2 dict ({input}, {labels}) into a dataset
        """
        
        train_ds_dict = {}
        for key, window_input_pd_dict in windowed_train_input_data_dict.items():
            
            window_label_pd_dict = windowed_train_label_data_dict[key]
            # print(key)
            # print(np.array(window_label_pd_dict).shape)
            

            counter_list = sorted(list(window_input_pd_dict.keys()))
            #############################
            input_list = [window_input_pd_dict[ii].values for ii in counter_list]
            label_list = [window_label_pd_dict[ii].values for ii in counter_list]

            train_ds = [input_list, label_list]
            train_ds_dict[key] = train_ds
        
        return train_ds_dict
    
    ###########################################################################################
    def evaluate_data(self, input_ds, output_columns, n_steps_in, n_steps_out,
                                    list_split_input_pd, list_split_label_pd, model):
        
        """
        predict values using the model and inverse transform the data to get final result
        """

        ############## Making forecast #################
        output_np = model.predict(input_ds)
        evaluate = model.evaluate(input_ds)
  
        ##################### Logging the evaluation for the testing inputs #############################
        def log_evaluation(file_name):
            f = open(file_name, "a")
            f.write("INPUT:{0}_OUTPUT:{1}_N-FEATURES:{2} -->> Evaluate [loss, MAE, RMSE]: {3}\n".format(str(n_steps_in), str(n_steps_out), str(len(output_columns)), str(evaluate)))
            f.close()  

        log_evaluation("Logs/Evaluation_" + self.Configuration.var_to_predict[0] + "_.txt")
        ################################################################################################
    
        # building a pd DataFrame
        shape_output = output_np.shape
        n_forecast = shape_output[0]
        forecast_len = shape_output[1]

        # preparing index
        input_forecast_hours = np.arange(1 - n_steps_in, 1)
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name="Forecast_hours")
        columns = output_columns

        #iterating through output
        list_output_pd = []
        # for index, split_input_pd in enumerate(list_split_input_pd):
        for index in range(n_forecast):
            # print("*"*80)
            # print("index=", index, "range=", n_forecast)
            split_input_pd = list_split_input_pd[index]
            split_label_pd = list_split_label_pd[index]
         
            # output_np = self.model.__call__(split_input_pd.values)
            timezone = split_input_pd.index.tz
            # print(timezone)
            # forecast_datetime = split_input_pd.index.values[-1]
            forecast_datetime = split_input_pd.index[-1]
            datetime_range = forecast_datetime + pd.to_timedelta(
                            np.arange(1, forecast_len+1) , unit="h")
            datetime_index = pd.DatetimeIndex(datetime_range, 
                            freq="H", name="datetime"
                        )#.tz_convert(timezone)
            

            # print("*"*80)
            # print("dt=",forecast_datetime,"tz=",split_input_pd.index.tz, "\nindex=",  datetime_index) 
            ### Create the df of outputs           
            df = pd.DataFrame(
                    data=output_np[index, :],
                    columns=columns,
                    index=datetime_index
                )

            # print(df)

        
            ##
            # rename the label columns to Target_{pollutant}_{station} for the referent OBS
            old_columns=split_label_pd.columns
            split_label_pd.columns = ["target_{ii}".format(ii = col) for col in split_label_pd.columns]
            
            ##
            # adding the forecast hours
            ## Index for history input hours from [-n_input: 0]
            split_input_pd["forecast_hours"] = input_forecast_hours
            # split_label_pd["forecast_hours"] = Forecast_hours_index
            ## Index for output forecast hours from [1: n_output]
            df["forecast_hours"] = Forecast_hours_index

            # concatenate
            # print(split_input_pd)
            df = pd.concat([split_input_pd, df,], axis = 0)
            # print(df)
            # print(split_label_pd)
        

           
            ### join the observation and prediction dataFrame together
            df = df.join(split_label_pd, how="outer")
            
            # df = pd.conca ([df, split_label_pd], axis = 1)

            # df = pd.concat([split_input_pd, df, split_label_pd], axis = 0)
            # add forecast number
            df["forecast_number"] = index
            # print(df)
            # print(df.columns, "\n",df.loc[:,list(old_columns)+list(split_label_pd.columns)])
            list_output_pd.append(df)
        
        
        ###
        # final concatenate all 
        output_pd = pd.concat(list_output_pd, axis = 0)
       
        list_output_pd = [output_pd]

        self.logger.info('Evaluation of train data'.ljust(self.justif-2,'.') + 'OK')
        return list_output_pd
###########################################################################################
    def evaluate_data_BNN(self, input_ds, output_columns, n_steps_in, n_steps_out,
                                    list_split_input_pd, list_split_label_pd, model):
        

        """
        predict values using the model and inverse transform the data to get final result

        input_ds: tensors of inputs generated from validation set
        
        """

         ############## Making forecast #################
        ##### Deterministic forecast
        output_np = model.predict(input_ds)
        determinstic_prediction = output_np

        ############################# Probabilistic forecast ###############################
        dist_flag = True
        if dist_flag:
            ############## BNN inference ################
            # Unzip the zipped dataset into separate datasets
            X_validate, y_validate  = zip(*input_ds)

            print("np.array(X_validate).shape", np.array(X_validate).shape)
            
            ### number of sampling
            preds_all = []
            num_samples = self.Configuration.prob_samples
            print(num_samples)
            ## PREDICTIONS ON THE WHOLE BATCHES
            for batch in X_validate:
                preds = [model(batch, training=True) for _ in range(num_samples)]
                preds_all.append(preds)

            preds_all = np.array(preds_all)

            print("preds_all", preds_all.shape)
            print("Number of nan in preds_all: ", np.isnan(preds_all).sum())
            if (np.isnan(preds_all).sum()>100):
                print("="*50,"WARNING NAN IN VALIDATING SET", "="*50 )
                preds_all = np.zeros(preds_all.shape)
           
            ### convert shape of predictive distributions of all validation set
            probs_predictions = preds_all.reshape(num_samples, output_np.shape[0],output_np.shape[1], output_np.shape[2])
            print("probs_predictions.shape", probs_predictions.shape)

            ### Get the mean of predictive distribution
            output_probs_mean = np.stack(probs_predictions).mean(axis=0) ### shape: [n_segments, n_timesteps, n_stations]
            print("output_probs_mean.shape", output_probs_mean.shape)
           
            
            ### get the observation for comparison
            y_validate = np.array(y_validate)
            ### Converting to allign with prediction shape
            y_validate = y_validate.reshape(output_np.shape)
            print("raw y_validate", (y_validate).shape)

            RealOBS_y = y_validate
                
            print("combined_y", RealOBS_y.shape)
            print("output_probs", probs_predictions.shape)

            #####################  Making calibration #######################
            ### Give some thresholds for large values of different pollutants
            calib_probs_mean = np.copy(output_probs_mean)
            ##########>>>> Calibration with high OBS threshold <<<<########## 
            Calib_class =  Calibration.Calibration_Class(self.Configuration)
            if self.Configuration.var_to_predict  == ["O3"]:
                obs_threshold = 3.0
            else: 
                obs_threshold = 15

            test_quantile = 90      ### test_quantile used for calib OBS

            calib_probs_mean, calib_dict, index_dict = Calib_class.Calib_OBS(list_split_input_pd, RealOBS_y, determinstic_prediction, calib_probs_mean, probs_predictions, test_quantile, obs_threshold)

            ##########>>>> Calibration with Temperature
            # thresholds = [24]       ### threshold used for calib temperature
            # for threshold in thresholds:
            #     Calib_class.Calib_temperature(list_split_input_pd, RealOBS_y, determinstic_prediction, output_probs_mean,probs_predictions, preds_all, threshold)

            ### Prepare the columns names of predictive distribution
            ### BNN mean columns
            meanBNN_columns = ["BNN_{ii}".format(ii = col) for col in output_columns]
            print("meanBNN_columns: ", meanBNN_columns)

            ### Calib mean columns
            calibBNN_columns = ["Calib_{ii}".format(ii = col) for col in output_columns]
            print("calibBNN_columns: ", calibBNN_columns)
            

            ### Combine both deterministic and BNN to an array
            output_np_combined = np.concatenate((output_np, output_probs_mean, calib_probs_mean), axis = -1)
            columns = output_columns + meanBNN_columns + calibBNN_columns
            print("---------------------")
            print(columns)
            
        else:
            output_np_combined = output_np
                    # output_columns.extend(meanBNN_columns)
            columns = output_columns
            print("---------------------")
            print(columns)
        
        # ########################## REVERSED THE DIFFERENCING ###############################
        # if self.Configuration.difference_flag:
        #     print("*"*50)
        #     print(">"*20, "UN-DIFFERENCING DATA")
        #     print("*"*50)
        #     reverse = np.copy(output_np_combined)
        #     reversed_data = np.cumsum(reverse, axis=0)
        #     # reversed_data = np.insert(reversed_data, 0, reverse[0,:], axis = 0)  # Add the first value back

        #     print(output_np_combined.shape)
        #     print(reversed_data.shape)
        #     output_np_combined[:] = reversed_data

       
        ########################################################################################
        shape_output = output_np_combined.shape
        n_forecast = shape_output[0]
        forecast_len = shape_output[1]

        # # preparing index
        input_forecast_hours = np.arange(1 - n_steps_in, 1)
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name="Forecast_hours")

   
        # iterating through output
        list_output_pd = []
        
        for index in range(n_forecast):
            # print("*"*80)
            # print("index=", index, "range=", n_forecast)
            split_input_pd = list_split_input_pd[index]
            split_label_pd = list_split_label_pd[index]

    
            # output_np = self.model.__call__(split_input_pd.values)
            timezone = split_input_pd.index.tz
            # print(timezone)
            # forecast_datetime = split_input_pd.index.values[-1]
            forecast_datetime = split_input_pd.index[-1]
            datetime_range = forecast_datetime + pd.to_timedelta(
                            np.arange(1, forecast_len+1) , unit="h")
            datetime_index = pd.DatetimeIndex(datetime_range, 
                            freq="H", name="datetime"
                        )#.tz_convert(timezone)
            

            # print("*"*80)
            # print("dt=",forecast_datetime,"tz=",split_input_pd.index.tz, "\nindex=",  datetime_index) 
            ### Create the df of outputs 
            #### Normal predictions          
            # print(df.describe())
            # #### Probabilistic predictions  
            # print(datetime_index.shape)
            # print(output_np_combined.shape)

            # adsd

            df = pd.DataFrame(
                    data=output_np_combined[index, :],
                    columns=columns,
                    index=datetime_index
                )
            
            

            
            # rename the label columns to Target_{pollutant}_{station} for the referent OBS
            old_columns=split_label_pd.columns
            split_label_pd.columns = ["target_{ii}".format(ii = col) for col in split_label_pd.columns]
            
            ##
            # adding the forecast hours
            ## Index for history input hours from [-n_input: 0]
            split_input_pd["forecast_hours"] = input_forecast_hours
            # split_label_pd["forecast_hours"] = Forecast_hours_index
            ## Index for output forecast hours from [1: n_output]
            df["forecast_hours"] = Forecast_hours_index
            
            # concatenate
            # print(split_input_pd.columns)
            # print(split_label_pd.columns)
            df = pd.concat([split_input_pd, df], axis = 0)

            # print(df)
            # print(split_label_pd)
        

            # print("-------------------")
            ### join the observation and prediction dataFrame together
            df = df.join(split_label_pd, how="outer")
            # print(df_bnn.columns)
        
            # df = pd.conca ([df, split_label_pd], axis = 1)

            # df = pd.concat([split_input_pd, df, split_label_pd], axis = 0)
            # add forecast number
            df["forecast_number"] = index
    
            # print(df.columns, "\n",df.loc[:,list(old_columns)+list(split_label_pd.columns)])
            list_output_pd.append(df)
        
        # final concatenate all 
        output_pd = pd.concat(list_output_pd, axis = 0)

        print(output_pd.describe())

    
        list_output_pd = [output_pd]


        # output_pd = tfds.as_dataframe(output_ds, output_ds.info)
        # output_pd = TTP.convert_tf_to_pd(output_ds, limit=-1)
        # print(output_pd)
        # [n_samples, timesteps, vars]
        self.logger.info('Evaluation of train data'.ljust(self.justif-2,'.') + 'OK')
        return list_output_pd

################################################################################
    def evaluate_data_BNN_single_station(self, input_ds, output_columns, n_steps_in, n_steps_out,
                                    list_split_input_pd, list_split_label_pd, model, station):
        

        """
        predict values using the model and inverse transform the data to get final result
        """
        ############## Making forecast #################
        ##### Deterministic forecast
        output_np = model.predict(input_ds)
        determinstic_prediction = output_np

        # print(determinstic_prediction.shape)
        # asds

        ############################# Probabilistic forecast ###############################
        dist_flag = True
        if dist_flag:
            ############## BNN inference ################
            # Unzip the zipped dataset into separate datasets
            X_validate, y_validate  = zip(*input_ds)

            print("np.array(y_validate).shape", np.array(y_validate).shape)
            
            ### number of sampling
            num_samples = self.Configuration.prob_samples
            print(num_samples)
       
            preds_all = []
            for i in X_validate:
                preds = [model(i, training=True) for _ in range(num_samples)]
                preds_all.append(preds)

            preds_all = np.array(preds_all)

            print("preds_all", preds_all.shape)
            print("Number of nan in preds_all: ", np.isnan(preds_all).sum())

         
           

            if preds_all.shape[0] > 1: 
                combined_array = np.concatenate((preds_all[0,:,:,:], preds_all[1,  :self.Configuration.batch_size, :,:]), axis=1)
            else:
                combined_array = preds_all
                combined_array = np.squeeze(combined_array)

            ### Predictive distribution
            probs_predictions = np.stack(combined_array)
            ### Get the mean of predictive distribution
            output_probs_mean = np.stack(combined_array).mean(axis=0)



            ### get the observation for comparison
            y_validate = np.array(y_validate)
            # print("raw", (y_validate).shape)
            # print((y_validate)[0,:,:,:].shape)
            # print((y_validate)[1,:,:,:].shape)
            if y_validate.shape[0] > 1: 
                RealOBS_y = np.concatenate(((y_validate)[0,:,:,:], (y_validate)[1,  :self.Configuration.batch_size, :,:]), axis=0)
            else:
                RealOBS_y = y_validate
                RealOBS_y = np.squeeze(RealOBS_y)
            
                
            print("combined_y", RealOBS_y.shape)
            print("output_probs", probs_predictions.shape)
            

            idx = []
            ### give some threshold for large values of different pollutants
            if self.Configuration.var_to_predict  == ["O3"]:
                threshold = 3
            else: 
                threshold = 15
            
            step_i = 2 ## 12th steps

            #### Pick-up some high values to compare with probabilistic
            for i, value in enumerate (RealOBS_y[:, int(n_steps_out/step_i)]):    ## get the middle steps of sequence
                if value > threshold:
                    # print(value)
                    idx.append(i)
            # print(idx)
            KDEST = []
            ### Create the histogram
            # step_i = np.random.choice(n_steps_out)       ### Choose randomly in the sequence of output 
            plot_histograms = Plots.Plot_Class(self.logger, self.justif, self.Configuration)

            for i in idx:
                plot_histograms.plot_distribution_singlestation(i, probs_predictions, RealOBS_y, determinstic_prediction, step_i, station)
                
            ### Get the column of predictive distribution
            ### mean columns
            meanBNN_columns = ["BNN_{ii}".format(ii = col) for col in output_columns]
            print(meanBNN_columns)

        print(output_np.shape)
        print(output_probs_mean.shape)

        output_np_combined = np.concatenate((output_np, output_probs_mean), axis = -1) 
    
        ########################################################################################
        shape_output = output_np_combined.shape
        n_forecast = shape_output[0]
        forecast_len = shape_output[1]

        # # preparing index
        input_forecast_hours = np.arange(1 - n_steps_in, 1)
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name="Forecast_hours")

        # output_columns.extend(meanBNN_columns)

        columns = list(output_columns) + meanBNN_columns
        print("---------------------")
        print(columns)
        
        # iterating through output
        list_output_pd = []
        
        for index in range(n_forecast):
            # print("*"*80)
            # print("index=", index, "range=", n_forecast)
            split_input_pd = list_split_input_pd[index]
            split_label_pd = list_split_label_pd[index]

    
            # output_np = self.model.__call__(split_input_pd.values)
            timezone = split_input_pd.index.tz
            # print(timezone)
            # forecast_datetime = split_input_pd.index.values[-1]
            forecast_datetime = split_input_pd.index[-1]
            datetime_range = forecast_datetime + pd.to_timedelta(
                            np.arange(1, forecast_len+1) , unit="h")
            datetime_index = pd.DatetimeIndex(datetime_range, 
                            freq="H", name="datetime"
                        )#.tz_convert(timezone)
            

            # print("*"*80)
            # print("dt=",forecast_datetime,"tz=",split_input_pd.index.tz, "\nindex=",  datetime_index) 
            ### Create the df of outputs 
            #### Normal predictions          
            # print(df.describe())
            #### Probabilistic predictions  

            print(datetime_index.shape)
            print(output_np_combined.shape)

            
            df = pd.DataFrame(
                    data=output_np_combined[index, :],
                    columns=columns,
                    index=datetime_index
                )
            
            

            
            # rename the label columns to Target_{pollutant}_{station} for the referent OBS
            old_columns=split_label_pd.columns
            split_label_pd.columns = ["target_{ii}".format(ii = col) for col in split_label_pd.columns]
            
            ##
            # adding the forecast hours
            ## Index for history input hours from [-n_input: 0]
            split_input_pd["forecast_hours"] = input_forecast_hours
            # split_label_pd["forecast_hours"] = Forecast_hours_index
            ## Index for output forecast hours from [1: n_output]
            df["forecast_hours"] = Forecast_hours_index
            
            # concatenate
            # print(split_input_pd.columns)
            # print(split_label_pd.columns)
            df = pd.concat([split_input_pd, df], axis = 0)

            # print(df)
            # print(split_label_pd)
        

            # print("-------------------")
            ### join the observation and prediction dataFrame together
            df = df.join(split_label_pd, how="outer")
            # print(df_bnn.columns)
        
            # df = pd.conca ([df, split_label_pd], axis = 1)

            # df = pd.concat([split_input_pd, df, split_label_pd], axis = 0)
            # add forecast number
            df["forecast_number"] = index
    
            # print(df.columns, "\n",df.loc[:,list(old_columns)+list(split_label_pd.columns)])
            list_output_pd.append(df)
        
        # final concatenate all 
        output_pd = pd.concat(list_output_pd, axis = 0)

        print(output_pd.describe())

    
        list_output_pd = output_pd


        # output_pd = tfds.as_dataframe(output_ds, output_ds.info)
        # output_pd = TTP.convert_tf_to_pd(output_ds, limit=-1)
        # print(output_pd)
        # [n_samples, timesteps, vars]
        self.logger.info('Evaluation of train data'.ljust(self.justif-2,'.') + 'OK')
        return list_output_pd
    
###########################################################################################
    def prepare_data_forecast(self, input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, forecast_datetime):
        """
        method to filter data per season, scale them, and split into train/test
        """
        
        input_data_pd.drop_duplicates(subset=output_columns,  inplace = True)
        # print(input_data_pd.columns)
        
        forecast_datetime = input_data_pd.index[-1]
    
        # ########### Use hard code for trimming the time at the present
        # # input_data_pd = input_data_pd[input_data_pd.index < forecast_datetime]

        self.SC = Split.Splitting_Class(self.logger, self.justif,)

            # Add datetime properties to the input and filter by season
        input_data_pd, list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd = self.SC.split_season(input_data_pd)


        input_data_pd["forecast_hours"] = forecast_datetime - input_data_pd.index

        ### NOTE: Check again this code for approriate input!!!!!
        if n_steps_in == 48: 
            extra = 1
        elif n_steps_in == 72:
            extra = 2
        else:
            extra = 0

        mask_datetime_input_data = (
        #    (input_data_pd["forecast_hours"] <= (pd.Timedelta(n_steps_in + extra, unit="h"))) ##### Ozone with input 24
        #    (input_data_pd["forecast_hours"] <= (pd.Timedelta(n_steps_in+1, unit="h"))) ##### Ozone with input 48
        #    (input_data_pd["forecast_hours"] <= (pd.Timedelta(n_steps_in+2, unit="h"))) ##### Ozone with input 72
            (input_data_pd["forecast_hours"] <= (pd.Timedelta(n_steps_in -1, unit="h")))  #### PM2.5 48h 
        )
        print("len(mask_datetime_input_data)" , len(mask_datetime_input_data))
        
        # mask_datetime_label_data = (
        #    (input_data_pd["forecast_hours"] >= pd.Timedelta(1, unit="h"))
        #   &(input_data_pd["forecast_hours"] <= pd.Timedelta(n_steps_out, unit="h"))  
        # )

        datetime_filtered_input_data_pd = input_data_pd.loc[mask_datetime_input_data, :]
        # print(len(datetime_filtered_input_data_pd))
        # print((datetime_filtered_input_data_pd.columns))
        # asa
        ######-------------------------------------------------------------
        # datetime_filtered_input_data_pd = input_data_pd.iloc[-n_steps_in:, :] ### whole Dataframe

        # print(datetime_filtered_input_data_pd.tail(len(datetime_filtered_input_data_pd)))
        # print(datetime_filtered_input_data_pd.shape)

        ###############################################
        # hack to keep the right number of columns
        # columns to drop
        
        # columns_to_drop = datetime_filtered_input_data_pd.columns[~ datetime_filtered_input_data_pd.columns.isin(input_columns)]

        print(len(datetime_filtered_input_data_pd))
        
        datetime_filtered_input_data_pd = datetime_filtered_input_data_pd[input_columns] ### Empty dataset with input columns
        datetime_filtered_input_data_pd.sort_index(axis=1, inplace=True)

       ###############################################
        # create tensorflow tensor
        datetime_filtered_input_data_tf = tf.convert_to_tensor(datetime_filtered_input_data_pd)
         ###############################################
        # create tensorflow dataset
        datetime_filtered_input_data_shaped_tf = tf.expand_dims(
            datetime_filtered_input_data_tf, axis=0
            )
       
        datetime_filtered_input_data_ds = tf.data.Dataset.from_tensors(datetime_filtered_input_data_shaped_tf)
        
        datetime_filtered_input_data_pd["forecast_hours"] = np.arange(1 - n_steps_in, 1)        ### 48-xx has problem!!
        # datetime_filtered_input_data_pd["forecast_hours"] = np.arange(- n_steps_in, n_steps_out +1)

        self.logger.info('Data preparation for forecast'.ljust(self.justif-2,'.') + 'OK')
        return datetime_filtered_input_data_pd, datetime_filtered_input_data_ds
###########################################################################################
    def make_forecast(self, input_tf, output_columns, datetime_filtered_input_data_pd, model):
        """
        predict values using the model
        """

        ############## Making forecast #################
        output_np = model.predict(input_tf)
        print(output_np)

        # dsa
        #### building a pd DataFrame
        shape_output = output_np.shape
        # n_forecast = shape_output[0]
        forecast_len = shape_output[1]
        input_len = self.Configuration.n_steps_in
 
        # forecast_len = shape_output[0]

        # preparing index
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name="Forecast_hours")
        # datetime_index = pd.DatetimeIndex(forecast_datetime + pd.to_timedelta(Forecast_hours_index, unit="h"))

        forecast_datetime = datetime_filtered_input_data_pd.index[-1]

        # starting_forecast_time = forecast_datetime + pd.to_timedelta(input_len-1, unit="h")
        starting_forecast_time = forecast_datetime
        # datetime_range = starting_forecast_time + ra+nge
        print("Starting_forecast_time:", starting_forecast_time)
        
        datetime_range = starting_forecast_time + pd.to_timedelta(
            np.arange(1, forecast_len+1) , unit="h")
        

        datetime_index = pd.DatetimeIndex(datetime_range, 
                        freq="H", name="datetime"
                        )
        columns = output_columns

        forecast_pd = pd.DataFrame(
            data=output_np[0, :],
            columns=columns,
            index=datetime_index
            )
        forecast_pd["forecast_hours"] = Forecast_hours_index
        print(forecast_pd.columns)
        
        self.logger.info('Forecast computation'.ljust(self.justif-2,'.') + 'OK')
        return forecast_pd

