"""
..  module:: Training_evaluations
    :platform: Unix
    :synopsis: Training evaluation helpers (moved under Core_iHPC/Evaluation).

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
"""
import sys
import os
import stat
import numpy as np
import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.layers as TKL 

# from ..Tools import Splitting as Split
# from ..Tools import TimeSeriesWindowGenerator as TSWG
# from ..Tools import Timeseries_segmentation as TSS
from ..Tools import TensorToPandas as TTP
# from ..Tools import PandasToTensor as PTT
# from ..Tools import Denormalization_keras_bug_layer as DKBL
# from ..Tools import Add_vars as AV
from ..Evaluation import Plots 
from ..Tools import Calibration


###########################################################################################
class Training_evaluations_Class(object):

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
        self.logger.info('Evaluation of model training'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        self.KERAS_NORMALISATION_MODEL_NAME = "model"
        self.KERAS_NORMALISATION_LAYER_NAME = "normalisation_layer"


        return

    ###########################################################################################
    def evaluate_cnn_lstm_training_data(self, 
                                        input_ds, 
                                        output_columns, 
                                        n_steps_in, 
                                        n_steps_out,
                                        list_split_input_pd, 
                                        list_split_label_pd, 
                                        model,
                                        ):
        
        """
        predict values using the model and inverse transform the data to get final result
        """
        # ## grabbing normalisation layer parameters to build the inverse:
        # # norm_layer = model.get_layer(self.KERAS_NORMALISATION_MODEL_NAME).get_layer(self.KERAS_NORMALISATION_LAYER_NAME)
        # norm_layer = model.get_layer(self.KERAS_NORMALISATION_LAYER_NAME)
        # inv_norm = TKL.Normalization(
        #     mean=norm_layer.mean,
        #     variance=norm_layer.variance,
        #     axis=norm_layer.axis,
        #     invert=True)
        ############## Making forecast #################
        # output_np = inv_norm(model.predict(input_ds))
        output_np = model.predict(input_ds)
        evaluate = model.evaluate(input_ds)
  
        # ##################### Logging the evaluation for the testing inputs #############################
        # def log_evaluation(file_name):
        #     f = open(file_name, "a")
        #     f.write("INPUT:{0}_OUTPUT:{1}_N-FEATURES:{2} -->> Evaluate [loss, MAE, RMSE]: {3}\n".format(str(n_steps_in), str(n_steps_out), str(len(output_columns)), str(evaluate)))
        #     f.close()  

        # log_evaluation("Logs/Evaluation_" + self.Configuration.var_to_predict[0] + "_.txt")
        # ################################################################################################
    
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
                            freq="h", name="datetime"
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
    def evaluate_cnn_lstm_bnn_training_data(self, 
                                            input_ds, 
                                            output_columns, 
                                            n_steps_in, 
                                            n_steps_out,
                                            list_split_input_pd, 
                                            list_split_label_pd, 
                                            model
                                            ):
        

        """
        predict values using the model and inverse transform the data to get final result

        input_ds: tensors of inputs generated from validation set
        
        # """
        # ## grabbing normalisation layer parameters to build the inverse:
        # norm_layer = model.get_layer(self.KERAS_NORMALISATION_MODEL_NAME).get_layer(self.KERAS_NORMALISATION_LAYER_NAME)
        # inv_norm = TKL.Normalization(
        #     mean=norm_layer.mean,
        #     variance=norm_layer.variance,
        #     axis=norm_layer.axis,
        #     invert=True)

         ############## Making forecast #################
        ##### Deterministic forecast
        # output_np = inv_norm(model.predict(input_ds))
        output_np = model.predict(input_ds)
        determinstic_prediction = output_np

        ############################# Probabilistic forecast ###############################
        dist_flag = True
        if dist_flag:
            ############## BNN inference ################
            # Unzip the zipped dataset into separate datasets
            # X_validate, y_validate  = zip(*inv_norm(input_ds))
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
                obs_threshold = 3.5
            else: 
                obs_threshold = 40

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
                            freq="h", name="datetime"
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
    
