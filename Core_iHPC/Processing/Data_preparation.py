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
from sklearn.preprocessing import MinMaxScaler

from ..Tools import Splitting as Split
# from ..Tools import TimeSeriesWindowGenerator as TSWG
from ..Tools import Timeseries_segmentation as TSS
from ..Tools import TensorToPandas as TTP
from ..Tools import PandasToTensor as PTT
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

        
        # self.train_percent = 0.98
        # self.validation_percent = 0.01
        # self.test_percent = 0.01
        self.train_percent = 0.7
        self.validation_percent = 0.2
        self.test_percent = 0.1

        self.FORECAST_HOURS_COL = "forecast_hours"
        return
    ##############################################################################
    def prepare_data_training(self, 
                            input_data_pd, 
                            n_steps_in, 
                            n_steps_out, 
                            input_columns, 
                            output_columns, 
                            batch_size, 
                            update_number
                            ):
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
        # self.input_data_pd = input_data_pd

        ###############################
        ## Add internal variables, such as based on time, seasons, etc...
        self.Add_var = AV.Add_vars_Class(self.logger, self.justif,
                                        self.Configuration,)
        input_data_pd  = self.Add_var.add_var_as_defined_in_config_file(input_data_pd)


        ### Update input column names
        # self.Configuration.input_column_names = input_data_pd.columns
        input_columns = self.Configuration.input_column_names
        

        self.SC = Split.Splitting_Class(self.logger, self.justif,)
        ######
        # filter by season
        # it returns lists of continuous pd timeseries for each season per invidual year.
        #  list_[season]_data_pd = 1 season -> [[continous pd timeseries for year m]
        #                                       [continous pd timeseries for year n]
        #                                               ]
        # self.input_data_pd, list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd = self.SC.split_season(self.input_data_pd)

        # ##############################
        # # build a dict  by season
        # # it returns lists of continuous pd timeseries for each season per invidual year.
        # #  self.splitted_input_data_dict = 
        # # {       season :   list_[season]_data_pd = 1 season -> [[continous pd timeseries for year m]
        # #                                       [continous pd timeseries for year n]
        # #                                               ]


        # self.splitted_input_data_dict = {
        #         1 : list_summer_data_pd,
        #         2 : list_autumn_data_pd, 
        #         3 : list_winter_data_pd, 
        #         4 : list_spring_data_pd,
        # }

        # ###############################################
        # # hack to keep the right number of columns
        # # iterate through a list of individual [continous pd timeseries for year m] 
        # # to drop the excess columns to match the model inputs.
        # # each of them is then column-sorted to ensure the proper ordering to match data sources inputs and the model variable input order.


        # # #### Calculate mean and variance for later normalization
        # # self.all_mean  = np.mean(df)
        # # self.all_variance  = np.var(df)

        # # This below doesn't update the original pd I think
        # # list = [df.drop(columns=columns_to_drop, inplace=False) for df in list]
        # ### Divide the batches 
        
        # self.logger.info('removing unused variables'.ljust(self.justif-2,'.') + 'OK')

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

        
        #### assign empty dict to both var and dict, hence, it links between 2 dicts and the contain 2 dicts are the same
        self.windowed_input_data_dict = {}
        self.windowed_label_data_dict = {}

        WG = TSS.WindowGenerator(self.logger, self.justif,
                    input_width, 
                    label_width, 
                    shift,
                    input_columns=input_columns, 
                    label_columns=label_columns,
                    )
        
        WG.segment_inputs(
                    input_data_pd ,
                    self.windowed_input_data_dict,
                    self.windowed_label_data_dict,
                    )
        
        
        self.logger.info('windowing timeseries'.ljust(self.justif-2,'.') + 'OK')

        #############################
        # splitting the final dataset into train/validation/test
        (windowed_train_input_data_dict, 
                 windowed_train_label_data_dict,
                 windowed_validation_input_data_dict,
                 windowed_validation_label_data_dict, 
                 windowed_test_input_data_dict, 
                 windowed_test_label_data_dict,
                 ordered_train_index_list,
                 ordered_val_index_list,
                 ordered_test_index_list,
               ) = self.SC.pd_dict_split_train_val_test( 
            self.windowed_input_data_dict,
            self.windowed_label_data_dict,
            self.train_percent,
            self.validation_percent,
            self.test_percent,
            shuffle=False,
            )

        self.logger.info('train/validation/test splits'.ljust(self.justif-2,'.') + 'OK')
        ############
        # convert to datasets

        Pandas_to_tensor = PTT.PandasToTensor(self.logger, self.justif,)
        
        # train dataset
        final_train_ds = Pandas_to_tensor.convert_pd_dict_to_ds(
            windowed_train_input_data_dict, 
            windowed_train_label_data_dict, 
            ordered_train_index_list,
            batch_size)
        # validation dataset
        final_validation_ds = Pandas_to_tensor.convert_pd_dict_to_ds(
            windowed_validation_input_data_dict, 
            windowed_validation_label_data_dict, 
            ordered_val_index_list,
            batch_size)
        # test dataset
        final_test_ds = Pandas_to_tensor.convert_pd_dict_to_ds(
            windowed_test_input_data_dict, 
            windowed_test_label_data_dict, 
            ordered_test_index_list,
            batch_size)

        # # ##################################################
        # # print("Train batches")
        # # for i, (batch_input, batch_label)  in final_train_ds.enumerate():
        # #     print(i,batch_input.shape, batch_label.shape)

        # # print("Validate batches")
        # # for i, (batch_input, batch_label)  in final_validation_ds.enumerate():
        # #     print(i,batch_input.shape, batch_label.shape)

        # # print("Test batches")
        # # for i, (batch_input, batch_label)  in final_test_ds.enumerate():
        # #     print(i,batch_input.shape, batch_label.shape)

        
        self.logger.info('Concatenation to datasets'.ljust(self.justif-2,'.') + 'OK')

        ###################################################
        # pack everything in a dict
        dict_split_final_datasets = {
                "train" : final_train_ds,
                "validation" : final_validation_ds,
                "test" : final_test_ds,
        }

        dict_split_final_pd = {
                "input_train" :      [windowed_train_input_data_dict[kk]        for kk in ordered_train_index_list],
                "input_validation" : [windowed_validation_input_data_dict[kk]   for kk in ordered_val_index_list],
                "input_test" :       [windowed_test_input_data_dict[kk]         for kk in ordered_test_index_list],
                "label_train" :      [windowed_train_label_data_dict[kk]        for kk in ordered_train_index_list],
                "label_validation" : [windowed_validation_label_data_dict[kk]   for kk in ordered_val_index_list],
                "label_test" :       [windowed_test_label_data_dict[kk]         for kk in ordered_test_index_list],
                "ordered_train_index_list" : ordered_train_index_list,
                "ordered_val_index_list" : ordered_val_index_list,
                "ordered_test_index_list" : ordered_test_index_list,
        }

        self.logger.info('Data preparation for training'.ljust(self.justif-2,'.') + 'OK')
        
        return dict_split_final_datasets, dict_split_final_pd, #stop_rolling

###########################################################################################
    def prepare_data_forecast(self, 
                              input_data_pd, 
                              n_steps_in, 
                              n_steps_out, 
                              input_columns, 
                              output_columns, 
                              forecast_datetime,
                              ):
        """
        method to filter data per season, scale them, and split into train/test
        """

        ###############################
        ## Add internal variables, such as based on time, seasons, etc...
        self.Add_var = AV.Add_vars_Class(self.logger, self.justif,
                                        self.Configuration,)
        input_data_pd  = self.Add_var.add_var_as_defined_in_config_file(input_data_pd)


        ### Update input column names
        input_columns = self.Configuration.input_column_names
        

        self.SC = Split.Splitting_Class(self.logger, self.justif,)
        
        ###################################
        ## filter input hours before forecast datetime to get the right input lenght
       
        forecast_datetime = input_data_pd.index[-1]
        input_data_pd[self.FORECAST_HOURS_COL] = (input_data_pd.index - forecast_datetime )

        # print(input_data_pd[self.FORECAST_HOURS_COL])
        
        mask_datetime_input_data = (
            (input_data_pd[self.FORECAST_HOURS_COL] >= (pd.Timedelta(-(n_steps_in - 1), unit="h")))  
        )

        # print("len(mask_datetime_input_data)" , len(mask_datetime_input_data))
        datetime_filtered_input_data_pd = input_data_pd.loc[mask_datetime_input_data, :]

        #######################
        ## convert timedeltas into forecasthours float
        datetime_filtered_input_data_pd = datetime_filtered_input_data_pd.assign(
            **{self.FORECAST_HOURS_COL:  datetime_filtered_input_data_pd[self.FORECAST_HOURS_COL] / np.timedelta64(1, "h"),}
        )

        #################
        ## sort the columns        
        datetime_filtered_input_data_pd.sort_index(axis=1, inplace=True)

        ############
        # convert to datasets
        Pandas_to_tensor = PTT.PandasToTensor(self.logger, self.justif,)

        datetime_filtered_input_data_ds = Pandas_to_tensor.convert_forecast_pd_to_tensor_ds(
            datetime_filtered_input_data_pd[input_columns].sort_index(axis=1, inplace=False))


        self.logger.info('Data preparation for forecast'.ljust(self.justif-2,'.') + 'OK')
        return datetime_filtered_input_data_pd, datetime_filtered_input_data_ds
###########################################################################################
    def prepare_data_forecast_bnn(self, 
                              input_data_pd, 
                              n_steps_in, 
                              n_steps_out, 
                              input_columns, 
                              output_columns, 
                              forecast_datetime,
                              ):
        """
        method to filter data per season, scale them, and split into train/test
        """

        ###############################
        ## Add internal variables, such as based on time, seasons, etc...
        self.Add_var = AV.Add_vars_Class(self.logger, self.justif,
                                        self.Configuration,)
        input_data_pd  = self.Add_var.add_var_as_defined_in_config_file(input_data_pd)


        ### Update input column names
        input_columns = self.Configuration.input_column_names
        

        self.SC = Split.Splitting_Class(self.logger, self.justif,)
        
        ###################################
        ## filter input hours before forecast datetime to get the right input lenght
       
        forecast_datetime = input_data_pd.index[-1]
        input_data_pd[self.FORECAST_HOURS_COL] = (input_data_pd.index - forecast_datetime )

        # print(input_data_pd[self.FORECAST_HOURS_COL])
        
        mask_datetime_input_data = (
            (input_data_pd[self.FORECAST_HOURS_COL] >= (pd.Timedelta(-(n_steps_in - 1), unit="h")))  
        )

        # print("len(mask_datetime_input_data)" , len(mask_datetime_input_data))
        datetime_filtered_input_data_pd = input_data_pd.loc[mask_datetime_input_data, :]

        #######################
        ## convert timedeltas into forecasthours float
        datetime_filtered_input_data_pd = datetime_filtered_input_data_pd.assign(
            **{self.FORECAST_HOURS_COL:  datetime_filtered_input_data_pd[self.FORECAST_HOURS_COL] / np.timedelta64(1, "h"),}
        )

        #################
        ## sort the columns        
        datetime_filtered_input_data_pd.sort_index(axis=1, inplace=True)

        ############
        # convert to datasets
        Pandas_to_tensor = PTT.PandasToTensor(self.logger, self.justif,)

        datetime_filtered_input_data_ds = Pandas_to_tensor.convert_forecast_pd_to_tensor_bnn(
            datetime_filtered_input_data_pd[input_columns].sort_index(axis=1, inplace=False))


        self.logger.info('Data preparation for forecast'.ljust(self.justif-2,'.') + 'OK')
        return datetime_filtered_input_data_pd, datetime_filtered_input_data_ds
###########################################################################################
    def make_forecast(self, 
                      input_tf, 
                      output_columns, 
                      datetime_filtered_input_data_pd, 
                      model,
                      ):
        """
        predict values using the model
        """

        ############## Making forecast #################
        output_np = model.predict(input_tf)
        print(output_np)

        ##############################
        #### building a pd DataFrame
        shape_output = output_np.shape
        # n_forecast = shape_output[0]
        forecast_len = shape_output[1]
        input_len = self.Configuration.n_steps_in
 
        ##############
        # preparing index
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name=self.FORECAST_HOURS_COL)

        starting_forecast_time = datetime_filtered_input_data_pd.index[-1]
        # print("Starting_forecast_time:", starting_forecast_time)
        
        datetime_range = starting_forecast_time + pd.to_timedelta(np.arange(1, forecast_len+1) , unit="h")
        datetime_index = pd.DatetimeIndex(datetime_range, freq="h", name="datetime")

        ##############
        # building the forecast only dataframe
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name=self.FORECAST_HOURS_COL)
        forecast_pd = pd.DataFrame(
            data=output_np[0, :],
            columns=output_columns,
            index=datetime_index
            )
        forecast_pd[self.FORECAST_HOURS_COL] = Forecast_hours_index
        # print(forecast_pd.columns)
        
        ##############
        # building the full forecast dataframe, inputs + outputs (+ forecast_hours columsn ;) )
        datetime_filtered_input_data_pd = datetime_filtered_input_data_pd[output_columns + [self.FORECAST_HOURS_COL]]
        full_forecast_pd = pd.concat([datetime_filtered_input_data_pd,forecast_pd], axis = 0)
        
        # print(full_forecast_pd)
        
        self.logger.info('Forecast computation'.ljust(self.justif-2,'.') + 'OK')
        return forecast_pd, full_forecast_pd

###########################################################################################
    def make_forecast_BNN(self, input_tf, output_columns, datetime_filtered_input_data_pd, model):
        """
        predict values using the model
        """

        ############## Making forecast #################
        distribution_forecast = [
            model.call(input_tf, training=True) for ii in range(self.Configuration.prob_samples)
        ]
        
        distribution_forecast_np = np.stack(distribution_forecast)
        
        print(distribution_forecast_np)
        

        ### Get the mean of predictive distribution
        output_probabilistic_mean = distribution_forecast_np.mean(axis=0) 
        print("output_probs_mean", output_probabilistic_mean)

         ##############################
        #### building a pd DataFrame
        shape_output = output_probabilistic_mean.shape
        # n_forecast = shape_output[0]
        forecast_len = shape_output[1]
        input_len = self.Configuration.n_steps_in
 
        ##############
        # preparing index
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name=self.FORECAST_HOURS_COL)

        starting_forecast_time = datetime_filtered_input_data_pd.index[-1]
        # print("Starting_forecast_time:", starting_forecast_time)
        
        datetime_range = starting_forecast_time + pd.to_timedelta(np.arange(1, forecast_len+1) , unit="h")
        datetime_index = pd.DatetimeIndex(datetime_range, freq="h", name="datetime")

        ##############
        # building the forecast only dataframe
        Forecast_hours_index = pd.RangeIndex(start=1, stop=forecast_len+1, step=1, name=self.FORECAST_HOURS_COL)
        forecast_BNN_pd = pd.DataFrame(
            data=output_probabilistic_mean[0, :],
            columns=output_columns,
            index=datetime_index
            )
        forecast_BNN_pd[self.FORECAST_HOURS_COL] = Forecast_hours_index
        # print(forecast_pd.columns)
        
        ##############
        # building the full forecast dataframe, inputs + outputs (+ forecast_hours columsn ;) )
        datetime_filtered_input_data_pd = datetime_filtered_input_data_pd[output_columns + [self.FORECAST_HOURS_COL]]
        full_forecast_BNN_pd = pd.concat([datetime_filtered_input_data_pd,forecast_BNN_pd], axis = 0)


        # forecast_BNN_pd = pd.DataFrame(
        #     data=output_probs_mean[0, :],
        #     columns=columns,
        #     index=datetime_index
        #     )

        # forecast_pd["forecast_hours"] = Forecast_hours_index
        # forecast_BNN_pd["forecast_hours"] = Forecast_hours_index

        # print("--------------> Deterministic forecast:")
        # print(forecast_pd)
        # print("--------------> Probabilistic forecast:")
        # print(full_forecast_BNN_pd)
        
        self.logger.info('Forecast computation'.ljust(self.justif-2,'.') + 'OK')
        return forecast_BNN_pd, full_forecast_BNN_pd 
###########################################################################################
    # ##############################################################################
    # def prepare_data_training_scaled(self, input_data_pd, n_steps_in, n_steps_out, input_columns, output_columns, batch_size, update_number):
    #     """
    #     Scaled PM2.5 data before converted to tensor

    #     """
    #     # self.input_data_pd = input_data_pd

    #     # self.Add_var = AV.Add_vars_Class(self.logger, self.justif,
    #     #                               self.Configuration,)
    #     # self.input_data_pd = self.Add_var.add_hour_sine_signal(self.input_data_pd)
    #     # self.input_data_pd = self.Add_var.add_month_sine_signal(self.input_data_pd)

    #     # print(self.input_data_pd.columns)

    #     ### Update input column names
    #     self.Configuration.input_column_names = self.input_data_pd.columns
    #     input_columns = self.Configuration.input_column_names

    #     self.SC = Split.Splitting_Class(self.logger, self.justif,)
    #     ######
    #     # filter by season
    #     # it returns lists of continuous pd timeseries for each season per invidual year.
    #     #  list_[season]_data_pd = 1 season -> [[continous pd timeseries for year m]
    #     #                                       [continous pd timeseries for year n]
    #     #                                               ]
    #     self.input_data_pd, list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd = self.SC.split_season(self.input_data_pd)

    #     ##############################
    #     # build a dict  by season
    #     # it returns lists of continuous pd timeseries for each season per invidual year.
    #     #  self.splitted_input_data_dict = 
    #     # {       season :   list_[season]_data_pd = 1 season -> [[continous pd timeseries for year m]
    #     #                                       [continous pd timeseries for year n]
    #     #                                               ]


    #     self.splitted_input_data_dict = {
    #         1 : list_summer_data_pd,
    #         2 : list_autumn_data_pd, 
    #         3 : list_winter_data_pd, 
    #         4 : list_spring_data_pd,
    #     }

    #     ###############################################
    #     # hack to keep the right number of columns
    #     # iterate through a list of individual [continous pd timeseries for year m] 
    #     # to drop the excess columns to match the model inputs.
    #     # each of them is then column-sorted to ensure the proper ordering to match data sources inputs and the model variable input order.

    #     # list of pd with columns to drop
    #     list_of_list_column_drop = [list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd]
    #     # list_of_list_column_drop = [list_summer_data_pd, ]
    #     list_of_split_keys_for_column_drop = [1,2,3,4]

    #     for key, llist in self.splitted_input_data_dict.items():
    #         for df in llist:
    #             columns_to_drop = df.columns[~ df.columns.isin(input_columns)]
    #             df.drop(columns=columns_to_drop, inplace=True)
    #             df.sort_index(axis=1, inplace=True)
        
    #     # #### Calculate mean and variance for later normalization
    #     # self.all_mean  = np.mean(df)
    #     # self.all_variance  = np.var(df)

    #     # This below doesn't update the original pd I think
    #     # list = [df.drop(columns=columns_to_drop, inplace=False) for df in list]
    #     ### Divide the batches 
        
    #     self.logger.info('removing unused variables'.ljust(self.justif-2,'.') + 'OK')

    #     ###############################################
    #     ### Splitting using a sliding window to create tensorflow datasets 
    #     # This below splits each [continous pd timeseries for year m] into a list of [all n_hours_in splitted pds  year m]
    #     # inputs:
    #     # list_of_list_split_input_pd = [[all pds for 1 season][all pds for another season][etc...]]
    #     # output:
    #     # list_of_list_of_list_split_input_pd = [ 
    #     #             [       1 season ->[all n_hours_in splitted pds year n] [all n_hours_in splitted pds  year m]] 
    #     #             [ another season ->[all n_hours_in splitted pds year t] [...] ]
    #     #                               [etc...]
    #     #             ]

    #     input_width = n_steps_in
    #     label_width = n_steps_out
    #     shift = n_steps_out
    #     label_columns = output_columns
    #     list_of_list_sliding_window = list_of_list_column_drop

    #     list_of_list_datasets = []
    #     list_of_list_of_list_split_input_pd = []
    #     list_of_list_of_list_split_label_pd = []
        
    #     self.windowed_input_data_dict = {}
    #     self.windowed_label_data_dict = {}

    #     #####################################################################################################
    #     ### Skip the seasonal splitting, using the whole dataset 
    #     train_pd = self.input_data_pd          
    #     validation_pd = None
    #     test_pd = None
    #     df = train_pd

        

    #     SWG = TSWG.SlidingWindowGenerator(input_width, label_width, shift,
    #                                         train_pd, validation_pd, test_pd,
    #                                         label_columns=label_columns)
    #     list_of_list_input_indices, list_of_list_label_indices = SWG.make_list_indices_train()

    #     #### assign empty dict to both var and dict, hence, it links between 2 dicts and the contain 2 dicts are the same
    #     split_dict_input = self.windowed_input_data_dict[key] = {}
    #     split_dict_label = self.windowed_label_data_dict[key] = {}
    #     # total_splits_counter = 0

    #     ########### NOTE: CHECK AGAIN THE COLUMNS OF INPUTS and COLUMNS OF self.input_data_pd  @@@@

    #     for index_2, indices_input in enumerate(list_of_list_input_indices):
                    
    #         # print("key=",key, "index=", index, "index_2=", index_2, "total counter=", total_splits_counter)
    #         indices_label = list_of_list_label_indices[index_2]
    #         # windowed_input_dict[index_2] = df.iloc[indices_input]
    #         # windowed_label_dict[index_2] = df.iloc[indices_label]
    #         # split_dict_input[total_splits_counter] = df.iloc[indices_input].loc[:, input_columns]
    #         # split_dict_label[total_splits_counter] = df.iloc[indices_label].loc[:, output_columns]
    #         # total_splits_counter += 1
    #         split_dict_input[index_2] = df.iloc[indices_input].loc[:, input_columns]
    #         split_dict_label[index_2] = df.iloc[indices_label].loc[:, output_columns]

        
    #     self.logger.info('windowing timeseries'.ljust(self.justif-2,'.') + 'OK')


    #     #############################
    #     # splitting the final dataset into train/validation/test
    #     windowed_train_input_data_dict = {}
    #     windowed_train_label_data_dict = {}
    #     windowed_validation_input_data_dict = {}
    #     windowed_validation_label_data_dict = {}
    #     windowed_test_input_data_dict = {}
    #     windowed_test_label_data_dict = {}

        
    #     for key, window_input_pd_dict in self.windowed_input_data_dict.items():
    #         #   print(key)
    #         window_label_pd_dict = self.windowed_label_data_dict[key]
            
    #         counter_list = sorted(list(window_input_pd_dict.keys()))
    #         len_list = len(counter_list)

    #         ###########################################################
    #         train_size = int(self.train_percent * len_list)
    #         val_size = int(self.validation_percent * len_list)
    #         test_size = int(self.test_percent * len_list)
    #         ###########################################################
    #         ### Testing with proper val-size
    #         num_batches = self.Configuration.num_batches

    #         #   ### NOTE: If the Val loss is NaN and BNN evaluation part is NaN, then check the size of validation set!!!!
    #         #   val_size = int(batch_size*num_batches - 1)
    #         #   test_size = int(batch_size - 1)
    #         #   train_size = int(len_list -  val_size - test_size - 1)
        

    #         ##########[HB] update values by rolling update window
    #         #   rolling_values = update_number*batch_size

    #         #   stop_rolling = False
    #         #   if rolling_values < val_size:
    #         #     train_index_list = counter_list[:train_size + rolling_values]
    #         #     val_index_list = counter_list[train_size + rolling_values: train_size + val_size]
    #         #     test_index_list = counter_list[-test_size:]

    #         #   else:
    #         #     train_index_list = counter_list[:train_size]
    #         #     val_index_list = counter_list[train_size: train_size + val_size]
    #         #     test_index_list = counter_list[-test_size:]

    #         #     stop_rolling = True

    #         train_index_list = counter_list[:train_size]
    #         val_index_list = counter_list[train_size: train_size + val_size]
    #         test_index_list = counter_list[-test_size:]
            
    #         print("*"*100)
    #         print("lenght data")
    #         print("sums", train_size, val_size, test_size, train_size+val_size+test_size, len_list)
    #         print("starting indexes", train_size-1, train_size, train_size+val_size-1, len_list-test_size,)
    #         print(len(train_index_list), len(val_index_list), len(test_index_list))
    #         stop()
                
    #         windowed_train_input_data_dict[key] = {ii:window_input_pd_dict[k] for ii, k in enumerate(train_index_list)}
    #         windowed_train_label_data_dict[key] = {ii:window_label_pd_dict[k] for ii, k in enumerate(train_index_list)}

    #         windowed_validation_input_data_dict[key] = {ii:window_input_pd_dict[k] for ii, k in enumerate(val_index_list)}
    #         windowed_validation_label_data_dict[key] = {ii:window_label_pd_dict[k] for ii, k in enumerate(val_index_list)}

    #         windowed_test_input_data_dict[key] = {ii:window_input_pd_dict[k] for ii, k in enumerate(test_index_list)}
    #         windowed_test_label_data_dict[key] = {ii:window_label_pd_dict[k] for ii, k in enumerate(test_index_list)}   
        
    #     self.logger.info('train/validation/test splits'.ljust(self.justif-2,'.') + 'OK')
    #     ############
    #     # convert to datasets
    #     # train dataset

    #     train_ds_dict = self.convert_pd_dict_to_ds(windowed_train_input_data_dict, windowed_train_label_data_dict, batch_size)
    
    #     # validation dataset
    #     validation_ds_dict = self.convert_pd_dict_to_ds(windowed_validation_input_data_dict, windowed_validation_label_data_dict, batch_size)

    #     # test dataset
    #     test_ds_dict = self.convert_pd_dict_to_ds(windowed_test_input_data_dict, windowed_test_label_data_dict, batch_size)

    #     self.logger.info('Conversion to datasets'.ljust(self.justif-2,'.') + 'OK')

    #     ################################################
    #     # concatenate datasets
    #     ################################################
    #     ## From dict --> list --> array ??? Too clumsy?

    #     llist = list(train_ds_dict.values())
        
    #     print("Total set: ", self.input_data_pd.shape)

        
    #     ### First making an empty array to convert from list to array!!!
    #     input_arr = np.empty((1, n_steps_in, len(input_columns)))
    #     label_arr = np.empty((1, n_steps_out, len(output_columns)))

    #     for i in llist:     ### i[inoput, label]
    #         input_arr = np.vstack((np.array(i[0]), input_arr))
    #         label_arr = np.vstack((np.array(i[1]), label_arr))

    #     print("Training set: ", input_arr.shape)
    #     print("Number of nan", np.isnan(input_arr).sum())
    
    #     ### modify for LSTM model
    #     input_ts = tf.data.Dataset.from_tensor_slices(input_arr[1:len(input_arr)])
    #     label_ts = tf.data.Dataset.from_tensor_slices(label_arr[1:len(label_arr)])

    #     final_train_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size)

    #     #################################################
    #     llist = list(validation_ds_dict.values())
    #     # print(llist)
    #     input_arr = np.empty((1, n_steps_in, len(input_columns)))
    #     label_arr = np.empty((1, n_steps_out, len(output_columns)))
    #     for i in llist:
    #             input_arr = np.vstack((np.array(i[0]), input_arr))
    #             label_arr = np.vstack((np.array(i[1]), label_arr))

    #     print("Validating set: ", input_arr.shape)
    #     if(np.any(np.isnan(input_arr))):
    #         # Fill NaN values with the specified fill value
    #         input_arr = np.nan_to_num(input_arr, nan=0)
    #     print("Number of nan", np.isnan(input_arr).sum())


    #     input_ts = tf.data.Dataset.from_tensor_slices(input_arr[1:len(input_arr)])
    #     label_ts = tf.data.Dataset.from_tensor_slices(label_arr[1:len(label_arr)])

    #     if(len(input_arr)> batch_size):
    #             final_validation_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size, drop_remainder=True)
    #     else: 
    #             final_validation_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size)

    
        
    #     ##################################################
    #     llist = list(test_ds_dict.values())
    #     input_arr = np.empty((1, n_steps_in, len(input_columns)))
    #     label_arr = np.empty((1, n_steps_out, len(output_columns)))
    #     for i in llist:
    #             input_arr = np.vstack((np.array(i[0]), input_arr))
    #             label_arr = np.vstack((np.array(i[1]), label_arr))
                
    #     # if Conv2D:
    #     #     input_arr = np.expand_dims(input_arr, axis=-1)
    #     #     label_arr = np.expand_dims(label_arr, axis=-1)

    #     print("Testing set: ",input_arr.shape)
    #     if(np.any(np.isnan(input_arr))):
    #         # Fill NaN values with the specified fill value
    #         input_arr = np.nan_to_num(input_arr, nan=0)
    #     print("Number of nan", np.isnan(input_arr).sum())


        
    #     input_ts = tf.data.Dataset.from_tensor_slices(input_arr[1:len(input_arr)])
    #     label_ts = tf.data.Dataset.from_tensor_slices(label_arr[1:len(label_arr)])
    #     if(len(input_arr)> batch_size):
    #             final_test_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size, drop_remainder=True)
    #     else: 
    #             final_test_ds = tf.data.Dataset.zip((input_ts, label_ts)).batch(batch_size)

    #     # ##################################################
    #     # print("Train batches")
    #     # for i, (batch_input, batch_label)  in final_train_ds.enumerate():
    #     #     print(i,batch_input.shape, batch_label.shape)

    #     # print("Validate batches")
    #     # for i, (batch_input, batch_label)  in final_validation_ds.enumerate():
    #     #     print(i,batch_input.shape, batch_label.shape)

    #     # print("Test batches")
    #     # for i, (batch_input, batch_label)  in final_test_ds.enumerate():
    #     #     print(i,batch_input.shape, batch_label.shape)

        
    #     self.logger.info('Concatenation to datasets'.ljust(self.justif-2,'.') + 'OK')

    #     ###################################################
    #     # pack everything in a dict
    #     dict_split_final_datasets = {
    #             "train" : final_train_ds,
    #             "validation" : final_validation_ds,
    #             "test" : final_test_ds,
    #     }
    #     dict_split_final_pd = {
    #             "input_train" : [v2 for k1, v1 in windowed_train_input_data_dict.items() for k2, v2 in v1.items()],
    #             "input_validation" : [v2 for k1, v1 in windowed_validation_input_data_dict.items() for k2, v2 in v1.items()],
    #             "input_test" : [v2 for k1, v1 in windowed_test_input_data_dict.items() for k2, v2 in v1.items()],
    #             "label_train" : [v2 for k1, v1 in windowed_train_label_data_dict.items() for k2, v2 in v1.items()],
    #             "label_validation" : [v2 for k1, v1 in windowed_validation_label_data_dict.items() for k2, v2 in v1.items()],
    #             "label_test" : [v2 for k1, v1 in windowed_test_label_data_dict.items() for k2, v2 in v1.items()],
    #     }

    #     self.logger.info('Data preparation for training'.ljust(self.justif-2,'.') + 'OK')
        
    #     return dict_split_final_datasets, dict_split_final_pd, stop_rolling
