"""
..  module:: Splitting
    :platform: Unix
    :synopsis: Definition of the basic object class to split data.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

import sys
import os
import stat
import numpy as np
import pandas as pd
import random 

###########################################################################################
class Splitting_Class(object):
    """ 
    This class defines a Splitting_Class, that contains the capacity to manage the Splitting.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, logger, justif,
                ):

        self.logger = logger
        self.justif = justif


        return
###########################################################################################
    #################### Splitting function ######################
    def split_sequences(self, sequence, df, look_back, forecast_horizon):
        X, y, X_time, y_time = list(), list(), list(), list()
        for i in range(len(sequence)): 
                lag_end = i + look_back
                forecast_end = lag_end + forecast_horizon
                if forecast_end > len(sequence):
                    break
                seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
                ind_x, ind_y = df.index[i:lag_end], df.index[lag_end:forecast_end]
                X.append(seq_x)
                y.append(seq_y)
                X_time.append(ind_x)
                y_time.append(ind_y)

        return np.array(X), np.array(y), np.array(X_time), np.array(y_time)

###########################################################################################
    #################### Splitting function for dataFrame ######################
    def split_sequences_df(self, df, look_back, forecast_horizon):
        X, y = list(), list()
        for i in range(len(df)): 
                lag_end = i + look_back
                forecast_end = lag_end + forecast_horizon
                if forecast_end > len(df):
                    break
                seq_x, seq_y = df.iloc[i:lag_end], df.iloc[lag_end:forecast_end]
                X.append(seq_x)
                y.append(seq_y)
        return np.array(X), np.array(y)

###########################################################################################
    def split_season_Hubert(self, df, raw_CTM):
        """
        method to split dataframe into summer and winter season
        """
        #### Get the index
        dt = df.index
        ######################## Filter data for training and forecast ############################
        data_train = df[(df.index < '2020-12-30 00:00:00')]
        data_test_summer = df[(df.index > '2021-01-01 00:00:00')& (df.index < '2021-03-30 00:00:00')]
        data_test_winter = df[(df.index > '2021-07-01 00:00:00')& (df.index < '2021-09-30 00:00:00')]

        # bug here -> raw_CTM_s = raw_CTM[(raw_CTM.index > '2021-01-01 00:00:00')& (df.index < '2021-01-15 00:00:00')]
        raw_CTM_s = raw_CTM[(raw_CTM.index > '2021-01-01 00:00:00')& (raw_CTM.index < '2021-01-15 00:00:00')]
        raw_CTM_w = raw_CTM[(raw_CTM.index > '2021-07-01 00:00:00')& (raw_CTM.index < '2021-09-30 00:00:00')]

        return data_train, data_test_summer, data_test_winter, raw_CTM_s, raw_CTM_w

###########################################################################################
    def split_season(self, input_data_pd):
        """
        method to split dataframe into summer and winter season
        """
     
        # add a day and month columns
        input_data_pd["day_of_year"] = input_data_pd.index.dayofyear    ### Pandas 1.1.5
        input_data_pd["day_of_week"] = input_data_pd.index.dayofweek    ### Pandas 1.1.5
        input_data_pd["month"] = input_data_pd.index.month
        input_data_pd["hour"] = input_data_pd.index.hour

        season_definition = "astronomical"
        season_definition = "meteorological"

        if season_definition == "astronomical":
            input_data_pd["season_number"] = (input_data_pd.index + pd.DateOffset(months=1, days=-21)).quarter
        if season_definition == "meteorological":
            input_data_pd["season_number"] = (input_data_pd.index + pd.DateOffset(months=1)).quarter

        # season mask and split per year
        mask = input_data_pd["season_number"] == 1
        summer_data_pd = input_data_pd.loc[mask, :]
        list_summer_data_pd = [df for i, df in summer_data_pd.groupby(pd.Grouper(freq='YE'))]

        mask = input_data_pd["season_number"] == 2
        autumn_data_pd = input_data_pd.loc[mask, :]
        list_autumn_data_pd = [df for i, df in autumn_data_pd.groupby(pd.Grouper(freq='YE'))]

        mask = input_data_pd["season_number"] == 3
        winter_data_pd = input_data_pd.loc[mask, :]
        list_winter_data_pd = [df for i, df in winter_data_pd.groupby(pd.Grouper(freq='YE'))]

        mask = input_data_pd["season_number"] == 4
        spring_data_pd = input_data_pd.loc[mask, :]
        list_spring_data_pd = [df for i, df in spring_data_pd.groupby(pd.Grouper(freq='YE'))]

        self.logger.info('Data split based on datetime'.ljust(self.justif-8,'.') + 'COMPLETE')

        return input_data_pd, list_summer_data_pd, list_autumn_data_pd, list_winter_data_pd, list_spring_data_pd
###########################################################################################
    def split_train_test_Hubert(self, 
        train_scaled, data_train,  n_steps_in, n_steps_out,
        test_scaled_summer, data_test_summer, test_scaled_winter, data_test_winter,
        raw_CTM_s,raw_CTM_w):
        """
        method to split between train and test 
        """
        ########### Data splitting for segements to train and forecast evaluation ##################
        X_train, y_train, _, _ = self.split_sequences(train_scaled, data_train,  n_steps_in, n_steps_out)
        X_test_s, y_test_s, x_test_s_dt, y_test_s_dt = self.split_sequences(test_scaled_summer, data_test_summer, n_steps_in, n_steps_out)
        X_test_w, y_test_w, x_test_w_dt, y_test_w_dt = self.split_sequences(test_scaled_winter, data_test_winter, n_steps_in, n_steps_out)
        _, CTM_split_s, _, _ = self.split_sequences(raw_CTM_s,raw_CTM_s, n_steps_in, n_steps_out)
        #   bug here --> _, CTM_split_w, _, _ = split_sequences(raw_CTM_w,raw_CTM_s, n_steps_in, n_steps_out)
        _, CTM_split_w, _, _ = self.split_sequences(raw_CTM_w,raw_CTM_w, n_steps_in, n_steps_out)

        CTM_split_s = CTM_split_s.reshape(-1,n_steps_out,1)
        CTM_split_w = CTM_split_w.reshape(-1,n_steps_out,1)
        return X_train, y_train, X_test_s, y_test_s, x_test_s_dt, y_test_s_dt, X_test_w, y_test_w, x_test_w_dt, y_test_w_dt, CTM_split_s, CTM_split_w 
###########################################################################################
    def pd_dict_split_train_val_test(self, 
            windowed_input_data_dict,
            windowed_label_data_dict,
            train_percent,
            validation_percent,
            test_percent,
            shuffle=False,
            ):
        """
        method to split a dict of pandas segemented timeseries into train-val-test pd dict, and return key order
        """
        key_list = sorted(list(windowed_input_data_dict.keys()))
        if shuffle:
            random.shuffle(key_list)
            
        len_list = len(key_list)

        ###########################################################
        train_size = int(train_percent * len_list)
        val_size = int(validation_percent * len_list)
        test_size = int(test_percent * len_list)
        ###########################################################

        train_index_list = key_list[:train_size]
        val_index_list = key_list[train_size: train_size + val_size]
        test_index_list = key_list[-(test_size+1):]
        
        print("*"*100)
        print("lenght data")
        print("sums", train_size, val_size, test_size, train_size+val_size+test_size, len_list)
        print("starting indexes", train_size-1, train_size, train_size+val_size-1, len_list-test_size,)
        print(len(train_index_list), len(val_index_list), len(test_index_list))

        windowed_train_input_data_dict = {key : windowed_input_data_dict[key] for key in train_index_list}
        windowed_train_label_data_dict = {key : windowed_label_data_dict[key] for key in train_index_list}

        windowed_validation_input_data_dict = {key : windowed_input_data_dict[key] for key in val_index_list}
        windowed_validation_label_data_dict = {key : windowed_label_data_dict[key] for key in val_index_list}

        windowed_test_input_data_dict = {key : windowed_input_data_dict[key] for key in test_index_list}
        windowed_test_label_data_dict = {key : windowed_label_data_dict[key] for key in test_index_list} 
        
        return  (windowed_train_input_data_dict, 
                 windowed_train_label_data_dict,
                 windowed_validation_input_data_dict,
                 windowed_validation_label_data_dict, 
                 windowed_test_input_data_dict, 
                 windowed_test_label_data_dict,
                 train_index_list,
                 val_index_list,
                 test_index_list,
                )

        



###########################################################################################
if __name__ == '__main__':
    import logging
    from Core.Tools import InitLogging as IL
    
    justif = 102
