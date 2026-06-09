##### This code is used to calibrate for the Ozone's high concentrations with Met. (T, WSP) and Precursors (NOx, VOC)
'''
Principle:
      - Training and make evaluation with validation set as normal
      - Set a threshold for high concentration of Ozone (e.g., 4 pphm) --> infer the temperature?
      - Collect the real data on validation set with concentraions being equal or higher than threshold (Real Calib set)
      - Compare the predition on validation which is equivalent to the high concentration set (Predict Calib set)
      - Normally the Real Calib set have distributions shift away to the Predict Calib set (under prediction)

      - From Validation set get the Temperature (future) woth 
'''

import sys
import os
import stat

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot  as plt
import datetime

from ..Evaluation import Plots
import logging
from ..Tools import InitLogging as IL

########################################################
class Calibration_Class(object):
    """ 
    This class defines calibration fucntions for future met. awareness

    Attributes
    -----------

       
    """
    def __init__(self, Configuration,
                 ):

        """
        
        Parameters
        -----------
        Configuration : Configuration.Configuration object
            instance of a configuration object that holds all th parameter of the model.
		list_output_pd : list outputs from the normal evaluation on the validation set 
        Attributes
        ----------        
 
        """
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = 102

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Calibration'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        # self.input_data_pd = list_output_pd     

        return

#####################################################################################################################
    def Calib_OBS(self, list_split_input_pd, RealOBS_y, determinstic_prediction, calib_probs_mean, probs_predictions, test_quantile, obs_threshold
            ):
        
        """
        This calibration function by high OBS --> Naive calibration with trial and errors of different test quantiles on validation set.
        The best test quantile will be used to make adjustment on testing set (forecast)
        
        Parameters
        -----------
       
        ----------        
 
        """    
       
        ### Define instance of plot class        
        plot_histograms = Plots.Plot_Class(self.logger, self.justif, self.Configuration)

        #### Get a fixed quantile from OBS distributions in summer and early Authum [Dec, Jan, Feb, Mar] to adjust the difference
        #### Each year, the distribution is quite different --> require an automatic method

        #### Number of step out
        n_steps_out = self.Configuration.n_steps_out
        #### Ozone on output stations
        stations = self.Configuration.output_column_names
        
     
        #### Pick-up some high values of Ozone to compare with probabilistic
        calib_dict = dict()
        index_dict = dict()

        #### Go through Data from each station
        calib_stations = {}

        for index_station, station in enumerate(stations):
            # print(index_station, station)
            idx = []
            calib_values = []
            ### Go through number of output step for each segment (24, 48, 72)
            for step_i in range(n_steps_out): 
                ### Go through all samples in validation set      
                for i, value in enumerate(RealOBS_y[:, step_i, index_station]): 

                    ### Normal deterministic forecast without BNN
                    Deterministic_forecast = determinstic_prediction[i, step_i, index_station]

                    ### Get real OBS at equivalent segement and prediction
                    Real_OBS = RealOBS_y[i, step_i, index_station]      ## same as value 

                    ### Compare 
                    if value >= obs_threshold: ### When value of OBS > any threshold --> trigger the calibration with quantile
                        ### Update the inference with new value with max of distribution ????? 
                        # output_probs_mean[i, step_i, index_station] = np.max(preds_all[:, i, step_i, index_station]) ### get max[n_sampling_dist, n_segments, n_steps_out, n_stations]
                        
                        ### Calculate quantiles using NumPy's percentile function --> Fixed quantile for demonstration only
                        Calib_inference = np.percentile(probs_predictions[:, i, step_i, index_station], [test_quantile])[0]   
                        calib_probs_mean[i, step_i, index_station] = Calib_inference
                        ### Append all the new inference values

                        calib_values.append(Calib_inference)
                        idx.append(i)   ### get the id of data being triggered (high concentration)

                        ### Plot histrogram with calib
                        if(Calib_inference >= 3):
                            plot_histograms.plot_distribution(i, probs_predictions, Real_OBS, Deterministic_forecast, Calib_inference, step_i, station)
                    
                    else:   
                        pass
                        # Mean_dist = -1
                        # plot_histograms.plot_distribution(i, probs_predictions, Real_OBS, Deterministic_forecast, Mean_dist, step_i, station)
                        
            calib_dict[station] =  calib_values
            index_dict[station] =  idx     
        return calib_probs_mean, calib_dict, index_dict

###############################################################################################
    def Calib_temperature(self, list_split_input_pd, RealOBS_y, determinstic_prediction, output_probs_mean,
                            probs_predictions, preds_all, threshold ):
        
        """
        This calibration function by high temperatures (set at 27 deg C).
        From the threshold of temperature 
            --> infer a threshold value of ozone (distribution of all ozone data from 27 +- 2 degree C) 
                --> infer a quantile by this threshold (use cummulative distribution function - cdf) on validating set
                    --> calibe new inferences with the quantile
       
        Parameters
        -----------
       
        ----------        
 
        """    
        
        stations = self.Configuration.output_column_names
        
        #### get all ozone in validation input at temperature equal or over threshold
        ### Make dummy dataframe with 1st batch
        filtered_df =  list_split_input_pd[0]

        # ### Filter data with high temperature --> Inspect ozone
        for segment in list_split_input_pd:
            for j in segment.index:
                temp_columns = segment.filter(like='TEMP_').columns
                # Get index of rows having at least 1 station > the temperature threshold
                high_temp_index =  segment[segment.loc[:, temp_columns].gt(threshold).any(axis=1)].index                
                # Filter rows where at least one temperature column has a value > temperature threshold
                filtered_df = filtered_df.append(segment.loc[high_temp_index], ignore_index=False)

        #########################################################################
        ### Filter data with high ozone --> Inspect temperatures
        # for segment in list_split_input_pd:
        #     for j in segment.index:
        #         oz_columns = segment.filter(like='OZONE_').columns
        #         # Get index of rows having at least 1 station > the temperature threshold
        #         high_oz_index =  segment[segment.loc[:, oz_columns].gt(threshold).any(axis=1)].index                
        #         # Filter rows where at least one temperature column has a value > temperature threshold
        #         filtered_df = filtered_df.append(segment.loc[high_oz_index], ignore_index=False)
        ##########################################################################
        ### Drop all duplicates
        filtered_df = filtered_df.drop_duplicates()   
        ### Remove 1st batch as dummy data
        filtered_df = filtered_df.iloc[len(list_split_input_pd[0]):] 
        ### The filtered data include ozone and temperature columns for all stations in the regions
        print(filtered_df.describe())
        ### Save filtered
        # filtered_df.to_csv("Result_Outputs/high_concentration_over_{}.csv".format(threshold))
        filtered_df.to_csv("Result_Outputs/high_T_over_{}.csv".format(threshold))

        ### Next, the quantile are get by the mean/median of temperatures distribution 
        
        T_data = filtered_df[temp_columns]
        print(T_data.columns)
        mean_T = np.mean(T_data, axis = 0)
        median_T = np.median(T_data, axis = 0)

        print("mean_T: ", mean_T)
        print("median_T: ", median_T)

   
        ### Next step would be map the distributions' means/medians of Temperatures to quantiles for calibration at each station.
        ## One method is: 
        ## Normalised distribution of high temperatures and predictive distribution, then convert the values from normalised high temperature to quantile on predictive distribution!!

                   
        return 

