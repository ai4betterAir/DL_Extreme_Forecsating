import os
import sys
import pandas as pd
import numpy as np
from os.path import exists
import logging
from Core_iHPC.Tools import InitLogging as IL

from . import Evaluation_singlemodel as ESM
from . import Evaluation_hourly as EHL
###########################################################################################

###########################################################################################
###########################################################################################
### Making the evaluation
### 1) Average metrics: Calculate over the all segments
### 2) Hourly metrics: Calculate over each hour: Oth, 1st, ... nth_forecast length
### 3) Segments in a segment: Divide a forecast segment into smaller segments (6h/12h), then calculate the average metrics in each segment  
class Composition_class(object):
    ###########################################################################################
    ###########################################################################################
    def __init__(self, 
                 Configuration,
                 metric_names,
                #  logger, justif, n_steps_in, n_steps_out, region, config
                ):
        
        
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Composition_class'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        # self.config = config

        # self.pollutant = config.var_to_predict[0]
        # self.input_steps = n_steps_in
        # self.output_steps = n_steps_out
        # self.iteration = config.n_iteration[0]
        # self.forecast_method = config.models.forecast_method[0]
        # self.selected_region = region
        # self.target_folder = 'training'
        # self.subdir_list = config.additional_var_to_select[0] ### this subfolder named with extraneous var: ex: T, NO, NO2...

        # self.base_output_dir = "Result_Outputs/Result_files"
        # self.target_folder = 'training'
        self.pollutant = self.Configuration.var_to_predict[0]
        self.input_steps = self.Configuration.n_steps_in#[0]
        self.output_steps = self.Configuration.n_steps_out#[0]
        # self.iteration = config.n_iteration[0]
        # self.forecast_method = config.models.forecast_method[0]
        self.selected_region = self.Configuration.selected_region#[0]
        # self.target_folder = 'training'
        # self.subdir_list = config.additional_var_to_select[0] ### this subfolder named with extraneous var: ex: T, NO, NO2...

        self.base_output_dir = self.Configuration.Main_Model_training_evaluation_full_dir
        # self.target_folder = 'training'
        

        self.unique_splitted_station_name = None
        self.unique_splitted_variable_name = None

        # self.metric_names = config.evaluation_metrics[0]
        self.metric_names = metric_names

        return
    ###########################################################################################
    ###########################################################################################
    def group_data(self, training_pd):
        '''
            Input: training_pd --> issued by evaluation model
            Output: dict_group --> group by forecast numbers and clean all unrelevent vars (e.g., hour, month, T, NO...)
            
            *Each sample is a dict of all stations:
            {   
                sample 0:   DF0: [FC0 ---> FC23],
                sample 1:   DF1: [FC1 ---> FC24],
                                ...............
                sample n:   DFn: [FCn ---> FCn+23]},
            }     
                
        '''
  
        #### Set forecast number = number of sequences being forecast 
        training_pd.set_index(["forecast_number"], append=True, drop=True, inplace=True)
        # print(training_pd)
        # print("Total length", len(training_pd))

        column_all = list(training_pd.columns)
        #### There are only name with station being capirtalized
        column_name =  [word for word in column_all if word.isupper()] 
        #### List of station names
        station_name = [x for x in column_name if 'BNN' not in x]
        print(station_name)
        
        splitted_station_name = [col.split("_",1)[-1] for col in station_name] # [var, rest]
        print(splitted_station_name)
       
        #### List of other variale names
        splitted_variable_name = [col.split("_",1)[0] for col in column_name]
        #### Unique station names
        self.unique_splitted_station_name = sorted(list(set(splitted_station_name)))
        #### Unique variable names
        self.unique_splitted_variable_name = sorted(list(set(splitted_variable_name)))

        #### Get th var_to_predict (Ozone,)
        target_cols = training_pd.columns[training_pd.columns.str.contains('target')]
        # print(target_cols)
        var_to_predict = list(set([col.split("_",2)[1] for col in target_cols]))
        # print(var_to_predict[0] == 'OZONE')

        #### Group by forecast number
        Indexes = ["forecast_number"]
        grouped = training_pd.groupby(level = Indexes, dropna=False, as_index = True)
        # print(training_pd)
        
        dict_group = {}
        dict_group_BNN = {}
        for index, group in grouped:

            ### Drop the input sequence for statistic calculation
            group_all = group[group.columns[group.columns.str.contains(var_to_predict[0])]].dropna()

            ### normal forecast DF
            normal_cols = group_all.columns[~group_all.columns.str.contains('BNN')]
            group_append = group_all[normal_cols]

            ### BNN forecast DF
            BNN_cols = group_all.columns[group_all.columns.str.contains('BNN')]
            target_cols = group_all.columns[group_all.columns.str.contains('target')]
            BNN_cols = BNN_cols.append(target_cols)
            group_append_BNN = group_all[BNN_cols]
            # print(group_append.columns)
            # print(group_append_BNN.columns)
            # sdsa
            ### Convert to the dictionary with keys being the forecast number (samples)
            dict_group[index] = group_append
            dict_group_BNN[index] = group_append_BNN

      
        return dict_group, dict_group_BNN
        
    
    ###########################################################################################
    def dict_evaluation(self, dict_group):
        '''
            Input: dict_group --> group by forecast numbers and clean all unrelevent vars (e.g., hour, month, T, NO...)
            Output: dict_eval --> Making statistics and produce a dict according to samples
           
           *Each sample is a dict of all stations:
            {   
                sample 0: { station 1: [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error],
                            station 2: [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error],
                                ...............
                            station n: [....................................................................................................]},
                sample 1: {.................................................................................................................]},
                sample 2: {.................................................................................................................]},
                ...........
                }
            
        '''

        dict_eval = dict()

        for key, value in dict_group.items():   ### key is the sample count      
            # print(value.columns) 
            dict_station = dict()

            for s in self.unique_splitted_station_name:
                
                ### extract the forecast and obs from a certain station
                station = value[value.columns[value.columns.str.contains(s)]]
                # print(station.columns)
                ### Each station has 2 columns: (1) normal forecast (2) BNN forecast and (3) Ground truth
                forecast_OBS = station.values[:,0]      
                actual_OBS = station.values[:,-1]

                if (len(actual_OBS) > 2) and (len(forecast_OBS) == len(actual_OBS)):  ## Check the final segment if the conditions are satisfied or not
                  ### Call the class of average evaluate - this cass insert the 
                #   print(len(forecast_OBS), len(actual_OBS))
                  eval = ESM.Evaluation_class(actual_OBS, forecast_OBS)
                  [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error] = eval.mae(),  eval.rmse(), eval.mape(), eval.pearson_r(), \
                                                                        eval.r_square(), eval.mean_forecast(), eval.mean_obs(),  eval.std_forecast(), eval.std_obs(), eval.max_bias_error()
                  ### dictionary for all station at each forecast number
                  dict_station[s] = [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error]
                
               
                else:
                    print('NOOOOOOOOO')
                    ### dictionary for all station at each forecast number
                    mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error = 0,0,0,0,0,0,0,0,0,0
                    dict_station[s] = [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error]
            
            dict_eval[key] = dict_station

        return dict_eval
    ###########################################################################################    
    ###########################################################################################
    def average_evaluations(self, dict_eval, BNN_flag = False):

        mae_ave, rmse_ave, mape_ave, pearson_r_ave, \
        r_square_ave, mean_forecast_ave, mean_obs_ave, \
        std_forecast_ave, std_obs_ave, max_bias_error_ave = list(), list(),list(),list(),list(),list(),list(),list(),list(),list()

        #### Extract metrics from each station
        for sta in self.unique_splitted_station_name:
            mae_st, rmse_st, mape_st, pearson_r_st, \
            r_square_st, mean_forecast_st, mean_obs_st, \
            std_forecast_st, std_obs_st, max_bias_error_st = list(), list(),list(),list(),list(),list(),list(),list(),list(),list()
            
            #### Extract metrics from each sample segment
            for sample_key, val in dict_eval.items():
                # print(sample_key)
   
                if sum(val[sta]) != 0:  ### Some sample has zeros in all metrics which should be excluded in averaging
                   
                    #### Gather all metrics of each station to each variable
                    mae_st.append(val[sta][0])            ### mae
                    rmse_st.append(val[sta][1])           ### rmse
                    mape_st.append(val[sta][2])           ### mape
                    pearson_r_st.append(val[sta][3])      ### pearson_r
                    r_square_st.append(val[sta][4])       ### r_square
                    mean_forecast_st.append(val[sta][5])  ### mean_forecast
                    mean_obs_st.append(val[sta][6])       ### mean_obs
                    std_forecast_st.append(val[sta][7])   ### std_forecast
                    std_obs_st.append(val[sta][8])        ### std_obs
                    max_bias_error_st.append(val[sta][9])  ### max_bias_error
                
            
            ##### After that group all stations to the same metrics
            mae_ave.append(mae_st)
            rmse_ave.append(rmse_st)
            mape_ave.append(mape_st)
            pearson_r_ave.append(pearson_r_st)
            r_square_ave.append(r_square_st)
            mean_forecast_ave.append(mean_forecast_st)
            mean_obs_ave.append(mean_obs_st)
            std_forecast_ave.append(std_forecast_st)
            std_obs_ave.append(std_obs_st)
            max_bias_error_ave.append(max_bias_error_st)

        #### Averaging the metrics along the station axis
        average_metrics = [ np.mean(mae_ave, axis= 1),
                            np.mean(rmse_ave, axis= 1),
                            np.mean(mape_ave, axis= 1),
                            np.mean(pearson_r_ave, axis= 1),
                            np.mean(r_square_ave, axis= 1),
                            np.mean(mean_forecast_ave, axis= 1),
                            np.mean(mean_obs_ave, axis= 1),
                            np.mean(std_forecast_ave, axis= 1),
                            np.mean(std_obs_ave, axis= 1),
                            np.mean(max_bias_error_ave, axis= 1)]
        

        #### Adding the name of metric for columns of dataframe
        out_average_metrics = dict()
        for i, name in enumerate(self.metric_names):
            out_average_metrics[name] = average_metrics[i]

        ### Convert to data frame
        average_metric_df = pd.DataFrame(columns=self.metric_names, data = out_average_metrics, index= self.unique_splitted_station_name)
        ### Update the index name of Dataframe
        if(BNN_flag):
            average_metric_df.index.name='STATISTICS (BNN)'#.format(self.selected_region)
        else: 
            average_metric_df.index.name='STATISTICS'#.format(self.selected_region)

        ########################### logger init #############################
        self.logger.info('AVERAGE METRICS DATAFRAME'.ljust(self.justif-2,'.') + 'OK')
        # print(average_metric_df)

        return average_metric_df
    
###########################################################################################
    ###########################################################################################
    def hourly_evaluation(self, dict_group):

        #### Hourly metrics: Calculate over each hour: Oth, 1st, ... nth_forecast length
        #### dict_group: group by forecast numbers and clean all unrelevent vars (e.g., hour, month, T, NO...)

        #### Get the length of forecast (e.g., 24h, 48h, ...)
        forecast_length = len(dict_group[0]) ### get the length of the 1st df in the dict
        print("Forecast Length: ", forecast_length)

        #### Number of samples (segments)
        n_segments = len(dict_group.keys())
        print("Number forecast segements: ", n_segments)


        ### Rearrange forecast data for each hour to evaluate hourly statistic
        ### Staking all forecast and obs according to hours (e.g., all forecast and obs of forecast hour 1, forecast hour 2, ... )
        forecast_hours = []
        trueOBS_hours = []

        for t in range(forecast_length):        ### along each timestep t = 0 -> 23 (for 24h forecast)
            
            fc_hour = []
            obs_hour = []

            for segment in range(n_segments):         ### along each segment i
                
                ### Extract the forecast and obs from a certain station
                trueOBS_station = dict_group[segment][dict_group[segment].columns[dict_group[segment].columns.str.contains('target')]]                ###  get OBS
                forecast_station = dict_group[segment][dict_group[segment].columns[dict_group[segment].columns.str.contains('target') == False]]      ###  get Forecast

                if(forecast_station.shape[0] == forecast_length):
                    ### Append all segments for the t-th forecast 
                    obs_hour.append(trueOBS_station.values[t, :])
                    fc_hour.append(forecast_station.values[t, :])


            ### Append all the forecast hours
            trueOBS_hours.append(obs_hour)      #### Dimension: (n_timesteps, n_samples, n_stations)
            forecast_hours.append(fc_hour)      #### Dimension: (n_timesteps, n_samples, n_stations)


        # print (np.array(forecast_hours).shape)   
        # print (np.array(trueOBS_hours).shape) 
        
        #### Define an object for hourly metrics
        eval_hourly = EHL.Evaluation_hourly_class(np.array(forecast_hours), np.array(trueOBS_hours))

        mae_vector, rmse_vector, mape_vector, pearson_r_vector, \
        r_square_vector, mean_forecast_vector, mean_obs_vector, \
        std_forecast_vector, std_obs_vector, max_bias_error_vector, IOA_vector = eval_hourly.run_all()

        hourly_metrics = [ mae_vector, rmse_vector, mape_vector, pearson_r_vector, \
                            r_square_vector, mean_forecast_vector, mean_obs_vector, \
                            std_forecast_vector, std_obs_vector, max_bias_error_vector, IOA_vector]

        # print(np.array(hourly_metrics).shape)
        dict_hourly_df  = dict()
        # for station_name in unique_splitted_station_name:
        # col_name =  []
   
        for i, metric in enumerate(self.metric_names):
            col_name = ([metric + "_" + x for x in self.unique_splitted_station_name])
            # print(col_name)
            # print(len(col_name))

            dict_hourly_df[metric]  = pd.DataFrame(columns=col_name , data=np.array(hourly_metrics)[i,:,:], index = range(forecast_length) )
            # print(dict_hourly_df[name])

        # print(dict_hourly_df)
        ### Concatenate all values of dict into a dataframe
        df_hourly_metrics = pd.concat([x for x in dict_hourly_df.values()], axis=1)
        ### Update the index name of Dataframe
        df_hourly_metrics.index.name='forecast_hour'

        ############################################################################
        self.logger.info('HOUR METRICS DATAFRAME'.ljust(self.justif-2,'.') + 'OK')

        # print(df_hourly_metrics)
        return df_hourly_metrics
    ###########################################################################################

    ###########################################################################################
    def starting_hour_evaluation(self, dict_group, BNN_flag = False):

        #### Hourly metrics: Calculate stats for each starting hour: OAM, 1AM, ... 23PM
        #### dict_group: group by forecast numbers and clean all unrelevent vars (e.g., hour, month, T, NO...)

        #### Get the length of forecast (e.g., 24h, 48h, ...)
        forecast_length = len(dict_group[0]) ### get the length of the 1st df in the dict
        # print("Forecast Length: ", forecast_length)

        #### Number of samples (segments)
        n_segments = len(dict_group.keys())
        # print("Number forecast segements: ", n_segments)


        ### Rearrange forecast data for each hour to evaluate hourly statistic
        ### Staking all forecast and obs according to hours (e.g., all forecast and obs of forecast hour 1, forecast hour 2, ... )
        forecast_hours = []
        trueOBS_hours = []
        
        ### 24 hours of real time
        real_hours = range(24)
        fc_starting_hour = {}
        obs_starting_hour= {}
      
        for t in real_hours:        ### along each timestep t = 0 -> 23AM (for 24h forecast)
            fc_temp = []
            obs_temp = []
            for segment in range(n_segments):         ### along each segment 
                
                ### Extract the forecast and obs columns from all station
                trueOBS_station = dict_group[segment][dict_group[segment].columns[dict_group[segment].columns.str.contains('target')]]                ###  get OBS
                forecast_station = dict_group[segment][dict_group[segment].columns[dict_group[segment].columns.str.contains('target') == False]]      ###  get Forecast
                
                ### Revert index for both OBS and forecast df
                trueOBS_station = trueOBS_station.reset_index()
                trueOBS_station = trueOBS_station.drop(['forecast_number'], axis = 1)
                trueOBS_station = trueOBS_station.set_index('datetime')

                forecast_station = forecast_station.reset_index()
                forecast_station = forecast_station.drop(['forecast_number'], axis = 1)
                forecast_station = forecast_station.set_index('datetime')

                # print(forecast_station.index[0])
                if(forecast_station.shape[0] == forecast_length): ### Ignore some remaining forecast df
                    if forecast_station.index[0].hour == t:
                        fc_temp.append(forecast_station)
                        obs_temp.append(trueOBS_station)
            
            ### Segments grouped with the same starting hour forecast
            fc_starting_hour[t] = fc_temp
            obs_starting_hour[t] = obs_temp

        ################### Making average forecast #######################
        dict_average_starting_hour = {}
        dict_min_starting_hour = {}
        dict_max_starting_hour = {}
        for starting_hour, llist_df in fc_starting_hour.items(): ### 24 hours list from 0AM - 23PM
            '''
                llist_df: list of dataframe 
                key: the starting forecast hour 0 AM -> 23 PM

            ''' 
            dict_hour_all_stations = {}
            for idx, fc_df in enumerate(llist_df):  ### dataframe forecast all stations for the same starting time
                '''
                    idx: ranging from [0 -> (n_outstep - 1)] (ex: 0 --> 11 for next 12h forecast)
                    fc_df: dataframe of each segment
                '''
                # dict_station = {}  
                ### create an empty dict with available keys of stations
                dict_station = {station: None for station in self.unique_splitted_station_name} 
             
                ### extract columns of each station in each forecast segment
                for station in self.unique_splitted_station_name:
                    obs_df = obs_starting_hour[starting_hour][idx]
                    ### extract data for each segment per station
                    actual_OBS = obs_df[obs_df.columns[obs_df.columns.str.contains(station)]].values
                    forecast_OBS = fc_df[fc_df.columns[fc_df.columns.str.contains(station)]].values

                    ### feed to average statistic for each segment
                    if (len(actual_OBS) > 2) and (len(forecast_OBS) == len(actual_OBS)):  ## Check the final segment if the conditions are satisfied or not
                    ### Call the class of average evaluate - this cass insert the 
                        eval = ESM.Evaluation_class(actual_OBS, forecast_OBS)
                        ### Calculate the metric for each station per forecast segment
                        mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error = \
                                                    eval.mae(),  eval.rmse(), eval.mape(), eval.pearson_r(), \
                                                    eval.r_square(), eval.mean_forecast(), eval.mean_forecast(),  \
                                                    eval.std_forecast(), eval.std_obs(), eval.max_bias_error()
                        ### dictionary for all station at each forecast number
                        stats = [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error]
                        ## Making dataframe of state for each segment per station
                        stats_cols =["mae", "rmse", "mape", "pearson_r", "r_square", "mean_forecast", "mean_obs", "std_forecast", "std_obs", "max_bias_error"]

                        dict_temp = {label: [number] for label, number in zip(stats_cols, stats)}
                        # dict_station[station] = pd.DataFrame(dict_temp)
                        if dict_station[station] == None:        
                            dict_station[station] = stats  
                        else:
                            dict_station[station].append(stats)  
                                   
                    else:
                        print('NOOOOOOOOO')
                        ### there is NaN or weird dimension then assign 0 for all stats
                        mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error = 0,0,0,0,0,0,0,0,0,0
                        # dict_station[station] = [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error]
                        stats = [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error]
                        if dict_station[station] == None:        
                            dict_station[station] = stats  
                        else:
                            dict_station[station].append(stats)         
                    #### After finishing this loop, states of ALL stations for ONE segemnt per starting hours have been calculated
                # print(dict_station)    
                # zsdsa
            
                dict_hour_all_stations[idx] = dict_station

            #### After finishing this loop, states of ALL stations for ALL segemnts per starting hours have been calculated
            # print(dict_hour_all_stations)
            
            ### Aggregate all states in each starting hour:

            # dummy_stats = [999 for _ in range(len(stats_cols))]
            
            aggregate_dict = {station: None for station in self.unique_splitted_station_name}
            
            for segment, stats_all_stations in dict_hour_all_stations.items():
                # print(stats_all_stations)
                # print("--"*50)
                for station_name, station_stats in stats_all_stations.items():
                    # print("---------------------")
                    # print(station_name)
                    # print(station_stats)
                    ### get a state in a staion
                    if aggregate_dict[station_name] == None:        
                        aggregate_dict[station_name] = [station_stats]  
                    else:
                        aggregate_dict[station_name].append(station_stats)  
                    
     
        
            ############### Aggregation of all stats #################
            ave_stats = {station: None for station in self.unique_splitted_station_name}
            min_stats = {station: None for station in self.unique_splitted_station_name}
            max_stats = {station: None for station in self.unique_splitted_station_name}
            for station_name, station_stats in aggregate_dict.items():
                ### Average stats for all segments
                # print("--"*50)
                # print(station_name, "--> LEN: ", len(station_stats))
                # print(station_stats)
                ave_stats[station_name] = [round((sum(sublist[i] for sublist in station_stats) / len(station_stats)),3) for i in range(len(station_stats[0]))]
                ### Maximum stats for all segments 
                max_stats[station_name] = [max(sublist) for sublist in zip(*station_stats)] ### zip(*station_stats) to transpose the sublists, which effectively groups the elements at the same positions across the sublists
                ### Minimum stats for all segments
                min_stats[station_name] = [min(sublist) for sublist in zip(*station_stats)] ### zip(*station_stats) to transpose the sublists,  which effectively groups the elements at the same positions across the sublists
            # print("AVERAGE_STATS")
            # print(ave_stats)
            # print("--"*50)
            # print("MIN_STATS")
            # print(min_stats)
            # print("--"*50)
            # print("MAX_STATS")            
            # print(max_stats)
            # print("--"*50)

        #### After finishing this loop, states of ALL stations for ALL segemnts for ALL starting hours (24) have been calculated
            dict_average_starting_hour[starting_hour] = ave_stats
            dict_min_starting_hour[starting_hour] = min_stats
            dict_max_starting_hour[starting_hour] = max_stats

        # print(dict_average_starting_hour.keys())
        # print("--AVERAGE_STATS--"*8)
        # print(dict_average_starting_hour)
        # print("--MIN_STATS--"*8)
        # print(dict_min_starting_hour)
        # print("--MAX_STATS--"*8)
        # print(dict_max_starting_hour)
        
        # Create a DataFrame with both layers of keys as index
        df_average = pd.DataFrame.from_dict({(i, j): dict_average_starting_hour[i][j] for i in dict_average_starting_hour.keys() for j in dict_average_starting_hour[i].keys()}, orient='index', columns = stats_cols)
        df_min = pd.DataFrame.from_dict({(i, j): dict_min_starting_hour[i][j] for i in dict_min_starting_hour.keys() for j in dict_average_starting_hour[i].keys()}, orient='index', columns = stats_cols)
        df_max = pd.DataFrame.from_dict({(i, j): dict_max_starting_hour[i][j] for i in dict_max_starting_hour.keys() for j in dict_average_starting_hour[i].keys()}, orient='index', columns = stats_cols)

        
        # Reset the index names
        df_average.index.names = ['Starting_hours, Stations']
        df_min.index.names = ['Starting_hours, Stations']
        df_max.index.names = ['Starting_hours, Stations']


        # df_average.to_csv("all_average_metrics.csv")    
        # df_min.to_csv("all_min_metrics.csv") 
        # df_max.to_csv("all_max_metrics.csv") 
        # print(df_average.head())
        # print(df_min.head())
        # print(df_max.head())
        ######################################################################

        self.logger.info('STARTING-HOUR STATS METRICS'.ljust(self.justif-2,'.') + 'OK')
       
        return df_average, df_min, df_max
    ###########################################################################################


    ###########################################################################################
    def save_output_file(self, region, output_pd, vars, file_name, iterates, model, BNN_flag= False):
        ############ Making evaluation files for dashboard ##################
        ###########################################################################################
        ### vars: name of folder ex: T_NO, NO2_NO, OBS_only
        if (BNN_flag):
            saved_template = '{region}_{file_name}_metrics_{pol}_{inputs}_{outputs}_{vars}_{iterates}_{model}_BNN.csv'
        else:
            saved_template = '{region}_{file_name}_metrics_{pol}_{inputs}_{outputs}_{vars}_{iterates}_{model}.csv'

        evaluation_filename = saved_template.format(region = region, file_name = file_name, pol = self.pollutant[0], \
                                                    inputs = self.input_steps, outputs = self.output_steps, vars = vars, iterates =iterates, model = model)

        evaluation_file = os.path.join(self.base_output_dir, "training/evaluation", evaluation_filename)
        if exists(evaluation_file):
            os.remove(evaluation_file)   ### Remove an existing file before writing new
        # ### save multiple df in one csv file
        #     with open(evaluation_file,'a') as f:
        output_pd.to_csv(evaluation_file)

        print("|||||||>>>>>> Saved To --> " + evaluation_file)
###########################################################################################
   