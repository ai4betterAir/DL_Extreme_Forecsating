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

### Making the evaluation
### 1) Average metrics: Calculate over the all segments
### 2) Hourly metrics: Calculate over each hour: Oth, 1st, ... nth_forecast length
### 3) Segments in a segment: Divide a forecast segment into smaller segments (6h/12h), then calculate the average metrics in each segment  
class Composition_class(object):
    ###########################################################################################
    def __init__(self, logger, justif, config
                ):
        
        self.logger = logger
        self.justif = justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Composition_class'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        self.config = config

        self.pollutant = config.var_to_predict[0]
        self.input_steps = config.n_steps_in#[0]
        self.output_steps = config.n_steps_out#[0]
        # self.iteration = config.n_iteration[0]
        # self.forecast_method = config.models.forecast_method[0]
        self.selected_region = config.selected_region#[0]
        # self.target_folder = 'training'
        # self.subdir_list = config.additional_var_to_select[0] ### this subfolder named with extraneous var: ex: T, NO, NO2...

        self.base_output_dir = config.Main_Model_training_evaluation_full_dir
        # self.target_folder = 'training'
        

        self.unique_splitted_station_name = None
        self.unique_splitted_variable_name = None


        self.metric_names = config.evaluation_metrics#[0]
        
        return
    
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
        splitted_station_name = [col.split("_",1)[-1] for col in column_name] # [var, rest]
        #### List of other variale names
        splitted_variable_name = [col.split("_",1)[0] for col in column_name]
        #### Unique station names
        self.unique_splitted_station_name = sorted(list(set(splitted_station_name)))
        #### Unique variable names
        self.unique_splitted_variable_name = sorted(list(set(splitted_variable_name)))

        # print(self.unique_splitted_station_name)
        # print(self.unique_splitted_variable_name)

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
        for index, group in grouped:
            # print(index)
            # print("index=", index,"columns=", group.columns, "\ngroup=",group)
            ### filter the target stations according to predictive pollutants
            filter_cols = group.columns[group.columns.str.contains(var_to_predict[0])]
            # print(filter_cols)
        
            ### Drop the input sequence for statistic calculation
            group_append = group[group.columns[group.columns.str.contains(var_to_predict[0])]].dropna()
            ### Convert to the dictionary with keys being the forecast number (samples)
            dict_group[index] = group_append
        
        return dict_group
        
    
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
            dict_station = dict()

            for s in self.unique_splitted_station_name:
                
                ### extract the forecast and obs from a certain station
                station = value[value.columns[value.columns.str.contains(s)]]
            #     print(station)
                ### Each station has 2 columns: (1) ground truth OBS and (2) forecast values to be compared
                forecast_OBS = station.values[:,0]        ### list of all forecast length - 1 forecast segement
                actual_OBS = station.values[:,1]


            #     print(actual_OBS.shape)
            #     print(forecast_OBS.shape)
                if (len(actual_OBS) > 0) and (len(actual_OBS) == len(forecast_OBS)):  ## Check the final segment if the conditions are satisfied or not
                  ### Call the class of average evaluate - this cass insert the 
                  eval = ESM.Evaluation_class(actual_OBS, forecast_OBS)
                  [mae, rmse, mape, pearson_r, r_square, mean_forecast, mean_obs, std_forecast, std_obs, max_bias_error] = eval.mae(),  eval.rmse(), eval.mape(), eval.pearson_r(), \
                                                                        eval.r_square(), eval.mean_forecast(), eval.mean_forecast(),  eval.std_forecast(), eval.std_obs(), eval.max_bias_error()
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
    def average_evaluations(self, dict_eval):
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
                    max_bias_error_st.append(val[sta][0])  ### max_bias_error
                
            
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

        #### Convert to data frame
        average_metric_df = pd.DataFrame(columns=self.metric_names, data = out_average_metrics, index= self.unique_splitted_station_name)
        ### Update the index name of Dataframe
        average_metric_df.index.name='Stations'
        ################ logger init ######################
        loggername='Evaluation'
        logger = IL.Initialise_logging(loggername)
        logger.info('AVERAGE METRICS DATAFRAME'.ljust(self.justif-2,'.') + 'OK')
        print(average_metric_df)

        return average_metric_df
    

    ###########################################################################################
    def hourly_evaluation(self, dict_group):

        #### Hourly metrics: Calculate over each hour: Oth, 1st, ... nth_forecast length
        #### dict_group: group by forecast numbers and clean all unrelevent vars (e.g., hour, month, T, NO...)

        #### Get the length of forecast (e.g., 24h, 48h, ...)
        forecast_length = len(dict_group[0])
        print("Forecast Length: ", forecast_length)

        #### Number of samples (segments)
        n_segments = len(dict_group.keys())
        print("Number forecast segements: ", n_segments)

        ### Rearrange forecast data for each hour to evaluate hourly statistic
        forecast_hours = []
        trueOBS_hours = []

        for t in range(forecast_length):        ### along each timestep t
            
            fc_hour = []
            obs_hour = []

            for i in range(n_segments):         ### along each segment i
                ### Extract the forecast and obs from a certain station
                trueOBS_station = dict_group[i][dict_group[i].columns[dict_group[i].columns.str.contains('target')]]                ###  OBS
                forecast_station = dict_group[i][dict_group[i].columns[dict_group[i].columns.str.contains('target') == False]]      ###  Forecast

                if(forecast_station.shape[0] == forecast_length):
                    ### Append all segments for the t-th forecast 
                    fc_hour.append(forecast_station.values[t, :])
                    obs_hour.append(trueOBS_station.values[t, :])

            ### Append all the forecast hours
            forecast_hours.append(fc_hour)      #### Dimension: (n_timesteps, n_samples, n_stations)
            trueOBS_hours.append(obs_hour)      #### Dimension: (n_timesteps, n_samples, n_stations)

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
            dict_hourly_df[metric]  = pd.DataFrame(columns=col_name , data=np.array(hourly_metrics)[i,:,:], index = range(forecast_length) )
            # print(dict_hourly_df[name])

        # print(dict_hourly_df)
        ### Concatenate all values of dict into a dataframe
        df_hourly_metrics = pd.concat([x for x in dict_hourly_df.values()], axis=1)
        ### Update the index name of Dataframe
        df_hourly_metrics.index.name='forecast_hour'
        
        loggername='Evaluation'
        logger = IL.Initialise_logging(loggername)
        logger.info('HOUR METRICS DATAFRAME'.ljust(self.justif-2,'.') + 'OK')

        print(df_hourly_metrics)
        return df_hourly_metrics
    
    ###########################################################################################
    def save_output_file(self, region, output_pd, vars, file_name, iterates, model):
        ############ Making evaluation files for dashboard ##################
        ###########################################################################################
        ### vars: name of folder ex: T_NO, NO2_NO, OBS_only
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

############################################################################
