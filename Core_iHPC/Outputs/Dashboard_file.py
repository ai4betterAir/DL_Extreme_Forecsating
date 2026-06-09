import os
import sys
import pandas as pd
import numpy as np
from os.path import exists
import json
import logging
from Core_iHPC.Tools import InitLogging as IL

# from .Evaluation import Evaluation_singlemodel as ESM
# from .Evaluation import Evaluation_hourly as EHL
from .Evaluation import Composition as CL_Composition
from .Evaluation import Composition_BNN as CL_Composition_BNN
from .Outputs import Output_manager as OM

###########################################################################################
### Hydra package is used to define configuration for testing multiple scenarios
import hydra
# from hydra import compose, initialize
# from omegaconf import OmegaConf, DictConfig
###########################################################################################

### Making the evaluation
### 1) Average metrics: Calculate over the all segments
### 2) Hourly metrics: Calculate over each hour: Oth, 1st, ... nth_forecast length
### 3) Segments in a segment: Divide a forecast segment into smaller segments (6h/12h), then calculate the average metrics in each segment  
class Dashboard_file_class(object):
    ###########################################################################################
    def __init__(self, Configuration,
                ):
        
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Dashboard file preparation'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        self.metrics_output_filename_template = Configuration.metrics_output_filename_template
        self.metrics_bnn_output_filename_template = Configuration.metrics_bnn_output_filename_template
        
        return

############################################################################
    def load_training_file(
        self, 
        training_dir,
        training_filename,
        iteration, 
        calib_flag
        ):
        
        training_filename_fullpath = os.path.join(training_dir, training_filename)
        
        # ### Template of evaluation files
        # file_template = '{region}_training_{pollutant}_{n_steps_in}_{n_steps_out}_{iteration}_{model}.csv'
        
        # training_output_file = file_template.format(
        #     region = region, 
        #     pollutant = self.pollutant, 
        #     n_steps_in = n_steps_in, 
        #     n_steps_out = n_steps_out, 
        #     iteration = iteration, 
        #     model = self.forecast_method
        #     )
        
        # ### Load predicted values vs. input values
        # file = os.path.join(self.base_output_dir, sub_folder_name, 'training', training_output_file)
        
        training_pd = pd.read_csv(
            training_filename_fullpath, 
            infer_datetime_format=True, 
            parse_dates=["datetime"], 
            index_col=[ "datetime"])
        
        print(training_pd.head())
        print("All columns of evaluation training file: \n", training_pd.columns)
        print("Extra_variable: ", self.vars)
        
        if calib_flag == False:
            drop_columns = training_pd.columns[training_pd.columns.str.contains('Calib_')]
            training_pd = training_pd.drop(columns = drop_columns)
        print("Remain columns of evaluation training file: \n", training_pd.columns)

        return training_pd


    def load_forecast_file_for_combination(self,
                                           forecast_dir,
                                           forecast_filename
                                            ):

        # forecast_dir =  os.path.join(
        #     self.base_output_dir, 
        #     sub_folder_name, 
        #     'forecast', 
        #     )
        # forecast_filename = training_output_file
        
        forecast_filename_fullpath =  os.path.join(forecast_dir, forecast_filename)
        
        forecast_pd = pd.read_csv(
            forecast_filename_fullpath, 
            infer_datetime_format=True, 
            parse_dates=["datetime"], 
            index_col=[ "datetime"])
        
        print(forecast_pd)
        return forecast_pd

############################################################################
    def load_evaluation_file_for_combination(self,
                                             evaluation_stats_dir,
                                             evaluation_stats_filename,
                                            ):


        evaluation_stats_filename_fullpath =  os.path.join(evaluation_stats_dir, evaluation_stats_filename)
        

        evaluation_pd = pd.read_csv(
            evaluation_stats_filename_fullpath, 
            infer_datetime_format=True
            )

 
        return evaluation_pd


############################################################################
    def save_combination_forecast_evaluation(self,
                                             forecast_pd,
                                             evaluation_pd,
                                             dashboard_viz_dir,
                                             dashboard_viz_filename,
                                             ):

        dashboard_viz_filename_template = (
            "forecast_dashboard_{region}_{var}_{inputs}_{outputs}_{model}.csv"
        )

        dashboard_viz_filename = dashboard_viz_filename_template.format(
            region=self.Configuration.selected_region,
            var=self.Configuration.input_var_dir,
            inputs=self.Configuration.n_steps_in,
            outputs=self.Configuration.n_steps_out,
            model=self.Configuration.forecast_method,
        )
        
        dashboard_viz_filename_fullpath =  os.path.join(
            dashboard_viz_dir,
            dashboard_viz_filename)
        

        ######################### save forecast   ##############################

        forecast_pd.to_csv(dashboard_viz_filename_fullpath)
        
        ######################### 1st way to insert statistic in csv ##############################
        
        # Append the second DataFrame (df2) as a footer to the same CSV file
        with open(dashboard_viz_filename_fullpath, 'a') as f:
            f.write('\n')  # Add a newline to separate df1 and df2
            evaluation_pd.to_csv(f, index=False)

        # ######################## 2nd way with JSON attched in csv ##############################
        # # Convert the DataFrame to JSON
        # json_data = evaluation_pd.to_json(orient='records')

        # ######## JSON at the bottom of CSV
        # # # Append the JSON data to a CSV file
        # # with open(dashboard_viz_filename_fullpath, 'a') as csv_file:
        # #     csv_file.write('\n')  # Add a newline to separate JSON from existing CSV data
        # #     csv_file.write(json_data)


        # # evaluation_pd.to_csv(dashboard_viz_filename_fullpath)
        # # # Append the JSON data to a CSV file
        # # with open(dashboard_viz_filename_fullpath, 'a') as csv_file:
        # #     csv_file.write('\n')  # Add a newline to separate JSON from existing CSV data
        # #     csv_file.write(json_data)

        # ######## JSON on top of CSV 
        # # Write the JSON data to a temporary CSV file
        # metadata_csv = dashboard_viz_filename_fullpath
        # with open(metadata_csv, 'w', newline='') as csv_file:
        #     csv_file.write(json.dumps(json_data))
        #     csv_file.write('\n')  # Add a newline separator

        # # Append the original CSV data to the temporary CSV file
        # with open(metadata_csv, 'a') as csv_file:
        #     forecast_pd.to_csv(csv_file, header=True, index=True)
        
        ########################### logger init #############################
        print(">>>>> Saved to: {}".format(dashboard_viz_filename_fullpath))
        self.logger.info('AVERAGE STATS INTEGRATED IN FORECAST FILE'.ljust(self.justif-2,'.') + 'OK')
        return

############################################################################
    def combine(self,
                forecast_dir,
                forecast_filename,
                evaluation_stats_dir,
                evaluation_stats_filename,
                dashboard_viz_dir,
                dashboard_viz_filename,
                ):

        forecast_pd = self.load_forecast_file_for_combination(
                                                            forecast_dir,
                                                            forecast_filename,
                                                            )
        evaluation_pd = self.load_evaluation_file_for_combination(
                                             evaluation_stats_dir,
                                             evaluation_stats_filename,
                                            )
        
        ### Drop MAPE !!!!
        evaluation_pd = evaluation_pd.drop('MAPE', axis = 1)
        print(evaluation_pd)


        self.save_combination_forecast_evaluation(
                                             forecast_pd,
                                             evaluation_pd,
                                             dashboard_viz_dir,
                                             dashboard_viz_filename,
                                             )

        return
############################################################################
    def compute_statistics(self,
                           training_pd,
                           ):
        #################################################################                   
        calib_flag = False

        if calib_flag == False:
            drop_columns = training_pd.columns[training_pd.columns.str.contains('Calib_')]
            training_pd = training_pd.drop(columns = drop_columns)
        print("Remain columns of evaluation training file: \n", training_pd.columns)
        #################################################################

        ##### Evaluation files with BNN differ from others without BNN 
        if 'BNN' in self.Configuration.forecast_method:
            evaluation_metrics_dict = self.compute_eval_bnn(training_pd,)
        else:
            evaluation_metrics_dict = self.compute_eval_normal(training_pd,)
            
        return  evaluation_metrics_dict
############################################################################
if __name__=="__main__":


    justif = 102
################ logger init ######################
    loggername='Evaluation'
    logger = IL.Initialise_logging(loggername)
###################################################

    evaluation = Evaluation_class(logger, justif)
    # ### Global initialization for model configuration with hydra
    
    evaluation.configure_from_main()

    ### Global initialization for model configuration with hydra
    initialize(version_base=None, config_path='Tools/Config_testing/')
    evaluation.config = compose(config_name = 'main_LSTM_BNN_24')

    
    # evaluation.config = compose(config_name = 'main_CNN_LSTM_BNN')
    # evaluation.config = compose(config_name = 'main_CNN_LSTM')
    
    evaluation.pollutant = evaluation.config.var_to_predict[0][0]
    evaluation.input_steps = evaluation.config.n_inputs
    evaluation.output_steps = evaluation.config.n_outputs

    evaluation.iteration = evaluation.config.n_iteration[0]
    evaluation.forecast_method = evaluation.config.models.forecast_method[0]
    evaluation.selected_region = evaluation.config.selected_region

    if evaluation.config.additional_var_to_select[0] == []:
        evaluation.subdir_list = ['OBS_only_test']
    else:
        evaluation.subdir_list = evaluation.config.additional_var_to_select[0]
    
    ####################################################################
    ### Define flag of calibration which contain the calib columns or not!
    calib_flag = False 
    print(evaluation.subdir_list)
    ### Additional variables (e.g., T, RH, NO...)
    for sub_folder_name in evaluation.subdir_list:
        asdas
        evaluation.vars = sub_folder_name

        ########## Multiple regions from config .yaml file #############
        regions = evaluation.selected_region
        print("List of regions: ", regions)

        ### Region selections
        for region in regions:
            ### Timestep selections (12, 24, 36, ..., 72)
            for n_steps_in in evaluation.input_steps:
                # for n_steps_out in evaluation.output_steps: 
                    n_steps_out = n_steps_in
                    print(n_steps_in)
                    print(n_steps_out)
                    
                    ### Number of iterations
                    for i in range(evaluation.iteration):
                        
                        training_pd = evaluation.load_training_file(
                                                                    training_dir,
                                                                    training_filename,
                                                                    i, 
                                                                    calib_flag,
                                                                    )

             
                        ##### Evaluation files with BNN differ from others without BNN 
                        if 'BNN' in evaluation.forecast_method:
                            evaluation_metrics_dict = self.compute_eval_bnn( training_pd)
                        else:
                            evaluation_metrics_dict = self.compute_eval_normal(training_pd)
                        self.OMC = OM.Output_manager_Class(self.Configuration,) 
                        self.OMC.output_metrics(evaluation_metrics_dict, i)


    # ################## Combine average evaluaton to the forecast file for dashboard visualisation ############
                        evaluation.combine(
                                        forecast_dir,
                                        forecast_filename,
                                        evaluation_stats_dir,
                                        evaluation_stats_filename,
                                        dashboard_viz_dir,
                                        dashboard_viz_filename,
                                        )
