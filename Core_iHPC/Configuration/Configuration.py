"""
.. module:: Whole_simulation_config
   :platform: Unix
   :synopsis: contains the routine to configure and connect the multiple domain run.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>


"""
import os
import sys
import stat
import re
# import datetime as dtime

import Core_iHPC.Tools.FrozenClass as FC
import Core_iHPC.Tools.DateMagics as DM
import Core_iHPC.Configuration.Name_convention as NC
# from . import DPE_region_stations as DPERT

###########################################################################################
class Configuration_Class(FC.FrozenClass):
    ''' This class configure the whole Forecast.
    '''
    def __init__(self, logger, justif,
             train_start_date_utc, train_end_date_utc, 
             start_date_utc, end_date_utc, timestamp_aedt,
             input_data_dir,
             Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir, Second_dashboard_viz_file_output_dir, 
             Forecast_RunTimeHours, Forecast_NumberofCpu, Forecast_Partition, 
             **kwargs 
             ):

        self.logger = logger
        #justification of the messages
        self.justif = justif


       
        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Configuring Forecast Runs'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        
        self.ComputationMode = None
        self.yaml_config_filename = None
            
        ######
        # Date magics!
        self.train_start_date_utc, self.train_start_date_aest, self.train_start_date_aedt = DM.DateMagics(train_start_date_utc, 'UTC')
        self.train_end_date_utc, self.train_end_date_aest, self.train_end_date_aest = DM.DateMagics(train_end_date_utc, 'UTC')
        self.start_date_utc, self.start_date_aest, self.start_date_aedt = DM.DateMagics(start_date_utc, 'UTC')
        self.end_date_utc, self.end_date_aest, self.end_date_aest = DM.DateMagics(end_date_utc, 'UTC')
        self.timestamp_utc, self.timestamp_aest, self.timestamp_aedt = DM.DateMagics(timestamp_aedt, 'AEDT')

        ######
        #  DL model parameters
        self.forecast_method = None
        # self.var_to_predict = var_to_predict
        self.var_to_predict = ['PM2.5'] #None
        self.additional_var_to_select = None
        self.n_steps_in = None
        self.n_steps_out = None
        self.n_epochs = None

        self.dpie_input_station_list = None
        self.lcs_input_station_list = None 
        self.custom_input_station_list = None
        self.dpie_output_station_list = None
        self.lcs_output_station_list = None
        self.custom_output_station_list = None

        self.var_data_list_from_input_pd = None
        self.batch_size = None

        # AI model parameters
        self.model_parameters_dict = None
        self.sparse_lstm_num_heads = 4
        self.sparse_lstm_window_size = 32
        self.sparse_lstm_ff_dim = 256
        self.sparse_lstm_d_model = 64
        self.sparse_lstm_dropout = 0.1
        self.sparse_lstm_units = 100
        self.sparse_lstm_cnn_filters = 64
        self.sparse_lstm_cnn_kernel_size = 3

        # DL model saving dir
        # self.model_forecast_save_dir = model_forecast_save_dir
        # train
        self.train_model = None
        # save whole model weigths
        self.save_model = None

        ######
        #  input parameters
        self.Data_input_flow = None
        self.input_data_dir = input_data_dir

        self.input_column_names = None
        self.output_column_names = None
        self.specie_properties_dict = None

        self.use_file_training = False

        self.difference_flag = True     ## flag for differencing raw data to stationary data

        self.use_file = True
        self.save_obs_processed_data = None
        ######
        # Output parameters
        self.result_file_subdir = "Forecast"
        self.prob_samples = 100
        self.num_batches = 1
        
        self.Code_configuration_main_dir = Code_configuration_main_dir
        self.Code_configuration_optimal_dir = None
        self.Code_configuration_optimal_history_dir = None
        
        self.Main_training_dir = Main_training_dir
        self.Main_output_dir = Main_output_dir
        self.Main_model_data_dir = Main_model_data_dir
        self.Second_dashboard_viz_file_output_dir = Second_dashboard_viz_file_output_dir

        self.Main_Model_training_full_dir = None
        self.Main_Model_training_plots_full_dir = None
        self.Main_output_run_full_dir = None
        self.Main_output_run_status_full_dir = None
        self.Main_output_run_plots_full_dir = None
        self.Main_model_data_full_dir = None
        self.Main_Test_results_full_dir = None
        self.Main_Model_training_evaluation_full_dir = None
        self.Main_Model_training_evaluation_plot_full_dir = None
        self.Main_Model_training_evaluation_plot_histogram_full_dir = None
        self.Main_Model_training_evaluation_plot_training_loss_full_dir =  None
        self.Main_Model_dashboard_viz_full_dir = None

        self.Model_dir = None
        self.plot_subdir = "Plots"
        self.evaluation_subdir = "Evaluation"
        self.evaluation_histogram_subdir = "Histogram"
        self.evaluation_training_loss_subdir = "Training_loss"
        self.dashboard_viz_subdir = "Dashboard"
        self.Run_configuration_dir = 'Tools/Config_testing'
        
        self.shared_data_dir = None # Shared data directory (from YAML)
        self.Main_output_run_shared_dir = None 
        self.Main_imputed_data_shared_dir = None
        
        self.Code_configuration_optimal_subdir = "Optima" 
        self.Code_configuration_optimal_history_subdir = "Optimisation_history"       
        self.Output_run_status_dir    = None      

        self.input_var_dir = None
        self.train_forecast_dir = None
        self.additional_var_dir = None
        self.output_full_path = None
        self.internal_var_to_add_list = None

        self.forecast_output_filename_template = '{region}_{mode}_{parameters}_{inputs}_{outputs}_{station}_{date}.csv'
        self.forecast_forward_output_filename_template = '{region}_{mode}_{parameters}_{inputs}_{outputs}_{station}_{iter}_{date}.csv'
        self.training_output_filename_template = '{region}_{mode}_{parameters}_{inputs}_{outputs}_{station}.csv'
        self.training_forward_output_filename_template = '{region}_{mode}_{parameters}_{inputs}_{outputs}_{station}_{iter}.csv'
        self.imputation_output_filename_template = 'Imputed_{region}_{var}_{additional_vars}.csv'
        self.obs_output_filename_template = 'Allobs_{raw_processed}_{input_stream}_{region}_{var}_{additional_vars}.csv'

        self.metrics_output_filename_template = '{region}_{metrics}_metrics_{var}_{inputs}_{outputs}_{additional_vars}_{iteration}_{model}.csv'
        self.metrics_bnn_output_filename_template = '{region}_{metrics}_metrics_{var}_{inputs}_{outputs}_{additional_vars}_{iteration}_{model}_BNN.csv'
        self.dashboard_viz_filename_template = "{region}_{var}_{inputs}_{outputs}_{model}_{date}_{run_hour}.csv"        

        self.forecast_plots_filename_template = "{region}_{mode}_{var}_{inputs}_{outputs}_{model}_{date}.png"        

        ###### region dictionary
        # self.DPE_region = DPERT.DPE_region_stations(self.var_to_predict)   
        # self.DPE_region_stations_dict = self.DPE_region.DPE_region_stations_dict     
        # self.selected_region = selected_region
        self.DPE_region_stations_dict = None     
        self.selected_region = None
        # Long/legacy region token used for filesystem naming (e.g. Central_Coast),
        # while `selected_region` remains the short region code (e.g. CC).
        self.selected_region_long_name = None
        
        ### Adding this for later making forecast file with original (no shift Temperature data)
        # self.full_input_pd = full_input_pd

        ######
        # Imputation
        self.imputation_method = None
        self.save_imputed_data = None
        ######
        
        ######
        # Decomposition
        self.vmd_n_imfs = None
        self.vmd_alpha = None
        self.vmd_tau = None
        self.vmd_DC = None
        self.vmd_init = None
        self.vmd_tol = None
        self.vmd_output_dir = None
        self.using_vmd_data = False              
        self.use_decomposition = True
        self.use_vmd_decomposition = True
        self.use_one_model_per_imf = True
        self.model_pipeline_mode = "per_imf"
        
        #file name convention
        self.Name_convention = NC.Name_configuration(self) 
        #####
        # evaluation metrics
        self.evaluation_metrics = None      

        
        self.StaticDataDir = None
        self.RunDir = None
        self.SimulationDir = None
        
        self.SimulationDirPattern = '{rundate}UTC---tstamp---{tstamp}UTC'
        self.SimulationDirPattern = '{rundate}UTC---tstamp---{tstamp}AEDT'
        
        self.StateFullPath = None
        self.StateFileNameTemplate = 'CCAM_{rundate}UTC---tstamp---{tstamp}UTC.json'       
       
        self.Forecast_RunTimeHours = Forecast_RunTimeHours
        self.Forecast_NumberofCpu = Forecast_NumberofCpu
        self.Forecast_Partition = Forecast_Partition

        # Model management parameters (populated from YAML via run_all)
        self.model_base_path = None
        self.model_name = None
        self.model_version = None
        self.time_steps = None
        self.lags = None
        self.strict_version_check = False

    # Added to handle shared data directory
    def configure_shared_data_dir(self, shared_data_dir):
        """
        Set a fixed shared data directory, bypassing the timestamped SimulationDir.
        All configs in a multi-config run should point to the same path.
        """
        self.Main_output_run_shared_dir = shared_data_dir  # Only raw/processed obs go here
        self.logger.info('Shared data dir = {d}'.format(d=shared_data_dir).ljust(self.justif-2,'.') + 'SET')

    def configure_imputed_data_dir(self, imputed_data_dir):
        """
        Set a fixed imputed-data cache directory.
        """
        self.Main_imputed_data_shared_dir = imputed_data_dir
        self.logger.info('Imputed data dir = {d}'.format(d=imputed_data_dir).ljust(self.justif-2,'.') + 'SET')

    def set_decomposition_pipeline_mode(self, use_decomposition, use_one_model_per_imf):
        """
        Central place for decomposition pipeline switches.

        Other model classes should read these attributes instead of hard-coding
        their own copy of the mode.
        """
        self.use_decomposition = bool(use_decomposition)
        self.use_vmd_decomposition = self.use_decomposition
        self.use_one_model_per_imf = bool(use_one_model_per_imf)
        if not self.use_decomposition:
            self.vmd_n_imfs = 0
        self.model_pipeline_mode = (
            "per_imf" if self.use_one_model_per_imf else "joint_imfs"
        )
        return

    def set_vmd_pipeline_mode(self, use_vmd_decomposition, use_one_model_per_imf):
        """
        Backward-compatible alias for set_decomposition_pipeline_mode.
        """
        return self.set_decomposition_pipeline_mode(use_vmd_decomposition, use_one_model_per_imf)

    def get_decomposition_mode_label(self):
        """
        Return a compact label that identifies how decomposition was used.

        Examples
        --------
        IMF_individual_IMF
        IMF_ALLIMF
        no_decomposition
        """
        if not self.use_decomposition:
            return "no_decomposition"
        if self.use_one_model_per_imf:
            return "IMF_individual_IMF"
        return "IMF_ALLIMF"

    def get_vmd_mode_label(self):
        """
        Backward-compatible alias for get_decomposition_mode_label.
        """
        return self.get_decomposition_mode_label()

    def get_metrics_model_token(self):
        """
        Return the model token used in metrics filenames.

        Keeps the existing model name and appends the decomposition mode label so the
        saved CSV clearly shows how the forecast was produced.
        """
        return f"{self.Model_dir}_{self.get_decomposition_mode_label()}"

    def _path_segments(self):
        """
        Return the variable path segments without duplicated copies.

        For many pollutant runs input_var_dir and additional_var_dir are the
        same token (for example O3/O3). In that case keep only one segment so
        the directory layout stays compact and readable.
        """
        segments = [self.input_var_dir, self.additional_var_dir]
        cleaned = []
        for segment in segments:
            if not segment:
                continue
            if segment in cleaned:
                continue
            cleaned.append(segment)
        return cleaned

    def _slug_region_token(self, name):
        token = str(name or "").strip()
        if not token:
            return token
        token = token.replace("-", " ")
        token = re.sub(r"[^A-Za-z0-9 ]+", " ", token)
        token = re.sub(r"\s+", "_", token).strip("_")
        return token


############################################################################################################################
############################################################################################################################
        self._freeze() #cannot add other vars in the class except the one defined in the init() (see FrozenClass.py)
############################################################################################################################
############################################################################################################################
         
        return
###########################################################################################
    def configure_model(self, forecast_method,
                            var_to_predict, additional_var_to_select, 
                            DPE_region_stations_dict, selected_region,  
                            dpie_input_station_list, lcs_input_station_list, custom_input_station_list,
                            dpie_output_station_list, lcs_output_station_list, custom_output_station_list,
                            n_steps_in, n_steps_out, n_epochs, batch_size, 
                            train_model, save_model, 
                            model_parameters_dict, internal_var_to_add_list
                            # selected_region,  full_input_pd,
                                            ):
        """
        configure model parameters
        """                                    
        self.forecast_method = forecast_method
        self.Model_dir = forecast_method
        
        self.var_to_predict = var_to_predict
        self.additional_var_to_select = additional_var_to_select
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.dpie_input_station_list = dpie_input_station_list
        self.lcs_input_station_list =  lcs_input_station_list
        self.custom_input_station_list = custom_input_station_list
        self.dpie_output_station_list = dpie_output_station_list
        self.lcs_output_station_list = lcs_output_station_list
        self.custom_output_station_list = custom_output_station_list

        self.DPE_region_stations_dict = DPE_region_stations_dict     
        self.selected_region = selected_region
        # Compute a backward-compatible "long" region name for filenames/dirs.
        self.selected_region_long_name = selected_region
        try:
            from Core_iHPC.Configuration.DPE_region_stations import (
                REGION_CODE_TO_API_REGION,
                REGION_CODE_TO_API_REGION_NORMALIZED,
            )
            region_code = str(selected_region).strip().upper()
            api_region = (
                REGION_CODE_TO_API_REGION_NORMALIZED.get(region_code)
                or REGION_CODE_TO_API_REGION.get(str(selected_region).strip())
            )
            if api_region:
                self.selected_region_long_name = self._slug_region_token(api_region) or selected_region
        except Exception:
            self.selected_region_long_name = selected_region
        # self.full_input_pd = full_input_pd

        # train
        self.train_model = train_model

        # save whole model weigths
        self.save_model = save_model
        
        # AI model parameters
        self.model_parameters_dict = model_parameters_dict
        self.sparse_lstm_num_heads = model_parameters_dict.get("sparse_lstm_num_heads", self.sparse_lstm_num_heads)
        self.sparse_lstm_window_size = model_parameters_dict.get("sparse_lstm_window_size", self.sparse_lstm_window_size)
        self.sparse_lstm_ff_dim = model_parameters_dict.get("sparse_lstm_ff_dim", self.sparse_lstm_ff_dim)
        self.sparse_lstm_d_model = model_parameters_dict.get("sparse_lstm_d_model", self.sparse_lstm_d_model)
        self.sparse_lstm_dropout = model_parameters_dict.get("sparse_lstm_dropout", self.sparse_lstm_dropout)
        self.sparse_lstm_units = model_parameters_dict.get("sparse_lstm_units", self.sparse_lstm_units)
        self.sparse_lstm_cnn_filters = model_parameters_dict.get("sparse_lstm_cnn_filters", self.sparse_lstm_cnn_filters)
        self.sparse_lstm_cnn_kernel_size = model_parameters_dict.get("sparse_lstm_cnn_kernel_size", self.sparse_lstm_cnn_kernel_size)
        requested_window_size = model_parameters_dict.get("sparse_lstm_window_size", self.sparse_lstm_window_size)
        max_window_size = self.time_steps if self.time_steps is not None else n_steps_in
        if requested_window_size > max_window_size:
            self.logger.warning(
                "Clamping sparse_lstm_window_size from {requested} to {max_window} "
                "to match available time steps".format(
                    requested=requested_window_size,
                    max_window=max_window_size,
                )
            )
        self.sparse_lstm_window_size = min(requested_window_size, max_window_size)
        
        #### additional internal vars to add, such as hour, month, etc...
        self.internal_var_to_add_list = internal_var_to_add_list
        
        return

###########################################################################################
    def configure_input(self, Data_input_flow, use_file,
                        save_obs_processed_data):
        """
        configure input parameter
        """
        self.Data_input_flow = Data_input_flow
        self.use_file = use_file
        self.save_obs_processed_data = save_obs_processed_data

        return

###########################################################################################
    def configure_imputation(self, imputation_method, save_imputed_data
                        ):
        """
        configure imputation parameter
        """
        self.imputation_method = imputation_method
        self.save_imputed_data = save_imputed_data
        return
###########################################################################################
    def configure_evaluation_metrics(self, evaluation_metrics
                        ):
        """
        configure imputation parameter
        """
        self.evaluation_metrics = evaluation_metrics
        return
        
###########################################################################################
    def configure_vmd(self, n_imfs, alpha, tau, DC, init, tol, output_dir):
        """Configure decomposition parameters"""
        self.vmd_n_imfs = n_imfs
        self.vmd_alpha = alpha
        self.vmd_tau = tau
        self.vmd_DC = DC
        self.vmd_init = init
        self.vmd_tol = tol
        self.vmd_output_dir = output_dir
        
        return

###########################################################################################
    def configure_output(self,
                        ):
        """
        configure output parameter
        self.base_output_dir / Forecast (Training)/ region/ self.additional_var_dir / self.SimulationDir

        3 main directory trees
        
        This is where all the directories are precomputed following the conventions

        self.Code_configuration_main_dir / self.Run_configuration_dir / conf files
        self.Code_configuration_main_dir / self.Code_configuration_optimal_subdir / optimal files
        
        self.Main_training_dir 
            / self.Region_dir 
                / sel.input_var_dir 
                    / self.additional_var_dir 
                        / self.Model_dir 
                            / files
                                / self.Plot_dir 
                                    / plot files 
                                / self.Evaluation_dir
                                    / self.Plot_dir  
                                        /self.evaluation_histogram_subdir 
                                        /self.evaluation_training_loss_subdir           
                        / Model_n / files
        
        self.Main_output_dir   
            / self.Region_dir 
                / sel.input_var_dir 
                    / self.additional_var_dir 
                        / self.Model_dir 
                            / self.run_output_dir (name convention) 
                                / files
                                |-> self.Plot_dir   --> plot files            

                        / Model_n  ...                                                                                                                         |-> self.Output_run_status_dir (json)

        self.Main_model_data_dir 
            / self.Model_dir   
                / models weights files
            Model_n        
                / model weights files
       
        """
        
        self.ComputeDirectories()
        region_dir = self.selected_region_long_name or self.selected_region

        ##########
        # compute the full paths
        # print(self.Main_training_dir,) 
        # print(    self.selected_region,)
        # print(    self.input_var_dir,)
        # print(    self.additional_var_dir,)
        # print(    self.Model_dir,)
        self.Main_Model_training_full_dir =  os.path.join(
            self.Main_training_dir, 
            region_dir,
            *self._path_segments(),
            self.Model_dir,
            )
        self.Main_Model_training_plots_full_dir =  os.path.join(
            self.Main_Model_training_full_dir,
            self.plot_subdir,
            )

        self.Main_Model_training_evaluation_full_dir =  os.path.join(
            self.Main_Model_training_full_dir,
            self.evaluation_subdir,
            )

        self.Main_Model_training_evaluation_plot_full_dir =  os.path.join(
            self.Main_Model_training_full_dir,
            self.plot_subdir,
            )

        self.Main_Model_training_evaluation_plot_histogram_full_dir =  os.path.join(
            self.Main_Model_training_evaluation_plot_full_dir,
            self.evaluation_histogram_subdir,
            )
       
        self.Main_Model_training_evaluation_plot_training_loss_full_dir =  os.path.join(
            self.Main_Model_training_evaluation_plot_full_dir,
            self.evaluation_training_loss_subdir,
            )
       
        self.Main_output_run_full_dir = os.path.join(
            self.Main_output_dir,
            region_dir,
            *self._path_segments(),
            self.Model_dir,
            self.SimulationDir,
        )
        
        self.Main_output_run_status_full_dir = os.path.join(
            self.Main_output_run_full_dir, 
            self.Output_run_status_dir
            )
        
        self.Main_output_run_plots_full_dir = os.path.join(
            self.Main_output_run_full_dir, 
            self.plot_subdir,
            )
        
        self.Main_model_data_full_dir = os.path.join(
            self.Main_model_data_dir, 
            self.Model_dir,
            )
    
        self.Main_Model_dashboard_viz_full_dir = os.path.join(
            self.Main_output_run_full_dir,
            self.dashboard_viz_subdir,
            )

        self.Main_Test_results_full_dir = os.path.join(
            os.path.dirname(self.Main_output_dir),
            "Test_Results",
        )
        
        self.Code_configuration_optimal_dir = os.path.join(
            self.Code_configuration_main_dir,
            self.Code_configuration_optimal_subdir,
        )
        
        self.Code_configuration_optimal_history_dir = os.path.join(
            self.Code_configuration_optimal_dir,
            self.Code_configuration_optimal_history_subdir,
        )
        return

###########################################################################################
    def ensure_all_output_directories(self):
        """
        Create every output/config directory that may be used by a run.
        """
        required_dirs = [
            self.Main_output_run_shared_dir,
            self.Main_imputed_data_shared_dir,
            self.Main_Model_training_full_dir,
            self.Main_Model_training_plots_full_dir,
            self.Main_Model_training_evaluation_full_dir,
            self.Main_Model_training_evaluation_plot_full_dir,
            self.Main_Model_training_evaluation_plot_histogram_full_dir,
            self.Main_Model_training_evaluation_plot_training_loss_full_dir,
            self.Main_output_run_full_dir,
            self.Main_output_run_plots_full_dir,
            self.Main_output_run_status_full_dir,
            self.Main_Model_dashboard_viz_full_dir,
            self.Main_model_data_full_dir,
            self.Main_Test_results_full_dir,
            self.Code_configuration_optimal_dir,
            self.Code_configuration_optimal_history_dir,
            self.Second_dashboard_viz_file_output_dir,
        ]

        for ddir in required_dirs:
            if ddir and not os.path.exists(ddir):
                os.makedirs(ddir, exist_ok=True)
                mod775 = (
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                    stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                    stat.S_IROTH | stat.S_IXOTH
                )
                os.chmod(ddir, mod775)
                self.logger.info('Directory = {msg}'.format(msg=ddir).ljust(self.justif-7,'.') + 'CREATED')
        return

###########################################################################################
    def ComputeDirectories(self):
        ''' 
        This is where the directory names are computed
        
        '''
        # Here is the computation of the main simulation dir

        ##########
        self.Output_run_status_dir = "status"
        
        ##########
        self.SimulationDir = self.SimulationDirPattern.format(rundate = self.start_date_utc.strftime('%Y%m%d%H'),
                                                            tstamp = self.timestamp_utc.strftime('%Y%m%d_%H%M%S')
                 )

        self.input_var_dir = self.var_to_predict[0]
        ##########
        # additional vars dir
        if (len(self.additional_var_to_select) > 1):
                self.additional_var_dir = '_'.join(self.additional_var_to_select)

        elif (len(self.additional_var_to_select) == 0):
                self.additional_var_dir = "OBS_only_test"
        else: 
                self.additional_var_dir = self.additional_var_to_select[0]

        ##########
        # train/forecast dir
        if self.train_model:
            self.train_forecast_dir = "Training"
        else:
            self.train_forecast_dir = "Forecast"

        
        return

###########################################################################################
         
###########################################################################################
    def Save_State_File(self):
        self.StateFileName = self.StateFileNameTemplate.format(
                     rundate = self.start_date_utc.strftime('%Y%m%d%H'),
                     tstamp=self.timestamp_utc.strftime('%Y%m%d_%H%M%S')
                     )
        self.StateFile = SF.StateFile(self.logger, self.justif, self.StateFullPath, self.StateFileName)
        self.StateFile.Create_state_dir()
        self.StateFile.Write_state_file(self.CCAM_config)
        return
###########################################################################################
    def Run_Everything(self, DomainToRun, PushTheButton, External_Job_Number_Trigger):

        ####
        #Compute Directories
        self.ComputeDirectories()
        
        self.Init_CCAMRuns(DomainToRun)
        ListofSbatchFilename = []
        ListofSbatchFilename_Pcc2hist = []
    
        for run in  self.CCAM_Runs:
            SbatchFilename, SBatchConfig, SbatchFilename_Pcc2hist, SBatchConfig_Pcc2hist = PC.Prep_CCAM(run)
            ListofSbatchFilename.append(SbatchFilename)
            ListofSbatchFilename_Pcc2hist.append(SbatchFilename_Pcc2hist)
        
            #export the config for the state file
            self.CCAM_config[run.domain_name] = run.ExportConfig()

        SBatchConfig.WriteScriptSubmit(ListofSbatchFilename, 'Submit_ccam')
        SBatchConfig.WriteScriptSubmit(ListofSbatchFilename_Pcc2hist, 'Submit_CC2H')
        SBatchConfig.WriteScriptSubmit_branching(ListofSbatchFilename, ListofSbatchFilename_Pcc2hist, External_Job_Number_Trigger=External_Job_Number_Trigger)
    
        #Save the state
        self.Save_State_File()
        
        #push the button
        if PushTheButton:
            SBatchConfig.RunScriptSubmit()
        return   
