"""
..  module:: Output_manager
    :platform: Unix
    :synopsis: Definition of the basic object class to output .

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
.. moduleauthor:: Sagthitharan Karalasingham <d9630120@umail.usq.edu.au>
"""

import sys
import os
import stat
import pickle
import shutil
import subprocess
import numpy as np
import pandas as pd

import Core_iHPC.Outputs.Yaml_file_writer as YFW

###########################################################################################
class Output_manager_Class(object):
    """ 
    This class defines a Output_Class, that contains the capacity to manage the outputs.

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
        self.logger.info('Configuring the Outputs'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))


        # self.base_run_dir = Configuration.base_run_dir
        # self.base_output_dir = Configuration.base_output_dir
        # self.result_file_subdir = Configuration.result_file_subdir
        # self.plot_subdir = Configuration.plot_subdir
        
        self.Main_training_dir = Configuration.Main_training_dir
        self.Main_output_dir = Configuration.Main_output_dir
        self.Main_model_data_dir = Configuration.Main_model_data_dir
        
        self.Model_dir = Configuration.Model_dir
        self.plot_subdir = Configuration.plot_subdir
        self.Run_configuration_dir = Configuration.Run_configuration_dir        
        self.input_var_dir = Configuration.input_var_dir        
        
        self.additional_var_to_select = Configuration.additional_var_to_select
        self.selected_region = (
            getattr(Configuration, "selected_region_long_name", None)
            or Configuration.selected_region
        )
        
        # self.Compute_output_dirs()
        
        # self.forecast_training_output_filename_template = '{region}_{mode}_{parameters}_{inputs}_{outputs}.csv'
        # self.forecast_training_forward_output_filename_template = '{region}_{mode}_{parameters}_{inputs}_{outputs}_{iter}.csv'
        # self.imputation_output_filename_template = 'Imputed_{imputation_method}_{region}_{var}_{additional_vars}.csv'
        # self.obs_output_filename_template = 'Allobs_{raw_processed}_{input_stream}_{region}_{var}_{additional_vars}.csv'
        self.forecast_output_filename_template = Configuration.forecast_output_filename_template
        self.forecast_forward_output_filename_template = Configuration.forecast_forward_output_filename_template
        self.training_output_filename_template = Configuration.training_output_filename_template
        self.training_forward_output_filename_template = Configuration.training_forward_output_filename_template
        self.imputation_output_filename_template = Configuration.imputation_output_filename_template
        self.obs_output_filename_template = Configuration.obs_output_filename_template
        self.metrics_output_filename_template = Configuration.metrics_output_filename_template
        self.metrics_bnn_output_filename_template = Configuration.metrics_bnn_output_filename_template
        self.dashboard_viz_filename_template = Configuration.dashboard_viz_filename_template        
        
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
    def _mirror_dashboard_directory(self, source_dir, target_dir):
        """
        Mirror dashboard CSV files from the per-run dashboard directory to the
        shared dashboard directory.

        The run-specific Dashboard folder is the authoritative source; this
        copy step keeps the operational `AI_dashboard_files` directory in sync.
        """
        if not source_dir or not target_dir:
            return

        if os.path.abspath(source_dir) == os.path.abspath(target_dir):
            return

        if not os.path.isdir(source_dir):
            return

        self.MakeDir(target_dir)
        for filename in os.listdir(source_dir):
            if not filename.lower().endswith(".csv"):
                continue

            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            if not os.path.isfile(source_path):
                continue

            shutil.copy2(source_path, target_path)
            self.Change_permissions(target_path)

    def _generate_live_dashboard_files(self):
        """
        Run the shared live-dashboard generator after the dashboard CSV is written.
        """
        script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "scripts",
                "generate_live_dashboards.py",
            )
        )
        if not os.path.isfile(script_path):
            self.logger.warning(
                "Live dashboard script".ljust(self.justif - 2, '.') + "NOT FOUND"
            )
            return False

        env = os.environ.copy()
        target_list = getattr(self.Configuration, "var_to_predict", None) or []
        target = target_list[0] if target_list else None
        start_date_aest = getattr(self.Configuration, "start_date_aest", None)

        if getattr(self.Configuration, "selected_region", None) is not None:
            env["LIVE_DASHBOARD_SELECTED_REGION"] = str(self.Configuration.selected_region)
        if target is not None:
            env["LIVE_DASHBOARD_TARGET"] = str(target)
        if getattr(self.Configuration, "n_steps_in", None) is not None:
            env["LIVE_DASHBOARD_INPUTS"] = str(self.Configuration.n_steps_in)
        if getattr(self.Configuration, "n_steps_out", None) is not None:
            env["LIVE_DASHBOARD_OUTPUTS"] = str(self.Configuration.n_steps_out)
        if getattr(self.Configuration, "Model_dir", None) is not None:
            env["LIVE_DASHBOARD_MODEL"] = str(self.Configuration.Model_dir)
        if start_date_aest is not None:
            env["LIVE_DASHBOARD_DATE"] = start_date_aest.strftime("%Y%m%dAEST")

        try:
            subprocess.run([sys.executable, script_path], check=True, env=env)
            return True
        except Exception as exc:
            self.logger.warning(
                "Live dashboard files".ljust(self.justif - 2, '.') + f"SKIPPED ({exc})"
            )
            return False

###########################################################################################
    def Write_pickle_files(self, 
            ytest_s, yhat_s, y_test_s_dt, ytest_w, yhat_w, y_test_w_dt,
            ):
        ''' This method write the results of the  forecast in a pickle file.
        
        '''

        n_steps_in = self.Configuration.n_steps_in
        n_steps_out = self.Configuration.n_steps_out
        pollutant = self.Configuration.var_to_predict

        ##### Using filetemplate to save the objects easily ####
        filenametemplate= 'y{test}_{se}_{pol}_{nstepin}_{nstepout}.pickle'
        filerun = "toto"
        filename_summer_test = filenametemplate.format(
                pol=pollutant, nstepin = n_steps_in, 
                nstepout=n_steps_out, filerun=filerun,
                se='s', test="test"
        )
        filename_summer_forecast = filenametemplate.format(
                pol=pollutant, nstepin = n_steps_in, 
                nstepout=n_steps_out, filerun=filerun,
                se='s', test="hat"
        )
        filename_summer_dt = filenametemplate.format(
                pol=pollutant, nstepin = n_steps_in, 
                nstepout=n_steps_out, filerun=filerun,
                se='s', test="test_dt"
        )


        filename_winter_test = filenametemplate.format(
                pol=pollutant, nstepin = n_steps_in, 
                nstepout=n_steps_out, filerun=filerun,
                se='w', test="test"
        )
        filename_winter_forecast = filenametemplate.format(
                pol=pollutant, nstepin = n_steps_in, 
                nstepout=n_steps_out, filerun=filerun,
                se='w', test="hat"
        )

        filename_winter_dt = filenametemplate.format(
                pol=pollutant, nstepin = n_steps_in, 
                nstepout=n_steps_out, filerun=filerun,
                se='w', test="test_dt"
        )
        self.MakeDir(os.path.join(self.base_output_dir, self.result_file_subdir))
        with open(os.path.join(self.base_output_dir, self.result_file_subdir, filename_summer_test), 'wb') as file:
                pickle.dump(ytest_s, file, protocol=pickle. HIGHEST_PROTOCOL)
        with open(os.path.join(self.base_output_dir, self.result_file_subdir, filename_summer_forecast), 'wb') as file:
                pickle.dump(yhat_s, file, protocol=pickle. HIGHEST_PROTOCOL)
        with open(os.path.join(self.base_output_dir, self.result_file_subdir, filename_summer_dt), 'wb') as file:
                pickle.dump(y_test_s_dt, file, protocol=pickle. HIGHEST_PROTOCOL)

        with open(os.path.join(self.base_output_dir, self.result_file_subdir, filename_winter_test), 'wb') as file:
                pickle.dump(ytest_w, file, protocol=pickle. HIGHEST_PROTOCOL)
        with open(os.path.join(self.base_output_dir, self.result_file_subdir, filename_winter_forecast), 'wb') as file:
                pickle.dump(yhat_w, file, protocol=pickle. HIGHEST_PROTOCOL)
        with open(os.path.join(self.base_output_dir, self.result_file_subdir, filename_winter_dt), 'wb') as file:
                pickle.dump(y_test_w_dt, file, protocol=pickle. HIGHEST_PROTOCOL)
        return

###########################################################################################
    def output_forecast(self, list_output_pd):
        """
        method to output the necessary for the produced forecast
        """ 
        
        for index, output_pd in enumerate(list_output_pd):
            if self.Configuration.train_model:               
                output_filename_template = self.training_output_filename_template
                output_dir = self.Configuration.Main_Model_training_full_dir
            else:
                output_filename_template = self.forecast_output_filename_template
                output_dir = self.Configuration.Main_output_run_full_dir
            
            # Extract station name from DataFrame
            station_name = output_pd['station'].iloc[0]
            
            output_filename = output_filename_template.format(
                region = self.selected_region,
                mode = self.Configuration.train_forecast_dir.lower(),   
                inputs = self.Configuration.n_steps_in, 
                outputs = self.Configuration.n_steps_out, 
                parameters = self.Configuration.var_to_predict[0],
                station = station_name,
                date = self.Configuration.start_date_aest.strftime("%Y%m%dAEST"),
            )
            
            self.MakeDir(output_dir)
            output_file_full_path = os.path.join(output_dir, output_filename)
            output_pd.to_csv(output_file_full_path)
            self.Change_permissions(output_file_full_path)
            self.logger.info('Forecast file  = {msg}'.format(msg=output_filename).ljust(self.justif-7,'.') + 'WRITTEN')
        
        return

#################################################################################
    def output_forecast_forward(self, list_output_pd, iteration):
        """
        method to output the necessary for the produced forecast
        """ 
    
        for index, output_pd in enumerate(list_output_pd):
            
            if self.Configuration.train_model:               
                output_filename_template = self.training_forward_output_filename_template
                output_dir = self.Configuration.Main_Model_training_full_dir
            else:
                output_filename_template = self.forecast_forward_output_filename_template
                output_dir = self.Configuration.Main_output_run_full_dir
            
            # Extract station name from DataFrame
            station_name = output_pd['station'].iloc[0]
            
            output_filename = output_filename_template.format(
                region = self.selected_region, 
                mode = self.Configuration.train_forecast_dir.lower(),
                inputs = self.Configuration.n_steps_in, 
                outputs = self.Configuration.n_steps_out, 
                parameters = self.Configuration.var_to_predict[0],
                station = station_name,
                iter = iteration,
                model = self.Configuration.Model_dir,
                date = self.Configuration.start_date_aest.strftime("%Y%m%dAEST"),
            )
    
            self.MakeDir(output_dir)
            output_file_full_path = os.path.join(output_dir, output_filename)
            output_pd.to_csv(output_file_full_path)
            self.Change_permissions(output_file_full_path)
    
            self.logger.info('Forecast file  = {msg}'.format(msg=output_file_full_path).ljust(self.justif-7,'.') + 'WRITTEN')
        
        return

#################################################################################
    def output_imputed_data(self, output_pd, station_name=None):
        """
        Method to output the imputed data for the produced forecast.
        Supports both per-station and combined output files.
        
        Parameters
        -----------
        output_pd : pd.DataFrame
            Imputed dataframe to save
        station_name : str, optional
            Station name for per-station files. If None, saves combined file.
        """ 
        # Determine directory
        if self.Configuration.train_model:
            dir = self.Configuration.Main_Model_training_full_dir
        else:
            dir = self.Configuration.Main_output_run_full_dir    
        
        # Build filename based on whether it's per-station or combined
        if station_name:
            # Per-station filename
            output_filename_template = 'Imputed_{region}_{var}_{station}.csv'
            filename = output_filename_template.format(
                region=self._region_token(),
                var=self.Configuration.input_var_dir,
                station=station_name
            )
        else:
            # Combined filename (original format)
            output_filename_template = self.imputation_output_filename_template
            filename = output_filename_template.format(
                region=self._region_token(),
                var=self.Configuration.input_var_dir,
                additional_vars=self.Configuration.additional_var_dir, 
            )
        
        # Save file
        filename_full_path = os.path.join(dir, filename)
        self.MakeDir(dir)
        output_pd.to_csv(filename_full_path)
        self.Change_permissions(filename_full_path)
        
        # Log message
        if station_name:
            self.logger.info('Imputed data ({station}) = {msg}'.format(
                station=station_name, msg=filename).ljust(self.justif-7,'.') + 'WRITTEN')
        else:
            self.logger.info('Imputed data = {msg}'.format(
                msg=filename).ljust(self.justif-7,'.') + 'WRITTEN')
        
        return
#################################################################################
    def output_obs_data(self, output_pd, input_stream):
        """
        method to output the necessary for the produced forecast
        """ 

        # output_filename_template = 'Allobs_{raw_processed}_{input_stream}_{region}_{var}_{additional_vars}.csv'
        output_filename_template = self.obs_output_filename_template        
        
        # Processed obs goes to the shared API input cache when configured.
        if self.Configuration.Main_output_run_shared_dir is not None:
            dir = self.Configuration.Main_output_run_shared_dir
        else:
            dir = self.Configuration.Main_output_run_full_dir
        
        filename = output_filename_template.format(
            raw_processed = "processed",
            input_stream = input_stream,
            region = self._region_token(),
            var = self.Configuration.input_var_dir,
            additional_vars = self.Configuration.additional_var_dir, 
            )
        filename_full_path = os.path.join(dir, filename)
        self.MakeDir(dir)
        if os.path.isfile(filename_full_path):
            self.logger.info('Existing processed data found, merging with new output'.ljust(self.justif,'.'))
            try:
                existing_pd = pd.read_csv(filename_full_path, index_col=0, parse_dates=True)
            except pd.errors.EmptyDataError:
                self.logger.info(
                    f'Existing processed file is empty at {filename_full_path}. Overwriting with new output'.ljust(
                        self.justif, '.'
                    ) + 'OK'
                )
                existing_pd = None
            if existing_pd is not None and not existing_pd.empty:
                output_pd = pd.concat([existing_pd, output_pd], axis=0)
                output_pd = output_pd[~output_pd.index.duplicated(keep='last')].sort_index()
        output_pd.to_csv(filename_full_path)
        self.Change_permissions(filename_full_path)
        self.logger.info('OBS data = {msg}'.format(msg=filename).ljust(self.justif-7,'.') + 'WRITTEN')
        
        return

#################################################################################
    def output_metrics(self, evaluation_metrics_dict, iteration):
        """
        method to output the necessary for the produced forecast
        """ 
        
        for key, pd_table in evaluation_metrics_dict.items():
            
            dict_key_processed = key.rsplit("_", 1)[0]

            if 'BNN' in dict_key_processed:
                output_filename_template = self.metrics_bnn_output_filename_template
            else:
                output_filename_template = self.metrics_output_filename_template

            filename = output_filename_template.format(
                region = self._region_token(),
                metrics = dict_key_processed,
                var = self.Configuration.input_var_dir,
                inputs = self.Configuration.n_steps_in,
                outputs = self.Configuration.n_steps_out,
                additional_vars = self.Configuration.additional_var_dir,
                iteration=iteration,
                model = self.Configuration.get_metrics_model_token(),
                )
            dir = self.Configuration.Main_Model_training_evaluation_full_dir
            
            filename_full_path = os.path.join(dir, filename)
            self.MakeDir(dir)
            pd_table.to_csv(filename_full_path)
            self.Change_permissions(filename_full_path)
            self.logger.info('Evaluation metrics = {msg}'.format(msg=filename).ljust(self.justif-7,'.') + 'WRITTEN')
            
        return   

#################################################################################
    def _region_token(self):
        return (
            getattr(self.Configuration, "selected_region_long_name", None)
            or self.Configuration.selected_region
        )

#################################################################################
    def _dashboard_filename(self):
        config_name = os.path.splitext(
            os.path.basename(getattr(self.Configuration, "yaml_config_filename", "") or "run")
        )[0]
        return self.dashboard_viz_filename_template.format(
            region=self._region_token(),
            var=self.Configuration.input_var_dir,
            inputs=self.Configuration.n_steps_in,
            outputs=self.Configuration.n_steps_out,
            additional_vars=self.Configuration.additional_var_dir,
            model=self.Configuration.Model_dir,
            config=config_name,
            date=self.Configuration.start_date_aest.strftime("%Y%m%d"),
            run_hour=self._dashboard_run_hour(),
        )

#################################################################################
    def _dashboard_run_hour(self):
        timestamp = (
            getattr(self.Configuration, "timestamp_aest", None)
            or getattr(self.Configuration, "timestamp_aedt", None)
            or getattr(self.Configuration, "start_date_aest", None)
        )
        if timestamp is not None and hasattr(timestamp, "strftime"):
            return timestamp.strftime("%HAEST")
        return "00AEST"

#################################################################################
    def _dashboard_metrics_filename(self):
        dashboard_filename = self._dashboard_filename()
        base_name, ext = os.path.splitext(dashboard_filename)
        return f"{base_name}_training_metrics{ext}"

#################################################################################
    def _test_results_dir(self):
        base_dir = getattr(self.Configuration, "Main_Test_results_full_dir", None)
        if not base_dir:
            base_dir = os.path.join(os.path.dirname(self.Configuration.Main_output_dir), "Test_Results")

        region = str(self._region_token())
        variable = self.Configuration.input_var_dir
        model = str(self.Configuration.Model_dir)
        simulation_dir = getattr(self.Configuration, "SimulationDir", None)
        if simulation_dir:
            return os.path.join(base_dir, region, variable, model, simulation_dir)
        return os.path.join(base_dir, region, variable, model)

#################################################################################
    def _test_results_filename(self, test_pd):
        horizon = getattr(self.Configuration, "n_steps_out", None)
        if horizon is None and test_pd is not None and "forecast_hours" in test_pd.columns:
            try:
                horizon = int(pd.to_numeric(test_pd["forecast_hours"], errors="coerce").max())
            except Exception:
                horizon = None
        horizon_token = str(horizon if horizon is not None else "na")
        return "{region}_{var}_{inputs}_{horizon}_test_results_{model}_{date}.csv".format(
            region=str(self._region_token()),
            var=self.Configuration.input_var_dir,
            inputs=self.Configuration.n_steps_in,
            horizon=horizon_token,
            model=str(self.Configuration.Model_dir),
            date=self.Configuration.start_date_aest.strftime("%Y%m%dAEST"),
        )

#################################################################################
    def _normalise_dashboard_name(self, value):
        return ''.join(ch for ch in str(value).upper() if ch.isalnum())

#################################################################################
    def _dashboard_target_column(self, station_df):
        candidates = []

        var_data = getattr(self.Configuration, 'var_data_list_from_input_pd', None)
        if var_data:
            candidates.extend(var_data)

        candidates.extend([
            getattr(self.Configuration, 'input_var_dir', None),
            self.Configuration.var_to_predict[0],
        ])

        for candidate in candidates:
            if candidate in station_df.columns:
                return candidate

        normalised_candidates = {
            self._normalise_dashboard_name(candidate)
            for candidate in candidates
            if candidate is not None
        }
        for column in station_df.columns:
            if self._normalise_dashboard_name(column) in normalised_candidates:
                return column

        self.logger.warning(
            "Cannot find target pollutant column in station history: {cols}".format(
                cols=list(station_df.columns)
            ).ljust(self.justif - 2, '.') + 'SKIP'
        )
        return None

#################################################################################
    def _combined_history_for_dashboard(self, station_history_dict):
        if not station_history_dict:
            return None

        dashboard_var = self.Configuration.input_var_dir
        station_order = getattr(self.Configuration, 'dpie_output_station_list', None)
        if not station_order:
            station_order = sorted(station_history_dict.keys())

        history_series = []
        for station in station_order:
            station_df = station_history_dict.get(station)
            if station_df is None or station_df.empty:
                continue

            target_column = self._dashboard_target_column(station_df)
            if target_column is None or target_column not in station_df.columns:
                continue
            station_series = station_df[target_column].copy()
            station_series.name = f"{dashboard_var}_{station}"
            history_series.append(station_series)

        if not history_series:
            return None

        history_pd = pd.concat(history_series, axis=1).sort_index()
        history_pd = history_pd.tail(self.Configuration.n_steps_in)
        history_pd['forecast_hours'] = list(range(-(len(history_pd) - 1), 1))
        history_pd.index.name = 'datetime'
        return history_pd

#################################################################################
    def _combined_forecast_for_dashboard(self, list_output_pd):
        if list_output_pd is None:
            return None

        if isinstance(list_output_pd, pd.DataFrame):
            output_tables = [list_output_pd]
        else:
            output_tables = list(list_output_pd)

        if len(output_tables) == 0:
            return None

        dashboard_var = self.Configuration.input_var_dir
        metadata_columns = {
            'forecast_hours',
            'forecast_number',
            'station',
            'model_version',
            'generated_at',
        }

        forecast_series = []
        forecast_hours = None
        for output_pd in output_tables:
            if output_pd is None or output_pd.empty:
                continue

            station = None
            if 'station' in output_pd.columns and not output_pd['station'].empty:
                station = output_pd['station'].iloc[0]

            value_columns = [col for col in output_pd.columns if col not in metadata_columns]
            if station is not None:
                preferred_column = f"{dashboard_var}_{station}"
                if preferred_column in value_columns:
                    value_columns = [preferred_column]

            if not value_columns:
                continue

            value_column = value_columns[0]
            if station is None and '_' in value_column:
                station = value_column.split('_', 1)[1]

            output_series = output_pd[value_column].copy()
            if station is not None:
                output_series.name = f"{dashboard_var}_{station}"
            else:
                output_series.name = value_column
            forecast_series.append(output_series)

            if forecast_hours is None and 'forecast_hours' in output_pd.columns:
                forecast_hours = output_pd['forecast_hours'].copy()

        if not forecast_series:
            return None

        forecast_pd = pd.concat(forecast_series, axis=1).sort_index()
        if forecast_hours is not None:
            forecast_pd['forecast_hours'] = forecast_hours.reindex(forecast_pd.index).values
        else:
            forecast_pd['forecast_hours'] = list(range(1, len(forecast_pd) + 1))
        forecast_pd.index.name = 'datetime'
        return forecast_pd

#################################################################################
    def _wide_forecast_for_dashboard(self, list_output_pd):
        """
        Build a wide forecast table with one row per timestamp and one column
        per station, matching the O3 / PM10 dashboard layout.
        """
        if list_output_pd is None:
            return None

        if isinstance(list_output_pd, pd.DataFrame):
            output_tables = [list_output_pd]
        else:
            output_tables = list(list_output_pd)

        if len(output_tables) == 0:
            return None

        dashboard_var = self.Configuration.input_var_dir
        metadata_columns = {
            'forecast_hours',
            'forecast_number',
            'station',
            'model_version',
            'generated_at',
        }

        forecast_series = []
        forecast_hours = None

        for output_pd in output_tables:
            if output_pd is None or output_pd.empty:
                continue

            station_df = output_pd.copy()

            if 'datetime' in station_df.columns:
                station_df['datetime'] = pd.to_datetime(station_df['datetime'])
                station_df = station_df.set_index('datetime', drop=True)
            elif 'timestamp' in station_df.columns:
                station_df['timestamp'] = pd.to_datetime(station_df['timestamp'])
                station_df = station_df.set_index('timestamp', drop=True)
            elif not isinstance(station_df.index, pd.DatetimeIndex):
                continue

            station = None
            if 'station' in station_df.columns and not station_df['station'].empty:
                station = station_df['station'].iloc[0]

            value_columns = [col for col in station_df.columns if col not in metadata_columns]
            if station is not None:
                preferred_column = f"{dashboard_var}_{station}"
                if preferred_column in value_columns:
                    value_columns = [preferred_column]

            if not value_columns:
                continue

            value_column = value_columns[0]
            if station is None and '_' in value_column:
                station = value_column.split('_', 1)[1]

            station_series = station_df[value_column].copy()
            station_series.name = f"{dashboard_var}_{station}" if station is not None else value_column
            forecast_series.append(station_series)

            if forecast_hours is None and 'forecast_hours' in station_df.columns:
                forecast_hours = station_df['forecast_hours'].copy()

        if not forecast_series:
            return None

        forecast_pd = pd.concat(forecast_series, axis=1).sort_index()
        forecast_pd.index.name = 'datetime'
        forecast_pd = forecast_pd.reset_index()

        if forecast_hours is not None:
            if not isinstance(forecast_hours.index, pd.DatetimeIndex):
                forecast_hours.index = pd.to_datetime(forecast_hours.index)
            forecast_hours = forecast_hours.reindex(pd.to_datetime(forecast_pd['datetime']))
            forecast_pd['forecast_hours'] = forecast_hours.values
        elif 'forecast_hours' not in forecast_pd.columns:
            forecast_pd['forecast_hours'] = list(range(1, len(forecast_pd) + 1))

        return forecast_pd

#################################################################################
    def _limit_dashboard_forecast_horizon(self, forecast_pd):
        """
        Keep only forecast rows 1..n_steps_out for live dashboard CSVs.
        Historical rows are produced separately with negative forecast_hours.
        """
        if forecast_pd is None or forecast_pd.empty:
            return forecast_pd

        horizon = getattr(self.Configuration, "n_steps_out", None)
        if horizon is None:
            return forecast_pd

        try:
            horizon = int(horizon)
        except Exception:
            return forecast_pd

        if horizon <= 0:
            return forecast_pd

        forecast_pd = forecast_pd.copy()
        if "forecast_hours" not in forecast_pd.columns:
            forecast_pd["forecast_hours"] = list(range(1, len(forecast_pd) + 1))

        forecast_hours = pd.to_numeric(forecast_pd["forecast_hours"], errors="coerce")
        clipped_pd = forecast_pd[(forecast_hours >= 1) & (forecast_hours <= horizon)].copy()

        if clipped_pd.empty:
            clipped_pd = forecast_pd.head(horizon).copy()
            clipped_pd["forecast_hours"] = list(range(1, len(clipped_pd) + 1))

        return clipped_pd

#################################################################################
    def output_combined_dashboard_viz_file(self,
                                          list_output_pd,
                                          station_history_dict=None,
                                          evaluation_pd=None,
                                          forecast_pd=None):
        """
        Write a forecast-ready CSV:
        datetime, {POLLUTANT}_{STATION}..., forecast_hours

        The history block is labelled -(n_inputs - 1)..0. The forecast block
        keeps the model output forecast_hours. If evaluation metrics are
        available, they are appended after a blank line.
        """
        history_pd = self._combined_history_for_dashboard(station_history_dict)
        wide_forecast_pd = self._wide_forecast_for_dashboard(list_output_pd)
        if wide_forecast_pd is not None:
            forecast_pd = wide_forecast_pd
        elif forecast_pd is None:
            forecast_pd = self._combined_forecast_for_dashboard(list_output_pd)
        forecast_pd = self._limit_dashboard_forecast_horizon(forecast_pd)

        if history_pd is None and forecast_pd is None:
            raise ValueError("No history or forecast data available for dashboard CSV")

        dashboard_viz_filename = self._dashboard_filename()
        dashboard_dirs = [
            self.Configuration.Main_Model_dashboard_viz_full_dir,
            self.Configuration.Second_dashboard_viz_file_output_dir,
        ]

        for dashboard_viz_dir in dashboard_dirs:
            dashboard_viz_filename_fullpath = os.path.join(
                dashboard_viz_dir,
                dashboard_viz_filename,
            )

            self.MakeDir(dashboard_viz_dir)
            data_blocks = []
            if history_pd is not None:
                history_block = history_pd.copy()
                history_block.index.name = "datetime"
                history_block = history_block.reset_index()
                data_blocks.append(history_block)
            if forecast_pd is not None:
                forecast_block = forecast_pd.copy()
                forecast_block.index.name = "datetime"
                if "datetime" in forecast_block.columns:
                    forecast_block = forecast_block.set_index("datetime", drop=True)
                forecast_block = forecast_block.reset_index()
                data_blocks.append(forecast_block)

            if not data_blocks:
                raise ValueError("No forecast or metrics data available for dashboard CSV")

            combined_data_pd = pd.concat(data_blocks, axis=0, ignore_index=True, sort=False)
            blocks = [combined_data_pd]

            if evaluation_pd is not None and not evaluation_pd.empty:
                blocks.append(evaluation_pd.copy())

            with open(dashboard_viz_filename_fullpath, "w", newline="") as f:
                blocks[0].to_csv(f, index=False)
                for block in blocks[1:]:
                    f.write("\n")
                    block.to_csv(f, index=False)

            self.Change_permissions(dashboard_viz_filename_fullpath)

        self._mirror_dashboard_directory(
            self.Configuration.Main_Model_dashboard_viz_full_dir,
            self.Configuration.Second_dashboard_viz_file_output_dir,
        )

        self.logger.info(
            'Dashboard file = {msg}'.format(msg=dashboard_viz_filename).ljust(
                self.justif - 7, '.') + 'WRITTEN')

        return forecast_pd

#################################################################################
    def output_dashboard_metrics_file(self, metrics_pd):
        """
        Write a dashboard CSV containing just the station metrics table.

        This is used when training mode has no forecast dataframe, but we still
        want a dashboard artifact saved alongside the forecast-mode file.
        """
        if metrics_pd is None or metrics_pd.empty:
            self.logger.warning(
                'Dashboard metrics'.ljust(self.justif - 2, '.') + 'SKIPPED')
            return None

        dashboard_viz_filename = self._dashboard_metrics_filename()
        dashboard_dirs = [
            self.Configuration.Main_Model_dashboard_viz_full_dir,
            self.Configuration.Second_dashboard_viz_file_output_dir,
        ]

        for dashboard_viz_dir in dashboard_dirs:
            dashboard_viz_filename_fullpath = os.path.join(
                dashboard_viz_dir,
                dashboard_viz_filename,
            )
            self.MakeDir(dashboard_viz_dir)
            metrics_pd.to_csv(dashboard_viz_filename_fullpath, index=False)
            self.Change_permissions(dashboard_viz_filename_fullpath)

        self._mirror_dashboard_directory(
            self.Configuration.Main_Model_dashboard_viz_full_dir,
            self.Configuration.Second_dashboard_viz_file_output_dir,
        )
        
        self.logger.info(
            'Dashboard file = {msg}'.format(msg=dashboard_viz_filename).ljust(
                self.justif - 7, '.') + 'WRITTEN')

        return metrics_pd

#################################################################################
    def output_dashboard_viz_file(self, 
                                forecast_pd,
                                evaluation_pd,
                                ):
        """
        method to output the necessary for the produced forecast
        """ 
        dashboard_viz_filename = self._dashboard_filename()
        
        dashboard_viz_dir = self.Configuration.Main_Model_dashboard_viz_full_dir
        
        dashboard_viz_filename_fullpath =  os.path.join(
            dashboard_viz_dir,
            dashboard_viz_filename)

        self.MakeDir(dashboard_viz_dir)
        
        ######################### save forecast   ##############################

        forecast_pd.to_csv(dashboard_viz_filename_fullpath)

        ######################### 1st way to insert statistic in csv ##############################
        
        # Append the second DataFrame (df2) as a footer to the same CSV file
        if evaluation_pd is not None:
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
        

        self.Change_permissions(dashboard_viz_filename_fullpath)

        self._mirror_dashboard_directory(
            self.Configuration.Main_Model_dashboard_viz_full_dir,
            self.Configuration.Second_dashboard_viz_file_output_dir,
        )
        
        ################
        # second file to operational dashboard viz dir
        
        dashboard_viz_dir = self.Configuration.Second_dashboard_viz_file_output_dir
        dashboard_viz_filename_fullpath =  os.path.join(
            dashboard_viz_dir,
            dashboard_viz_filename)

        self.MakeDir(dashboard_viz_dir)
        
        ######################### save forecast   ##############################

        forecast_pd.to_csv(dashboard_viz_filename_fullpath)

        ######################### 1st way to insert statistic in csv ##############################
        
        # Append the second DataFrame (df2) as a footer to the same CSV file
        if evaluation_pd is not None:
            with open(dashboard_viz_filename_fullpath, 'a') as f:
                f.write('\n')  # Add a newline to separate df1 and df2
                evaluation_pd.to_csv(f, index=False)
        self.Change_permissions(dashboard_viz_filename_fullpath)
        
        
        self.logger.info('Dashboard file = {msg}'.format(msg=dashboard_viz_filename).ljust(self.justif-7,'.') + 'WRITTEN')
            
        return   

#################################################################################
    def output_test_results_file(self, test_pd):
        """
        Save the per-station testing dataframe for a run.

        The output is organised under AI_Runs/Test_Results so every region and
        forecast horizon gets a durable CSV with the observed/predicted test
        series.
        """
        if test_pd is None or getattr(test_pd, "empty", True):
            self.logger.warning(
                'Test results'.ljust(self.justif - 2, '.') + 'SKIPPED')
            return None

        test_dir = self._test_results_dir()
        test_filename = self._test_results_filename(test_pd)
        test_filepath = os.path.join(test_dir, test_filename)

        self.MakeDir(test_dir)

        output_pd = test_pd.copy()
        if "region" not in output_pd.columns:
            output_pd.insert(0, "region", self._region_token())
        if "variable" not in output_pd.columns:
            output_pd.insert(1, "variable", self.Configuration.input_var_dir)
        if "model" not in output_pd.columns:
            output_pd.insert(2, "model", self.Configuration.Model_dir)
        if "forecast_number" not in output_pd.columns:
            output_pd["forecast_number"] = getattr(self.Configuration, "n_steps_out", None)

        output_pd.to_csv(test_filepath, index=False)
        self.Change_permissions(test_filepath)

        self.logger.info(
            'Test results file = {msg}'.format(msg=test_filepath).ljust(
                self.justif - 7, '.') + 'WRITTEN')
        return test_filepath

#################################################################################
    def output_optimal_yaml_config_file(self, optimisation_history_dict, target_order_np,
                                ):
        Yml_writer = YFW.Yaml_file_writer_Class(self.Configuration,)
        nownownow = Yml_writer.write_yaml_optimal_config_file(self.Configuration.yaml_config_filename)
        # Also keep a stable "best" YAML alongside the optimisation history so
        # runtime config can consume it without guessing timestamps.
        best_target = None
        try:
            if target_order_np is not None and len(target_order_np) > 0:
                best_target = float(max(target_order_np))
        except Exception:
            best_target = None
        Yml_writer.write_yaml_optimal_best_config_file(
            self.Configuration.yaml_config_filename,
            best_target=best_target,
        )
        self.output_optimisation_history_file(optimisation_history_dict, target_order_np, nownownow,)
        
        return
#################################################################################
    def output_optimisation_history_file(self, optimisation_history_dict, target_order_np, nownownow,
                                ):
        
        
        reorganised_dict_list = []
        for iindex, val in enumerate(target_order_np):
            for ddict in optimisation_history_dict:
                if ddict["target"] == val:
                    reorganised_dict  = {
                        "index" : iindex,
                        "cost_function" : val,      
                        }
                    
                    update_dict = {
                            key: int(val2)
                            for key,val2 in ddict["params"].items()
                            }
                    reorganised_dict.update(update_dict)
                    reorganised_dict_list.append(reorganised_dict)
        # print("reorganised_dict_list", reorganised_dict_list)            
        optimisation_history_pd = pd.DataFrame.from_dict(reorganised_dict_list)
        
        # print(optimisation_history_pd)

        filename_template = "{ff}_optimisation_history_{date}.csv"
        self.MakeDir(self.Configuration.Code_configuration_optimal_history_dir)
        safe_name = os.path.basename(str(self.Configuration.yaml_config_filename)).replace(os.sep, "_")
        
        filename_fullpath = os.path.join(self.Configuration.Code_configuration_optimal_history_dir, 
                                         filename_template.format(
                                             ff=safe_name,
                                             date = nownownow
                                             )
                                         )
        
        optimisation_history_pd.to_csv(filename_fullpath, index=False)
        self.Change_permissions(filename_fullpath)
        self.logger.info('optimisation history file  = {msg}'.format(msg=filename_fullpath).ljust(self.justif-7,'.') + 'WRITTEN')
        
        return
        
###########################################################################################
    def output_vmd_decomposition(self, output_pd, target_variable, station_name):
        """
        Method to output VMD decomposed data per station.
        
        Parameters
        -----------
        output_pd : pd.DataFrame
            Decomposed dataframe with IMFs and residual
        target_variable : str
            Target variable name (e.g., 'PM2.5', 'PM10', 'Ozone')
        station_name : str
            Station name
        """
        # Get VMD output directory from configuration
        vmd_output_dir = getattr(self.Configuration, 'vmd_output_dir', './outputs/vmd')
        
        # Determine base directory
        if self.Configuration.train_model:
            base_dir = self.Configuration.Main_Model_training_full_dir
        else:
            base_dir = self.Configuration.Main_output_run_full_dir
        
        # Create decomposition subdirectory
        decomp_dir = os.path.join(base_dir, 'decomposition')
        
        # Build filename
        output_filename_template = 'IMF_{var}_{region}_{station}.csv'
        filename = output_filename_template.format(
            var=target_variable,
            region=self._region_token(),
            station=station_name
        )
        
        # Save file
        filename_full_path = os.path.join(decomp_dir, filename)
        self.MakeDir(decomp_dir)
        output_pd.to_csv(filename_full_path)
        self.Change_permissions(filename_full_path)
        
        # Log message
        self.logger.info('VMD data ({station}) = {msg}'.format(
            station=station_name, msg=filename).ljust(self.justif-7,'.') + 'WRITTEN')
        
        return

        
# ###########################################################################################
#     def make_plots(self, pd_tbl, show_plots=False, write_plots=False):
#         """
#         plot the results
#         """
#         self.Plots = PC.Plot_Class(self.logger, self.justif,
#                     os.path.join(self.base_output_dir,  self.result_dir), self.base_plot_subdir,
#                     pd_tbl,            )

#         self.Plots.plot_all(show_plots=show_plots, write_plots=write_plots)            

 
