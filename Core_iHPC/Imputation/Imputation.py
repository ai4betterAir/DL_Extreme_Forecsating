"""
..  module:: Imputation
    :platform: Unix
    :synopsis: Definition of the basic object class to configure and run data imputation.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
.. moduleauthor:: Sagthitharan Karalasingham <d9630120@umail.usq.edu.au>   
"""

import sys
import os
import stat
import numpy as np
import pandas as pd
import sklearn as SK
import sklearn.experimental as SKE
import sklearn.impute as SKI
import sklearn.preprocessing as SKP
import Core_iHPC.Outputs.Output_manager as OM
import Core_iHPC.Configuration.DPE_region_stations as DPERT
from .Impute import KNN as KNNI
from . import Impute as IMPUTE_MODELS


def resolve_imputer_class(method):
    """
    Resolve an imputation method name to the corresponding implementation class.

    This keeps the pipeline generic by centralising imputation model selection
    inside the Imputation package.
    """
    token = str(method or "").strip()
    if not token:
        token = "NONE"
    token_upper = token.upper()
    class_map = {
        "NONE": IMPUTE_MODELS.NoneImputation,
        "MICE": IMPUTE_MODELS.MICEImputation,
        "KNN": IMPUTE_MODELS.KNNImputation,
        "TEMPORALMICE": IMPUTE_MODELS.TemporalMICEImputation,
        "AQUISTIL": IMPUTE_MODELS.AQUISTILImputation,
    }
    if token_upper not in class_map:
        raise ValueError(
            "Unsupported imputation method {method}. Choose one of: {choices}".format(
                method=method,
                choices=", ".join(sorted(class_map)),
            )
        )
    return class_map[token_upper]

###########################################################################################
class Imputation_Class(object):
    """ 
    This class defines a Imputation_Class, that contains the capacity to impute data.
    This is the entry point for all imputation methods

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
        self.logger.info('Configuring the data imputation'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        return
        
###########################################################################################
    def extract_station_data(self, input_data_pd, var_to_predict=None):
        """
        Extract data organized by station from combined dataframe.
        Handles multi-word station names like CAMPBELLTOWN_WEST correctly.
        Uses DPE_region_stations class to get known stations according to corporate standards.
        
        Parameters
        -----------
        input_data_pd : pd.DataFrame
            Combined dataframe with columns like VAR_STATION
        var_to_predict : list, optional
            Variable to predict (e.g., ['O3'] or ['PM2.5']). If None, attempts to detect from data.
            
        Returns
        --------
        dict
            Dictionary with station names as keys and station dataframes as values
        """
        # Determine variable to predict if not provided
        if var_to_predict is None:
            # Try to detect from column names
            if any('O3_' in col for col in input_data_pd.columns):
                var_to_predict = ['O3']
                self.logger.info("Auto-detected variable: O3")
            else:
                var_to_predict = ['PM2.5']
                self.logger.info("Auto-detected variable: PM2.5")
        
        # Get known stations from DPE_region_stations class
        dpe_stations = DPERT.DPE_region_stations(var_to_predict)
        
        # Build comprehensive list of known stations from all regions
        # CRITICAL: Multi-word stations FIRST (order matters!)
        known_stations = []
        
        # First add multi-word stations to ensure proper matching
        multi_word_stations = ['CAMPBELLTOWN_WEST', 'MACQUARIE_PARK', 'ALBION_PARK_SOUTH', 
                               'KEMBLA_GRANGE', 'PARRAMATTA_NORTH', 'COOK_AND_PHILLIP',
                               'ROUSE_HILL', 'ST_MARYS']
        
        for region_name, region_stations in dpe_stations.DPE_region_stations_dict.items():
            for station in region_stations:
                if station in multi_word_stations and station not in known_stations:
                    known_stations.append(station)
        
        # Then add remaining single-word stations
        for region_name, region_stations in dpe_stations.DPE_region_stations_dict.items():
            for station in region_stations:
                if station not in known_stations:
                    known_stations.append(station)
        
        self.logger.info("Using {n} known stations from DPE_region_stations".format(
            n=len(known_stations)))
        
        # Find which stations are present in the data
        stations = set()
        for col in input_data_pd.columns:
            for station in known_stations:
                if col.endswith('_' + station):
                    stations.add(station)
                    break
        
        if len(stations) == 0:
            self.logger.warning("No known stations found - using fallback method")
            # Fallback: use old method for unknown station names
            for col in input_data_pd.columns:
                if '_' in col:
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2:
                        stations.add(parts[1])
        
        stations = sorted(list(stations))
        self.logger.info("Found {n} stations: {s}".format(
            n=len(stations), s=', '.join(stations)))
        
        # Extract data for each station
        station_dict = {}
        for station in stations:
            # Get columns for this station
            station_cols = [col for col in input_data_pd.columns if col.endswith('_' + station)]
            
            if len(station_cols) == 0:
                continue
                
            # Create station dataframe
            station_df = input_data_pd[station_cols].copy()
            
            # Remove station suffix from column names
            new_cols = []
            for col in station_df.columns:
                # Remove '_STATION' from end
                new_col = col[:-len('_' + station)]
                new_cols.append(new_col)
            station_df.columns = new_cols
            
            station_dict[station] = station_df
            
            self.logger.info("  Station {s}: {n} variables ({vars})".format(
                s=station, n=len(station_df.columns),
                vars=', '.join(station_df.columns.tolist())))
        
        return station_dict

###########################################################################################
    def combine_station_data(self, station_dict):
        """
        Combine per-station imputed data back into single dataframe
        
        Parameters
        -----------
        station_dict : dict
            Dictionary with station names as keys and dataframes as values
            
        Returns
        --------
        pd.DataFrame
            Combined dataframe with VAR_STATION column format
        """
        combined_dfs = []
        
        for station, station_df in station_dict.items():
            # Add station suffix back to column names
            station_df_renamed = station_df.copy()
            station_df_renamed.columns = [col + '_' + station for col in station_df.columns]
            combined_dfs.append(station_df_renamed)
        
        # Concatenate all stations
        combined_df = pd.concat(combined_dfs, axis=1)
        
        # Sort columns alphabetically
        combined_df = combined_df[sorted(combined_df.columns)]
        
        return combined_df

###########################################################################################
    def MICE_imputation(self, input_data_pd):
        """
        configure and impute data using the sklearn mice impute method
        https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87
        """
        from sklearn.experimental import enable_iterative_imputer

        empty_cols = [col for col in input_data_pd.columns if input_data_pd[col].isna().all()]
        working_data_pd = input_data_pd.drop(columns=empty_cols) if empty_cols else input_data_pd

        if working_data_pd.shape[1] == 0:
            self.logger.warning(
                "MICE received only all-missing columns; filling with zeros".ljust(self.justif - 2, '.') + 'FALLBACK'
            )
            return input_data_pd.fillna(0)

        # Define MICE Imputer and fill missing values
        mice_imputer = SKI.IterativeImputer(
                    estimator=SK.linear_model.BayesianRidge(), 
                    missing_values=np.nan, 
                    sample_posterior=False, 
                    max_iter=10, 
                    tol=0.001, 
                    n_nearest_features=None, 
                    initial_strategy="mean", 
                    imputation_order="ascending"
                    )


        imputed_values = mice_imputer.fit_transform(working_data_pd)
        imputed_data_pd = pd.DataFrame(
            imputed_values,
            columns=working_data_pd.columns,
            index=working_data_pd.index)

        if empty_cols:
            self.logger.warning(
                "MICE skipped all-missing columns {cols}; filling with zeros".format(cols=empty_cols).ljust(
                    self.justif - 2, '.') + 'FALLBACK'
            )
            for col in empty_cols:
                imputed_data_pd[col] = 0.0
            imputed_data_pd = imputed_data_pd[input_data_pd.columns]

        return imputed_data_pd
###########################################################################################
    def KNN_imputation(self, input_data_pd):
        """
        configure and impute data using the sklearn K-Nearest Neighbours impute method
        https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87
        """

        # Define KNN Imputer and fill missing values
        # Define scaler to set values between 0 and 1

        scaler = SKP.MinMaxScaler(feature_range=(0, 1))
        df_knn = pd.DataFrame(
            scaler.fit_transform(input_data_pd),
            columns = input_data_pd.columns,
            index=input_data_pd.index,
            )

        # Define KNN imputer and fill missing values
        knn_imputer = SKI.KNNImputer(
            missing_values=np.nan,
            n_neighbors=5, 
            weights='uniform', 
            metric='nan_euclidean'
            )
        imputed_data_pd = pd.DataFrame(
            knn_imputer.fit_transform(df_knn), 
            columns=df_knn.columns,
            index=df_knn.index,
            )

        return imputed_data_pd
        
        
###########################################################################################

    def TemporalMICE_imputation(self, input_data_pd):
        """
        Configure and impute data using Temporal MICE with lag/lead features.
        This method extends standard MICE by incorporating temporal dependencies
        through lagged and lead features, and uses Bayesian posterior sampling
        across multiple imputations.
        
        Parameters
        -----------
        input_data_pd : pd.DataFrame
            Input dataframe with missing values
            
        Returns
        --------
        pd.DataFrame
            Pooled imputed dataframe (averaged across multiple imputations)
        """
        from sklearn.experimental import enable_iterative_imputer
        
        # Get configuration parameters
        n_imputations = getattr(self.Configuration, 'temporal_n_imputations', 5)
        max_iter = getattr(self.Configuration, 'temporal_max_iter', 10)
        temporal_lags = getattr(self.Configuration, 'temporal_lags', [1, 2, 3, 24])
        temporal_leads = getattr(self.Configuration, 'temporal_leads', [1, 2])
        random_state = getattr(self.Configuration, 'temporal_random_state', 42)
        refinement_iterations = getattr(self.Configuration, 'temporal_refinement_iterations', 2)
        
        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('TemporalMICE Configuration:'.ljust(self.justif,'|'))
        self.logger.info("  n_imputations = {n}".format(n=n_imputations).ljust(self.justif,'|'))
        self.logger.info("  max_iter = {m}".format(m=max_iter).ljust(self.justif,'|'))
        self.logger.info("  lags = {l}".format(l=temporal_lags).ljust(self.justif,'|'))
        self.logger.info("  leads = {l}".format(l=temporal_leads).ljust(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        
        # Store original columns and missing value counts
        original_columns = input_data_pd.columns.tolist()
        original_missing = input_data_pd.isnull().sum()
        empty_columns = [col for col in original_columns if original_missing[col] == len(input_data_pd)]

        working_input_pd = input_data_pd.drop(columns=empty_columns) if empty_columns else input_data_pd

        if working_input_pd.shape[1] == 0:
            self.logger.warning(
                "TemporalMICE received only all-missing columns; filling with zeros".ljust(
                    self.justif - 2, '.') + 'FALLBACK'
            )
            return input_data_pd.fillna(0)
        
        self.logger.info("Original missing values:".ljust(self.justif,'|'))
        for col in original_columns:
            if original_missing[col] > 0:
                pct = 100.0 * original_missing[col] / len(input_data_pd)
                self.logger.info("  {col}: {n} ({pct:.2f}%)".format(
                    col=col, n=original_missing[col], pct=pct).ljust(self.justif,'|'))
        
        # List to store all imputed datasets
        imputed_datasets = []
        
        # Perform multiple imputations
        for m in range(n_imputations):
            self.logger.info("Processing imputation set {curr}/{total}".format(
                curr=m+1, total=n_imputations).ljust(self.justif - 2, '.') + 'RUN')
            
            df_m = working_input_pd.copy()
            current_random_state = random_state + m
            
            # Stage 1: Create lag/lead features
            df_m = self.create_lag_lead_features(
                df_m, 
                original_columns, 
                temporal_lags, 
                temporal_leads
            )
            
            self.logger.info("  Extended to {n} columns with temporal features".format(
                n=df_m.shape[1]).ljust(self.justif - 2, '.') + 'OK')
            
            # Stage 2: Initial Bayesian MICE imputation with posterior sampling
            mice_imputer = SKI.IterativeImputer(
                estimator=SK.linear_model.BayesianRidge(),
                sample_posterior=True,  # Key difference from standard MICE
                missing_values=np.nan,
                max_iter=max_iter,
                tol=0.001,
                n_nearest_features=None,
                initial_strategy="mean",
                imputation_order="ascending",
                random_state=current_random_state
            )
            
            df_m_values = mice_imputer.fit_transform(df_m)
            df_m = pd.DataFrame(
                df_m_values,
                columns=df_m.columns,
                index=df_m.index
            )
            
            self.logger.info("  Initial imputation".ljust(self.justif - 2, '.') + 'OK')
            
            # Stage 3: Refinement iterations for improved convergence
            for refinement in range(refinement_iterations):
                mice_refiner = SKI.IterativeImputer(
                    estimator=SK.linear_model.BayesianRidge(),
                    sample_posterior=True,
                    missing_values=np.nan,
                    max_iter=max_iter,
                    tol=0.001,
                    n_nearest_features=None,
                    initial_strategy="mean",
                    imputation_order="ascending",
                    random_state=current_random_state + refinement + 1
                )
                
                df_m_values = mice_refiner.fit_transform(df_m)
                df_m = pd.DataFrame(
                    df_m_values,
                    columns=df_m.columns,
                    index=df_m.index
                )
            
            self.logger.info("  Refinement ({n} iterations)".format(
                n=refinement_iterations).ljust(self.justif - 2, '.') + 'OK')
            
            # Return to original columns only (drop temporal features)
            df_m_final = df_m.reindex(columns=original_columns).copy()
            if empty_columns:
                for col in empty_columns:
                    df_m_final[col] = 0.0
                df_m_final = df_m_final[original_columns]
            imputed_datasets.append(df_m_final)
        
        # Stage 4: Pool results using Rubin's rules
        self.logger.info("Pooling {n} imputation sets".format(
            n=n_imputations).ljust(self.justif - 2, '.') + 'RUN')
        
        # Calculate pooled mean across all imputations
        pooled_data = sum(imputed_datasets) / n_imputations
        
        # Preserve original index and columns
        imputed_data_pd = pd.DataFrame(
            pooled_data,
            columns=original_columns,
            index=input_data_pd.index
        )
        
        # Verify no missing values remain
        remaining_missing = imputed_data_pd.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning("Remaining missing values after imputation: {n}".format(
                n=remaining_missing).ljust(self.justif - 2, '.') + 'WARN')
        else:
            self.logger.info("All missing values imputed successfully".ljust(self.justif - 2, '.') + 'OK')
        
        return imputed_data_pd


###########################################################################################

    def create_lag_lead_features(self, df, variables, lags, leads):
        """
        Create lagged and lead features for temporal imputation
        
        Parameters
        -----------
        df : pd.DataFrame
            Input dataframe
        variables : list
            List of variable names to create lags/leads for
        lags : list
            List of lag periods (e.g., [1, 2, 3, 24])
        leads : list
            List of lead periods (e.g., [1, 2])
            
        Returns
        --------
        pd.DataFrame
            Extended dataframe with lag and lead features
        """
        df_extended = df.copy()
        
        for var in variables:
            if var not in df.columns:
                self.logger.warning("Variable {vv} not found in dataframe".format(vv=var).ljust(self.justif - 2, '.') + 'SKIP')
                continue
            
            # Create lag features
            for lag in lags:
                df_extended["{var}_lag{lag}".format(var=var, lag=lag)] = df[var].shift(lag)
            
            # Create lead features
            for lead in leads:
                df_extended["{var}_lead{lead}".format(var=var, lead=lead)] = df[var].shift(-lead)
        
        return df_extended        
        
        
###########################################################################################
    def save_imputed_data(self, imputed_data_pd, station_name=None):
        """
        Save imputed data 
        
        Parameters
        -----------
        imputed_data_pd : pd.DataFrame
            Imputed dataframe
        station_name : str, optional
            Station name for per-station saving
        """
        Output_Manager = OM.Output_manager_Class(self.Configuration)
        Output_Manager.output_imputed_data(imputed_data_pd, station_name=station_name)
        
###########################################################################################
    def imputation(self, input_data_pd, save_data=False):
        """
        Impute data using the configuration defined for method and parameters.
        Performs per-station imputation if data contains multiple stations.
        
        Parameters
        -----------
        input_data_pd : pd.DataFrame
            Input dataframe with missing values
        save_data : bool
            Whether to save imputed data
            
        Returns
        --------
        pd.DataFrame
            Imputed dataframe
        dict
            Dictionary of per-station imputed dataframes
        """
        if self.Configuration.imputation_method == "NONE":
            imputed_data_pd = input_data_pd
            self.logger.warning("Bypass Imputation".ljust(self.justif - 2 ,'.') + 'OK')
            station_imputed_dict = {}
            
        else:
            # Extract per-station data
            # Pass var_to_predict from Configuration if available, otherwise let method auto-detect
            var_to_predict = getattr(self.Configuration, 'var_to_predict', None)
            station_dict = self.extract_station_data(input_data_pd, var_to_predict=var_to_predict)
            
            # Impute each station separately
            station_imputed_dict = {}
            
            for station, station_df in station_dict.items():
                self.logger.info(''.ljust(self.justif,'-'))
                self.logger.info("Imputing station: {s}".format(s=station).center(self.justif,'|'))
                self.logger.info(''.ljust(self.justif,'-'))
                
                # Select imputation method
                if self.Configuration.imputation_method == "MICE":
                    imputed_station_df = self.MICE_imputation(station_df)
                    method_name = "MICE"
                    
                elif self.Configuration.imputation_method == "KNN":
                    KNN = KNNI.KNN_Class(self.Configuration)
                    imputed_data_pd = KNN.fit(pandas)
                    
                    imputed_station_df = self.KNN_imputation(station_df)
                    method_name = "KNN"
                    
                elif self.Configuration.imputation_method == "TemporalMICE":
                    imputed_station_df = self.TemporalMICE_imputation(station_df)
                    method_name = "TemporalMICE"
                    
                else:
                    self.logger.error("Imputation method".ljust(self.justif - 15 ,'.') + 'NOT IMPLEMENTED')
                    sys.exit("Sorry :(")
                
                station_imputed_dict[station] = imputed_station_df
                self.logger.info("{m} Imputation for {s}".format(
                    m=method_name, s=station).ljust(self.justif - 2, '.') + 'OK')
                
                # Save per-station data if requested
                if save_data:
                    self.save_imputed_data(imputed_station_df, station_name=station)
            
            # Combine all stations back
            imputed_data_pd = self.combine_station_data(station_imputed_dict)
            self.logger.info("Combined {n} stations".format(
                n=len(station_imputed_dict)).ljust(self.justif - 2, '.') + 'OK')
        
        # Save combined data if requested
        if save_data:
            self.save_imputed_data(imputed_data_pd, station_name=None)
        
        return imputed_data_pd, station_imputed_dict
