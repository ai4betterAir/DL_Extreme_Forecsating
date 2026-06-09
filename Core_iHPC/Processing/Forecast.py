"""
..  module:: Forecast
    :platform: Unix
    :synopsis: Definition of the basic object class to configure and forecast data.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""
import sys
import os
import stat
import importlib
import numpy as np
import pandas as pd
from ..Models import CNN_LSTM_v1 as CL_CNN_LSTM
from ..Models import CNN_LSTM_BNN_24_PM25 as CL_CNN_LSTM_BNN_24_PM25
from ..Models import CNN_LSTM_BNN_24_PM10 as CL_CNN_LSTM_BNN_24_PM10
from ..Models import CNN_LSTM_24_PM10 as CL_CNN_LSTM_24_PM10
from ..Models import CNN_LSTM_24_PM10_v2 as CL_CNN_LSTM_24_PM10_v2
from ..Models import Sparse_LSTM_v1 as CL_Sparse_LSTM
from ..Models import model_loader as MODEL_LOADER
#from Core_iHPC.Models.model import create_sparse_lstm_model as CL_Sparse_LSTM
from ..Models.layers.sparse_attention import SparseAttention


###########################################################################################
class Forecast_Class(object):
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
        self.logger.info('Configuring the Forecast'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        return
###########################################################################################
    def CNN_LSTM_model_v1(self, input_data_pd, n_retrain):
        """
        configure and run forecast using the cnn_lstm_model_v1 model
        """

        self.CLC = CL_CNN_LSTM.CNN_LSTM_Class(self.Configuration)
        result = self.CLC.run_all(input_data_pd, n_retrain)
        
        # Handle both single return (list) and tuple return (list, hist)
        if isinstance(result, tuple):
            list_output_pd, train_hist = result
        else:
            list_output_pd = result
            train_hist = None

        return list_output_pd, train_hist

###########################################################################################
    def CNN_LSTM_BNN_24_PM25_model_v1(self, input_data_pd, n_retrain):
        """
        configure and run forecast using the cnn_lstm_model_v1 model
        """

        self.CLC = CL_CNN_LSTM_BNN_24_PM25.CNN_LSTM_BNN_Class(self.Configuration)
        list_output_pd = self.CLC.run_all(input_data_pd, n_retrain)   


        return list_output_pd
###########################################################################################
    def CNN_LSTM_BNN_24_PM10_model_v1(self, input_data_pd, n_retrain):
        """
        configure and run forecast using the cnn_lstm_model_v1 model
        """

        self.CLC = CNN_LSTM_BNN_24_PM10_model_v1.CNN_LSTM_BNN_Class(self.Configuration)
        list_output_pd = self.CLC.run_all(input_data_pd, n_retrain)   


        return list_output_pd
###########################################################################################
    def CNN_LSTM_24_PM10_model_v1(self, input_data_pd, n_retrain):
        """
        configure and run forecast using the cnn_lstm_model_v1 model
        """

        self.CLC = CNN_LSTM_24_PM10_model_v1.CNN_LSTM_Class(self.Configuration)
        list_output_pd = self.CLC.run_all(input_data_pd, n_retrain)   


        return list_output_pd
###########################################################################################
    def CNN_LSTM_24_PM10_model_v2(self, input_data_pd, n_retrain):
        """
        configure and run forecast using the cnn_lstm_model_v1 model
        """

        self.CLC = CNN_LSTM_24_PM10_model_v2.CNN_LSTM_Class(self.Configuration)
        list_output_pd = self.CLC.run_all(input_data_pd, n_retrain)   


        return list_output_pd
        
###########################################################################################
    def Sparse_LSTM_v1(self, input_data_pd, n_retrain):
         """Run forecast using Sparse_LSTM_v1 model"""
         self.CLC = CL_Sparse_LSTM.Sparse_LSTM_Class(self.Configuration)
         list_output_pd, train_hist = self.CLC.run_all(input_data_pd, n_retrain)
         
         return list_output_pd, train_hist     

###########################################################################################
    def _run_generic_model(self, input_data_pd, n_retrain):
        """
        Resolve and run a model module selected by configuration.
        """
        model_module_name = str(getattr(self.Configuration, "forecast_method", "") or "").strip()
        if not model_module_name:
            raise ValueError("forecast_method is not set")

        module = importlib.import_module(f"Core_iHPC.Models.{model_module_name}")
        self.CLC = MODEL_LOADER.instantiate_model(module, self.Configuration)
        result = self.CLC.run_all(input_data_pd, n_retrain)

        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
            if len(result) == 1:
                return result[0], None
        return result, None


###########################################################################################
    def run_forecast(self, input_data_pd, n_retrain):
        """
        Run forecast using the configuration defined for method and parameters
        
        Returns
        --------
        tuple
            (list_output_pd, train_hist) - Output DataFrames and training history
        """
        if self.Configuration.forecast_method == "CNN_LSTM_model_v1":
            list_output_pd, train_hist = self.CNN_LSTM_model_v1(input_data_pd, n_retrain)
            self.logger.info("CNN_LSTM_model_v1 forecast".ljust(self.justif - 2 ,'.') + 'OK')
        elif self.Configuration.forecast_method == "Sparse_LSTM_v1":
            list_output_pd, train_hist = self.Sparse_LSTM_v1(input_data_pd, n_retrain)
            self.logger.info("Sparse_LSTM_v1 forecast".ljust(self.justif - 2 ,'.') + 'OK')
        else:
            list_output_pd, train_hist = self._run_generic_model(input_data_pd, n_retrain)
            self.logger.info(
                str(self.Configuration.forecast_method).ljust(self.justif - 2 ,'.') + ' forecast OK'
            )
                
        return list_output_pd, train_hist

###########################################################################################
