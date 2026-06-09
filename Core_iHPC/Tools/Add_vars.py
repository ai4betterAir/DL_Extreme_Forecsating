"""
..  module:: Add_vars
    :platform: Unix
    :synopsis: Definition of the basic object class to spliot data.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

import sys
import os
import stat
import numpy as np
import pandas as pd

###########################################################################################
class Add_vars_Class(object):
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
                 Configuration,
                ):

        self.logger = logger
        self.justif = justif
        self.Configuration = Configuration

        return
###########################################################################################
    def add_hour_sine_signal(self, input_pd):
        #### Adding cycle patern of hour
        hour_index = input_pd.index.hour

        input_pd['hour_sin'] = np.sin(2 * np.pi * hour_index/24.0)
        input_pd['hour_cos'] = np.cos(2 * np.pi * hour_index/24.0)
        ### Update input columns
        self.Configuration.input_column_names = list(input_pd.columns)

        self.logger.info('Hour signal and columns update'.ljust(self.justif-8,'.') + 'COMPLETE')
        # print("*"*50)
        # print("---->>>>>>>>>> UPDATED Input_column_names: ", self.Configuration.input_column_names)
        # print("*"*50)
        return input_pd
###########################################################################################
    def add_month_sine_signal(self, input_pd):
        #### Adding cycle patern of hour
        month_index = input_pd.index.month

        input_pd['month_sin'] = np.sin(2 * np.pi * month_index/12.0)
        input_pd['month_cos'] = np.cos(2 * np.pi * month_index/12.0)

        ### Update input columns
        self.Configuration.input_column_names = list(input_pd.columns)

        self.logger.info('Month signal and columns update'.ljust(self.justif-8,'.') + 'COMPLETE')
        # print("*"*50)
        # print("---->>>>>>>>>> UPDATED Input_column_names: ", self.Configuration.input_column_names)
        # print("*"*50)
        return input_pd
###########################################################################################
    def add_season(self, input_pd, season_definition = "meteorological"):
        """ add season number to the input

        season_definition = "astronomical"
        season_definition = "meteorological"


        Args:
            input_pd (_type_): _description_
            season_definition (str, optional): _description_. Defaults to "meteorological".

        Returns:
            _type_: _description_
        """

        if season_definition == "astronomical":
            input_pd["season_number"] = (input_pd.index + pd.DateOffset(months=1, days=-21)).quarter
        if season_definition == "meteorological":
            input_pd["season_number"] = (input_pd.index + pd.DateOffset(months=1)).quarter

        ### Update input columns
        self.Configuration.input_column_names = list(input_pd.columns)

        self.logger.info('Seasons and columns update'.ljust(self.justif-8,'.') + 'COMPLETE')

        return input_pd
###########################################################################################
    def add_day_of_year(self, input_pd,):
        """ add day of year column to the input

        Args:
            input_pd (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_pd["day_of_year"] = input_pd.index.dayofyear    ### Pandas 1.1.5


        ### Update input columns
        self.Configuration.input_column_names = list(input_pd.columns)

        self.logger.info('Day of the year and columns update'.ljust(self.justif-8,'.') + 'COMPLETE')

        return input_pd
###########################################################################################
    def add_day_of_week(self, input_pd,):
        """ add day of year column to the input

        Args:
            input_pd (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_pd["day_of_week"] = input_pd.index.dayofweek    ### Pandas 1.1.5


        ### Update input columns
        self.Configuration.input_column_names = list(input_pd.columns)

        self.logger.info('Day of the week and columns update'.ljust(self.justif-8,'.') + 'COMPLETE')

        return input_pd
###########################################################################################
    def update_config(self,):

        return self.Configuration
###########################################################################################
    def add_var_as_defined_in_config_file(self, input_pd,):
        
        if len(self.Configuration.internal_var_to_add_list) > 0:
            for internal_var in self.Configuration.internal_var_to_add_list:
                match internal_var:
                    case "hour":
                        input_pd = self.add_hour_sine_signal(input_pd)
                    case "month":
                        input_pd = self.add_month_sine_signal(input_pd) 
                    case "season_meteorological":
                        input_pd = self.add_season(input_pd, season_definition = "meteorological")      
                    case "season_astronomical":
                        input_pd = self.add_season(input_pd, season_definition = "astronomical")
                    case "day_of_year":
                        input_pd = self.add_day_of_year(input_pd,) 
                    case "day_of_week":             
                        input_pd = self.add_day_of_week(input_pd,)
                    case _ :
                        self.logger.error("unrecognised additional internal var")    
        
        input_pd.sort_index(axis=1, inplace=True)
        self.Configuration.input_column_names = list(input_pd.columns)
        
        return input_pd #, self.Configuration                
    