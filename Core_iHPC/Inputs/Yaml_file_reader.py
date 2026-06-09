"""
..  module:: Yaml_file_reader
    :platform: Unix
    :synopsis: Definition of the basic object class to read yaml files.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
   
"""

import sys
import os
import stat

import yaml
import importlib

###########################################################################################
###########################################################################################
class Yaml_file_reader_Class(object):
    """ 
    This class defines a Yaml_file_reader_Class, that contains the capacity to read yaml config files.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, Configuration,
                ):
        """_summary_

        Args:
            Configuration (_type_): _description_
        """
        self.Configuration = Configuration
        
        self.Code_configuration_main_dir = self.Configuration.Code_configuration_main_dir
        
        return
###########################################################################################
    def read_yaml_config_file(self, filename):
        """_summary_

        Args:
            filename (_type_): _description_
        """

        filename = str(filename)
        if filename.endswith(".yaml"):
            rel_path = filename
        else:
            rel_path = f"{filename}.yaml"

        if os.path.isabs(rel_path):
            filename_fullpath = rel_path
        else:
            filename_fullpath = os.path.join(self.Code_configuration_main_dir, rel_path)
        
        with open(filename_fullpath, 'r') as ff:
            yaml_file_dict = yaml.safe_load(ff)
        # If the YAML references a `config_file`, allow top-level values
        # set to the literal string 'from_config' to be replaced by the
        # corresponding attribute from that module.
        if isinstance(yaml_file_dict, dict) and 'config_file' in yaml_file_dict:
            config_file = yaml_file_dict.get('config_file')
            if isinstance(config_file, str) and config_file.endswith('.py'):
                module_name = os.path.splitext(config_file)[0]
                # Temporarily add the config dir to path and import
                sys.path.insert(0, self.Code_configuration_main_dir)
                try:
                    cfg_mod = importlib.import_module(module_name)
                except Exception:
                    cfg_mod = None
                finally:
                    try:
                        sys.path.pop(0)
                    except Exception:
                        pass

                if cfg_mod is not None:
                    for key, val in list(yaml_file_dict.items()):
                        if isinstance(val, str) and val.strip().lower() == 'from_config':
                            # Map YAML key (e.g. n_epoch) to config attr (e.g. N_EPOCH)
                            attr_name = key.upper()
                            if hasattr(cfg_mod, attr_name):
                                yaml_file_dict[key] = getattr(cfg_mod, attr_name)

        return yaml_file_dict



###########################################################################################
if __name__ == '__main__':
    import datetime as dtime
    from Core_iHPC.Tools import InitLogging as IL
    from Core_iHPC.Configuration import Configuration as CC

    ################ logger init #################
    justif = 102
    config_name = "yaml_reader"
    loggername= config_name
    logger = IL.Initialise_logging(loggername)
   
    ### Define list of configuration files, files are stored in "/Tools/Config_testing" folder
    # list_configs = ['main_LSTM_BNN', 'main_CNN_LSTM_BNN']
    # list_configs = ['main_LSTM_BNN']            ### 'main_LSTM_BNN.yaml'
    # list_configs = ['main_CNN_LSTM_BNN']            ### 'main_CNN_LSTM_BNN.yaml'
    # list_configs = ['main_CNN_LSTM']            ### 'main_CNN_LSTM_BNN.yaml'
    list_configs = ['main_LSTM_BNN_24']            ### 'main_LSTM_BNN.yaml'
    # list_configs = ['main_LSTM_BNN_72']            ### 'main_LSTM_BNN.yaml'
    list_configs = [
                    # #"CE_24_Ozone_LSTM_BNN",
                    "Illawara_24_Ozone_LSTM_BNN",
                    "Illawara_48_Ozone_LSTM_BNN",
                    "Illawara_72_Ozone_LSTM_BNN",
                    "NW_24_Ozone_LSTM_BNN",
                    "NW_48_Ozone_LSTM_BNN",
                    "NW_72_Ozone_LSTM_BNN",
                    # #"SW_24_Ozone_LSTM_BNN",
                    "SW_48_Ozone_LSTM_BNN",
                    "SW_72_Ozone_LSTM_BNN",
                    # #"LH_24_Ozone_LSTM_BNN",
                    # #"LH_48_Ozone_LSTM_BNN",
                    "LH_72_Ozone_LSTM_BNN",
                    ]
    # list_configs = [ 'main_LSTM_BNN_48', 'main_LSTM_BNN_72']            ### 'main_LSTM_BNN.yaml'
    #Code_configuration_main_dir = "/home/barthelemyx/Projects/Deep_learning/cnn_lstm_forecast/Core_iHPC/Tools/Config_testing"
    
    current_date = dtime.datetime.today()
    
###########################################################
    Configuration = CC.Configuration_Class(logger, justif,
        current_date, current_date, 
        current_date, current_date, current_date, 
        None,
        None, None, None, Code_configuration_main_dir,  
        None, None, None, 
        )

    YFRC = Yaml_file_reader_Class(Configuration)
    YFRC.read_yaml_config_file(list_configs[0])
