"""
..  module:: Yaml_file_writer
    :platform: Unix
    :synopsis: Definition of the basic object class to read yaml files.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
   
"""

import sys
import os
import stat
import datetime

import yaml
###########################################################################################
###########################################################################################
class Yaml_file_writer_Class(object):
    """ 
    This class defines a Yaml_file_reader_Class, that contains the capacity to write yaml config files.

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
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif
        
        self.Code_configuration_optimal_dir = self.Configuration.Code_configuration_optimal_dir
        self.Code_configuration_optimal_history_dir = self.Configuration.Code_configuration_optimal_history_dir
        self.nownownow = datetime.datetime.now().strftime("%Y%m%d_%H")
        
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
    def write_yaml_optimal_config_file(self, filename):
        """_summary_

        Args:
            filename (_type_): _description_
        """
        
        safe_name = os.path.basename(str(filename)).replace(os.sep, "_")
        filename_template = "{ff}_optimal_{date}.yaml"
        
        filename_fullpath = os.path.join(
            self.Code_configuration_optimal_dir,
            filename_template.format(ff=safe_name, date=self.nownownow),
        )
        
        yml_dict = {
           "model_parameters" :  self.Configuration.model_parameters_dict,
            "run_parameters" : {
                "n_epoch": self.Configuration.n_epochs,
                "batchsize": self.Configuration.batch_size,
                "num_batches": self.Configuration.num_batches,
                },
            "use_file": self.Configuration.use_file,
            "n_iteration": 1,
            "prob_samples":  self.Configuration.prob_samples,
            "internal_var_to_add": self.Configuration.internal_var_to_add_list,
            "additional_var_to_select": self.Configuration.additional_var_to_select,
            "evaluation_metrics" : self.Configuration.evaluation_metrics,
        }
        
        self.MakeDir(self.Code_configuration_optimal_dir)
        
        with open(filename_fullpath, 'w') as ff:
            yaml.dump(yml_dict, ff, default_flow_style=False, sort_keys=False)
        
        self.Change_permissions(filename_fullpath)
        self.logger.info('optimal configuration file  = {msg}'.format(msg=filename_fullpath).ljust(self.justif-7,'.') + 'WRITTEN')
        return self.nownownow

###########################################################################################
    def write_yaml_optimal_best_config_file(self, filename, best_target=None):
        """
        Write/overwrite a stable "best" optimal YAML file for the given base config name.

        This is designed for runtime consumption (train/forecast) without having
        to guess the latest timestamp.
        """
        safe_name = os.path.basename(str(filename)).replace(os.sep, "_")
        filename_template = "{ff}_optimal_best.yaml"
        filename_fullpath = os.path.join(
            self.Code_configuration_optimal_history_dir,
            filename_template.format(ff=safe_name),
        )

        yml_dict = {
            "model_parameters": self.Configuration.model_parameters_dict,
            "run_parameters": {
                "n_epoch": self.Configuration.n_epochs,
                "batchsize": self.Configuration.batch_size,
                "num_batches": self.Configuration.num_batches,
            },
            "use_file": self.Configuration.use_file,
            "n_iteration": 1,
            "prob_samples": self.Configuration.prob_samples,
            "internal_var_to_add": self.Configuration.internal_var_to_add_list,
            "additional_var_to_select": self.Configuration.additional_var_to_select,
            "evaluation_metrics": self.Configuration.evaluation_metrics,
        }
        if best_target is not None:
            yml_dict["optimisation_best_target"] = float(best_target)
            yml_dict["optimisation_generated_at"] = self.nownownow

        self.MakeDir(self.Code_configuration_optimal_history_dir)

        with open(filename_fullpath, "w") as ff:
            yaml.dump(yml_dict, ff, default_flow_style=False, sort_keys=False)

        self.Change_permissions(filename_fullpath)
        self.logger.info(
            "optimal BEST configuration file  = {msg}".format(msg=filename_fullpath).ljust(self.justif - 7, ".")
            + "WRITTEN"
        )
        return filename_fullpath
        



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
    Code_configuration_main_dir = "/home/barthelemyx/Projects/Deep_learning/cnn_lstm_forecast/Core_iHPC/Tools/Config_testing"
    
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
