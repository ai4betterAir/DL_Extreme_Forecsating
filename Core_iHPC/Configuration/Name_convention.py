"""
.. module:: Name_convention
   :platform: Unix
   :synopsis: contains the routines to define the standard names for AI forecast.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>


"""
import os
import sys
import base64
import hashlib
###########################################################################################
class Name_configuration(object):
    ''' This class define the naming convention.
    '''
    def __init__(self, Configuration, 
             ):

        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif


       
        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Configuring Names'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        
        return
###########################################################################################
    def b32_hash(self, content):
        """Return the b32 encoded sha1 hash of the input string as a string."""
        sha = hashlib.sha1(content.encode("utf-8"))
        b32_hash = base64.b32encode(sha.digest()).lower()
        b32_hash = b32_hash.decode("utf-8")
        return b32_hash
###########################################################################################
    def model_file_name(self,):
        """
        Method to compute the name convention and the hash to save the trained model file.
        """
        # model_file_name_template = "model_{model_name}_inputs_{inputs}_outputs_{outputs}_parameters_{parameters}.h5"
        # model_file_name_template = "model_{model_name}_{option_hash}.h5"
        model_file_name_template = "model_{model_name}_{option_hash}.keras"
        all_options_template = 'inputs_{inputs}_outputs_{outputs}_parameters_{parameters}'

        model_name = "forecast"

        input_column_names = sorted(self.Configuration.input_column_names)#.lower()
        inputs = "_".join(input_column_names).lower() 

        output_column_names = sorted(self.Configuration.output_column_names)#.lower()
        outputs = "_".join(output_column_names).lower()

        n_steps_in = str(self.Configuration.n_steps_in)
        n_steps_out = str(self.Configuration.n_steps_out)
        n_epochs = str(self.Configuration.n_epochs)
        batch_size = str(self.Configuration.batch_size)

        parameters = "_".join([n_steps_in, n_steps_out, n_epochs, batch_size])

        
        


        all_options = all_options_template.format(inputs = inputs, outputs = outputs, parameters = parameters)
        
        print(all_options)
        # ASDA
        #################
        all_option_hash = self.b32_hash(all_options)
        print("===============> Name of model:")
        print("name", all_options, "\n", all_option_hash, "\n", )

        model_file_name = model_file_name_template.format(model_name=model_name, option_hash=all_option_hash)
        return model_file_name