"""
.. module:: CommandLineParsing
   :platform: Unix
   :synopsis: Parser of the options passed to the main script.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>


"""
import argparse
import os
import sys



###########################################################################################
class CommandLineParsingClass(object):
    """ 
    This class defines a Input_Class, that contains the capacity to manage the intputs.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, logger, justif,
                    
                ):

        self.logger = logger
        self.justif = justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Argument line parsing'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))

        self._config()

        return

###########################################################################################
    def _config(self,):

        """
        This function parse the argument line send to the main CCAM script.
        it helps to define the ensemble of runs.
        if no argument, then it's a manual run

        Parameters
        ---------- 
            logger : logging.logger
                logger instance
            all_mec : list of str
                list of all lumping mechanisms available
        Returns
        -------
            Startdate : datetime
                Date to run the forecast in AEDT. The date should be naive (no tz).
            timezone : str
                String reprenseting the timezone of the startdate. It will be used later to localise the startdate. 
        """
    
        self.parser = argparse.ArgumentParser()
        # parser.add_argument("--emission_input_netcdf_fullpath", help="full path of the emission file to speciate, in netcdf format")
        # parser.add_argument("--lumping_chemistry", help='''\
        #     Define lumping mechanism, one of 
        #     'CB05_CF2', 'CB6R3_AE7', 'CB6R3_AE7_TRACER', 'CB6R3_AE8', 'CB6R4_CF2',
        #     'CRI_AE7', 'CRI_AE8', 
        #     'SAPRC07_CF2', 'SAPRC07TC_AE7', 'SAPRC07TC_AE8',
        #     'RACM2_AE7', 'RACM2_AE8',]''')
        # parser.add_argument("--output_path", help="Directory where to save speciated file")
        # parser.add_argument('--intervals',  help= 'update intervals')
        self.parser.add_argument('--pollutant',  help= 'define ', type=int)
        self.parser.add_argument('--inputs',  help= 'define ', type = int)
        self.parser.add_argument('--outputs',  help= 'define ', type = int)
        self.parser.add_argument('--epochs',  help= 'define ', type = int)
        # parser.add_argument('--cases',  help= 'define ')

###########################################################################################
    def parse(self,):

        args = self.parser.parse_args()

        if ((args.inputs is not None) & 
            (args.outputs is not None) & 
            (args.pollutant is not None) & 
            (args.epochs is not None)
            ) :
            self.n_steps_in = args.inputs  
            self.n_steps_out = args.outputs 
            self.pollutant = args.pollutant  
            self.n_epochs = args.epochs  
            # cases = str(arg.cases)
        else:
            self.n_steps_in = None
            self.n_steps_out = None
            self.pollutant = None
            self.n_epochs = None
            # cases = None
            
        
        return self.n_steps_in, self.n_steps_out, self.pollutant, self.n_epochs

