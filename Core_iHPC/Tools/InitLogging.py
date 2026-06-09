"""
.. module:: InitLogging
   :platform: Unix
   :synopsis: Everything needed to initialise a logger.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
                  

"""
import logging
import os
############################################################################# 

def Initialise_logging(loggername, log_dir=None):
    """
    This function defines and configures a logger instance.
     
    Parameters
    ---------- 
        loggername : str
            Name of the logger that will be also the log filename.
        log_dir : str, optional
            Directory where the log file should be written. Defaults to the
            current working directory when not provided.
    Returns
    -------
        logger : logging.logger
            instance of the initialised logger.        
    """
# create logger with 'plotting'
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid stacking handlers when the same logger name is initialised multiple times.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, loggername + '.log')
    else:
        log_path = loggername + '.log'

    # create file handler which logs even debug messages
    fileh = logging.FileHandler(log_path)
    fileh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    consoleh = logging.StreamHandler()
    consoleh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatterfile = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatterconsole = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    fileh.setFormatter(formatterfile)
    consoleh.setFormatter(formatterconsole)
    # add the handlers to the logger
    logger.addHandler(fileh)
    logger.addHandler(consoleh)
    
#    logger.info('creating an instance of auxiliary_module.Auxiliary')
#    a = auxiliary_module.Auxiliary()
#    logger.info('created an instance of auxiliary_module.Auxiliary')
#    logger.info('calling auxiliary_module.Auxiliary.do_something')
#    a.do_something()
#    logger.info('finished auxiliary_module.Auxiliary.do_something')
#    logger.info('calling auxiliary_module.some_function()')
#    auxiliary_module.some_function()
#    logger.info('done with auxiliary_module.some_function()')
    return logger

############################################################################# 
def PrintDiff(f1,f2,logger):
    """
    This function compute and print the difference between 2 files.
    It try to reproduce the *NIX "diff" command
     
    Parameters
    ---------- 
        f1 : str
            Name of the first file to compare.
        f2 : str
            Name of the second file to compare.
        logger : logging.logger
             instance of an initialised logger.        
   Returns
   -------
        Not much : str
            write in the logger the diff line by line 
    """
    import difflib
    diff = difflib.ndiff(open(f1,'r').readlines(),open(f2,'r').readlines())
#    diff = difflib.unified_diff(open(f1,'r').readlines(),open(f2,'r').readlines(), fromfile=f1, tofile=f2, n=("--lines 3"))
    changes = [l for l in diff if l.startswith('+ ') or l.startswith('- ')]
    logger.debug('-------------------------------')
    logger.debug('- %s',f1)
    logger.debug('+ %s',f2)
    logger.debug('-------------------------------')
    for c in changes:
       logger.debug('%s',c[:-1])
    return   
