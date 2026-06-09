"""
..  module:: PandasToTensor
    :platform: Unix
    :synopsis: Definition of the basic object class to convert Pandas dtaframe into Tensorflow datasets.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
"""

import sys
import os
import stat
import numpy as np
import pandas as pd
import tensorflow as tf

###########################################################################################
class PandasToTensor(object):
    """ 
    This class defines a PandasToTensor, that contains the capacity to convert Pandas dtaframe into Tensorflow datasets.

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


        return
###########################################################################################
    ###########################################################################################
    def convert_pd_dict_to_ds(self, 
                              windowed_train_input_data_dict, 
                              windowed_train_label_data_dict, 
                              ordered_keys,
                              batch_size, 
                              debug=False,
                              ):
        """
        Convert 2 dict ({input}, {labels}) into a dataset
        """
        debug = True

        if debug:
            print("*"*100)
            print("windowed_input_data_dict")
            # print(windowed_train_input_data_dict)
        


        # key_list = sorted(list(windowed_train_input_data_dict.keys()))
        #############################
        input_list = [[windowed_train_input_data_dict[ii].values] for ii in ordered_keys]
        label_list = [[windowed_train_label_data_dict[ii].values] for ii in ordered_keys]
                
        #############################
        train_ds = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(
                    np.stack(input_list,  axis=0,)
                    ), 
                tf.data.Dataset.from_tensor_slices(
                    np.stack(label_list,  axis=0,)
                    ), 
                ),
            ).unbatch().batch(batch_size)#.batch(batch_size),# drop_remainder=True)


        if debug:
            print("*"*100)
            print("converted tensorflow dataset")
            print(train_ds)
            # print(list(train_ds.as_numpy_iterator()))
            
            # stop()      
            print(batch_size)  
        return train_ds
    ###########################################################################################
    def convert_forecast_pd_to_tensor_ds(self, 
                            input_pd,
                            debug=False,
                            ):
        """
        Convert a forecast panda into the tensorflow dataset shape to feed th model
        """
       ###############################################
        # create tensorflow tensor
        input_tf = tf.convert_to_tensor(input_pd)
         ###############################################
        # create tensorflow dataset
        input_shaped_tf = tf.expand_dims(
            input_tf, axis=0
            )
       
        input_ds = tf.data.Dataset.from_tensors(input_shaped_tf)
        return input_ds
    ###########################################################################################
    def convert_forecast_pd_to_tensor_bnn(self, 
                            input_pd,
                            debug=False,
                            ):
        """
        Convert a forecast panda into the tensorflow dataset shape to feed th model
        """
       ###############################################
        # create tensorflow tensor
        input_tf = tf.convert_to_tensor(input_pd)
         ###############################################
        # create tensorflow dataset
        input_shaped_tf = tf.expand_dims(
            input_tf, axis=0
            )
       
        # input_ds = tf.data.Dataset.from_tensors(input_shaped_tf)
        return input_shaped_tf


###########################################################################################
if __name__ == '__main__':
    import logging
    from Core.Tools import InitLogging as IL
    
    justif = 102
