"""
..  module:: TimeSeriesSegmentation
    :platform: Unix
    :synopsis: Definition of an object to flatten inputs dictionnary and segmentaing it into input/label.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""
import sys
import numpy as np
import pandas as pd
# import tensorflow as tf
from Core_iHPC.Tools import TimeSeriesWindowGenerator as TSWG
from Core_iHPC.Tools import InitLogging as IL

###########################################################################################
class WindowGenerator():
    def __init__(self, logger, justif,
                input_width, 
                label_width, 
                shift,
                input_columns=None, 
                label_columns=None,
                debug = False,
                ):

        self.logger = logger
        self.justif = justif

        self.label_columns = label_columns
        self.input_columns = input_columns

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.start_index = 0
        self.debug = debug
        return
###########################################################################################
    def segment_inputs(self,
                input_obj,
                split_indexed_dict_input,
                split_indexed_dict_label,
                ):
        
        if self.debug:
            print("type = ", type(input_obj))
            print("*"*100)


        match input_obj:
            case pd.DataFrame():
                # segment
                end_index = self.segment_pd_dataframe(
                                    input_obj,
                                    split_indexed_dict_input,
                                    split_indexed_dict_label,
                                    self.start_index,
                                    )
                self.start_index = end_index + 1
            case list():
                # dig into list and recurse
                for obj in input_obj:
                    if self.debug:
                        print("-"*100)
                        print("type = ", type(obj))
                        # print(obj)
                    end_index = self.segment_inputs(
                                        obj,
                                        split_indexed_dict_input,
                                        split_indexed_dict_label,
                                        )
                    # self.start_index = end_index + 1
                    
            case dict():
                # dig into dict  and recurse
                for key, obj in input_obj.items():
                    if self.debug:
                        print("-"*100)
                        print("type = ", type(obj))
                        # print(obj)
                    end_index = self.segment_inputs(
                                        obj,
                                        split_indexed_dict_input,
                                        split_indexed_dict_label,
                                        )
                    # self.start_index = end_index + 1
            case _:
                self.logger.error(
                    "Input object class pattern = {type} segmentation".format(type=type(input_obj)
                                                                              ).ljust(self.justif - 15 ,'.') + 'NOT IMPLEMENTED')
                sys.exit("Sorry :(")
        
        return self.start_index    
###########################################################################################
    def segment_pd_dataframe(self,
                input_pd,
                split_indexed_dict_input,
                split_indexed_dict_label,
                start_index,
                ):
        
        
        SWG = TSWG.WindowGenerator(
            self.input_width, 
            self.label_width, 
            self.shift,
            input_pd,
            None,
            None,
            input_columns=self.input_columns,
            label_columns=self.label_columns
            )
        list_of_list_input_indices, list_of_list_label_indices = SWG.make_list_indices_train()
            
        for index, indices_input in enumerate(list_of_list_input_indices):

            index_2 = index + start_index
            # print("key=",key, "index=", index, "index_2=", index_2, "total counter=", total_splits_counter)
            indices_label = list_of_list_label_indices[index]

            split_indexed_dict_input[index_2] = input_pd.iloc[indices_input].loc[:, self.input_columns].sort_index(axis=1)
            split_indexed_dict_label[index_2] = input_pd.iloc[indices_label].loc[:, self.label_columns].sort_index(axis=1)

        return index_2
        
        self.logger.info('windowing timeseries'.ljust(self.justif-2,'.') + 'OK')


###########################################################################################
if __name__ == '__main__':
   
    ################ logger init #################
    justif = 102
    loggername= "time_segmentation"
    logger = IL.Initialise_logging(loggername)
    
    input_width = 5
    label_width = 3
    shift = 1

    size = 10
    test_pd = pd.DataFrame(
        {
            "A":np.arange(size),
            "B":np.arange(size),
        },
    )
    print(test_pd)

    input_columns=["A", "B"] 
    label_columns=[ "B"] 
###############################################################################    
    WG = WindowGenerator(logger, justif,
                input_width, 
                label_width, 
                shift,
                input_columns=input_columns, 
                label_columns=label_columns,
                )

    
    split_indexed_dict_input = {}
    split_indexed_dict_label = {}
    
    ################################    
    # simple pd
    WG.segment_inputs(
                test_pd,
                split_indexed_dict_input,
                split_indexed_dict_label,
                )

    print("*"*100)
    print("simple pd")
    print("segmented input dict", split_indexed_dict_input)
    print("segmented label dict", split_indexed_dict_label)
    
###############################################################################    
    WG = WindowGenerator(logger, justif,
                input_width, 
                label_width, 
                shift,
                input_columns=input_columns, 
                label_columns=label_columns,
                )

    
    split_indexed_dict_input = {}
    split_indexed_dict_label = {}
    
    ################################    
    ##############################
    # list of pd
    
    input_obj = [test_pd,test_pd, [test_pd,test_pd]]
    WG.segment_inputs(
                input_obj,
                split_indexed_dict_input,
                split_indexed_dict_label,
                )

    print("*"*100)
    print("list of pd")
    print("segmented input dict", split_indexed_dict_input)
    print("segmented label dict", split_indexed_dict_label)
    
###############################################################################    
    WG = WindowGenerator(logger, justif,
                input_width, 
                label_width, 
                shift,
                input_columns=input_columns, 
                label_columns=label_columns,
                )

    
    split_indexed_dict_input = {}
    split_indexed_dict_label = {}
    
    ################################    
    ##############################
    # dict of pd
    
    input_obj = {
        "AA":test_pd,
        "BB": test_pd, 
        "CC":[test_pd,test_pd],
        "DD":{"EE":test_pd, "FF":[test_pd,test_pd],},
    }
    WG.segment_inputs(
                input_obj,
                split_indexed_dict_input,
                split_indexed_dict_label,
                )

    print("*"*100)
    print("dicts")
    print("segmented input dict", split_indexed_dict_input)
    print("segmented label dict", split_indexed_dict_label)
    
        