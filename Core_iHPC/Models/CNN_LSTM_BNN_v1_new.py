"""
..  module:: CNN_LSTM_BNN
    :platform: Unix
    :synopsis: Definition of the basic object class to configure and run CNN_LSTM model.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

import tensorflow.keras as TFK
import tensorflow.keras.layers as TKL 
import tensorflow.keras.models as TKM
# from tensorflow.keras.utils import plot_model
# # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.callbacks as TKC
import tensorflow.keras.losses as TKLoss
import tensorflow.keras.metrics as TKMetrics
import tensorflow.keras.optimizers as TKO
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization
# import bayes_opt



from . import CNN_LSTM_BNN_v1_generic as generic

### Set ramdom seed at fix value to make fixed reproduced simulation
# tf.random.set_seed(1)
###########################################################################################
class CNN_LSTM_BNN_Class(generic.CNN_LSTM_BNN_generic_Class):
    """ 
    This class defines a CNN_LSTM_Class, that configure a CNN LSTM model, prepare its data, train the model and produce a forecast. 

    Attributes
    -----------
       
    """
    def __init__(self, Configuration,
                 ):

        """
        This method defines the trainin stopping criteria and train the model using the train and validation dataset.

        Parameters
        -----------
        Configuration : Configuration.Configuration object
            instance of a configuration object that holds all th parameter of the model.

        Attributes
        ----------        
        model_save_dir : str    
            directory where to save/load the model file.

        """
        super().__init__(Configuration)
        
        # set the global paramaters bounds
        self.global_optimisation_parameters_bounds = {
            "lstm_1_units" : (10, 250),
            "cnn_1_kernel_size": (1, self.Configuration.n_steps_in), 
            "cnn_1_filters": (1, self.Configuration.n_steps_in),
        }
        
        self.Bayesian_global_opmimiser_params = {
            "init_points" : 7, 
            "n_iter" : 70,
        }
        
        # # test purpose
        # self.Bayesian_global_opmimiser_params = {
        #     "init_points" : 2, 
        #     "n_iter" : 2,
        # }

        # n_steps_in = self.Configuration.n_steps_in
        # n_steps_out = self.Configuration.n_steps_out
        # n_epochs = self.Configuration.n_epochs

        return
        
###########################################################################################
    def define_model(self,
                    n_steps_in, 
                    n_steps_out, 
                    n_features_in, 
                    n_features_out, 
                    build_architecture,
                    ):
        """
        This is Hubert's v1 model
        Input -> CNN1 -> CNN2 -> LSTM1 -> Dropout -> LSTM2 -> Dense -> Output
        A static normalisation/denormalisation layer have been added to the data flow so the normalising coefficients are saved into the model, 
        and constant once the training is done, so every other run will be made using the same values.
        """                     

        self.model_save_filename_template = "cnn_lstm_bnn_model_v1_{vars}_{nstepin}_{nstepout}_{nepoch}.h5"

        if build_architecture:
            # #############
            # # for multi GPUs
            # # Create a MirroredStrategy.
            # strategy = tf.distribute.MirroredStrategy()
            # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            # #############


            ####################### CNN-LSTM old version ##########################
            if self.Configuration.model_parameters_dict["lstm_1_units"] is not None:
                lstm_1_units = int(self.Configuration.model_parameters_dict["lstm_1_units"])
            else:
                lstm_1_units=  128

            if self.Configuration.model_parameters_dict["cnn_1_filters"] is not None:
                cnn_1_filters = int(self.Configuration.model_parameters_dict["cnn_1_filters"])
            else:
                cnn_1_filters = 12

            if self.Configuration.model_parameters_dict["cnn_1_kernel_size"] is not None:
                cnn_1_kernel_size = int(self.Configuration.model_parameters_dict["cnn_1_kernel_size"])
            else:
                cnn_1_kernel_size = 12

            # print("*"*100)
            # print("model configuration")
            # print("cnn_1_kernel_size: {}".format(cnn_1_kernel_size))
            # print("cnn_1_filters: {}".format(cnn_1_filters))
            # print("lstm_1_units: {}".format(lstm_1_units))
            # print("*"*100)
            
            
            dropout_rate = 0.4
            lstm_2_units= 64
            padding = 'same'
            activation = 'relu'
            activation_lstm = 'tanh'

            Dense = n_steps_out*n_features_out

            # ### 24-24 
            # input_layer =   TKL.Input(shape=(n_steps_in, n_features_in)) 
            # conv =       TKL.Conv1D(filters=cnn_1_filters, kernel_size=cnn_1_kernel_size, activation=activation, padding = padding, name='conv1')(input_layer)
            # conv =       TKL.Dropout(dropout_rate)(conv)
            # lstm =       TKL.LSTM(lstm_1_units, return_sequences=True, activation=activation_lstm, name='lstm1', bias_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(conv)
            # lstm =       TKL.Dropout(dropout_rate)(lstm)
            # output_layer =      TKL.Dense(n_features_out, activation=activation, name='dense1')(lstm)
            # self.model =    TKM.Model([input_layer], [output_layer])

            # #############
            # # for multi GPUs
            # # Open a strategy scope.
            # with strategy.scope():

            ### 48-48 
            input_layer = TKL.Input(shape=(n_steps_in, n_features_in)) 
            input_layer = TKL.Dropout(dropout_rate)(input_layer)
            conv =      TKL.Conv1D(filters=cnn_1_filters, kernel_size=cnn_1_kernel_size, activation=activation, padding = padding, name='conv1')(input_layer)
            conv =      TKL.Dropout(dropout_rate)(conv)
            lstm =      TKL.LSTM(lstm_1_units, return_sequences=True, activation=activation_lstm, name='lstm1', bias_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(conv)
            lstm =      TKL.Dropout(dropout_rate)(lstm)
            lstm =      TKL.Dense(n_features_out*n_steps_out, activation=activation, name='dense1')(lstm)
            lstm =      TKL.Dropout(dropout_rate)(lstm)
            output_layer =  TKL.Dense(n_features_out, name='dense_out')(lstm)
            self.model =    TKM.Model([input_layer], [output_layer])


            # input_layer =   TKL.Input(shape=(n_steps_in, n_features_in)) 
            # conv =       TKL.Conv1D(filters=cnn_1_filters, kernel_size=cnn_1_kernel_size, activation=activation, padding = padding, name='conv1')(input_layer)
            # # conv = TKL.BatchNormalization()(conv)
            # # conv =       TKL.Conv1D(filters=cnn_2_filters, kernel_size=cnn_2_kernel_size, activation=activation, padding = padding, name='conv2')(conv)
            # conv = TKL.BatchNormalization()(conv)
            # lstm =       TKL.LSTM(lstm_1_units, return_sequences=True, activation=activation_lstm, name='lstm1', bias_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(conv)
            # lstm =  TKL.BatchNormalization()(lstm)
            # lstm =       TKL.Dropout(dropout_rate)(lstm)
            # lstm =       TKL.LSTM(lstm_2_units, activation=activation_lstm, name='lstm2')(lstm)
            # lstm =       TKL.Dropout(dropout_rate)(lstm)
            # lstm = TKL.BatchNormalization()(lstm)
            # dense =      TKL.Dense(n_steps_out*n_features_out, activation=activation, name='dense1')(lstm)
            # dense =      TKL.Dropout(dropout_rate)(dense)
            # dense = TKL.BatchNormalization()(dense)
            # output_layer =  TKL.Reshape((n_steps_out,n_features_out), name='out_reshape')(dense)
            # # denormalisation=self.denormalisation_layer(output_layer)
            # # denormalisation = output_layer
            # self.model =    TKM.Model([input_layer], [output_layer])
            


                
            # loss_function = [TKLoss.MeanSquaredError(), TKLoss.MeanAbsoluteError()]
            loss_function = self.custom_loss_function
            # loss_weights = [1.,1.]
            # loss_function = TKLoss.RootMeanSquaredError()

            metrics_list=[
                TKMetrics.MeanAbsoluteError(),
                TKMetrics.RootMeanSquaredError(),
                ]
            
            # optimizer = TKO.SGD()
            optimizer = TKO.Adam()
            
            self.model.compile(optimizer=optimizer, 
                                loss=loss_function,
                                # loss_weights = loss_weights,
                                metrics = metrics_list,
                                run_eagerly=False)

        # print(self.model.summary())
        
        
        self.logger.info('CNN-LSTM-BNN model v1'.ljust(self.justif-10,'.') + 'CONFIGURED')
        return




        
        
