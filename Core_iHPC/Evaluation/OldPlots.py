
import sys
import os
import stat
import datetime as dtime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import string



###########################################################################################
class Plot_Class(object):
    """ 
    This class defines a Plot_Class, that contains the functions for plotting
    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, logger, justif, Configuration,
                ):

        self.logger = logger
        self.justif = justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Plotting the results for evaluations'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        self.folder_path = '/home/nguyenh/Project/cnn_lstm_forecast/images/'

        self.Configuration = Configuration

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
        os.umask(0)
        mod664 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IROTH  
        mod666 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IROTH |stat.S_IWOTH 
        # os.chmod(path, mod664)
        os.chmod(path, mod666)
        return    
    ###########################################################
    def plot_training_loss (self, train_hist, iterative
                                  ):
                # # evaluate the model
        # print(train_hist.history.keys())
   
        
        if self.Configuration.train_model:

            #### plot training history losses
            history = {}

            history['train_{}'.format(str(iterative))] = train_hist.history['loss']
            history['val_{}'.format(str(iterative))] = train_hist.history['val_loss']

            plt.figure(figsize=(12,8))
        
            
            for key, value in history.items():
                plt.plot(value, label = key)
                plt.ylabel('Loss', fontsize=14)
                plt.xlabel('Epochs',fontsize=14)

                title = "Train_validate_losses_{0}_{1}_{2}_{3}_{4}_{5}_{6}.png".format(self.Configuration.n_steps_in, \
                    self.Configuration.n_steps_out, self.Configuration.var_to_predict[0], self.Configuration.selected_region, \
                    self.Configuration.forecast_method, self.Configuration.batch_size, self.Configuration.n_epochs)
                plt.title(title)
                plt.legend()
                
            dir = self.Configuration.Main_Model_training_evaluation_plot_training_loss_full_dir   
            full_path = os.path.join(dir, title)            
            self.MakeDir(dir)
            plt.savefig(full_path)
            self.Change_permissions(full_path)


            #### plot RMSE 
            history = {}
            history['train_{}'.format(str(iterative))] = train_hist.history['root_mean_squared_error']
            history['val_{}'.format(str(iterative))] = train_hist.history['val_root_mean_squared_error']

            plt.figure(figsize=(12,8))
        
            #### plot training history
            for key, value in history.items():
                plt.plot(value, label = key)
                plt.ylabel('RMSE', fontsize=14)
                plt.xlabel('Epochs',fontsize=14)

                title = "Training_loss/Train_validate_RMSE_{0}_{1}_{2}_{3}_{4}_{5}_{6}.png".format(self.Configuration.n_steps_in, \
                    self.Configuration.n_steps_out, self.Configuration.var_to_predict[0], self.Configuration.selected_region, \
                    self.Configuration.forecast_method, self.Configuration.batch_size, self.Configuration.n_epochs)
                plt.title(title)
                plt.legend()
            
            dir = self.Configuration.Main_Model_training_evaluation_plot_full_dir
            full_path = os.path.join(dir, title)            
            self.MakeDir(dir)
            plt.savefig(full_path)
            self.Change_permissions(full_path)

        
            # static_dir = os.path.join(self.folder_path, "Histograms", self.Configuration.forecast_method, str(self.Configuration.n_steps_out))
            # Check if the folder exists
            
        return
    # #################################################################################################
    # def plot_distribution(self, i, probs_predictions, RealOBS_y, determinstic_prediction, output_probs_mean, step_i):
    #         # print('Plotting histogram of significant values')
    #         plt.figure()
    #         ### get the histogram and infer the highest values
    #         hist, bins, _  = plt.hist(probs_predictions[:, i,  int(self.Configuration.n_steps_out/step_i), 1], bins=20, edgecolor='black', density=1)
            
    #         bin_index = np.argmax(hist)

    #         Kernel_inference = bins[bin_index]
    #         # KDEST.append(Kernel_inference)
    #         Deterministic_forecast = determinstic_prediction[i, int(self.Configuration.n_steps_out/step_i),1]
    #         Real_OBS = RealOBS_y[i, int(self.Configuration.n_steps_out/step_i),1]

    #         Accurate_improved = np.round(100*(np.abs(Kernel_inference-Real_OBS)/np.abs(Deterministic_forecast-Real_OBS)),2)

    #         # Add a vertical line
    #         plt.axvline(output_probs_mean[i, int(self.Configuration.n_steps_out/2),1], color='red', linewidth = 2, label = 'Calibrated_inference')
    #         plt.axvline(Deterministic_forecast, color='blue', linewidth = 2, label = 'Normal_forecast')
    #         plt.axvline(Real_OBS, color='green', linewidth = 2.5, label = 'Real_OBS')
    #         plt.axvline(Kernel_inference, color='purple', linewidth = 2, label = 'Mean_distribution')
    #         if self.Configuration.var_to_predict  == ["O3"]:
    #               plt.xlabel('Ozone_concentration [pphm]')
    #         else:
    #               plt.xlabel('Particle_concentration [ug/m^3]')
    #         plt.ylabel('Probability')
    #         plt.title("Improved: {} %".format(np.abs(100-Accurate_improved)))
    #         plt.legend()
    #         title = 'Histogram_{}_Step_{}th_{}_{}.jpg'.format(i, step_i, self.Configuration.var_to_predict[0], self.Configuration.forecast_method)
    #         plt.title(title)
    #         plt.savefig(self.folder_path + "Histograms/" + title )

    #################################################################################################
    def plot_distribution(self, i,  probs_predictions, Real_OBS, Deterministic_forecast, Calib_inference, step_i, station):
 
            plt.figure()
            # print("plotting")
            ### get the histogram and infer the highest values
            hist, bins, _  = plt.hist(probs_predictions[:, i, step_i , 0], bins=20, edgecolor='black', density=1)
            
            bin_index = np.argmax(hist)

            Kernel_inference = bins[bin_index]
            # KDEST.append(Kernel_inference)
            
            #### Percentage improvement
            BNN_improved = np.round(100*(np.abs(np.abs(Kernel_inference-Real_OBS)-np.abs(Deterministic_forecast-Real_OBS))/np.abs(Deterministic_forecast-Real_OBS)),2)
            Calibrated_improved = np.round(100*(np.abs(np.abs(Calib_inference-Real_OBS)-np.abs(Deterministic_forecast-Real_OBS))/np.abs(Deterministic_forecast-Real_OBS)),2)

            #### Distance - gaps
            BNN_gap = np.round((Real_OBS - Kernel_inference),2)
            Calibrated_gap = np.round((Real_OBS - Calib_inference),2)
            Deterministic_gap = np.round((Real_OBS - Deterministic_forecast),2)

            # Add a vertical line
            plt.axvline(Deterministic_forecast, color='blue', linewidth = 2, label = 'Normal_forecast')
            plt.axvline(Real_OBS, color='green', linewidth = 2.5, label = 'Real_OBS')

            file_name = 'Histogram_{}_Step_{}th_{}_{}.jpg'.format(i, step_i, self.Configuration.var_to_predict[0], self.Configuration.forecast_method)

            if (Calib_inference > 0):
                plt.axvline(Calib_inference, color='red', linewidth = 2.5, label = 'Calibrated_inference')
                # title = file_name   + "\n Original Gap: {} pphm".format(Deterministic_gap)\
                #                 + " -- Calib Gap: {} pphm".format(Calibrated_gap)\
                #                 + "\n Improved: {}%".format(np.round(100*(Deterministic_gap-Calibrated_gap)/Deterministic_gap),2)
                file_name = "Peak_" +  file_name
                # print(file_name)
                # sdsa
            else:
                plt.axvline(Kernel_inference, color='red', linewidth = 2, label = 'Mean_distribution')
                # title = file_name   + "\n Original Gap: {} pphm".format(Deterministic_gap)\
                #                 + " -- BNN Gap: {} pphm".format(BNN_gap)\
                #                 + "\n Improved: {}%".format(np.round(100*(Deterministic_gap-BNN_gap)/Deterministic_gap),2)

            if self.Configuration.var_to_predict  == ["O3"]:
                  plt.xlabel('Ozone_concentration [pphm]')
            else:
                  plt.xlabel('Particle_concentration [ug/m^3]')
            plt.ylabel('Probability')
            
            plt.legend()
            
            # title = file_name + "\n BNN Improved: {} %".format(BNN_improved)\
            #                     + "\n Calib Improved: {} %".format(Calibrated_improved)
            header =  (station + "_" + str(self.Configuration.n_steps_out) + "-hour forecast")
            # title = header + "\n BNN % Improved: {} %".format(BNN_improved)\
            #                     + "\n Calib % Improved: {} %".format(Calibrated_improved)
            title = header  + "\n Calibration Improved: {} %".format(Calibrated_improved)
            
            plt.title(title)
            
            
            dir = self.Configuration.Main_Model_training_evaluation_plot_histogram_full_dir
            self.MakeDir(dir)

        
            # static_dir = os.path.join(self.folder_path, "Histograms", self.Configuration.forecast_method, str(self.Configuration.n_steps_out))
            # Check if the folder exists


            full_path = os.path.join(dir, file_name)
            plt.savefig(full_path)
            self.Change_permissions(full_path)

            plt.clf()
            plt.close()
 

#################################################################################################            
    def plot_distribution_singlestation(self, i, probs_predictions, RealOBS_y, determinstic_prediction, step_i, station):
            # print('Plotting histogram of significant values')
            plt.figure()
            # print(probs_predictions.shape)

        
            ### get the histogram and infer the highest values
            hist, bins, _  = plt.hist(probs_predictions[:, i,  int(self.Configuration.n_steps_out/step_i)], bins=20, edgecolor='black', density=1)
            bin_index = np.argmax(hist)

            Kernel_inference = bins[bin_index]
            # KDEST.append(Kernel_inference)
            Deterministic_forecast = determinstic_prediction[i, int(self.Configuration.n_steps_out/step_i)]
            Real_OBS = RealOBS_y[i, int(self.Configuration.n_steps_out/step_i)]

            Accurate_improved = np.round(100*(np.abs(Kernel_inference-Real_OBS)/np.abs(Deterministic_forecast-Real_OBS)),2)

            # Add a vertical line
            # plt.axvline(output_probs_mean[i, int(n_steps_out/2),1], color='red', linewidth = 2, label = 'mean_dist')
            plt.axvline(Deterministic_forecast, color='blue', linewidth = 2, label = 'Normal_forecast')
            plt.axvline(Real_OBS, color='green', linewidth = 2.5, label = 'Real_OBS')
            plt.axvline(Kernel_inference, color='red', linewidth = 2.5, label = 'Probabilistic_forecast')
            if self.Configuration.var_to_predict  == ["O3"]:
                  plt.xlabel('Ozone_concentration [pphm]')
            else:
                  plt.xlabel('Particle_concentration [ug/m^3]')
            plt.ylabel('Probability')
            plt.title("Improved: {} %".format(np.abs(100-Accurate_improved)))
            plt.legend()
            title = 'histogram_{}_Step_{}th_{}_{}_{}.jpg'.format(i, step_i, self.Configuration.var_to_predict[0], self.Configuration.forecast_method, station)
            print(title)
            plt.title(title)


            dir = self.Configuration.Main_Model_training_evaluation_plot_histogram_full_dir
            self.MakeDir(dir)

            full_path = os.path.join(dir, title)
            plt.savefig(full_path)
            self.Change_permissions(full_path)


            plt.clf()
            plt.close()
#################################################################################################            
    def plot_forecast(self, forecast_pd):
        """Plot all stations for the forecast
        
        """
        columns = list(forecast_pd.columns)
        print(columns)
        
        #remove forecast_hours columns fronm list
        columns.remove("forecast_hours")
        # columns.remove("datetime")
        
        mask = forecast_pd["forecast_hours"]<=0
        historical_data = forecast_pd.loc[mask, :].reset_index()
        mask = forecast_pd["forecast_hours"]>=0
        forecast_data = forecast_pd.loc[mask, :].reset_index()
        fig = plt.figure()
        
        for col in columns:
            splitted = col.split("_")
            var = splitted[0]
            station = "_".join(splitted[1:])
            # plot historical data
            colour = "k"
            plt.plot(historical_data["datetime"], historical_data[col], color=colour, label=station )
            # plot forecst data
            colour = "b"
            plt.plot(forecast_data["datetime"], forecast_data[col], color=colour, label=station )
        plt.title(var)
        plt.legend()
        plt.tight_layout()
        
        forecast_plot_dir = self.Configuration.Main_output_run_plots_full_dir    
        filename_template = self.Configuration.forecast_plots_filename_template
        
        forecast_plot_filename = filename_template.format(
            region = self.Configuration.selected_region,
            mode = self.Configuration.train_forecast_dir.lower(),
            var = self.Configuration.input_var_dir,
            inputs = self.Configuration.n_steps_in, 
            outputs = self.Configuration.n_steps_out, 
            additional_vars = self.Configuration.additional_var_dir, 
            model = self.Configuration.Model_dir,
            date = self.Configuration.start_date_aest.strftime("%Y%m%dAEST"),
            )
        
        forecast_plot_filename_fullpath =  os.path.join(
                forecast_plot_dir,
                forecast_plot_filename)
        

        self.MakeDir(forecast_plot_dir)

        plt.savefig(forecast_plot_filename_fullpath)
        
        self.Change_permissions(forecast_plot_filename_fullpath)
        
        self.logger.info('Forecast plot = {msg}'.format(msg=forecast_plot_filename).ljust(self.justif-7,'.') + 'WRITTEN')
        return


            
            
        
            