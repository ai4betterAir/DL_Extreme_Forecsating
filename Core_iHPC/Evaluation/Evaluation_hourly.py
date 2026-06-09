import numpy as np
from scipy.stats import pearsonr

class Evaluation_hourly_class:
  def __init__(self, forecast_hours, trueOBS_hours):
    """
    Initializes the Evaluation class with forecast and actual values.
    This evaluation aims to calculate the forecast performance at each hour
    e.g., Calculate errorr for all forecast at T0, then T1, .... T_forecast_length
    The evaluation for each hour can evaluate the forecast capacity of model.
    --> Expected results: Performance of T0 > T1 > T2 > ... > T_forecast_length
    
    Parameters:
    forecast (numpy array): The forecasted values.
    actual (numpy array): The actual values.
    """
    self.forecast = 0  ### (initialized value)
    self.actual = 0   ### (initialized value)

    self.forecast_vector = []
    self.actual_vector = []
    
    self.forecast_array = forecast_hours  ### (n_timesteps, n_samples, n_stations)
    self.actual_array = trueOBS_hours   ### (n_timesteps, n_samples, n_stations)
    # print("array")
    # print(self.forecast_array.shape)
  
#######################################################################################
  def mae(self):
    """
    Calculates the mean absolute error (MAE) between the forecast and actual values.
    
    Returns:
    float: The mean absolute error between the forecast and actual values.
    """
    return np.round(np.mean(np.abs(self.forecast - self.actual)),3)
  
  def mape(self):
    """
    Calculates the mean absolute percentage error (MAPE) between the forecast and actual values.
    
    Returns:
    float: The mean absolute percentage error between the forecast and actual values.
    """
    return np.round(np.mean(np.abs((self.actual - self.forecast) / self.actual)) * 100,3)
  
  def rmse(self):
    """
    Calculates the root mean squared error (RMSE) between the forecast and actual values.
    
    Returns:
    float: The root mean squared error between the forecast and actual values.
    """
    return np.round(np.sqrt(np.mean((self.forecast - self.actual)**2)),3)
  
  def pearson_r(self):
    """
    Calculates the Pearson correlation coefficient (r) between the forecast and actual values.
    
    Returns:
    float: The Pearson correlation coefficient between the forecast and actual values.
    """

    return np.round(pearsonr(self.forecast_vector, self.actual_vector)[0],3)

  
  def r_square(self):
    """
    Calculates the coefficient of determination (R^2) between the forecast and actual values.
    
    Returns:
    float: The coefficient of determination between the forecast and actual values.
    """
    y_bar = np.mean(self.actual_vector)
    ss_tot = np.sum((self.actual_vector - y_bar)**2)
    ss_res = np.sum((self.forecast_vector - self.actual_vector)**2)
    return np.round(1 - (ss_res / ss_tot),3)
  
  def mean_forecast(self):
    """
    Calculates the mean of the forecast values.
    
    Returns:
    float: The mean of the forecast values.
    """
    return np.round(np.mean(self.forecast_vector),3)

  def mean_obs(self):
    """
    Calculates the mean of the forecast values.
    
    Returns:
    float: The mean of the forecast values.
    """
    return np.round(np.mean(self.actual_vector),3)

  def std_forecast(self):
    """
    Calculates the standard deviation of the forecast values.
    
    Returns:
    float: The standard deviation of the forecast values.
    """
    return np.round(np.std(self.forecast_vector),3)
  
  def std_obs(self):
    """
    Calculates the standard deviation of the forecast values.
    
    Returns:
    float: The standard deviation of the forecast values.
    """
    return np.round(np.std(self.actual_vector),3)
  
  
  def max_bias_error(self):
    """
    Calculates the maximum bias error between the forecast and actual values.
    
    Returns:
    float: The maximum bias error between the forecast and actual values.
    """
    return np.round(np.max(np.abs(self.forecast_vector - self.actual_vector)),3)

  def Index_of_Agreement(self):
      mean_obs = np.mean(self.forecast_vector)
      SSE = np.sum((self.actual_vector - self.forecast_vector)**2)
      PSE = np.sum((np.abs(self.actual_vector-mean_obs) + np.abs(self.forecast_vector-mean_obs))**2)
      d = 1-(SSE/PSE)
      return d

#######################################################################################
  def run_all(self):
    """
  
    Returns:
    float: array/list of hourly metrics
    """
    ### Get the dimension of input forecast_array(n_steps, n_segments, n_outputs) (output: deterministic & BNN)
    forecast_length = self.forecast_array.shape[0]  
    forecast_sample = self.forecast_array.shape[1]
    forecast_station = self.forecast_array.shape[2]

    ### Define vectors of metrics
    mae_vector, rmse_vector, mape_vector = list(),list(),list()
    pearson_r_vector, r_square_vector = list(), list()
    mean_forecast_vector, mean_obs_vector = list(), list()
    std_forecast_vector, std_obs_vector = list(), list()
    max_bias_error_vector = list()
    IOA_vector = list()


    ### Calculate metrics over each hour and for each station
    for hour in range(forecast_length):
        mae_sample, rmse_sample, mape_sample = list(),list(),list()
        pearson_r_sample, r_square_sample = list(), list()
        mean_forecast_sample, mean_obs_sample = list(), list()
        std_forecast_sample, std_obs_sample = list(), list()
        max_bias_error_sample = list()
        IOA_sample = list()

        self.forecast_vector = []
        self.actual_vector = []

        for station in range(forecast_station):
            ####### Calculate the errors for each forecast value/sample #########
            mae_station = list()
            rmse_station = list()
            mape_station = list()

            for sample in range(forecast_sample):
              
              ### Extract each value of each hour
              self.forecast = self.forecast_array[hour, sample, station]    ## a scaler 
              self.actual = self.actual_array[hour, sample, station]        ## a scaler      

              ### Calculate metric of each scaler in each timestep
              mae_station.append(self.mae())
              rmse_station.append(self.rmse())
              mape_station.append(self.mape())

            
            ### Accummulate the errors over all target stations at each hour
            mae_sample.append(np.mean(mae_station))
            rmse_sample.append(np.mean(rmse_station))
            mape_sample.append(np.mean(mape_station))
            ######################################################################

            ####### Calculate the other metrics WITH VECTORS over all samples at each hour ######
            self.forecast_vector = self.forecast_array[hour, :, station]
            self.actual_vector = self.actual_array[hour, :, station]

            pearson_r_sample.append(self.pearson_r())
            r_square_sample.append(self.r_square())
            mean_forecast_sample.append(self.mean_forecast())
            mean_obs_sample.append(self.mean_obs())
            std_forecast_sample.append(self.std_forecast())
            std_obs_sample.append(self.std_obs())
            max_bias_error_sample.append(self.max_bias_error())
            IOA_sample.append(self.Index_of_Agreement())

            ######################################################################

        ### Accummulate the errors and other metrics over all hours
        mae_vector.append(mae_sample)
        rmse_vector.append(rmse_sample)
        mape_vector.append(mape_sample)

        pearson_r_vector.append(pearson_r_sample)
        r_square_vector.append(r_square_sample)

        mean_forecast_vector.append(mean_forecast_sample)
        mean_obs_vector.append(mean_obs_sample)
        std_forecast_vector.append(std_forecast_sample)
        std_obs_vector.append(std_obs_sample)

        max_bias_error_vector.append(max_bias_error_sample)
        IOA_vector.append(IOA_sample)

    return mae_vector, rmse_vector, mape_vector, pearson_r_vector, \
          r_square_vector, mean_forecast_vector, mean_obs_vector, \
          std_forecast_vector, std_obs_vector, max_bias_error_vector, IOA_vector

