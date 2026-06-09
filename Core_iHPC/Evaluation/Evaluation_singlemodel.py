import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class Evaluation_class:
  ##############################################################
  def __init__(self, forecast, actual):
    """
    Initializes the Evaluation class with forecast and actual values.
    
    Parameters:
    forecast (numpy array): The forecasted values.
    actual (numpy array): The actual values.
    """
    self.forecast = forecast
    self.actual = actual
  
  def mae(self):
    """
    Calculates the mean absolute error (MAE) between the forecast and actual values.
    
    Returns:
    float: The mean absolute error between the forecast and actual values.
    """
    return np.round(np.mean(np.abs(self.forecast - self.actual)),3)

##############################################################  
  def mape(self):
    """
    Calculates the mean absolute percentage error (MAPE) between the forecast and actual values.
    
    Returns:
    float: The mean absolute percentage error between the forecast and actual values.
    """

    epsilon = 0.01 ### This small value to avoid infinite MAPE when actual values is = 0

    return np.round(np.mean(np.abs(self.actual - self.forecast) / (self.actual + epsilon)) * 100,3)
    
##############################################################  
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
    fc = self.forecast
    obs = self.actual

    # # Check if the object is a NumPy array
    # if isinstance(fc, np.ndarray) & isinstance(obs, np.ndarray):
    #     pass
    # else:### if this is not an array (n,) then must convert to vector
    #     fc = np.squeeze(fc.values)
    #     obs = np.squeeze(obs.values)

    ### Series converted into array has (n, 1) dim, that throws error for the pearson correlation --> (n,)
    if (fc.ndim > 1) or (obs.ndim > 1):
        fc = np.squeeze(fc)
        obs = np.squeeze(obs)

    return np.round(pearsonr(fc, obs)[0],3)

    # print(self.forecast)
    # print(self.actual)
    print(pearsonr(fc, obs))
    
    

##############################################################  
  def r_square(self):
    """
    Calculates the coefficient of determination (R^2) between the forecast and actual values.
    
    Returns:
    float: The coefficient of determination between the forecast and actual values.
    """
    y_bar = np.mean(self.actual)
    ss_tot = np.sum((self.actual - y_bar)**2)
    ss_res = np.sum((self.forecast - self.actual)**2)
    return np.round(1 - (ss_res / ss_tot),3)

 ############################################################## 
  def mean_forecast(self):
    """
    Calculates the mean of the forecast values.
    
    Returns:
    float: The mean of the forecast values.
    """
    return np.round(np.mean(self.forecast),3)

##############################################################
  def mean_obs(self):
    """
    Calculates the mean of the forecast values.
    
    Returns:
    float: The mean of the forecast values.
    """
    return np.round(np.mean(self.actual),3)

##############################################################
  def std_forecast(self):
    """
    Calculates the standard deviation of the forecast values.
    
    Returns:
    float: The standard deviation of the forecast values.
    """
    return np.round(np.std(self.forecast),3)
  
##############################################################  
  def std_obs(self):
    """
    Calculates the standard deviation of the forecast values.
    
    Returns:
    float: The standard deviation of the forecast values.
    """
    return np.round(np.std(self.actual),3)
  
##############################################################  
  def max_bias_error(self):
    """
    Calculates the maximum bias error between the forecast and actual values.
    
    Returns:
    float: The maximum bias error between the forecast and actual values.
    """
    return np.round(np.max(self.forecast - self.actual),3)

##############################################################  
  def max_hour_error(self):
    """
    Calculate the maximum error for each n_output-hour sequence in the predicted data.

    Parameters:
        observed (list or numpy array): A list or numpy array of observed data.
        predicted (list or numpy array): A list or numpy array of model-predicted data.

    Returns:
        list: A list containing the maximum error for each n_output-hour sequence.
    """

    predicted, observed = self.forecast, self.actual

    if len(observed) != len(predicted):
        raise ValueError("The length of observed and predicted arrays must be the same.")

    max_errors = []
    sequence_length = predicted.shape[-1]  # e.g., 12 hours per sequence

    for i in range(len(predicted)):
        # Calculate the maximum error for each sequence
        max_error = max(abs(observed[i] - predicted[i]))
        max_errors.append(max_error)

    return max_errors

##############################################################  
  def signed_error(self):
    """
    Calculates the difference between the forecast and actual values.
    This quantity can be used to evaluate if model more over-prediction (more positive errors) or under-prediction (more negatives)
    
    Returns:
    float: The the difference between the forecast and actual values.
    """
    return np.round(self.actual - self.forecast, 3)

##############################################################  
  def index_of_agreement(self):
    """
    Calculate the Index of Agreement (IA) for model evaluation.

    Parameters:
        observed (list or numpy array): A list or numpy array of observed data.
        predicted (list or numpy array): A list or numpy array of model-predicted data.

    Returns:
        float: The Index of Agreement (IA) value between 0 and 1.
    """
    if len(self.actual) != len(self.forecast):
        raise ValueError("The length of observed and predicted arrays must be the same.")

    # Calculate the mean of the observed values
    mean_observed = sum(self.actual) / len(self.actual)

    # Calculate the numerator and denominator for the Index of Agreement formula
    numerator = sum([(self.forecast[i] - self.actual[i]) ** 2 for i in range(len(self.actual))])
    denominator = sum([(abs(self.forecast[i] - mean_observed) + abs(self.actual[i] - mean_observed)) ** 2 for i in range(len(self.actual))])

    # Calculate the Index of Agreement
    ia = 1 - (numerator / denominator)

    return np.round(ia,3)
  
  ################# MAKE PLOTS FOR STATISTIC ######################

    def Taylor_plot(self,):
      
      return
    
    def Whisker_plot(self,):
      
      return
    
  def Compare_plot(self,):
      
      return


def bool_to_yes_no(value):
    return "yes" if bool(value) else "no"


def build_metrics_summary_row(configuration):
    variable = None
    if getattr(configuration, "var_to_predict", None):
        variable = configuration.var_to_predict[0]
    model_name = getattr(configuration, "model_name", None) or getattr(configuration, "Model_dir", None)
    if model_name is None:
        model_name = "unknown_model"
    return {
        "region": getattr(configuration, "selected_region", None),
        "variable": variable,
        "Decompose": bool_to_yes_no(getattr(configuration, "use_vmd_decomposition", False)),
        "USE_ONE_MODEL_PER_IMF": bool_to_yes_no(getattr(configuration, "use_one_model_per_imf", False)),
        "Model name": model_name,
        "n_steps_in": getattr(configuration, "n_steps_in", None),
        "n_steps_out": getattr(configuration, "n_steps_out", None),
    }


def build_evaluation_metrics_summary(configuration):
    report_dir = getattr(configuration, "Main_Model_training_full_dir", None)
    if not report_dir or not os.path.isdir(report_dir):
        return pd.DataFrame()

    rows = []
    for report_path in sorted(glob.glob(os.path.join(report_dir, "*_metrics_report_*.csv"))):
        report_name = os.path.basename(report_path)
        station_name = report_name.split("_metrics_report_", 1)[0]
        try:
            report_pd = pd.read_csv(report_path)
        except Exception:
            continue
        if not {"Metric", "Training", "Testing"}.issubset(report_pd.columns):
            continue

        base_row = build_metrics_summary_row(configuration)
        base_row["site"] = station_name
        base_row["source_file"] = report_name
        for _, metric_row in report_pd.iterrows():
            row = dict(base_row)
            row["Metric"] = metric_row.get("Metric")
            row["Training"] = metric_row.get("Training")
            row["Testing"] = metric_row.get("Testing")
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_dashboard_metrics_summary(configuration, dashboard_pd):
    if dashboard_pd is None or dashboard_pd.empty:
        return pd.DataFrame()

    frame = dashboard_pd.copy()
    if "Stations" in frame.columns and "site" not in frame.columns:
        frame = frame.rename(columns={"Stations": "site"})
    elif "site" not in frame.columns:
        frame["site"] = ""

    base_row = build_metrics_summary_row(configuration)
    for column, value in base_row.items():
        frame[column] = value
    ordered_columns = [
        "region",
        "site",
        "variable",
        "Decompose",
        "USE_ONE_MODEL_PER_IMF",
        "Model name",
        "n_steps_in",
        "n_steps_out",
    ]
    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame[ordered_columns + remaining_columns]


def write_summary_csv(output_path, summary_df):
    if summary_df is None or summary_df.empty:
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    return True
