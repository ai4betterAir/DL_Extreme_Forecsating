Sparse LSTM Air Quality Forecast
=================================

A deep learning pipeline for air quality forecasting (PM2.5, O3, etc.) using a Sparse LSTM
model with Variational Mode Decomposition (VMD) and TemporalMICE imputation. The pipeline
supports both training new models and running operational forecasts using pre-trained weights,
across multiple forecast horizons (3 h, 6 h, 9 h, 12 h).


REQUIREMENTS
------------
- Python 3.10 (python3.10, python3.10-venv, python3.10-dev)
- Git
- Access to the DPE AQMS API (for data download)


INSTALLATION
------------

1. Clone the repository

    git clone git@bitbucket.org:<your-org>/<repo-name>.git
    cd <repo-name>
    git checkout aidfan_20251014

2. Install Python 3.10 (Ubuntu)

    sudo apt install python3.10 python3.10-venv python3.10-dev

3. Create the virtual environment

    The environment MUST be named sparse-vmd.

    python3.10 -m venv sparse-vmd
    source sparse-vmd/bin/activate        # Linux / macOS
    # sparse-vmd\Scripts\activate         # Windows

4. Install dependencies

    pip install --upgrade pip
    pip install tensorflow==2.15.0
    pip install keras==2.15.0
    pip install numpy==1.25.2
    pip install pandas==2.1.0
    pip install matplotlib==3.7.3
    pip install scikit-learn==1.3.0
    pip install vmdpy==0.2
    pip install openpyxl==3.1.5
    pip install seaborn

5. Verify TensorFlow and Keras versions

    python3 -c "import tensorflow as tf; import keras; print('TF:', tf.__version__); print('Keras:', keras.__version__)"

    Both should report 2.15.0.


PROJECT STRUCTURE
-----------------

.
+-- main.py                              <- Entry point
+-- requirements.txt
+-- Core_iHPC/
|   +-- Configuration/
|   +-- Inputs/                          <- Input_manager, AQMS API
|   +-- Models/
|   |   +-- Sparse_LSTM_v1.py           <- Unified train/forecast model
|   +-- Processing/
|   |   +-- Imputation.py               <- TemporalMICE imputation
|   |   +-- VMD_decomposition.py        <- VMD signal decomposition
|   +-- Tools/
|       +-- Config_testing/             <- YAML configuration files
|           +-- SW_24_PM2.5_Sparse_LSTM_hr_3.yaml
|           +-- SW_24_PM2.5_Sparse_LSTM_hr_6.yaml
|           +-- SW_24_PM2.5_Sparse_LSTM_hr_9.yaml
|           +-- SW_24_PM2.5_Sparse_LSTM_hr_12.yaml
+-- AI_Runs/
|   +-- Training/                        <- Training outputs and model evaluation
|   +-- Forecast/                        <- Forecast outputs
|   |   +-- <region>/<var>/<var>/<model>/
|   |       +-- shared_data/            <- Shared imputed + VMD files (all horizons)
|   |       |   +-- Imputed_MICE_<region>_<var>_<station>.csv
|   |       |   +-- VMD_decomposition/
|   |       |       +-- VMD_<var>_<region>_<station>.csv
|   |       +-- <timestamp>/            <- Per-run forecast output files
|   +-- Model_weights/                   <- Saved model weights (.keras / .h5)
+-- AI_dashboard_files/                  <- Dashboard visualisation outputs


CONFIGURATION
-------------

Each forecast horizon has its own YAML file in Core_iHPC/Tools/Config_testing/.
The key switches are:

Training vs Forecast mode:
    train_model: false    # true = train, false = forecast with pre-trained model

Data download vs load from file:
    use_file: false       # false = download from API, true = load saved raw file

Imputation and VMD:
    impute: true          # Run TemporalMICE imputation
    decompose: true       # Run VMD decomposition

Forecast horizon (n_outputs):

    PM2.5
    Config file                            Horizon
    ------------------------------------   -------
    SW_24_PM2.5_Sparse_LSTM_hr_3.yaml      3 hours
    SW_24_PM2.5_Sparse_LSTM_hr_6.yaml      6 hours
    SW_24_PM2.5_Sparse_LSTM_hr_9.yaml      9 hours
    SW_24_PM2.5_Sparse_LSTM_hr_12.yaml    12 hours

    Ozone
    Config file                            Horizon
    ------------------------------------   -------
    Ozone_Sparse_LSTM_hr_3.yaml            3 hours
    Ozone_Sparse_LSTM_hr_6.yaml            6 hours
    Ozone_Sparse_LSTM_hr_9.yaml            9 hours
    Ozone_Sparse_LSTM_hr_12.yaml          12 hours


RUNNING THE PIPELINE
--------------------

Activate the environment first:
    source sparse-vmd/bin/activate

Run a single horizon
    To run a single pollutant and horizon, comment out the others in main.py:

        # PM2.5
        list_configs = [
            "SW_24_PM2.5_Sparse_LSTM_hr_3",
            #"SW_24_PM2.5_Sparse_LSTM_hr_6",
            #"SW_24_PM2.5_Sparse_LSTM_hr_9",
            #"SW_24_PM2.5_Sparse_LSTM_hr_12",
        ]

        # Ozone
        list_configs = [
            "Ozone_Sparse_LSTM_hr_3",
            #"Ozone_Sparse_LSTM_hr_6",
            #"Ozone_Sparse_LSTM_hr_9",
            #"Ozone_Sparse_LSTM_hr_12",
        ]

    Then run:
        python -m Core_iHPC.main

Run all forecast horizons
    To run all horizons for both pollutants in one go, enable all entries in list_configs:

        list_configs = [
            # PM2.5
            "SW_24_PM2.5_Sparse_LSTM_hr_3",
            "SW_24_PM2.5_Sparse_LSTM_hr_6",
            "SW_24_PM2.5_Sparse_LSTM_hr_9",
            "SW_24_PM2.5_Sparse_LSTM_hr_12",
            # Ozone
            "Ozone_Sparse_LSTM_hr_3",
            "Ozone_Sparse_LSTM_hr_6",
            "Ozone_Sparse_LSTM_hr_9",
            "Ozone_Sparse_LSTM_hr_12",
        ]

    When running multiple horizons, data is downloaded, imputed and decomposed only once
    (on hr_3) and automatically reused by the subsequent configs -- so runtime scales with
    the number of models, not the number of preprocessing steps.


OUTPUT FILE LOCATIONS
---------------------

Forecast outputs
    Each run creates a timestamped subdirectory:

    AI_Runs/Forecast/<region>/<var>/<var>/<model>/<timestamp>/
        +-- <region>_Forecast_<var>_<n_in>_<n_out>_<station>_0_<date>.csv

    Example:
    AI_Runs/Forecast/SLSYD/PM2.5/PM2.5/Sparse_LSTM_v1/2026030121UTC---tstamp---20260301_102316AEDT/

Shared preprocessing files (imputed + VMD)
    Written once and reused across all horizon runs:

    AI_Runs/Forecast/SLSYD/PM2.5/PM2.5/Sparse_LSTM_v1/shared_data/
        +-- Allobs_raw_DPE_station_api_SLSYD_PM2.5_PM2.5.csv   <- raw downloaded obs
        +-- Imputed_MICE_SLSYD_PM2.5_<station>.csv              <- one per station
        +-- VMD_decomposition/
            +-- VMD_PM2.5_SLSYD_<station>.csv                   <- one per station

Training outputs
    AI_Runs/Training/<region>/<var>/<var>/<model>/
        +-- <station>_metrics_report.csv    <- per-station R2, RMSE, MAE
        +-- Plots/

Model weights
    AI_Runs/Model_weights/<model_name>_<version>/models/
        +-- IMF_0_model.keras
        +-- IMF_1_model.keras
        +-- ...


SWITCHING BETWEEN TRAINING AND FORECAST
----------------------------------------

In the relevant YAML file set:

    # To train
    train_model: true
    use_file: false      # download fresh data

    # To forecast with a pre-trained model
    train_model: false
    use_file: false      # download latest obs

Training saves model weights to AI_Runs/Model_weights/. Forecast loads them from the same
location -- ensure model_name and model_version in the YAML match what was used during training.


TROUBLESHOOTING
---------------

KeyError: 'station' during training metrics:
    The old load_training_file() call is incompatible with Sparse_LSTM_v1. Per-station
    metrics are written directly by the model to <station>_metrics_report.csv. Remove the
    load_training_file() block from main.py and replace with:
        self.logger.info("Training metrics".ljust(self.justif - 8, '.') + 'SEE STATION REPORTS')


TF/Keras version mismatch on model load:
    source sparse-vmd/bin/activate
    pip install tensorflow==2.15.0 keras==2.15.0

    Models saved with Keras 2.15 cannot be loaded with 2.14 and vice versa.

ValueError: Layer 'conv1d' expected 2 variables, but received 0 variables during loading:
    This is a TF/Keras version mismatch. Run:
        pip install tensorflow==2.15.0 keras==2.15.0
