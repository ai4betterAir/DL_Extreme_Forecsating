"""
Default runtime constants for AI_Nowcasting/cnn_lstm_forecast.

This module is intentionally "data only": constants/defaults that can be
selected from YAML via `config_file`.
"""

from Core_iHPC.Configuration.runtime_config import TARGET_MODEL_CONFIG as TARGET_MODEL_CONFIG

# Region selection / expansion
ALL_REGION_SENTINELS = {"ALL", "AUTO", "ALL_AVAILABLE", "*", "SLSYD"}
# Use all-region sentinels like "ALL", "AUTO", "ALL_AVAILABLE", "*", or "SLSYD" to run every region.
# Change this one line to switch the default region for all runs.
DEFAULT_SELECTED_REGION = "CNT"

# Preferred short region code (dashboard/config codes like CNC/SyE/SyNW/...).
#
# Short-code reference (used by dashboard/configs -> NSW API region names):
#   CNC  -> Central Coast
#   SyE  -> Sydney East
#   SyNW -> Sydney North-west
#   SySW -> Sydney South-west
#   CNT  -> Central Tablelands
#   NRT  -> Northern Tablelands
#   SRT  -> Southern Tablelands
#   LHN  -> Lower Hunter
#   UHN  -> Upper Hunter
#   NCL  -> Newcastle Local
#   SYD  -> Sydney
#
# (Source of truth: Core_iHPC/Configuration/DPE_region_stations.py REGION_CODE_TO_API_REGION)

# Optional: split ALL-region runs into named region groups (HPC-friendly).
# one group per job (e.g. 3 SLURM jobs in parallel), by setting:
# - YAML: `region_group: "A"` (or B/C), or
# - env var: `PIPELINE_REGION_GROUP=A`
#
# If PARALLEL_RUN_REGIONS="no" (default), or no group is selected, ALL runs
# proceed sequentially as usual.
PARALLEL_RUN_REGIONS = "no"  # "yes"/"no"
PARALLEL_REGION_GROUPS = {
    # Balanced by total site-target workload across O3, PM2.5, and PM10.
    # A: 36 site-targets (SyE=24, NCL=7, CNC=3, NRT=2)
    "A": ["SyE", "NCL", "CNC", "NRT"],
    # B: 36 site-targets (UHN=21, LHN=9, CNT=6)
    "B": ["UHN", "LHN", "CNT"],
    # C: 36 site-targets (SyNW=18, SySW=15, SRT=3)
    "C": ["SyNW", "SySW", "SRT"],
}


# Run mode defaults
DEFAULT_TRAIN_MODEL = False  # Set to True only when you want to train; forecast/dashboard runs default to False.
DEFAULT_PLOT = True

# Pipeline controls (defaults; YAML can still override decompose/impute flags)
PIPELINE_MODE = "raw"  # Valid values: "raw", "imf_joint", "imf_per_imf", "custom"
# (if PIPELINE_MODE= imf_joint, select USE_ONE_MODEL_PER_IMF= no; if PIPELINE_MODE= imf_per_imf,
#  select USE_ONE_MODEL_PER_IMF= yes; if PIPELINE_MODE= custom, select the two switches below directly).
# In "custom" mode, the two switches below are used directly.
USE_DECOMPOSE = "no"  # Use IMF decomposition in the data preprocessing step. Set to "no" to skip it and feed imputed data directly into the model.
USE_ONE_MODEL_PER_IMF = "no"  # Use one model per IMF. Set to "no" to train one joint model on all IMFs together.

# Decomposition
TYPE_OF_DECOMPOSITION = "VMD"  # Valid: "VMD", "EMD", ...
DECOMPOSITION_BACKEND_ALIASES = {
    "VMD": "VMD",
    "EMD": "EMD", #EMD is not implemented yet, but we can add it in the future if desired.
}

PIPELINE_MODE_PRESETS = {
    "raw": {"use_decompose": False, "use_one_model_per_imf": False},
    "imf_joint": {"use_decompose": True, "use_one_model_per_imf": False},
    "imf_per_imf": {"use_decompose": True, "use_one_model_per_imf": True},
}

PIPELINE_MODE_ALIASES = {
    "vmd_joint": "imf_joint",
    "vmd_per_imf": "imf_per_imf",
}

# Forecast/model defaults
DEFAULT_VAR_TO_PREDICT = ["O3", "PM2.5", "PM10"]  # ["O3", "PM2.5", "PM10"]
DEFAULT_FORECAST_METHOD = "Sparse_LSTM_v1"

# Imputation defaults
AVAILABLE_IMPUTATION_METHODS = ["NONE", "MICE", "KNN", "TemporalMICE", "AQUISTIL"]
IMPUTATION_METHOD = "AQUISTIL"
DEFAULT_IMPUTATION_METHOD = "AQUISTIL"

# Default training epochs (can be referenced from YAML via 'from_config')
N_EPOCH = 25
