"""
..  module:: CNN_LSTM_forecast
    :platform: Unix
    :synopsis: Entry point for running the nowcasting/forecast pipeline.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>

This file keeps the "old repo spirit": a familiar main module that exposes a
class with `MakeDir()` and `run_all()`.

The actual pipeline implementation lives in `Core_iHPC/Processing/Pipeline.py`.
"""

import sys
import os
import stat
import datetime as dtime
import time
import logging
import itertools as it

# Heavy scientific deps are imported for backward familiarity, but guarded so
# tooling environments (e.g., lint/CI) without TF installed can still import.
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None  # type: ignore

try:
    import pandas as pd  # noqa: F401
except Exception:
    pd = None  # type: ignore

try:
    import numpy as np  # noqa: F401
except Exception:
    np = None  # type: ignore

try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # type: ignore

import Core_iHPC.Tools.InitLogging as IL  # noqa: F401

from Core_iHPC.Processing import Pipeline as _PIPELINE


# Region selection / expansion
#
# Use a short region code (CNC, SyE, SyNW, SySW, CNT, NRT, SRT, LHN, UHN,
# NCL, SYD) or an all-region sentinel ("ALL", "AUTO", "ALL_AVAILABLE", "*",
# "SLSYD"). This is the single default region selector for normal runs.
DEFAULT_SELECTED_REGION = ["SyE", "NCL", "CNC", "NRT"]
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

def configure_pipeline_defaults():
    _PIPELINE.DEFAULT_SELECTED_REGION = DEFAULT_SELECTED_REGION
    _PIPELINE.MAIN_DEFAULT_SELECTED_REGION = DEFAULT_SELECTED_REGION


configure_pipeline_defaults()


class PipelineRunner_Class(object):
    """
    Generic runner class for any model/pipeline.

    This class keeps a familiar `MakeDir()` + `run_all()` surface like the old
    repo, but delegates to the generic implementation in
    `Core_iHPC.Processing.Pipeline`.
    """

    def __init__(self, logger, justif, yaml_config_filename):
        self.logger = logger
        self.justif = justif
        self.yaml_config_filename = yaml_config_filename
        self._impl = _PIPELINE.ForecastPipeline_Class(logger, justif, yaml_config_filename)

    def MakeDir(self, ddir):
        return self._impl.MakeDir(ddir)

    def run_all(self, *args, **kwargs):
        return self._impl.run_all(*args, **kwargs)


# Preferred generic name (new code should use this).
ForecastPipeline_Class = _PIPELINE.ForecastPipeline_Class

# Legacy alias: some code still expects Sparse_LSTM_Class.
Sparse_LSTM_Class = _PIPELINE.Sparse_LSTM_Class

def main(*args, **kwargs):
    configure_pipeline_defaults()
    return _PIPELINE.main(*args, **kwargs)

__all__ = [
    "PipelineRunner_Class",
    "ForecastPipeline_Class",
    "Sparse_LSTM_Class",
    "DEFAULT_SELECTED_REGION",
    "main",
]


if __name__ == "__main__":
    main()
