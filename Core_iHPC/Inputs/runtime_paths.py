"""
Runtime path helpers.

This module centralises:
- Base directory discovery (project vs Core_iHPC).
- Common runtime directories (API_Input, logs, dashboard output, model weights).
- Directory creation helpers used by the pipeline.

Design intent:
Inputs = where data is found + cache paths + directory creation for caches.
"""

import os
from typing import Iterable, Optional


CORE_IHPC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_DIR = os.path.abspath(os.path.join(CORE_IHPC_DIR, os.pardir))


AI_RUNS_DIR_NAME = "AI_Runs"
DEFAULT_MODEL_BASE_PATH = os.path.join(AI_RUNS_DIR_NAME, "Model_weights")
RAW_DASHBOARD_FILES_DIR_NAME = "AI_dashboard_files"
RAW_DASHBOARD_FILES_DIR = "/mnt/scratch_lustre/ar_aichem_scratch/Nawcasting_Dashboard_Files"


class RuntimePaths:
    """
    Small value object so callers can override base_dir in tests/runs.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or PROJECT_DIR

    @property
    def api_input_root(self) -> str:
        return os.path.join(self.base_dir, "API_Input")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.base_dir, AI_RUNS_DIR_NAME)

    @property
    def api_inputs_dir(self) -> str:
        return os.path.join(self.api_input_root, "Inputs")

    @property
    def imputed_data_dir(self) -> str:
        return os.path.join(self.api_input_root, "model_data", "Imputed_data")

    @property
    def shared_data_dir(self) -> str:
        return os.path.join(self.api_input_root, "model_data", "Shared_data")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.base_dir, AI_RUNS_DIR_NAME, "logs")

    @property
    def raw_dashboard_dir(self) -> str:
        return RAW_DASHBOARD_FILES_DIR

    @property
    def model_weights_dir(self) -> str:
        return os.path.join(self.base_dir, DEFAULT_MODEL_BASE_PATH)

    @property
    def training_dir(self) -> str:
        return os.path.join(self.runs_dir, "Training")

    @property
    def forecast_dir(self) -> str:
        return os.path.join(self.runs_dir, "Forecast")

    @property
    def visualization_dir(self) -> str:
        return os.path.join(self.runs_dir, "Visualization")

    def iter_core_runtime_dirs(self) -> Iterable[str]:
        # Keep this list intentionally small: just the shared roots that are
        # safe to create early.
        return (
            self.data_dir,
            self.api_inputs_dir,
            self.shared_data_dir,
            self.imputed_data_dir,
            self.logs_dir,
            self.raw_dashboard_dir,
            self.model_weights_dir,
        )


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_runtime_dirs(base_dir: Optional[str] = None) -> RuntimePaths:
    paths = RuntimePaths(base_dir=base_dir or PROJECT_DIR)
    for runtime_dir in paths.iter_core_runtime_dirs():
        ensure_dir(runtime_dir)
    return paths


def runtime_base_dir_for_run(yaml_file_dict: Optional[dict] = None, default_base_dir: Optional[str] = None) -> str:
    """
    Resolve the runtime base directory for a run.

    YAML override keys (first match wins):
    - base_dir
    - runtime_base_dir
    - run_base_dir
    """
    yaml_file_dict = yaml_file_dict or {}
    base_dir = (
        yaml_file_dict.get("base_dir")
        or yaml_file_dict.get("runtime_base_dir")
        or yaml_file_dict.get("run_base_dir")
    )
    if base_dir:
        return os.path.abspath(str(base_dir))
    return os.path.abspath(default_base_dir or PROJECT_DIR)


def runtime_runs_dir_for(runtime_base_dir: str) -> str:
    return os.path.join(runtime_base_dir, AI_RUNS_DIR_NAME)


def shared_data_dir_for_run(yaml_file_dict: Optional[dict] = None, default_base_dir: Optional[str] = None) -> str:
    base_dir = runtime_base_dir_for_run(yaml_file_dict, default_base_dir=default_base_dir)
    return RuntimePaths(base_dir=base_dir).shared_data_dir


def imputed_data_dir_for_run(yaml_file_dict: Optional[dict] = None, default_base_dir: Optional[str] = None) -> str:
    base_dir = runtime_base_dir_for_run(yaml_file_dict, default_base_dir=default_base_dir)
    return RuntimePaths(base_dir=base_dir).imputed_data_dir
