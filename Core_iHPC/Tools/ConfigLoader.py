"""
Load a Python defaults module for a run, optionally selected from YAML.

The intent is to keep runtime defaults out of `main.py`, while still allowing
each YAML file to point at a different defaults module via `config_file`.
"""

import importlib
import importlib.util
import os
from types import ModuleType
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_MODULE = "Core_iHPC.Tools.Config_testing.config"


def load_config_module(config_file: Optional[str], config_dir: str) -> ModuleType:
    """
    Resolve `config_file` (from YAML) into an imported Python module.

    Supported forms:
    - None/empty: loads DEFAULT_CONFIG_MODULE
    - Dotted path: `Core_iHPC.Tools.Config_testing.config`
    - Filename: `config.py` or `my_config.py` (resolved relative to `config_dir`)
    - Relative/absolute path: `./my_config.py` or `/abs/path/my_config.py`
    - Basename without extension: `my_config` (resolved under `config_dir`)
    """
    if not config_file:
        return importlib.import_module(DEFAULT_CONFIG_MODULE)

    token = str(config_file).strip()
    if not token:
        return importlib.import_module(DEFAULT_CONFIG_MODULE)

    # File-based config (explicit .py or path separators)
    if token.endswith(".py") or os.sep in token or token.startswith("."):
        path = token
        if not os.path.isabs(path):
            path = os.path.join(config_dir, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"config_file not found: {path}")
        module_name = f"_run_config_{abs(hash(os.path.abspath(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import config_file from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Dotted module path
    if "." in token:
        return importlib.import_module(token)

    # Basename (optionally without .py) under the config_dir
    basename = token[:-3] if token.endswith(".py") else token
    candidate_path = os.path.join(config_dir, f"{basename}.py")
    if os.path.isfile(candidate_path):
        module_name = f"_run_config_{abs(hash(os.path.abspath(candidate_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import config_file from {candidate_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Finally, try it as a module inside Core_iHPC.Tools.Config_testing
    return importlib.import_module(f"Core_iHPC.Tools.Config_testing.{basename}")


def apply_config_defaults_to_yaml(
    yaml_file_dict: Dict[str, Any],
    config_module: ModuleType,
) -> Dict[str, Any]:
    """
    Apply Python defaults to the YAML dict only when keys are missing.
    """
    if hasattr(config_module, "DEFAULT_SELECTED_REGION"):
        yaml_file_dict.setdefault("selected_region", getattr(config_module, "DEFAULT_SELECTED_REGION"))
    yaml_file_dict.setdefault("train_model", getattr(config_module, "DEFAULT_TRAIN_MODEL", True))

    if not yaml_file_dict.get("var_to_predict"):
        yaml_file_dict["var_to_predict"] = list(getattr(config_module, "DEFAULT_VAR_TO_PREDICT", []))

    yaml_file_dict.setdefault(
        "IMPUTATION_METHOD",
        getattr(config_module, "DEFAULT_IMPUTATION_METHOD", getattr(config_module, "IMPUTATION_METHOD", "NONE")),
    )
    return yaml_file_dict


def export_selected_config_to_namespace(config_module: ModuleType, namespace: Dict[str, Any]) -> None:
    """
    Export a known set of constants into `namespace` (typically `globals()` in main.py).

    This keeps call sites unchanged while allowing YAML to switch defaults.
    """
    for key in (
        "ALL_REGION_SENTINELS",
        "DEFAULT_TRAIN_MODEL",
        "PIPELINE_MODE",
        "USE_DECOMPOSE",
        "USE_ONE_MODEL_PER_IMF",
        "TYPE_OF_DECOMPOSITION",
        "TARGET_MODEL_CONFIG",
        "DEFAULT_SELECTED_REGION",
        "DEFAULT_VAR_TO_PREDICT",
        "DEFAULT_FORECAST_METHOD",
        "AVAILABLE_IMPUTATION_METHODS",
        "IMPUTATION_METHOD",
        "DEFAULT_IMPUTATION_METHOD",
        "DEFAULT_PLOT",
        "PIPELINE_MODE_PRESETS",
        "PIPELINE_MODE_ALIASES",
        "DECOMPOSITION_BACKEND_ALIASES",
    ):
        if hasattr(config_module, key):
            namespace[key] = getattr(config_module, key)


def _deep_update_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def apply_optimal_yaml_overrides(
    yaml_file_dict: Dict[str, Any],
    config_root_dir: str,
    yaml_config_name: str,
    selector: str = "best",
) -> Dict[str, Any]:
    """
    Merge an optimisation-derived YAML into the current YAML dict.

    selector:
      - "best": loads `Optima/Optimisation_history/<config>_optimal_best.yaml`
      - "latest": loads the newest `Optima/<config>_optimal_*.yaml`
    """
    if not yaml_config_name:
        return yaml_file_dict

    base = os.path.basename(str(yaml_config_name))
    for suffix in (".yaml", ".yml"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break

    optima_dir = os.path.join(config_root_dir, "Optima")
    history_dir = os.path.join(optima_dir, "Optimisation_history")

    selector_norm = str(selector or "best").strip().lower()
    candidate_path = None

    if selector_norm in {"best", "optimal_best"}:
        candidate_path = os.path.join(history_dir, "{ff}_optimal_best.yaml".format(ff=base))
        if not os.path.isfile(candidate_path):
            return yaml_file_dict
    elif selector_norm in {"latest", "newest"}:
        pattern_prefix = "{ff}_optimal_".format(ff=base)
        if not os.path.isdir(optima_dir):
            return yaml_file_dict
        candidates = [
            os.path.join(optima_dir, fname)
            for fname in os.listdir(optima_dir)
            if fname.startswith(pattern_prefix) and fname.endswith(".yaml")
        ]
        if not candidates:
            return yaml_file_dict
        candidate_path = max(candidates, key=os.path.getmtime)
    else:
        # treat selector as a relative/absolute yaml path
        candidate_path = selector
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(config_root_dir, candidate_path)
        if not os.path.isfile(candidate_path):
            return yaml_file_dict

    with open(candidate_path, "r") as handle:
        overrides = yaml.safe_load(handle) or {}

    return _deep_update_dict(yaml_file_dict, overrides)
