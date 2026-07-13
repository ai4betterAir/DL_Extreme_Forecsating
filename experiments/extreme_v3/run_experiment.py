#!/usr/bin/env python3
"""Run one isolated extreme-PM2.5 v3 experiment.

The experiment YAML files live outside ``Core_iHPC/Tools/Config_testing`` so
normal operational runs do not auto-discover and launch them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
CONFIG_DIR = EXPERIMENT_DIR / "configs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core_iHPC.Processing.Pipeline import ForecastPipeline_Class  # noqa: E402
from Core_iHPC.Tools.InitLogging import Initialise_logging  # noqa: E402


def available_configs() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


def resolve_config(name: str) -> Path:
    candidate = Path(name)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")
    if not candidate.is_absolute():
        candidate = CONFIG_DIR / candidate.name
    candidate = candidate.resolve()
    if candidate.parent != CONFIG_DIR.resolve():
        raise ValueError(f"Config must be inside {CONFIG_DIR}")
    if not candidate.is_file():
        choices = ", ".join(available_configs())
        raise FileNotFoundError(f"Config not found: {candidate}. Available: {choices}")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one leakage-safe extreme PM2.5 v3 model experiment.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Config filename or stem, for example PM25_GraphWaveNet_Quantile",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available v3 experiment configurations and exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        print("\n".join(available_configs()))
        return 0
    if not args.config:
        raise SystemExit("Provide a config name or use --list.")

    config_path = resolve_config(args.config)
    os.chdir(PROJECT_ROOT)
    logger = Initialise_logging(f"extreme_v3_{config_path.stem}")
    pipeline = ForecastPipeline_Class(
        logger=logger,
        justif=102,
        yaml_config_filename=str(config_path),
    )
    pipeline.run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
