"""
Time-series plotting utilities for forecast test-phase outputs.

This module is intentionally lightweight and does not depend on the training
pipeline. It can consume either:

* a single forecast CSV file,
* a dataframe already loaded in memory, or
* a directory containing multiple forecast CSVs.

The default plot uses ``forecast_hours`` on the x-axis and draws one subplot
per site for each detected variable. If the input also contains an observed
series, the plot overlays it for comparison.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_TIME_SERIES_OUTPUT_DIR = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/"
    "cnn_lstm_forecast/AI_Runs/Visualization/Time_series"
)

META_COLUMNS = {
    "datetime",
    "timestamp",
    "forecast_hours",
    "forecast_number",
    "station",
    "region",
    "input_var",
    "additional_var",
    "model",
    "model_version",
    "generated_at",
    "filename",
    "run_output_dir",
    "forecast_date",
    "timestamp",
}

DEFAULT_OBS_CANDIDATES = (
    "actual",
    "obs",
    "observation",
    "truth",
    "target",
    "original",
    "observed",
)

METRICS_ONLY_COLUMNS = {
    "stations",
    "mae",
    "rmse",
    "mape",
    "pearson_r",
    "r_square",
    "mean_fc",
    "mean_obs",
    "std_fc",
    "std_obs",
    "mbe",
}


class Plot_Class(object):
    """
    Plot forecast test-phase series by forecast hour.
    """

    def __init__(self, logger=None, justif=102, Configuration=None):
        self.logger = logger or logging.getLogger(__name__)
        self.justif = justif
        self.Configuration = Configuration

        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self.logger.info("".ljust(self.justif, "-"))
        self.logger.info("Plotting forecast test-phase time series".center(self.justif, "|"))
        self.logger.info("".ljust(self.justif, "-"))

    def MakeDir(self, ddir):
        if not os.path.exists(ddir):
            os.makedirs(ddir, exist_ok=True)
            self.logger.info(f"Directory = {ddir}".ljust(self.justif - 7, ".") + "CREATED")
        return

    def Change_permissions(self, path):
        os.umask(0)
        # Keep behavior simple: the file is already created by matplotlib.
        try:
            os.chmod(path, 0o666)
        except OSError:
            self.logger.debug("Could not change permissions for %s", path)
        return

    def _default_output_dir(self):
        if self.Configuration is not None:
            for attr in ("Main_output_run_plots_full_dir", "Main_Model_training_evaluation_plot_full_dir"):
                value = getattr(self.Configuration, attr, None)
                if value:
                    return value
        return DEFAULT_TIME_SERIES_OUTPUT_DIR

    def _format_hour_label(self, x, pos=None):
        try:
            return pd.Timestamp(mdates.num2date(x)).strftime("%I%p").lstrip("0")
        except Exception:
            return ""

    def _set_datetime_axis(self, ax, timestamps):
        ts = pd.to_datetime(pd.Series(timestamps), errors="coerce").dropna()
        if ts.empty:
            return

        span = ts.max() - ts.min()
        if span <= pd.Timedelta(days=2):
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_major_formatter(FuncFormatter(self._format_hour_label))
        elif span <= pd.Timedelta(days=21):
            day_step = max(1, int(np.ceil(max(span.days, 1) / 7.0)))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d%b"))
        elif span <= pd.Timedelta(days=180):
            month_step = max(1, int(np.ceil(max(span.days, 1) / 90.0)))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b%y"))
        else:
            year_step = max(1, int(np.ceil(max(span.days, 1) / 365.0)))
            ax.xaxis.set_major_locator(mdates.YearLocator(base=year_step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    def _date_title(self, frame, variable):
        if "timestamp" in frame.columns and frame["timestamp"].notna().any():
            date_value = pd.Timestamp(frame["timestamp"].dropna().iloc[0]).strftime("%Y-%m-%d")
            return f"{date_value} | {variable}"
        return variable

    def _forecast_hour_label(self, forecast_hour):
        try:
            return f"{int(float(forecast_hour))} Hr Forecast"
        except Exception:
            return str(forecast_hour)

    def _forecast_hour_token(self, forecast_hour):
        try:
            return f"{int(float(forecast_hour))}HrForecast"
        except Exception:
            return self._safe_name(forecast_hour)

    def _run_horizon_token(self):
        configured_horizon = getattr(self.Configuration, "n_steps_out", None)
        if configured_horizon is not None:
            try:
                return f"{int(float(configured_horizon))}HrForecast"
            except Exception:
                pass
        return "TestPhase"

    def _safe_name(self, value):
        return str(value).replace(os.sep, "_").replace(" ", "_")

    def _temporal_plot_groups(self, frame: pd.DataFrame, split_mode: Optional[str] = None):
        """
        Split a variable frame into temporal chunks for alternate output trees.

        split_mode:
        - None: no chunking
        - "month": one slice per calendar month
        - "week": sequential 7-day slices named week01, week02, ...
        """
        if "timestamp" not in frame.columns or frame["timestamp"].isna().all():
            return [(None, None, frame)]
        if not split_mode:
            return [(None, None, frame)]

        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        valid_ts = ts.dropna()
        if valid_ts.empty:
            return [(None, None, frame)]

        groups = []
        if split_mode == "month":
            unique_months = list(dict.fromkeys(valid_ts.dt.to_period("M").astype(str).tolist()))
            if len(unique_months) <= 1:
                return [(None, None, frame)]
            for month_str in unique_months:
                month_period = pd.Period(month_str, freq="M")
                month_mask = ts.dt.to_period("M") == month_period
                month_frame = frame.loc[month_mask].copy()
                if month_frame.empty:
                    continue
                groups.append(
                    (
                        month_period.strftime("%b%Y"),
                        month_period.strftime("%b %Y"),
                        month_frame,
                    )
                )
        elif split_mode == "week":
            ordered_unique_ts = pd.Series(valid_ts.sort_values().unique())
            if len(ordered_unique_ts) <= 24 * 7:
                return [(None, None, frame)]
            for idx, start in enumerate(range(0, len(ordered_unique_ts), 24 * 7), start=1):
                chunk_ts = ordered_unique_ts.iloc[start:start + (24 * 7)]
                if chunk_ts.empty:
                    continue
                chunk_start = pd.Timestamp(chunk_ts.iloc[0])
                chunk_end = pd.Timestamp(chunk_ts.iloc[-1])
                mask = ts.isin(chunk_ts.tolist())
                week_frame = frame.loc[mask].copy()
                if week_frame.empty:
                    continue
                groups.append(
                    (
                        f"week{idx:02d}",
                        f"week {idx:02d} | {chunk_start:%d %b %Y} - {chunk_end:%d %b %Y}",
                        week_frame,
                    )
                )
        else:
            return [(None, None, frame)]

        return groups or [(None, None, frame)]

    def _known_stations(self):
        region_map = getattr(self.Configuration, "DPE_region_stations_dict", None)
        selected_region = getattr(self.Configuration, "selected_region", None)
        if isinstance(region_map, dict) and selected_region in region_map:
            return [str(station) for station in region_map[selected_region]]
        return []

    def _split_column_by_known_station(self, column):
        known_stations = sorted(self._known_stations(), key=len, reverse=True)
        for station in known_stations:
            if column.endswith(f"_{station}"):
                return column[: -(len(station) + 1)], station
        if "_" in column:
            variable, inferred_station = column.rsplit("_", 1)
            return variable, inferred_station
        return column, "site"

    def _grid_for_sites(self, n_sites):
        if n_sites <= 1:
            return 1, 1
        return 3, 3

    def _site_chunks(self, stations, capacity):
        for start in range(0, len(stations), capacity):
            yield stations[start:start + capacity]

    def _read_input(self, source):
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, (str, os.PathLike)):
            return pd.read_csv(source)
        raise TypeError(f"Unsupported input type: {type(source)!r}")

    def _discover_csv_files(self, input_dir):
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        csv_files = sorted(
            p for p in input_path.rglob("*.csv")
            if "_forecast_" in p.name or "forecast" in p.name.lower()
        )
        return csv_files

    def _infer_actual_column(self, df: pd.DataFrame, value_column: str) -> Optional[str]:
        lower_map = {col.lower(): col for col in df.columns}
        for candidate in DEFAULT_OBS_CANDIDATES:
            if candidate in lower_map:
                return lower_map[candidate]

        if value_column.lower().endswith("_actual"):
            return value_column

        return None

    def _infer_variable_station(self, column: str, station: Optional[str] = None) -> Tuple[str, str]:
        if station and column.endswith(f"_{station}"):
            variable = column[: -(len(station) + 1)]
            return variable, station

        known_stations = sorted(self._known_stations(), key=len, reverse=True)
        for candidate in known_stations:
            if column.endswith(f"_{candidate}"):
                variable = column[: -(len(candidate) + 1)]
                return variable, candidate

        if "_" in column:
            variable, inferred_station = column.rsplit("_", 1)
            return variable, inferred_station

        return column, station or "site"

    def _is_value_column(self, column: str, df: pd.DataFrame) -> bool:
        if column in META_COLUMNS:
            return False
        if column.lower() in DEFAULT_OBS_CANDIDATES:
            return False
        return is_numeric_dtype(df[column])

    def _looks_like_metrics_table(self, frame: pd.DataFrame) -> bool:
        columns = {str(col).strip().lower() for col in frame.columns}
        if "stations" not in columns:
            return False
        if any(name in columns for name in ("forecast_hours", "timestamp", "datetime")):
            return False
        return any(name in columns for name in METRICS_ONLY_COLUMNS)

    def _normalise_frame(self, df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
        frame = df.copy()

        if self._looks_like_metrics_table(frame):
            raise ValueError(
                f"{source_name or 'input dataframe'} looks like a metrics summary table, not forecast output"
            )

        if "datetime" in frame.columns and "timestamp" not in frame.columns:
            frame = frame.rename(columns={"datetime": "timestamp"})

        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

        if "forecast_hours" in frame.columns:
            frame["forecast_hours"] = pd.to_numeric(frame["forecast_hours"], errors="coerce")

        if "forecast_number" in frame.columns:
            frame["forecast_number"] = pd.to_numeric(frame["forecast_number"], errors="coerce")

        value_columns = [col for col in frame.columns if self._is_value_column(col, frame)]

        records: List[pd.DataFrame] = []
        lower_columns = {col.lower(): col for col in frame.columns}
        timestamp_values = (
            frame["timestamp"].values
            if "timestamp" in frame.columns
            else [pd.NaT] * len(frame)
        )
        forecast_number_values = (
            frame["forecast_number"].values
            if "forecast_number" in frame.columns
            else [np.nan] * len(frame)
        )

        if "predicted" in lower_columns and "observed" in lower_columns:
            predicted_col = lower_columns["predicted"]
            observed_col = lower_columns["observed"]
            variable_col = lower_columns.get("variable")

            if variable_col is None:
                variable_values = [source_name or "series"] * len(frame)
            else:
                variable_values = frame[variable_col].astype(str).values

            normalized = pd.DataFrame({
                "timestamp": timestamp_values,
                "forecast_hours": frame["forecast_hours"].values,
                "forecast_number": forecast_number_values,
                "station": frame["station"].astype(str).values if "station" in frame.columns else ["site"] * len(frame),
                "variable": variable_values,
                "observed": frame[observed_col].values,
                "predicted": frame[predicted_col].values,
            })
            records.append(normalized)

        # Long format: a single station with one or more explicit value columns.
        if not records and "station" in frame.columns and value_columns:
            station_values = frame["station"].astype(str)
            if len(value_columns) == 1:
                value_column = value_columns[0]
                actual_column = self._infer_actual_column(frame, value_column)
                variable = self._infer_variable_station(value_column, station_values.iloc[0])[0]

                normalized = frame[["forecast_hours", "station"]].copy()
                normalized["variable"] = variable
                normalized["value"] = frame[value_column].values
                if "forecast_number" in frame.columns:
                    normalized["forecast_number"] = frame["forecast_number"].values
                if "timestamp" in frame.columns:
                    normalized["timestamp"] = frame["timestamp"].values
                if actual_column is not None and actual_column in frame.columns:
                    normalized["actual"] = frame[actual_column].values
                else:
                    normalized["actual"] = np.nan
                records.append(normalized)
            elif not records:
                for value_column in value_columns:
                    variable = self._infer_variable_station(value_column, station_values.iloc[0])[0]
                    actual_column = self._infer_actual_column(frame, value_column)
                    normalized = frame[["forecast_hours", "station"]].copy()
                    normalized["variable"] = variable
                    normalized["value"] = frame[value_column].values
                    if "forecast_number" in frame.columns:
                        normalized["forecast_number"] = frame["forecast_number"].values
                    if "timestamp" in frame.columns:
                        normalized["timestamp"] = frame["timestamp"].values
                    if actual_column is not None and actual_column in frame.columns:
                        normalized["actual"] = frame[actual_column].values
                    else:
                        normalized["actual"] = np.nan
                    records.append(normalized)

        # Wide format: columns are named like ``PM2.5_Bondi`` or ``O3_Parramatta``.
        elif not records and value_columns:
            for value_column in value_columns:
                variable, station = self._split_column_by_known_station(value_column)
                normalized = pd.DataFrame({
                    "forecast_hours": frame.get("forecast_hours", pd.Series(range(len(frame)))),
                    "station": station,
                    "variable": variable,
                    "value": frame[value_column].values,
                })
                if "forecast_number" in frame.columns:
                    normalized["forecast_number"] = frame["forecast_number"].values
                if "timestamp" in frame.columns:
                    normalized["timestamp"] = frame["timestamp"].values
                actual_column = self._infer_actual_column(frame, value_column)
                if actual_column is not None and actual_column in frame.columns:
                    normalized["actual"] = frame[actual_column].values
                else:
                    normalized["actual"] = np.nan
                records.append(normalized)

        if not records:
            raise ValueError(
                f"No plottable series found in {source_name or 'input dataframe'}"
            )

        normalized_frame = pd.concat(records, ignore_index=True)
        normalized_frame["station"] = normalized_frame["station"].astype(str)
        normalized_frame["variable"] = normalized_frame["variable"].astype(str)
        normalized_frame = normalized_frame.sort_values(
            ["variable", "station", "forecast_hours"],
            kind="stable",
        )
        return normalized_frame

    def _plot_variable(
        self,
        frame: pd.DataFrame,
        variable: str,
        output_dir: str,
        region_name: str,
        prefix: str = "test_phase",
        split_mode: Optional[str] = None,
    ):
        variable_frame = frame[frame["variable"] == variable].copy()
        stations = list(dict.fromkeys(variable_frame["station"].tolist()))
        if not stations:
            return []

        n_stations = len(stations)
        nrows, ncols = self._grid_for_sites(n_stations)
        capacity = nrows * ncols
        station_groups = list(self._site_chunks(stations, capacity))
        generated_paths = []

        month_groups = self._temporal_plot_groups(variable_frame, split_mode=split_mode)

        for month_token, month_title, plot_frame in month_groups:
            for page_idx, page_stations in enumerate(station_groups, start=1):
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(7.5 * ncols, 4.2 * nrows),
                    sharex=True,
                    squeeze=False,
                )
                title = self._date_title(plot_frame, variable)
                if month_title is not None:
                    title = f"{title} | {month_title}"
                if len(station_groups) > 1:
                    title = f"{title} | page {page_idx}"
                fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

                for ax, station in zip(axes.ravel(), page_stations):
                    station_frame = plot_frame[plot_frame["station"] == station].copy()
                    if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
                        station_frame = station_frame.sort_values("timestamp")
                    else:
                        station_frame = station_frame.sort_values("forecast_hours")

                    if "predicted" in station_frame.columns and "observed" in station_frame.columns:
                        x_values = station_frame["timestamp"] if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any() else station_frame["forecast_hours"]
                        ax.plot(x_values, station_frame["observed"], color="tab:orange", linewidth=2, linestyle="--", label="Observed")
                        ax.plot(x_values, station_frame["predicted"], color="tab:blue", linewidth=2.2, label="Predicted")
                    elif "forecast_number" in station_frame.columns and station_frame["forecast_number"].notna().any():
                        for forecast_number, seg in station_frame.groupby("forecast_number", sort=True):
                            ax.plot(
                                seg["timestamp"] if "timestamp" in seg.columns and seg["timestamp"].notna().any() else seg["forecast_hours"],
                                seg["value"],
                                alpha=0.35,
                                linewidth=1.5,
                                label=f"Forecast {int(forecast_number)}" if pd.notna(forecast_number) else "Forecast",
                            )
                    else:
                        ax.plot(
                            station_frame["timestamp"] if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any() else station_frame["forecast_hours"],
                            station_frame["value"],
                            color="tab:blue",
                            linewidth=2,
                            label="Forecast",
                        )

                    if "actual" in station_frame.columns and station_frame["actual"].notna().any():
                        ax.plot(
                            station_frame["timestamp"] if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any() else station_frame["forecast_hours"],
                            station_frame["actual"],
                            color="tab:orange",
                            linewidth=2,
                            linestyle="--",
                            label="Actual",
                        )

                    ax.set_title(station)
                    ax.set_xlabel("Time" if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any() else "Forecast hour")
                    ax.set_ylabel(variable)
                    if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
                        self._set_datetime_axis(ax, station_frame["timestamp"])
                    elif station_frame["forecast_hours"].notna().any():
                        ax.set_xticks(list(range(0, int(max(station_frame["forecast_hours"].max(), 3)) + 1)))
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="best", fontsize=9)

                for ax in axes.ravel()[len(page_stations):]:
                    ax.axis("off")

                fig.tight_layout(rect=(0, 0, 1, 0.96))
                self.MakeDir(output_dir)
                region_token = self._safe_name(region_name)
                variable_token = self._safe_name(variable)
                base_name = f"{region_token}_{variable_token}_{self._run_horizon_token()}"

                if len(station_groups) > 1:
                    file_name = f"{base_name}_part{page_idx}.png"
                else:
                    file_name = f"{base_name}.png"
                if month_token is not None:
                    file_name = file_name.replace(".png", f"_{month_token}.png")
                full_path = os.path.join(output_dir, file_name)
                fig.savefig(full_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                self.Change_permissions(full_path)
                self.logger.info(f"Plot saved: {full_path}".ljust(self.justif - 2, ".") + "OK")
                generated_paths.append(full_path)

        return generated_paths

    def plot_test_phase_timeseries(
        self,
        source,
        output_dir: Optional[str] = None,
        region_name: Optional[str] = None,
        variables: Optional[Sequence[str]] = None,
        prefix: str = "test_phase",
        split_mode: Optional[str] = None,
    ) -> List[str]:
        """
        Plot test-phase time series by variable and site.

        Parameters
        ----------
        source:
            DataFrame, CSV file path, or directory containing forecast CSVs.
        output_dir:
            Where PNG files should be written. Defaults to the configured
            forecast plot directory, or a local ``plots`` folder.
        prefix:
            File-name prefix for generated figures.

        Returns
        -------
        list[str]
            Paths to generated plot files.
        """
        output_dir = output_dir or self._default_output_dir()
        if region_name is None and self.Configuration is not None:
            region_name = getattr(self.Configuration, "selected_region", "Region")
        region_name = region_name or "Region"

        if isinstance(source, (str, os.PathLike)) and Path(source).is_dir():
            normalized_frames: List[pd.DataFrame] = []
            for csv_file in self._discover_csv_files(source):
                frame = self._read_input(csv_file)
                source_name = csv_file.stem
                try:
                    normalized_frames.append(self._normalise_frame(frame, source_name=source_name))
                except ValueError as exc:
                    self.logger.warning("Skipping %s: %s", csv_file.name, exc)

            if not normalized_frames:
                return []

            combined = pd.concat(normalized_frames, ignore_index=True)
            if variables:
                target_set = {str(value) for value in variables}
                combined = combined[combined["variable"].isin(target_set)].copy()
            generated_paths: List[str] = []
            for variable in combined["variable"].unique():
                generated_paths.extend(
                    self._plot_variable(
                        combined,
                        variable,
                        output_dir=output_dir,
                        region_name=region_name,
                        prefix=prefix,
                        split_mode=split_mode,
                    )
                )
            return generated_paths

        frame = self._read_input(source)
        source_name = Path(source).stem if isinstance(source, (str, os.PathLike)) else ""
        normalized = self._normalise_frame(frame, source_name=source_name)
        if variables:
            target_set = {str(value) for value in variables}
            normalized = normalized[normalized["variable"].isin(target_set)].copy()

        generated: List[str] = []
        for variable in normalized["variable"].unique():
            generated.extend(self._plot_variable(
                normalized,
                variable,
                output_dir=output_dir,
                region_name=region_name,
                prefix=prefix,
                split_mode=split_mode,
            ))
        return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot test-phase forecast time series by forecast hour."
    )
    parser.add_argument(
        "source",
        help="CSV file or directory containing forecast CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write PNG files",
    )
    parser.add_argument(
        "--prefix",
        default="test_phase",
        help="Prefix for generated plot files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plotter = Plot_Class()
    plotter.plot_test_phase_timeseries(
        args.source,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
