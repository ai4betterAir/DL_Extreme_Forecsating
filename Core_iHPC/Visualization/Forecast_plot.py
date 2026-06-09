"""
Forecast horizon alignment plots.

This module draws one plot per site. Each plot shows:

* the observed/history trace up to time ``t``;
* forecast windows anchored at the same ``t`` for horizons 3, 6, 9, and 12
  hours by default;
* all forecast points aligned on the same datetime axis so shared timestamps
  land on top of each other.

The expected input is a forecast dataframe with at least:

* ``forecast_hours``
* ``station`` or station-specific value columns
* a datetime column or datetime index

The code is intentionally tolerant of the repo's different forecast CSV
layouts, including:

* per-station forecast files from ``Output_manager.output_forecast_forward``
* combined dashboard-style files
* wide frames with columns such as ``PM2.5_Bondi``
"""

import argparse
import logging
import os
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})


DEFAULT_FORECAST_PLOT_DIR = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/"
    "cnn_lstm_forecast/AI_Runs/Visualization/Forecast_Plot"
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
}

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
    Plot forecast windows for each site.
    """

    def __init__(self, logger=None, justif=102, Configuration=None):
        self.logger = logger or logging.getLogger(__name__)
        self.justif = justif
        self.Configuration = Configuration

        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self.logger.info("".ljust(self.justif, "-"))
        self.logger.info("Forecast horizon plotting".center(self.justif, "|"))
        self.logger.info("".ljust(self.justif, "-"))

    def MakeDir(self, ddir):
        if not os.path.exists(ddir):
            os.makedirs(ddir, exist_ok=True)
            self.logger.info(
                "Directory = {msg}".format(msg=ddir).ljust(self.justif - 7, ".") + "CREATED"
            )
        return

    def Change_permissions(self, path):
        try:
            os.chmod(path, 0o666)
        except OSError:
            self.logger.debug("Could not change permissions for %s", path)
        return

    def _default_output_dir(self):
        if self.Configuration is not None:
            candidate = getattr(self.Configuration, "Main_output_run_plots_full_dir", None)
            if candidate:
                return candidate
        return DEFAULT_FORECAST_PLOT_DIR

    def _hour_formatter(self, x, pos=None):
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
            ax.xaxis.set_major_formatter(FuncFormatter(self._hour_formatter))
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

    def _horizon_label(self, horizon):
        return "{0} Hr Forecast".format(int(horizon))

    def _date_title(self, station_frame, variable):
        candidate = None
        if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
            candidate = pd.Timestamp(station_frame["timestamp"].dropna().iloc[0])
        if candidate is None:
            return variable
        return "{0} | {1}".format(candidate.strftime("%Y-%m-%d"), variable)

    def _safe_name(self, value):
        return str(value).replace(os.sep, "_").replace(" ", "_")

    def _station_key(self, value):
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())

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

    def _region_sites_in_order(self, variable_frame):
        stations = [str(station) for station in variable_frame["station"].dropna().tolist()]
        if self.Configuration is None:
            return list(dict.fromkeys(stations))

        region_map = getattr(self.Configuration, "DPE_region_stations_dict", None)
        selected_region = getattr(self.Configuration, "selected_region", None)
        ordered = []
        if isinstance(region_map, dict) and selected_region in region_map:
            for station in region_map[selected_region]:
                station_name = str(station)
                station_key = self._station_key(station_name)
                match = None
                for candidate in stations:
                    if self._station_key(candidate) == station_key:
                        match = candidate
                        break
                if match is not None and match not in ordered:
                    ordered.append(match)
        for station in stations:
            if station not in ordered:
                ordered.append(station)
        return ordered

    def _match_station_frame(self, variable_frame, station_name):
        target_key = self._station_key(station_name)
        station_values = variable_frame["station"].astype(str)
        mask = station_values.map(self._station_key) == target_key
        return variable_frame[mask].copy()

    def _site_grid_shape(self, n_sites):
        if n_sites <= 1:
            return 1, 1
        ncols = min(4, max(1, int(math.ceil(math.sqrt(n_sites)))))
        nrows = int(math.ceil(float(n_sites) / float(ncols)))
        return nrows, ncols

    def _station_pages(self, stations, page_size=2):
        for start in range(0, len(stations), page_size):
            yield stations[start:start + page_size]

    def _read_input(self, source):
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, (str, os.PathLike)):
            return pd.read_csv(source)
        raise TypeError("Unsupported input type: {0!r}".format(type(source)))

    def _detect_datetime_column(self, df):
        for candidate in ("timestamp", "datetime"):
            if candidate in df.columns:
                return candidate
        return None

    def _value_columns(self, df):
        columns = []
        for column in df.columns:
            if column in META_COLUMNS:
                continue
            if is_numeric_dtype(df[column]):
                columns.append(column)
                continue
            coerced = pd.to_numeric(df[column], errors="coerce")
            if coerced.notna().any():
                columns.append(column)
        return columns

    def _looks_like_metrics_table(self, frame):
        columns = {str(col).strip().lower() for col in frame.columns}
        if "stations" not in columns:
            return False
        if any(name in columns for name in ("forecast_hours", "timestamp", "datetime")):
            return False
        return any(name in columns for name in METRICS_ONLY_COLUMNS)

    def _infer_station_from_column(self, column):
        return self._split_column_by_known_station(column)[1]

    def _infer_variable_from_column(self, column, station=None):
        if station and column.endswith("_" + station):
            return column[: -(len(station) + 1)]
        known_stations = sorted(self._known_stations(), key=len, reverse=True)
        for candidate in known_stations:
            if column.endswith(f"_{candidate}"):
                return column[: -(len(candidate) + 1)]
        if "_" in column:
            return column.rsplit("_", 1)[0]
        return column

    def _normalise(self, df, source_name=""):
        frame = df.copy()
        if self._looks_like_metrics_table(frame):
            raise ValueError(
                "{0} looks like a metrics summary table, not forecast output".format(
                    source_name or "input dataframe"
                )
            )
        datetime_col = self._detect_datetime_column(frame)
        if datetime_col is not None and datetime_col != "timestamp":
            frame = frame.rename(columns={datetime_col: "timestamp"})

        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

        if "forecast_hours" in frame.columns:
            frame["forecast_hours"] = pd.to_numeric(frame["forecast_hours"], errors="coerce")
        else:
            frame["forecast_hours"] = np.nan

        if "forecast_number" in frame.columns:
            frame["forecast_number"] = pd.to_numeric(frame["forecast_number"], errors="coerce")

        records = []
        value_columns = self._value_columns(frame)
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

        if "station" in frame.columns and len(value_columns) == 1:
            value_column = value_columns[0]
            station = frame["station"].astype(str)
            variable = self._infer_variable_from_column(value_column, station.iloc[0])
            out = pd.DataFrame({
                "timestamp": timestamp_values,
                "forecast_hours": frame["forecast_hours"].values,
                "forecast_number": forecast_number_values,
                "station": station.values,
                "variable": variable,
                # Coerce to numeric so a mixed-type CSV column does not drop
                # forecast hours during plotting.
                "value": pd.to_numeric(frame[value_column], errors="coerce").values,
            })
            records.append(out)
        elif value_columns:
            for value_column in value_columns:
                variable, station = self._split_column_by_known_station(value_column)
                out = pd.DataFrame({
                    "timestamp": timestamp_values,
                    "forecast_hours": frame["forecast_hours"].values,
                    "forecast_number": forecast_number_values,
                    "station": station,
                    "variable": variable,
                    # Coerce to numeric so a mixed-type CSV column does not drop
                    # forecast hours during plotting.
                    "value": pd.to_numeric(frame[value_column], errors="coerce").values,
                })
                records.append(out)

        if not records:
            raise ValueError(
                "No plottable forecast columns found in {0}".format(source_name or "input dataframe")
            )

        normalized = pd.concat(records, ignore_index=True)
        normalized["station"] = normalized["station"].astype(str)
        normalized["variable"] = normalized["variable"].astype(str)
        if "timestamp" in normalized.columns:
            normalized = normalized.sort_values(["station", "forecast_hours", "timestamp"], kind="stable")
        else:
            normalized = normalized.sort_values(["station", "forecast_hours"], kind="stable")
        return normalized

    def _pick_anchor_row(self, station_frame):
        # Use the latest non-forecast row as the common start point when it is
        # available. This is the shared t used by the forecast windows.
        if "forecast_hours" not in station_frame.columns:
            return None

        history = station_frame[station_frame["forecast_hours"] <= 0]
        if not history.empty:
            if "timestamp" in history.columns and history["timestamp"].notna().any():
                history = history.sort_values("timestamp")
            else:
                history = history.sort_values("forecast_hours")
            return history.iloc[-1]

        if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
            station_frame = station_frame.sort_values("timestamp")
        return station_frame.iloc[0] if len(station_frame) else None

    def _plot_variable_horizon_grid(self, variable_frame, output_dir, horizons, prefix):
        variable = variable_frame["variable"].iloc[0]
        region_token = self._safe_name(getattr(self.Configuration, "selected_region", "Region"))
        variable_token = self._safe_name(variable)
        stations = self._region_sites_in_order(variable_frame)
        n_sites = len(stations)
        nrows, ncols = self._site_grid_shape(n_sites)
        horizon_list = sorted({int(h) for h in horizons})
        max_horizon = max(horizon_list) if horizon_list else 0
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(12, 5.2 * ncols), max(6, 3.8 * nrows)),
            sharex=True,
            squeeze=False,
        )
        axes = axes.ravel()
        linestyle_map = {
            3: "-",
            6: "--",
            9: "-.",
            12: ":",
        }
        horizon_colors = {
            3: "tab:blue",
            6: "tab:orange",
            9: "tab:green",
            12: "tab:red",
        }

        for s_idx, station_name in enumerate(stations):
            ax = axes[s_idx]
            station_frame = self._match_station_frame(variable_frame, station_name)
            if station_frame.empty:
                ax.axis("off")
                continue

            if "forecast_hours" in station_frame.columns:
                station_frame = station_frame.sort_values("forecast_hours")
            elif "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
                station_frame = station_frame.sort_values("timestamp")

            forecast = station_frame[station_frame["forecast_hours"] > 0].copy()
            if forecast.empty or max_horizon <= 0:
                ax.axis("off")
                continue

            plot_horizons = sorted(horizon_list, reverse=True)
            for horizon in plot_horizons:
                cut = forecast[forecast["forecast_hours"] <= horizon].copy()
                if cut.empty:
                    continue

                line_x = np.arange(1, max_horizon + 1, dtype=float)
                line_y = np.full(max_horizon, np.nan, dtype=float)
                cut_hours = pd.to_numeric(cut["forecast_hours"], errors="coerce")
                cut_values = pd.to_numeric(cut["value"], errors="coerce")
                for hour_value, forecast_value in zip(cut_hours.tolist(), cut_values.tolist()):
                    if pd.isna(hour_value) or pd.isna(forecast_value):
                        continue
                    hour_index = int(hour_value) - 1
                    if 0 <= hour_index < max_horizon:
                        line_y[hour_index] = float(forecast_value)

                ax.plot(
                    line_x,
                    line_y,
                    color=horizon_colors.get(int(horizon), "tab:blue"),
                    linestyle=linestyle_map.get(int(horizon), "-"),
                    linewidth=2.2,
                    marker="o",
                    markersize=3,
                    label=self._horizon_label(horizon),
                )

            ax.set_title(station_name, fontsize=12, fontweight="semibold")
            ax.set_xlabel("Forecast hour", fontweight="semibold")
            ax.set_ylabel(variable, fontweight="semibold")
            if max_horizon > 0:
                xticks = sorted({1, 3, 6, 9, 12}.intersection(set(range(1, max_horizon + 1))))
                if not xticks:
                    xticks = list(range(1, max_horizon + 1))
                ax.set_xticks(xticks)
                ax.set_xlim(1, max_horizon)
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.grid(True, alpha=0.25)

        for ax in axes[n_sites:]:
            ax.axis("off")

        fig.suptitle(self._date_title(variable_frame, variable), fontsize=16, fontweight="semibold", y=0.995)
        horizon_handles = [
            Line2D([0], [0], color=horizon_colors.get(int(h), "tab:blue"), linestyle=linestyle_map.get(int(h), "-"), linewidth=2.2, label=self._horizon_label(h))
            for h in horizons
        ]
        fig.legend(
            handles=horizon_handles,
            loc="upper center",
            ncol=len(horizons),
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        self.MakeDir(output_dir)
        full_path = os.path.join(output_dir, "{0}_{1}_Sites_ForecastPlot.png".format(region_token, variable_token))
        fig.savefig(full_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        self.Change_permissions(full_path)
        self.logger.info("Plot saved: {0}".format(full_path).ljust(self.justif - 2, ".") + "OK")
        return full_path

    def _plot_station_page(self, variable_frame, stations_page, output_dir, horizons, prefix, page_index, total_pages):
        variable = variable_frame["variable"].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, squeeze=False)
        axes = axes.ravel()
        fig.suptitle(self._date_title(variable_frame, variable), fontsize=16, fontweight="semibold", y=0.99)
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(stations_page):
                ax.axis("off")
                continue

            station_name = stations_page[ax_idx]
            station_frame = variable_frame[variable_frame["station"] == station_name].copy()
            if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
                station_frame = station_frame.sort_values("timestamp")
            else:
                station_frame = station_frame.sort_values("forecast_hours")

            history = station_frame[station_frame["forecast_hours"] <= 0].copy()
            forecast = station_frame[station_frame["forecast_hours"] > 0].copy()
            anchor = self._pick_anchor_row(station_frame)

            if not history.empty and "timestamp" in history.columns and history["timestamp"].notna().any():
                ax.plot(history["timestamp"], history["value"], color="black", linewidth=2, label="History")

            for idx, horizon in enumerate(horizons):
                cut = forecast[forecast["forecast_hours"] <= horizon].copy()
                if cut.empty:
                    continue

                if anchor is not None and pd.notna(anchor.get("timestamp", np.nan)):
                    line_x = [anchor["timestamp"]]
                    line_y = [anchor["value"]]
                    line_x.extend(cut["timestamp"].tolist())
                    line_y.extend(cut["value"].tolist())
                else:
                    line_x = cut["timestamp"].tolist()
                    line_y = cut["value"].tolist()

                ax.plot(
                    line_x,
                    line_y,
                    color=colors[idx % len(colors)],
                    linewidth=2.2,
                    marker="o",
                    markersize=4,
                    label=self._horizon_label(horizon),
                )

            if anchor is not None and pd.notna(anchor.get("timestamp", np.nan)):
                ax.scatter([anchor["timestamp"]], [anchor["value"]], color="black", s=36, zorder=5, label="t")

            ax.set_title(station_name, fontweight="semibold")
            ax.set_xlabel("Time", fontweight="semibold")
            ax.set_ylabel(variable, fontweight="semibold")
            if "timestamp" in station_frame.columns and station_frame["timestamp"].notna().any():
                self._set_datetime_axis(ax, station_frame["timestamp"])
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        fig.tight_layout(rect=(0, 0, 1, 0.96))

        self.MakeDir(output_dir)
        region_token = self._safe_name(getattr(self.Configuration, "selected_region", "Region"))
        variable_token = self._safe_name(variable)
        if total_pages > 1:
            filename = "{0}_{1}_part{2}.png".format(prefix, "{0}_{1}".format(region_token, variable_token), page_index)
        else:
            filename = "{0}_{1}_{2}.png".format(prefix, region_token, variable_token)
        full_path = os.path.join(output_dir, filename)
        fig.savefig(full_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        self.Change_permissions(full_path)
        self.logger.info("Plot saved: {0}".format(full_path).ljust(self.justif - 2, ".") + "OK")
        return full_path

    def plot_horizon_windows(self, source, output_dir=None, horizons=(3, 6, 9, 12), variables=None, prefix="forecast_window"):
        """
        Plot aligned forecast windows for each site.

        Parameters
        ----------
        source:
            DataFrame, CSV path, or directory of CSV files.
        output_dir:
            Directory to write PNG files.
        horizons:
            Iterable of cutoffs. The default draws t+1..t+3, t+1..t+6,
            t+1..t+9, and t+1..t+12.
        prefix:
            Output filename prefix.

        Returns
        -------
        list[str]
            Generated image paths.
        """
        output_dir = output_dir or self._default_output_dir()
        frames = []

        if isinstance(source, (str, os.PathLike)) and Path(source).is_dir():
            for path in sorted(Path(source).glob("*.csv")):
                try:
                    frames.append(self._normalise(self._read_input(path), source_name=path.name))
                except ValueError as exc:
                    self.logger.warning("Skipping %s: %s", path.name, exc)
        else:
            frames.append(self._normalise(self._read_input(source)))

        if not frames:
            return []

        normalized = pd.concat(frames, ignore_index=True)
        if variables:
            target_set = {str(value) for value in variables}
            normalized = normalized[normalized["variable"].isin(target_set)].copy()
        generated = []
        for variable, variable_frame in normalized.groupby("variable", sort=False):
            generated.append(
                self._plot_variable_horizon_grid(
                    variable_frame.copy(),
                    output_dir,
                    horizons,
                    prefix,
                )
            )
        return generated


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot aligned forecast horizons for each site.")
    parser.add_argument("source", help="CSV file or directory containing forecast outputs")
    parser.add_argument("--output-dir", default=None, help="Directory to write plots")
    parser.add_argument("--prefix", default="forecast_window", help="Filename prefix")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    plotter = Plot_Class()
    plotter.plot_horizon_windows(
        args.source,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    return 0


Forecast_Plot_Class = Plot_Class


if __name__ == "__main__":
    raise SystemExit(main())
