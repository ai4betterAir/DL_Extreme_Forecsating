"""Generate a DPE-style region station module from AQMS observations.

This script reads the AQMS raw export or workbook, determines which stations
actually have data for each target pollutant, and writes a Python module that
mirrors ``Core_iHPC/Configuration/DPE_region_stations.py``.

The generated module keeps the original comment blocks and author lines, but
filters each regional station list so that only stations with data for that
target remain.
"""

import argparse
import re
from pathlib import Path
from pprint import pformat

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_DIR = REPO_ROOT / "API_Input" / "API_input_raw"
DEFAULT_RAW_CSV = DEFAULT_RAW_DIR / "CSV_File_1777613556.csv"
DEFAULT_METADATA_XLSX = DEFAULT_RAW_DIR / "Air Quality API Excel Power Query.xlsx"
DEFAULT_OUTPUT_PY = REPO_ROOT / "Core_iHPC" / "Configuration" / "DPE2_region_stations.py"
RAW_CSV_HEADER_MAX_SCAN_ROWS = 20


REGION_STATIONS_BY_TARGET = {
    "O3": {
        "Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
        "SW_Sydney": ["BRINGELLY", "CAMPBELLTOWN_WEST", "CAMDEN", "LIVERPOOL", "BARGO", "OAKDALE"],
        "NW_Sydney": ["PARRAMATTA_NORTH", "PENRITH", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
        "CE_Sydney": ["COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE"],
        "Illawara": ["WOLLONGONG", "ALBION_PARK_SOUTH", "KEMBLA_GRANGE"],
        "Lower_Hunter": ["NEWCASTLE", "BERESFIELD", "WALLSEND"],
        "Newcastle": ["MAYFIELD", "STOCKTON"],
    },
    "PM2.5": {
        "Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
        "SW_Sydney": ["BRINGELLY", "CAMPBELLTOWN_WEST", "LIVERPOOL"],
        "CW_Sydney": ["PARRAMATTA_NORTH", "PROSPECT", "ROUSE_HILL"],
        "CE_Sydney": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE"],
        "NW_Sydney": ["PENRITH", "RICHMOND", "ST_MARYS"],
        "Illawara": ["WOLLONGONG", "ALBION_PARK_SOUTH", "KEMBLA_GRANGE"],
        "Lower_Hunter": ["NEWCASTLE", "BERESFIELD", "WALLSEND"],
        "Newcastle": ["MAYFIELD", "STOCKTON"],
        "Upper_Hunter": ["MUSWELLBROOK", "SINGLETON", "MERRIWA"],
    },
    "PM10": {
        "Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
        "SW_Sydney": ["BRINGELLY", "CAMPBELLTOWN_WEST", "LIVERPOOL"],
        "CW_Sydney": ["PARRAMATTA_NORTH", "PROSPECT", "ROUSE_HILL"],
        "CE_Sydney": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE"],
        "NW_Sydney": ["PENRITH", "RICHMOND", "ST_MARYS"],
        "Illawara": ["WOLLONGONG", "ALBION_PARK_SOUTH", "KEMBLA_GRANGE"],
        "Lower_Hunter": ["NEWCASTLE", "BERESFIELD", "WALLSEND"],
        "Newcastle": ["MAYFIELD", "STOCKTON"],
        "Upper_Hunter": ["MUSWELLBROOK", "SINGLETON", "MERRIWA"],
    },
}


def strip_column_prefix(columns):
    return [re.sub(r"^Column1\.", "", col) for col in columns]


def normalize_station_name(value):
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_target_name(value):
    text = str(value).strip().upper()
    if text in {"OZONE", "O3"}:
        return "O3"
    if text in {"PM25", "PM2_5", "PM2.5", "PM_2.5"}:
        return "PM2.5"
    if text == "PM10":
        return "PM10"
    return text


def ensure_australia_brisbane_datetime(series):
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize("Australia/Brisbane")
    return dt.dt.tz_convert("Australia/Brisbane")


def find_wide_export_header_row(raw_csv: Path):
    with raw_csv.open("r", encoding="latin1") as handle:
        for row_idx, line in enumerate(handle):
            if row_idx >= RAW_CSV_HEADER_MAX_SCAN_ROWS:
                break
            first_cells = [cell.strip().strip('"') for cell in line.split(",")[:2]]
            if len(first_cells) >= 2 and first_cells[0].lower() == "date" and first_cells[1].lower() == "time":
                return row_idx
    return None


def load_site_details(metadata_xlsx: Path) -> pd.DataFrame:
    site_df = pd.read_excel(metadata_xlsx, sheet_name="SiteDetails")
    site_df.columns = strip_column_prefix(site_df.columns)
    rename_map = {
        "Site_Id": "Site_Id",
        "SiteName": "SiteName",
        "Longitude": "Longitude",
        "Latitude": "Latitude",
        "Region": "Region",
    }
    site_df = site_df.rename(columns=rename_map)
    site_df["SiteName"] = site_df["SiteName"].astype(str).str.replace(" ", "_", regex=False)
    site_df["SiteNameKey"] = site_df["SiteName"].map(normalize_station_name)
    site_df["Site_Id"] = pd.to_numeric(site_df["Site_Id"], errors="coerce").astype("Int64")
    return site_df.drop_duplicates(subset=["Site_Id"]).reset_index(drop=True)


def detect_wide_export(raw_csv: Path) -> bool:
    header_row = find_wide_export_header_row(raw_csv)
    if header_row is None:
        return False

    try:
        header = pd.read_csv(raw_csv, nrows=1, encoding="latin1", skiprows=header_row)
    except Exception:
        return False

    cols = list(header.columns)
    return any(col.lower() == "date" for col in cols) and any(col.lower() == "time" for col in cols)


def parse_wide_export(raw_csv: Path, site_details: pd.DataFrame) -> pd.DataFrame:
    header_row = find_wide_export_header_row(raw_csv)
    if header_row is None:
        raise ValueError("Could not locate the AQMS wide-export header row")

    df = pd.read_csv(raw_csv, encoding="latin1", skiprows=header_row)
    df.columns = [str(col).strip() for col in df.columns]

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Wide export is missing Date/Time columns")

    meta_cols = {"Date", "Time", "Site_Id", "SiteName", "Longitude", "Latitude"}
    value_cols = [col for col in df.columns if col not in meta_cols]
    if not value_cols:
        raise ValueError("Wide export does not contain any measurement columns")

    measurement_pattern = re.compile(
        r"^(?P<site_id>\d+)\s+(?P<parameter>.+?)\s+1h average\s+\[(?P<units>[^\]]+)\]$",
        re.IGNORECASE,
    )

    records = []
    date_values = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    time_values = df["Time"]

    def _time_to_timedelta(series):
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_timedelta(pd.to_numeric(series, errors="coerce"), unit="h")
        text = series.astype(str).str.strip()
        text = text.where(text.str.lower() != "nan", "")
        text = text.str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
        text = text.str.replace(r"^(\d+)$", r"\1:00:00", regex=True)
        text = text.str.replace(r"^\s*$", "", regex=True)
        parsed = pd.to_timedelta(text, errors="coerce")
        if parsed.notna().any():
            return parsed
        numeric_hours = pd.to_numeric(series, errors="coerce")
        return pd.to_timedelta(numeric_hours, unit="h")

    time_delta = _time_to_timedelta(time_values)
    base_datetime = ensure_australia_brisbane_datetime(date_values + time_delta)

    for col in value_cols:
        match = measurement_pattern.match(col)
        if not match:
            continue

        site_id = int(match.group("site_id"))
        parameter_code = match.group("parameter").strip()
        units = match.group("units").strip()

        series = pd.to_numeric(df[col], errors="coerce")
        part = pd.DataFrame(
            {
                "datetime": base_datetime,
                "Site_Id": site_id,
                "ParameterCode": parameter_code,
                "Units": units,
                "Value": series,
            }
        )
        records.append(part)

    if not records:
        raise ValueError("No measurement columns matched the expected AQMS wide-export pattern")

    long_df = pd.concat(records, ignore_index=True)
    long_df = long_df.dropna(subset=["datetime", "Value"])
    long_df = long_df.merge(
        site_details[["Site_Id", "SiteName", "SiteNameKey", "Longitude", "Latitude", "Region"]],
        on="Site_Id",
        how="left",
    )
    return long_df


def parse_current_observed(metadata_xlsx: Path, site_details: pd.DataFrame) -> pd.DataFrame:
    current_df = pd.read_excel(metadata_xlsx, sheet_name="CurrentObserved")
    current_df.columns = strip_column_prefix(current_df.columns)
    rename_map = {
        "Site_Id": "Site_Id",
        "Parameter.ParameterCode": "ParameterCode",
        "Parameter.ParameterDescription": "ParameterDescription",
        "Parameter.Units": "Units",
        "Parameter.UnitsDescription": "UnitsDescription",
        "Parameter.Frequency": "Frequency",
        "Parameter.Category": "Category",
        "Parameter.SubCategory": "SubCategory",
        "Date": "Date",
        "Hour": "Hour",
        "HourDescription": "HourDescription",
        "Value": "Value",
        "AirQualityCategory": "AirQualityCategory",
        "DeterminingPollutant": "DeterminingPollutant",
    }
    current_df = current_df.rename(columns=rename_map)

    current_df["Site_Id"] = pd.to_numeric(current_df["Site_Id"], errors="coerce").astype("Int64")
    current_df["Hour"] = pd.to_numeric(current_df["Hour"], errors="coerce")
    current_df["Value"] = pd.to_numeric(current_df["Value"], errors="coerce")

    if {"Frequency", "Category", "SubCategory"}.issubset(current_df.columns):
        current_df = current_df[
            (current_df["Frequency"] == "Hourly average")
            & (current_df["Category"] == "Averages")
            & (current_df["SubCategory"] == "Hourly")
        ].copy()

    if "Date" in current_df.columns:
        current_df["datetime"] = pd.to_datetime(current_df["Date"], errors="coerce") + pd.to_timedelta(
            current_df["Hour"].fillna(1) - 1, unit="h"
        )
    else:
        current_df["datetime"] = pd.NaT
    current_df["datetime"] = ensure_australia_brisbane_datetime(current_df["datetime"])

    current_df = current_df.merge(
        site_details[["Site_Id", "SiteName", "SiteNameKey", "Longitude", "Latitude", "Region"]],
        on="Site_Id",
        how="left",
    )

    return current_df


def normalize_observations(raw_csv: Path, metadata_xlsx: Path) -> pd.DataFrame:
    site_details = load_site_details(metadata_xlsx)
    long_df = parse_current_observed(metadata_xlsx, site_details)

    if long_df.empty and raw_csv.exists() and detect_wide_export(raw_csv):
        long_df = parse_wide_export(raw_csv, site_details)

    long_df = long_df.dropna(subset=["datetime", "SiteName", "SiteNameKey", "ParameterCode"])
    long_df["datetime"] = ensure_australia_brisbane_datetime(long_df["datetime"])
    long_df["TargetKey"] = long_df["ParameterCode"].map(normalize_target_name)
    return long_df


def infer_region_maps_from_processed_cache(search_dirs):
    region_maps = {target: {} for target in REGION_STATIONS_BY_TARGET}
    cache_pattern = re.compile(
        r"^Allobs_processed_DPE_station_api_(?P<region>.+?)_ALL\.csv$",
        re.IGNORECASE,
    )

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for cache_path in sorted(search_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv")):
            match = cache_pattern.match(cache_path.name)
            if not match:
                continue
            region_name = match.group("region")
            try:
                cache_columns = pd.read_csv(cache_path, nrows=0).columns.tolist()
            except Exception:
                continue
            if not cache_columns:
                continue

            for target, region_map in REGION_STATIONS_BY_TARGET.items():
                if region_name not in region_map:
                    continue
                base_stations = {normalize_station_name(station) for station in region_map[region_name]}
                filtered = region_maps[target].setdefault(region_name, [])
                for column in cache_columns:
                    if column == "datetime" or "_" not in column:
                        continue
                    column_target, station_name = column.split("_", 1)
                    if normalize_target_name(column_target) != target:
                        continue
                    station_key = normalize_station_name(station_name)
                    if station_key in base_stations and station_key not in filtered:
                        filtered.append(station_key)

    for target in region_maps:
        for region_name in list(region_maps[target].keys()):
            region_maps[target][region_name] = sorted(region_maps[target][region_name])
        region_maps[target] = {
            region_name: stations
            for region_name, stations in region_maps[target].items()
            if stations
        }
    return region_maps


def build_filtered_region_maps(long_df: pd.DataFrame):
    region_maps = {}
    for target, region_map in REGION_STATIONS_BY_TARGET.items():
        target_df = long_df[long_df["TargetKey"] == target].copy()
        filtered_region_map = {}
        for region_name, stations in region_map.items():
            available_stations = []
            for station in stations:
                station_key = normalize_station_name(station)
                station_rows = target_df[target_df["SiteNameKey"] == station_key]
                if not station_rows.empty:
                    available_stations.append(station_key)
            if available_stations:
                filtered_region_map[region_name] = sorted(set(available_stations))
        region_maps[target] = filtered_region_map
    return region_maps


def indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def render_region_module(region_maps) -> str:
    ozone_regions = pformat(region_maps.get("O3", {}), width=120)
    pm25_regions = pformat(region_maps.get("PM2.5", {}), width=120)
    pm10_regions = pformat(region_maps.get("PM10", {}), width=120)

    template = f'''"""
..  module:: DPE_region_stations
    :platform: Unix
    :synopsis: Definition of the basic object class to spliot data.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

###########################################################################################
class DPE2_region_stations(object):
    """ 
    This class defines a Splitting_Class, that contains the capacity to manage the Splitting.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, var_to_predict):
        self.region_aliases = {{
            "SYD": "Sydney",
            "ALLSYD": "Sydney",
            "SW": "SW_Sydney",
            "NW": "NW_Sydney",
            "CE": "CE_Sydney",
            "CW": "CW_Sydney",
            "LH": "Lower_Hunter",
            "UH": "Upper_Hunter",
        }}

        targets = self._normalise_targets(var_to_predict)
        if len(targets) != 1:
            raise ValueError(
                "DPE_region_stations expects one target pollutant at a time. "
                "Expand multi-target runs before building the region map."
            )
        target = targets[0]

        ### Input list of stations
        ### For Ozone
        ### No Ozone for "VINEYARD", not much for "PROSPECT"
        ### CE Sydney: No Ozone for "ALEXANDRIA", "CHULLORA", "LINDFIELD"
        ### Newcastle: No Ozone for "CARRINGTON"
        # Select_Sydney = ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE",  "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK",  "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS" ]  ### for PM2.5
        ozone_regions = {indent_block(ozone_regions, 8)}

        ### For PM2.5
        # SW_Sydney = ["BRINGELLY", "CAMPBELLTOWN_WEST",  "LIVERPOOL", "BARGO" ]      ### for PM2.5
        # SW_Sydney = ["BRINGELLY", "CAMPBELLTOWN_WEST", "CAMDEN", "LIVERPOOL" , "OAKDALE"]      ### for PM2.5
        # ALL_Sydney = ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE",  "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK",  "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS" ]  ### for PM2.5
        # Select_Sydney = ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE",  "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK",  "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS" ]  ### for PM2.5
        ### Upper Hunter only has PM2.5 data, no Ozone (maybe has PM10)
        pm25_regions = {indent_block(pm25_regions, 8)}

        ### For PM10
        ### Using the same regional station groups as PM2.5 unless refined later
        ### Upper Hunter has PM10 in the operational dashboard examples
        pm10_regions = {indent_block(pm10_regions, 8)}

        self.region_stations_by_target = {{
            "O3": ozone_regions,
            "PM2.5": pm25_regions,
            "PM10": pm10_regions,
        }}
        self.available_targets = sorted(self.region_stations_by_target)

        # self.DPE_region_stations_dict = {{
        #     "SW_Sydney" : SW_Sydney,
        #     "NW_Sydney" : NW_Sydney,
        #     "CE_Sydney" : CE_Sydney,
        #     "CW_Sydney" : CW_Sydney,
        #     "Illawara" : Illawara,
        #     "Lower_Hunter" : Lower_Hunter,
        #     "Upper_Hunter" : Upper_Hunter,
        #     "Newcastle" : Newcastle,
        # }}
        # self.DPE_region_stations_dict = {{
        #     #"ALLSYD" : ALL_Sydney,
        #     "SLSYD" : Select_Sydney,
        # }}

        if target not in self.region_stations_by_target:
            raise ValueError(
                "No DPE region station map is configured for target "
                "{{target}}. Available targets: {{available}}".format(
                    target=target,
                    available=", ".join(self.available_targets),
                )
            )

        print("==== Stations for {{target}} ====".format(target=target))
        self.target = target
        self.DPE_region_stations_dict = self.region_stations_by_target[target]

        return

    def _normalise_targets(self, var_to_predict):
        if isinstance(var_to_predict, str):
            return [var_to_predict]

        targets = []
        for value in var_to_predict:
            if isinstance(value, (list, tuple)):
                targets.extend(self._normalise_targets(value))
            else:
                targets.append(value)
        return targets

    def resolve_region_name(self, selected_region):
        return self.region_aliases.get(selected_region, selected_region)


# Backward-compatible alias for code that still imports the old class name.
DPE_region_stations = DPE2_region_stations
'''
    return template


def write_region_module(output_py: Path, module_text: str):
    output_py.parent.mkdir(parents=True, exist_ok=True)
    output_py.write_text(module_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a DPE-style region station Python module from AQMS data."
    )
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--metadata-xlsx", type=Path, default=DEFAULT_METADATA_XLSX)
    parser.add_argument("--output-py", type=Path, default=DEFAULT_OUTPUT_PY)
    args = parser.parse_args()

    if not args.raw_csv.exists() and not args.metadata_xlsx.exists():
        raise FileNotFoundError("No AQMS input files found")
    if not args.metadata_xlsx.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {args.metadata_xlsx}")

    long_df = normalize_observations(args.raw_csv, args.metadata_xlsx)
    region_maps = build_filtered_region_maps(long_df)
    print(f"Processed rows: {len(long_df):,}")
    module_text = render_region_module(region_maps)
    write_region_module(args.output_py, module_text)

    print(f"Generated module: {args.output_py}")
    for target, region_map in region_maps.items():
        print(f"{target}: {sum(len(stations) for stations in region_map.values())} stations across {len(region_map)} regions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
