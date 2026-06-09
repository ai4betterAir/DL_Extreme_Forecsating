"""Process AQMS raw exports into region-level wide CSV files.

This script supports two AQMS input shapes:
- a wide CSV export with columns like ``Date``, ``Time`` and
  ``1001 CO 1h average [ppm]``
- the Excel workbook export that contains ``SiteDetails`` and
  ``CurrentObserved`` sheets

It writes one wide CSV per region using the region station layout that is
embedded below. Each output keeps all available variables, not a single
pollutant.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_DIR = REPO_ROOT / "API_Input" / "API_input_raw"
DEFAULT_RAW_CSV = DEFAULT_RAW_DIR / "CSV_File_1777613556.csv"
DEFAULT_METADATA_XLSX = DEFAULT_RAW_DIR / "Air Quality API Excel Power Query.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "API_Input" / "Inputs"
RAW_CSV_HEADER_MAX_SCAN_ROWS = 20

REGION_ALIAS_MAP = {
    "Southern Tablelands": "SR_Table",
    "Northern Tablelands": "NR_Table",
    "Central Tablelands": "CN_Table",
    "Sydney East": "CE_Sydney",
    "Sydney South-west": "SW_Sydney",
    "Sydney North-west": "NW_Sydney",
    "Central Coast": "CC_Coast",
    "Lower Hunter": "Lower_Hunter",
    "Upper Hunter": "Upper_Hunter",
    "Newcastle Local": "Newcastle",
}

REGION_SHORT_TO_LONG = {
    "CC": "CC_Coast",
    "CE": "CE_Sydney",
    "CN": "CN_Table",
    "NR": "NR_Table",
    "SR": "SR_Table",
    "NW": "NW_Sydney",
    "SW": "SW_Sydney",
    "LH": "Lower_Hunter",
    "UH": "Upper_Hunter",
    "NC": "Newcastle",
    "SYD": "Sydney",
}

REGION_LONG_NAME_MAP = {
    "SR_Table": "Southern Tablelands",
    "NR_Table": "Northern Tablelands",
    "CN_Table": "Central Tablelands",
    "CE_Sydney": "Sydney East",
    "SW_Sydney": "Sydney South-west",
    "NW_Sydney": "Sydney North-west",
    "CC_Coast": "Central Coast",
    "Lower_Hunter": "Lower Hunter",
    "Upper_Hunter": "Upper Hunter",
    "Newcastle": "Newcastle Local",
    "Sydney": "Sydney",
}

ALLOWED_REGIONS = {
    "SR_Table",
    "NR_Table",
    "CN_Table",
    "CE_Sydney",
    "SW_Sydney",
    "NW_Sydney",
    "CC_Coast",
    "Lower_Hunter",
    "Upper_Hunter",
    "Newcastle",
    "Sydney",
}

REGION_OUTPUT_ORDER = [
    "SR_Table",
    "NR_Table",
    "CN_Table",
    "CE_Sydney",
    "SW_Sydney",
    "NW_Sydney",
    "CC_Coast",
    "Lower_Hunter",
    "Upper_Hunter",
    "Newcastle",
    "Sydney",
]

SYDNEY_REGION_SITES = [
    "BRINGELLY",
    "CAMDEN",
    "CAMPBELLTOWN_WEST",
    "COOK_AND_PHILLIP",
    "EARLWOOD",
    "LIDCOMBE",
    "LIVERPOOL",
    "MACQUARIE_PARK",
    "OAKDALE",
    "PARRAMATTA_NORTH",
    "PENRITH",
    "PROSPECT",
    "RANDWICK",
    "RICHMOND",
    "ROUSE_HILL",
    "ROZELLE",
    "ST_MARYS",
]


def strip_column_prefix(columns):
    return [re.sub(r"^Column1\.", "", col) for col in columns]


def normalize_station_name(value):
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_region_filename(value):
    """Convert a region label into a filesystem-friendly token."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


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
    site_df["SiteName"] = site_df["SiteName"].astype(str).str.strip()
    site_df["SiteNameKey"] = site_df["SiteName"].map(normalize_station_name)
    site_df["Region"] = site_df["Region"].astype(str).str.strip()
    site_df["Region"] = site_df["Region"].replace(REGION_ALIAS_MAP)
    # Allow short region codes (CE/NW/...) but always persist the long/original pipeline key.
    site_df["Region"] = site_df["Region"].replace(REGION_SHORT_TO_LONG)
    site_df["RegionKey"] = site_df["Region"].str.casefold()
    site_df["Site_Id"] = pd.to_numeric(site_df["Site_Id"], errors="coerce").astype("Int64")
    site_df = site_df[
        site_df["Region"].notna()
        & (site_df["RegionKey"] != "nan")
        & site_df["Region"].isin(ALLOWED_REGIONS)
    ].copy()
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

    base_datetime = date_values + time_delta
    base_datetime = ensure_australia_brisbane_datetime(base_datetime)

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
        site_details[["Site_Id", "SiteName", "SiteNameKey", "Longitude", "Latitude", "Region", "RegionKey"]],
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
        site_details[["Site_Id", "SiteName", "Longitude", "Latitude", "Region", "RegionKey"]],
        on="Site_Id",
        how="left",
    )

    return current_df


def normalize_observations(raw_csv: Path, metadata_xlsx: Path) -> pd.DataFrame:
    site_details = load_site_details(metadata_xlsx)

    if detect_wide_export(raw_csv):
        long_df = parse_wide_export(raw_csv, site_details)
    else:
        long_df = parse_current_observed(metadata_xlsx, site_details)

    long_df = long_df.dropna(subset=["datetime", "SiteName", "SiteNameKey", "ParameterCode", "RegionKey"])
    long_df["datetime"] = ensure_australia_brisbane_datetime(long_df["datetime"])
    # Ensure downstream outputs always use the long/original pipeline region key.
    long_df["Region"] = long_df["Region"].replace(REGION_SHORT_TO_LONG)
    return long_df


def region_to_wide(region_df: pd.DataFrame):
    if region_df.empty:
        return pd.DataFrame()

    region_df["Var_name"] = region_df["ParameterCode"].astype(str) + "_" + region_df["SiteName"].astype(str)

    wide_df = (
        region_df.pivot_table(
            index="datetime",
            columns="Var_name",
            values="Value",
            aggfunc="first",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    wide_df.index.name = "datetime"
    return wide_df


def save_region_outputs(long_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    long_df = long_df.copy()
    # Always use the long/original pipeline region key for output selection + filenames.
    if "Region" in long_df.columns:
        long_df["Region"] = long_df["Region"].replace(REGION_SHORT_TO_LONG)

    for region_name in REGION_OUTPUT_ORDER:
        if region_name == "Sydney":
            region_df = long_df[long_df["SiteNameKey"].isin(SYDNEY_REGION_SITES)].copy()
        else:
            region_df = long_df[long_df["Region"] == region_name].copy()

        wide_df = region_to_wide(region_df)
        if wide_df.empty:
            continue

        # File naming should prefer the long/display region name (e.g. 'Southern Tablelands')
        # rather than the pipeline key (e.g. 'SR_Table'). Use REGION_LONG_NAME_MAP when available.
        display_region = REGION_LONG_NAME_MAP.get(region_name, region_name)
        filename_region = normalize_region_filename(display_region)
        filename = f"Allobs_processed_DPE_station_api_{filename_region}_ALL.csv"
        output_path = output_dir / filename
        wide_df.to_csv(output_path, index=True)
        written_paths.append(output_path)

    return written_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Process AQMS raw exports into region CSVs")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--metadata-xlsx", type=Path, default=DEFAULT_METADATA_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.metadata_xlsx.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {args.metadata_xlsx}")

    if not args.raw_csv.exists() and not args.metadata_xlsx.exists():
        raise FileNotFoundError("No AQMS input files found")

    long_df = normalize_observations(args.raw_csv, args.metadata_xlsx)
    written_paths = save_region_outputs(long_df, args.output_dir)

    print(f"Processed rows: {len(long_df):,}")
    print(f"Region files written: {len(written_paths)}")
    for path in written_paths:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
