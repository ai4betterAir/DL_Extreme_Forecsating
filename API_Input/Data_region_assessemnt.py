import json
import os
import re
import zipfile
from pathlib import Path
import pandas as pd
from xml.sax.saxutils import escape

# --------------------------------------------------------------------
# 1. Paths
# --------------------------------------------------------------------
BASE_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast/API_Input"
API_INPUT_RAW = "API_input_raw"
SITE_META_XLSX = os.path.join(BASE_DIR, API_INPUT_RAW, "Air Quality API Excel Power Query.xlsx")
RAW_CSV = os.path.join(BASE_DIR, API_INPUT_RAW, "CSV_File_1777613556.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Variable_region_Check")
VARIABLE_REGION_SITE_MAP_JSON = os.path.join(OUTPUT_DIR, "variable_region_site_maps.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_measurement_csv(csv_path, **kwargs):
    """Read AquisNET exports with a small encoding fallback set."""
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Unable to read {csv_path} using encodings {encodings}: {last_error.reason}",
    )


def make_excel_sheet_name(name, used_names):
    """Create an Excel-safe sheet name and keep it unique."""
    cleaned = re.sub(r"[:\\/?*\[\]]", "_", str(name)).strip()
    cleaned = cleaned or "Sheet"
    cleaned = cleaned[:31]

    candidate = cleaned
    counter = 1
    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate


def join_sorted_unique(values):
    cleaned = {str(value).strip() for value in values if pd.notna(value) and str(value).strip()}
    return ", ".join(sorted(cleaned))


def normalize_site_name(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return "" if text in {"", "NA", "N_A", "NAN", "<NA>"} else text


def format_datetime_column(series):
    return series.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")


def dataframe_for_excel(df):
    """Convert a DataFrame to text-safe values for a lightweight XLSX writer."""
    excel_df = df.copy()
    for col in excel_df.columns:
        if pd.api.types.is_datetime64_any_dtype(excel_df[col]):
            excel_df[col] = format_datetime_column(excel_df[col])
        else:
            excel_df[col] = excel_df[col].astype("string").fillna("")
    return excel_df


def excel_column_name(col_idx):
    """Convert a 1-based column index to Excel letters."""
    name = []
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        name.append(chr(65 + remainder))
    return "".join(reversed(name))


def worksheet_xml_from_dataframe(df):
    """Build worksheet XML using inline strings only."""
    excel_df = dataframe_for_excel(df)
    rows = [list(excel_df.columns)] + excel_df.values.tolist()
    xml_rows = []

    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{excel_column_name(col_idx)}{row_idx}"
            text = escape("" if pd.isna(value) else str(value))
            text = text.replace("\n", "&#10;")
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
            )
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def write_simple_xlsx(workbook_path, sheets):
    """Write a basic multi-sheet XLSX workbook without external Excel packages."""
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheets>',
    ]
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]

    for idx, (sheet_name, _) in enumerate(sheets, start=1):
        safe_name = escape(sheet_name)
        workbook_xml.append(
            f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )

    workbook_xml.extend(['</sheets>', '</workbook>'])
    workbook_rels.append("</Relationships>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(workbook_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", "".join(content_types))
        workbook.writestr("_rels/.rels", root_rels)
        workbook.writestr("xl/workbook.xml", "".join(workbook_xml))
        workbook.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))

        for idx, (_, df) in enumerate(sheets, start=1):
            workbook.writestr(
                f"xl/worksheets/sheet{idx}.xml",
                worksheet_xml_from_dataframe(df),
            )


TARGET_REGION_SITE_MAPS = {
    "OZONE": {
        "Sydney": [
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
        ],
        "SW_Sydney": ["BRINGELLY", "CAMPBELLTOWN_WEST", "CAMDEN", "LIVERPOOL", "BARGO", "OAKDALE"],
        "NW_Sydney": ["PARRAMATTA_NORTH", "PENRITH", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
        "CE_Sydney": ["COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE"],
        "Illawara": ["WOLLONGONG", "ALBION_PARK_SOUTH", "KEMBLA_GRANGE"],
        "Lower_Hunter": ["NEWCASTLE", "BERESFIELD", "WALLSEND"],
        "Newcastle": ["MAYFIELD", "STOCKTON"],
    },
    "PM2.5": {
        "Sydney": [
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
        ],
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
        "Sydney": [
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
        ],
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

TARGET_REGION_JSON_KEYS = {
    "OZONE": "ozone_regions",
    "PM2.5": "pm25_regions",
    "PM10": "pm10_regions",
}

PROCESSED_REGION_CSV_PATTERN = re.compile(
    r"^Allobs_processed_DPE_station_api_(?P<region>.+?)_ALL\.csv$"
)
PROCESSED_VARIABLE_PATTERN = re.compile(r"^(?P<variable>[^_]+)_(?P<site>.+)$", re.I)


def _dedupe_keep_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _parse_processed_csv_header(csv_path):
    header_df = pd.read_csv(csv_path, nrows=0)
    region_match = PROCESSED_REGION_CSV_PATTERN.match(csv_path.name)
    if not region_match:
        return None, {}

    region_name = region_match.group("region")
    variable_sites = {}

    for column in header_df.columns:
        column = str(column).strip()
        if column.lower() == "datetime":
            continue

        match = PROCESSED_VARIABLE_PATTERN.match(column)
        if not match:
            continue

        variable = match.group("variable").upper()
        if variable in {"PM25", "PM2_5", "PM_2_5"}:
            variable = "PM2.5"

        site = normalize_site_name(match.group("site"))
        if not site:
            continue
        variable_sites.setdefault(variable, []).append(site)

    return region_name, {var: _dedupe_keep_order(sites) for var, sites in variable_sites.items()}


def build_variable_region_site_maps(available_data=None, processed_csv_dir=BASE_DIR):
    """Build per-variable region -> site maps from processed region CSV headers."""
    processed_csv_paths = sorted(Path(processed_csv_dir).glob("Allobs_processed_DPE_station_api_*_ALL.csv"))
    if processed_csv_paths:
        variable_region_maps = {}
        for csv_path in processed_csv_paths:
            region_name, variable_sites = _parse_processed_csv_header(csv_path)
            if not region_name:
                continue

            for variable, sites in variable_sites.items():
                if sites:
                    variable_region_maps.setdefault(variable, {})[region_name] = sites

        return variable_region_maps

    # Fallback for environments without the processed region CSVs.
    if available_data is None:
        return {}

    available_data = available_data.copy()
    available_data["SiteNameNormalized"] = available_data["SiteName"].map(normalize_site_name)
    available_data = available_data[available_data["SiteNameNormalized"] != ""].copy()

    variable_to_sites = (
        available_data.groupby("Variable", dropna=False)["SiteNameNormalized"]
        .agg(lambda s: sorted({str(value).strip() for value in s.dropna() if str(value).strip()}))
        .to_dict()
    )

    generic_region_maps = {}
    for variable, variable_df in available_data.groupby("Variable", dropna=False):
        region_pairs = []
        for region, region_df in variable_df.groupby("Region", dropna=False):
            if pd.isna(region):
                continue

            region_name = str(region).strip()
            if not region_name or region_name.lower() == "nan":
                continue

            sites = sorted(
                {
                    str(value).strip()
                    for value in region_df["SiteNameNormalized"]
                    if str(value).strip()
                }
            )
            if sites:
                region_pairs.append((region_name, sites))

        generic_region_maps[str(variable).strip()] = {
            region: sites for region, sites in sorted(region_pairs, key=lambda item: item[0])
        }

    variable_region_maps = {}
    for variable in sorted(variable_to_sites):
        if variable in TARGET_REGION_SITE_MAPS:
            observed_sites = set(variable_to_sites.get(variable, []))
            filtered_map = {}
            for region, site_list in TARGET_REGION_SITE_MAPS[variable].items():
                matched_sites = [site for site in site_list if site in observed_sites]
                if matched_sites:
                    filtered_map[region] = matched_sites
            variable_region_maps[variable] = filtered_map
        else:
            variable_region_maps[variable] = generic_region_maps.get(variable, {})

    return variable_region_maps


def build_exact_target_region_maps(available_data):
    """Return named region dictionaries using the DPE JSON key style."""
    available_maps = build_variable_region_site_maps(available_data)
    named_maps = {}

    for variable, region_map in available_maps.items():
        if variable == "OZONE":
            json_key = "ozone_regions"
        elif variable == "PM2.5":
            json_key = "pm25_regions"
        elif variable == "PM10":
            json_key = "pm10_regions"
        else:
            json_key = f"{variable.lower()}_regions"

        named_maps[json_key] = region_map

    return named_maps


def dump_region_site_maps_json(data):
    """Serialize nested region->sites maps with each site list kept inline."""
    chunks = ["{"]
    outer_items = list(data.items())

    for outer_idx, (outer_key, regions) in enumerate(outer_items):
        outer_comma = "," if outer_idx < len(outer_items) - 1 else ""
        chunks.append(f'  "{outer_key}": {{')

        region_items = list(regions.items())
        for region_idx, (region_name, sites) in enumerate(region_items):
            region_comma = "," if region_idx < len(region_items) - 1 else ""
            chunks.append(f'    "{region_name}": {json.dumps(sites, ensure_ascii=False)}{region_comma}')

        chunks.append(f"  }}{outer_comma}")

    chunks.append("}")
    return "\n".join(chunks) + "\n"

# --------------------------------------------------------------------
# 2. Load site metadata
# --------------------------------------------------------------------
meta = pd.read_excel(SITE_META_XLSX)

# Remove "Column1." prefix if present
meta.columns = [c.replace("Column1.", "") for c in meta.columns]

required_cols = ["Site_Id", "SiteName", "Longitude", "Latitude", "Region"]
missing = [c for c in required_cols if c not in meta.columns]
if missing:
    raise ValueError(f"Missing expected columns in metadata file: {missing}")

meta["Site_Id"] = meta["Site_Id"].astype(str)

# --------------------------------------------------------------------
# 3. Load measurement CSV
#    Skip first two rows (range/source)
# --------------------------------------------------------------------
data = read_measurement_csv(RAW_CSV, skiprows=2, low_memory=False)

# --------------------------------------------------------------------
# 4. Build a proper datetime index from Date + Time
# --------------------------------------------------------------------
# Try common formats; adjust if your file is different (e.g. day-first)
data["DateTime"] = pd.to_datetime(
    data["Date"].astype(str) + " " + data["Time"].astype(str),
    errors="coerce",
    dayfirst=True,
)

# If there are invalid rows, you can drop them
data = data.dropna(subset=["DateTime"]).reset_index(drop=True)

# Optionally sort by DateTime
data = data.sort_values("DateTime").reset_index(drop=True)

# --------------------------------------------------------------------
# 5. Parse measurement column names into (site_id, variable, unit)
# --------------------------------------------------------------------
id_time_cols = {"Date", "Time", "DateTime"}

col_info = []

# Example column: "1001 NOX 1h average [pphm]"
pattern = re.compile(r"^(\d+)\s+([^\[]+?)\s+1h average\s+(\[[^\]]+\])$")

for col in data.columns:
    if col in id_time_cols:
        continue

    m = pattern.match(col.strip())
    if m:
        site_id = m.group(1)
        var = m.group(2).strip()
        unit = m.group(3)
        col_info.append({
            "orig_col": col,
            "site_id": site_id,
            "variable": var,
            "unit": unit
        })
    else:
        # Columns that don't match pattern are ignored, or you can log them
        # print("Unrecognized measurement column:", col)
        pass

col_info_df = pd.DataFrame(col_info)

# --------------------------------------------------------------------
# 6. Compute stats for each (site_id, variable):
#    - start time
#    - end time
#    - total count
#    - non-null count
#    - % missing
# --------------------------------------------------------------------
records = []

for _, row in col_info_df.iterrows():
    col = row["orig_col"]
    site_id = row["site_id"]
    var = row["variable"]
    unit = row["unit"]

    series = data[col]
    total = series.shape[0]
    non_null = series.notna().sum()
    missing = total - non_null

    # Avoid division by zero
    pct_missing = (missing / total * 100) if total > 0 else 0.0
    pct_available = (non_null / total * 100) if total > 0 else 0.0

    # Start/end where data is actually present (non-null)
    if non_null > 0:
        non_null_idx = series.dropna().index
        start_time = data.loc[non_null_idx.min(), "DateTime"]
        end_time = data.loc[non_null_idx.max(), "DateTime"]
    else:
        start_time = pd.NaT
        end_time = pd.NaT

    records.append({
        "Site_Id": site_id,
        "Variable": var,
        "Unit": unit,
        "StartDateTime": start_time,
        "EndDateTime": end_time,
        "TotalCount": total,
        "NonNullCount": non_null,
        "MissingCount": missing,
        "PctMissing": pct_missing,
        "PctAvailable": pct_available,
    })

availability_df = pd.DataFrame(records)

# --------------------------------------------------------------------
# 7. Attach SiteName, Region (and keep as long format)
# --------------------------------------------------------------------
availability_with_meta = availability_df.merge(
    meta[["Site_Id", "SiteName", "Region"]],
    on="Site_Id",
    how="left"
)

availability_with_meta["SiteName"] = availability_with_meta["SiteName"].astype("string").str.strip()
availability_with_meta["Region"] = availability_with_meta["Region"].astype("string").str.strip()

# --------------------------------------------------------------------
# 8. Build one site-level summary table:
#    columns: Region, Site_Id, SiteName, AvailableVariables, TimeRange
# --------------------------------------------------------------------
available_data = availability_with_meta[
    availability_with_meta["NonNullCount"] > 0
].copy()

site_summary = (
    available_data.groupby(["Region", "Site_Id", "SiteName"], dropna=False)
    .agg(
        AvailableVariables=("Variable", join_sorted_unique),
        StartDateTime=("StartDateTime", "min"),
        EndDateTime=("EndDateTime", "max"),
    )
    .reset_index()
)

site_summary["TimeRange"] = (
    site_summary["StartDateTime"].dt.strftime("%Y-%m-%d %H:%M")
    + " to "
    + site_summary["EndDateTime"].dt.strftime("%Y-%m-%d %H:%M")
)

site_variable_sets = (
    available_data.groupby(["Region", "Site_Id", "SiteName"], dropna=False)["Variable"]
    .agg(lambda s: {str(value).strip() for value in s.dropna() if str(value).strip()})
    .reset_index(name="VariableSet")
)

region_records = []
for region, region_df in site_variable_sets.groupby("Region", dropna=False):
    variable_sets = [var_set for var_set in region_df["VariableSet"] if var_set]
    common_variables = set.intersection(*variable_sets) if variable_sets else set()

    region_records.append({
        "Region": region,
        "Sites": join_sorted_unique(region_df["SiteName"]),
        "CommonVariables": ", ".join(sorted(common_variables)),
    })

region_summary = pd.DataFrame(region_records).sort_values("Region").reset_index(drop=True)

# --------------------------------------------------------------------
# 9. Create one Excel workbook with a summary sheet and one sheet per variable
# --------------------------------------------------------------------
workbook_path = os.path.join(OUTPUT_DIR, "region_site_variable_data_summary.xlsx")
used_sheet_names = set()
sheets = [
    (
        make_excel_sheet_name("Summary", used_sheet_names),
        site_summary.sort_values(["Region", "SiteName"]).reset_index(drop=True),
    )
]

for variable in sorted(available_data["Variable"].dropna().unique()):
    variable_df = (
        available_data[available_data["Variable"] == variable][
            ["Region", "Site_Id", "SiteName", "Variable", "Unit", "StartDateTime", "EndDateTime"]
        ]
        .sort_values(["Region", "SiteName"])
        .reset_index(drop=True)
    )

    variable_df["TimeRange"] = (
        variable_df["StartDateTime"].dt.strftime("%Y-%m-%d %H:%M")
        + " to "
        + variable_df["EndDateTime"].dt.strftime("%Y-%m-%d %H:%M")
    )

    sheets.append(
        (
            make_excel_sheet_name(variable, used_sheet_names),
            variable_df,
        )
    )

write_simple_xlsx(workbook_path, sheets)

print(f"Saved variable workbook to {workbook_path}")

# --------------------------------------------------------------------
# 10. Create a second workbook with region-level and site-level summaries
# --------------------------------------------------------------------
overview_workbook_path = os.path.join(OUTPUT_DIR, "region_and_site_variable_overview.xlsx")
overview_sheets = [
    ("Regions", region_summary),
    ("Sites", site_summary.sort_values(["Region", "SiteName"]).reset_index(drop=True)),
]

write_simple_xlsx(overview_workbook_path, overview_sheets)

print(f"Saved overview workbook to {overview_workbook_path}")

# --------------------------------------------------------------------
# 11. Save the overview data as JSON for downstream use
# --------------------------------------------------------------------
overview_json_path = os.path.join(OUTPUT_DIR, "region_and_site_variable_overview.json")
sites_json_df = site_summary.sort_values(["Region", "SiteName"]).reset_index(drop=True).copy()
regions_json_df = region_summary.sort_values("Region").reset_index(drop=True).copy()

for col in ["StartDateTime", "EndDateTime"]:
    sites_json_df[col] = format_datetime_column(sites_json_df[col])

overview_json = {
    "Regions": regions_json_df.where(pd.notna(regions_json_df), None).to_dict(orient="records"),
    "Sites": sites_json_df.where(pd.notna(sites_json_df), None).to_dict(orient="records"),
}

with open(overview_json_path, "w", encoding="utf-8") as json_file:
    json.dump(overview_json, json_file, ensure_ascii=False, indent=2)

print(f"Saved overview JSON to {overview_json_path}")

# --------------------------------------------------------------------
# 12. Save per-variable region -> site maps for downstream use
# --------------------------------------------------------------------
variable_region_site_maps = build_exact_target_region_maps(available_data)

with open(VARIABLE_REGION_SITE_MAP_JSON, "w", encoding="utf-8") as json_file:
    json_file.write(dump_region_site_maps_json(variable_region_site_maps))

print(f"Saved variable region-site JSON to {VARIABLE_REGION_SITE_MAP_JSON}")
