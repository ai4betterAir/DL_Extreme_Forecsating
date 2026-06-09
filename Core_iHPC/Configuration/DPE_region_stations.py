"""
..  module:: DPE_region_stations
    :platform: Unix
    :synopsis: Definition of the basic object class to spliot data.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

###########################################################################################
import json
import os
import re
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


SITE_DETAILS_URL = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
PARAMETER_DETAILS_URL = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

# Short region codes used by the dashboard / configs -> API region names.
REGION_CODE_TO_API_REGION = {
    "CNC": "Central Coast",
    "SyE": "Sydney East",
    "SyNW": "Sydney North-west",
    "SySW": "Sydney South-west",
    "CNT": "Central Tablelands",
    "NRT": "Northern Tablelands",
    "SRT": "Southern Tablelands",
    "LHN": "Lower Hunter",
    "UHN": "Upper Hunter",
    "NCL": "Newcastle Local",
    "SYD": "Sydney",
}

# Case-insensitive lookup for region codes (so callers can safely use `.upper()`).
REGION_CODE_TO_API_REGION_NORMALIZED = {
    str(code).strip().upper(): api_region
    for code, api_region in REGION_CODE_TO_API_REGION.items()
}

# Map the API region names to the internal region keys used by the pipeline.
API_REGION_TO_REGION_CODE = {
    api_region: code
    for code, api_region in REGION_CODE_TO_API_REGION.items()
}


def _slug_station_name(name):
    name = str(name or "").strip().upper()
    name = name.replace("-", " ")
    name = re.sub(r"[^A-Z0-9 ]+", " ", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name


def fetch_site_details(timeout=20):
    request = Request(SITE_DETAILS_URL, headers={"accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        return []
    return payload


def build_region_station_map_from_api(timeout=20):
    """
    Build a region -> station list mapping from the NSW site details endpoint.

    Notes:
    - Station names are normalised to the pipeline naming convention (UPPER_SNAKE_CASE).
    - Placeholder rows where SiteName matches the Region label are excluded.
    """
    sites = fetch_site_details(timeout=timeout)
    grouped = {}
    for site in sites:
        api_region = (site.get("Region") or "").strip()
        if not api_region:
            continue
        region_code = API_REGION_TO_REGION_CODE.get(api_region)
        if not region_code:
            continue
        site_name = (site.get("SiteName") or "").strip()
        if not site_name:
            continue
        if site_name.strip().lower() == api_region.strip().lower():
            # Skip placeholder site rows named the same as the region
            continue
        station = _slug_station_name(site_name)
        if not station:
            continue
        grouped.setdefault(region_code, set()).add(station)

    return {region: sorted(stations) for region, stations in grouped.items()}


def _expand_region_keys(region_map):
    """
    Return a region map that supports BOTH:
    - short region codes (CE/NW/SW/...)
    - legacy pipeline region keys (CE_Sydney/NW_Sydney/...)
    """
    expanded = {}
    for key, stations in (region_map or {}).items():
        expanded[key] = stations
        # if key is a short code, add API region name alias
        api_name = REGION_CODE_TO_API_REGION.get(key)
        if api_name:
            expanded.setdefault(api_name, stations)
        # if key is an API region name, add short code alias
        short_code = API_REGION_TO_REGION_CODE.get(key)
        if short_code:
            expanded.setdefault(short_code, stations)
    return expanded


class DPE_region_stations(object):
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
        # Canonical region tokens are the short codes (CC/CE/NW/...) defined in
        # REGION_CODE_TO_API_REGION.
        #
        # Backward-compatibility: accept legacy pipeline keys (CC_Coast, CE_Sydney, ...)
        # by mapping them back to the short code.
        # Build region alias map from canonical short codes and API region names
        self.region_aliases = {}
        for short_code, api_region in REGION_CODE_TO_API_REGION.items():
            # short code -> canonical short code
            self.region_aliases[short_code] = short_code
            self.region_aliases[short_code.upper()] = short_code
            # API region name -> short code
            self.region_aliases[api_region] = short_code
            self.region_aliases[api_region.upper()] = short_code

        # No legacy pipeline aliases — rely on REGION_CODE_TO_API_REGION as source of truth
        # Ensure SYD is present if provided
        if "SYD" in REGION_CODE_TO_API_REGION:
            self.region_aliases.setdefault("SYD", "SYD")

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
        
        ozone_regions = {
            "CNC": ["WYONG"],
            "SyE": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE", "ULTIMO_UTS"],
            "CNT": ["BATHURST", "ORANGE"],
            "LHN": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
            "SyNW": ["PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
            "NCL": ["STOCKTON"],
            "SRT": ["GOULBURN"],
            "SySW": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "LIVERPOOL", "OAKDALE"],
            "SYD": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
            "UHN": ["MERRIWA", "MUSWELLBROOK", "SINGLETON"],
        }
        pm10_regions = {
            "CNC": ["WYONG"],
            "SyE": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE", "ULTIMO_UTS"],
            "CNT": ["BATHURST", "ORANGE"],
            "LHN": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
            "NRT": ["ARMIDALE"],
            "SyNW": ["PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
            "NCL": ["CARRINGTON", "MAYFIELD", "STOCKTON"],
            "SRT": ["GOULBURN"],
            "SySW": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "LIVERPOOL", "OAKDALE"],
            "SYD": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
            "UHN": ["ABERDEEN", "BULGA", "CAMBERWELL", "JERRYS_PLAINS", "MAISON_DIEU", "MERRIWA", "MOUNT_THORLEY", "MUSWELLBROOK", "MUSWELLBROOK_NW", "SINGLETON", "SINGLETON_NW", "SINGLETON_SOUTH", "WARKWORTH", "WYBONG"],
        }
        pm25_regions = {
            "CNC": ["WYONG"],
            "SyE": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE", "ULTIMO_UTS"],
            "CNT": ["BATHURST", "ORANGE"],
            "LHN": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
            "NRT": ["ARMIDALE"],
            "SyNW": ["PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
            "NCL": ["CARRINGTON", "MAYFIELD", "STOCKTON"],
            "SRT": ["GOULBURN"],
            "SySW": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "LIVERPOOL", "OAKDALE"],
            "SYD": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
            "UHN": ["CAMBERWELL", "MERRIWA", "MUSWELLBROOK", "SINGLETON"],
        }

        # Expand the short-code maps so legacy pipeline keys (CE_Sydney, etc.) keep working.
        ozone_regions = _expand_region_keys(ozone_regions)
        pm10_regions = _expand_region_keys(pm10_regions)
        pm25_regions = _expand_region_keys(pm25_regions)

        self.region_stations_by_target = {
            "O3": ozone_regions,
            "PM2.5": pm25_regions,
            "PM10": pm10_regions,
        }
        self.available_targets = sorted(self.region_stations_by_target)

        # Expose the API region short codes so configs can use them directly.
        self.api_region_codes = dict(REGION_CODE_TO_API_REGION)

        if target not in self.region_stations_by_target:
            raise ValueError(
                "No DPE region station map is configured for target "
                "{target}. Available targets: {available}".format(
                    target=target,
                    available=", ".join(self.available_targets),
                )
            )

        print("==== Stations for {target} ====".format(target=target))
        self.target = target
        self.DPE_region_stations_dict = self.region_stations_by_target[target]
        if isinstance(self.DPE_region_stations_dict, tuple):
            self.DPE_region_stations_dict = self.DPE_region_stations_dict[0]

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
        if selected_region is None:
            return selected_region
        token = str(selected_region).strip()
        if not token:
            return selected_region

        if token in self.region_aliases:
            return self.region_aliases[token]
        upper = token.upper()
        if upper in self.region_aliases:
            return self.region_aliases[upper]

        # Also accept the NSW API region names directly (e.g. "Sydney East")
        api_name = token
        if upper in REGION_CODE_TO_API_REGION_NORMALIZED:
            api_name = REGION_CODE_TO_API_REGION_NORMALIZED[upper]
        region_code = API_REGION_TO_REGION_CODE.get(api_name)
        if region_code:
            return region_code

        return token
