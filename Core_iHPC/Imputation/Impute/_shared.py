import os
import sys

import numpy as np
import pandas as pd

import Core_iHPC.Outputs.Output_manager as OM
import Core_iHPC.Configuration.DPE_region_stations as DPERT


class ImputationBase(object):
    def __init__(self, Configuration):
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif, '-'))
        self.logger.info('Configuring the data imputation'.center(self.justif, '|'))
        self.logger.info(''.ljust(self.justif, '-'))

    def _normalize_station_token(self, value):
        return "".join(ch for ch in str(value).upper() if ch.isalnum())

    def _station_variants(self, station):
        variants = {
            station,
            station.replace("_", " "),
            station.replace("_", "-"),
            station.replace("-", " "),
        }
        return [variant for variant in variants if variant]

    def _column_matches_station(self, column_name, station):
        column_name = str(column_name)
        for variant in self._station_variants(station):
            if column_name.endswith("_" + variant):
                prefix = column_name[: -(len(variant) + 1)]
                return True, prefix
        return False, None

    def extract_station_data(self, input_data_pd, var_to_predict=None):
        if var_to_predict is None:
            if any('O3_' in col for col in input_data_pd.columns):
                var_to_predict = ['O3']
                self.logger.info("Auto-detected variable: O3")
            else:
                var_to_predict = ['PM2.5']
                self.logger.info("Auto-detected variable: PM2.5")

        dpe_stations = DPERT.DPE_region_stations(var_to_predict)

        known_stations = []
        multi_word_stations = [
            'CAMPBELLTOWN_WEST', 'MACQUARIE_PARK', 'ALBION_PARK_SOUTH',
            'KEMBLA_GRANGE', 'PARRAMATTA_NORTH', 'COOK_AND_PHILLIP',
            'ROUSE_HILL', 'ST_MARYS'
        ]

        for _, region_stations in dpe_stations.DPE_region_stations_dict.items():
            for station in region_stations:
                if station in multi_word_stations and station not in known_stations:
                    known_stations.append(station)

        for _, region_stations in dpe_stations.DPE_region_stations_dict.items():
            for station in region_stations:
                if station not in known_stations:
                    known_stations.append(station)

        self.logger.info("Using {n} known stations from DPE_region_stations".format(
            n=len(known_stations)))

        stations = set()
        for col in input_data_pd.columns:
            for station in known_stations:
                matches_station, _ = self._column_matches_station(col, station)
                if matches_station:
                    stations.add(station)
                    break

        if len(stations) == 0:
            self.logger.warning("No known stations found - using fallback method")
            for col in input_data_pd.columns:
                if '_' in col:
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2:
                        stations.add(parts[1])

        stations = sorted(list(stations))
        self.logger.info("Found {n} stations: {s}".format(
            n=len(stations), s=', '.join(stations)))

        station_dict = {}
        for station in stations:
            station_cols = []
            station_col_names = []
            for col in input_data_pd.columns:
                matches_station, prefix = self._column_matches_station(col, station)
                if not matches_station:
                    continue
                station_cols.append(col)
                station_col_names.append(prefix)
            if len(station_cols) == 0:
                continue

            station_df = input_data_pd[station_cols].copy()
            station_df.columns = station_col_names
            station_dict[station] = station_df

            self.logger.info("  Station {s}: {n} variables ({vars})".format(
                s=station, n=len(station_df.columns),
                vars=', '.join(station_df.columns.tolist())))

        return station_dict

    def combine_station_data(self, station_dict):
        combined_dfs = []
        for station, station_df in station_dict.items():
            station_df_renamed = station_df.copy()
            station_df_renamed.columns = [col + '_' + station for col in station_df.columns]
            combined_dfs.append(station_df_renamed)

        combined_df = pd.concat(combined_dfs, axis=1)
        combined_df = combined_df[sorted(combined_df.columns)]
        return combined_df

    def save_imputed_data(self, imputed_data_pd, station_name=None):
        output_manager = OM.Output_manager_Class(self.Configuration)
        output_manager.output_imputed_data(imputed_data_pd, station_name=station_name)
