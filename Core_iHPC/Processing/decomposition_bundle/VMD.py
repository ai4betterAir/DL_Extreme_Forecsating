"""
Generic decomposition backend.

This module now contains both:
- the station decomposition backend class
- the joint-collapse helper used by the per-IMF/joint pipeline
"""

import sys
import os
import stat
import numpy as np
import pandas as pd
from vmdpy import VMD
import Core_iHPC.Outputs.Output_manager as OM
import re


###########################################################################################
class VMD_decomposition_Class(object):
    """Decompose time series data using Variational Mode Decomposition."""

    def __init__(self, Configuration):
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

###########################################################################################
    def decompose_single_column(self, data_series, column_name):
        """
        Decompose a single time series column using VMD.
        """
        n_imfs = int(getattr(self.Configuration, 'vmd_n_imfs', 20))
        alpha = float(getattr(self.Configuration, 'vmd_alpha', 2000))
        tau = float(getattr(self.Configuration, 'vmd_tau', 0.0))
        K = n_imfs
        DC = int(getattr(self.Configuration, 'vmd_DC', 0))
        init = int(getattr(self.Configuration, 'vmd_init', 1))
        tol_val = getattr(self.Configuration, 'vmd_tol', 1e-7)
        tol = float(tol_val)

        self.logger.info("Decomposing: {col}".format(col=column_name).ljust(self.justif-2,'.') + 'RUN')
        self.logger.info("  n_imfs = {n}, alpha = {a}".format(n=n_imfs, a=alpha))

        signal = data_series.values
        if np.isnan(signal).any():
            self.logger.error("NaN values found in signal".ljust(self.justif-2,'.') + 'FAIL')
            raise ValueError("Input signal contains NaN values. Please impute data first.")

        try:
            u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
            self.logger.info("VMD decomposition".ljust(self.justif-2,'.') + 'OK')
        except Exception as e:
            self.logger.error("VMD decomposition failed: {e}".format(e=str(e)))
            raise

        vmd_len = u.shape[1]
        if vmd_len != len(signal):
            self.logger.warning("VMD output length {v} != signal length {s}, trimming.".format(
                v=vmd_len, s=len(signal)))
            signal = signal[:vmd_len]
            data_series = data_series.iloc[:vmd_len]

        imf_dict = {}
        for i in range(K):
            imf_dict['IMF_{n}'.format(n=i+1)] = u[i, :]

        sum_imfs = np.sum(u, axis=0)
        residual = signal - sum_imfs

        imf_dict['Residual'] = residual
        imf_dict['Original'] = signal
        imf_dict['Reconstructed'] = sum_imfs

        result_df = pd.DataFrame(imf_dict, index=data_series.index)

        self.logger.info("Created {n} IMFs + Residual".format(n=K).ljust(self.justif-2,'.') + 'OK')
        return result_df

###########################################################################################
    def decompose_per_station(self, station_imputed_dict, target_variable):
        """
        Decompose target variable for each station separately.
        """
        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Per-Station Decomposition'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info("Target variable: {v}".format(v=target_variable))
        self.logger.info("Stations to process: {n}".format(n=len(station_imputed_dict)))

        station_decomposed_dict = {}

        for station, station_df in station_imputed_dict.items():
            self.logger.info(''.ljust(self.justif,'-'))
            self.logger.info("Station: {s}".format(s=station).center(self.justif,'|'))
            self.logger.info(''.ljust(self.justif,'-'))

            if target_variable not in station_df.columns:
                self.logger.warning("Variable {v} not found in station {s}".format(
                    v=target_variable, s=station).ljust(self.justif-2,'.') + 'SKIP')
                continue

            try:
                decomposed_df = self.decompose_single_column(
                    station_df[target_variable],
                    "{v}_{s}".format(v=target_variable, s=station)
                )
                station_decomposed_dict[station] = decomposed_df
                self.logger.info("Station {s} decomposed".format(s=station).ljust(self.justif-2,'.') + 'OK')
            except Exception as e:
                self.logger.error("Failed to decompose station {s}: {e}".format(
                    s=station, e=str(e)))
                continue

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Per-Station Decomposition Complete'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info("Successfully decomposed {n}/{t} stations".format(
            n=len(station_decomposed_dict), t=len(station_imputed_dict)))

        return station_decomposed_dict

###########################################################################################
    def save_decomposed_data(self, station_decomposed_dict, target_variable):
        """
        Save decomposed data per station.
        """
        Output_Manager = OM.Output_manager_Class(self.Configuration)

        for station, decomposed_df in station_decomposed_dict.items():
            output_df = decomposed_df.copy()
            if isinstance(decomposed_df.index, pd.DatetimeIndex):
                output_df.insert(0, 'ds', decomposed_df.index.strftime('%d/%m/%Y %H:%M'))

            Output_Manager.output_vmd_decomposition(
                output_df,
                target_variable,
                station_name=station
            )

            self.logger.info("Saved decomposition for station {s}".format(
                s=station).ljust(self.justif-2,'.') + 'OK')

###########################################################################################
    def decompose(self, imputed_data_pd, station_imputed_dict, target_column, save_data=False):
        """
        Main decomposition method for per-station processing.
        """
        station_decomposed_dict = self.decompose_per_station(
            station_imputed_dict,
            target_column
        )

        if save_data:
            self.save_decomposed_data(station_decomposed_dict, target_column)

        return station_decomposed_dict


###########################################################################################
def collapse_joint_imfs(station_decomposed_dict, logger=None, justif=100):
    """
    Collapse all IMF columns into a single residual-like series per station.
    """
    joint_dict = {}
    for station, station_df in station_decomposed_dict.items():
        component_cols = [
            col for col in station_df.columns
            if re.match(r"^IMF[_\s]?\d+$", str(col), re.IGNORECASE) or str(col).lower() == "residual"
        ]
        if not component_cols:
            fallback_cols = [
                col for col in station_df.columns
                if str(col).lower() not in {"original", "reconstructed", "ds"}
                and pd.api.types.is_numeric_dtype(station_df[col])
            ]
            if fallback_cols:
                component_cols = fallback_cols
                if logger is not None:
                    logger.warning(
                        f'  {station}: no IMF labels found; collapsing numeric columns {fallback_cols}'.ljust(
                            justif - 2, '.') + 'FALLBACK'
                    )
            else:
                if "Residual" in station_df.columns:
                    component_cols = ["Residual"]
                elif "Original" in station_df.columns:
                    component_cols = ["Original"]
                else:
                    raise ValueError(
                        f"No IMF component columns found for station {station}. "
                        f"Available columns: {list(station_df.columns)[:10]}"
                    )

        collapsed = station_df.copy()
        collapsed_residual = collapsed[component_cols].sum(axis=1)

        output_df = pd.DataFrame(index=collapsed.index)
        output_df["Residual"] = collapsed_residual

        if "Original" in collapsed.columns:
            output_df["Original"] = collapsed["Original"]
        else:
            output_df["Original"] = collapsed_residual

        output_df["Reconstructed"] = collapsed_residual
        joint_dict[station] = output_df

        if logger is not None:
            logger.info(
                f'  {station}: {len(component_cols)} IMF components collapsed'.ljust(
                    justif - 2, '.') + 'OK')

    return joint_dict


VMD_Class = VMD_decomposition_Class
collapse_joint_components = collapse_joint_imfs

