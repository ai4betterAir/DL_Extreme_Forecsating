import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb

from ._shared import ImputationBase

MODEL_NAME = "AQUISTIL"


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _resolve_target_column(columns, requested_target):
    if requested_target in columns:
        return requested_target

    alias_groups = {
        "O3": ("O3", "OZONE"),
        "OZONE": ("O3", "OZONE"),
        "PM25": ("PM25", "PM2.5", "PM_25"),
        "PM2.5": ("PM25", "PM2.5", "PM_25"),
        "PM_25": ("PM25", "PM2.5", "PM_25"),
        "PM10": ("PM10",),
    }

    normalized_columns = {_normalize_name(col): col for col in columns}
    normalized_target = _normalize_name(requested_target)

    for alias in alias_groups.get(normalized_target, (requested_target,)):
        resolved = normalized_columns.get(_normalize_name(alias))
        if resolved is not None:
            return resolved

    return None


def _target_prefixes(requested_target):
    normalized_target = _normalize_name(requested_target)
    alias_groups = {
        "O3": ("O3", "OZONE"),
        "OZONE": ("O3", "OZONE"),
        "PM25": ("PM25", "PM2.5", "PM_25"),
        "PM2.5": ("PM25", "PM2.5", "PM_25"),
        "PM_25": ("PM25", "PM2.5", "PM_25"),
        "PM10": ("PM10",),
    }
    return alias_groups.get(normalized_target, (requested_target,))


def _prepare_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    datetime_col = None
    for candidate in ("DateTime", "datetime", "timestamp"):
        if candidate in out.columns:
            datetime_col = candidate
            break

    if datetime_col is not None:
        dt = pd.to_datetime(out[datetime_col], errors="coerce")
        out[datetime_col] = dt
    elif isinstance(out.index, pd.DatetimeIndex):
        dt = pd.to_datetime(out.index, errors="coerce")
    else:
        return out

    if isinstance(dt, pd.DatetimeIndex):
        out["hour"] = dt.hour.astype(float)
        out["dayofweek"] = dt.dayofweek.astype(float)
        out["month"] = dt.month.astype(float)
        out["dayofyear"] = dt.dayofyear.astype(float)
    else:
        out["hour"] = dt.dt.hour.astype(float)
        out["dayofweek"] = dt.dt.dayofweek.astype(float)
        out["month"] = dt.dt.month.astype(float)
        out["dayofyear"] = dt.dt.dayofyear.astype(float)

    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)

    return out


def _build_features(df_work: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build a local feature table for AQUISTIL without external dependencies.
    The method stays standalone and uses only the current station dataframe.
    """
    df_work = _prepare_time_features(df_work)
    X = df_work.copy()

    if target in X.columns:
        y = pd.to_numeric(X[target], errors="coerce")
        X[f"{target}_lag_1"] = y.shift(1)
        X[f"{target}_lag_6"] = y.shift(6)
        X[f"{target}_lag_24"] = y.shift(24)
        X[f"{target}_lag_72"] = y.shift(72)
        X[f"{target}_roll_mean_24"] = y.shift(1).rolling(24, min_periods=6).mean()
        X[f"{target}_roll_std_24"] = y.shift(1).rolling(24, min_periods=6).std()
        X[f"{target}_roll_mean_72"] = y.shift(1).rolling(72, min_periods=12).mean()

        is_missing = y.isna().astype(int)
        run_id = (is_missing == 0).cumsum()
        X[f"{target}_gap_length"] = is_missing.groupby(run_id).cumcount()
        X[f"{target}_gap_is_long"] = (X[f"{target}_gap_length"] >= 24).astype(int)
        X[f"{target}_gap_is_very_long"] = (X[f"{target}_gap_length"] >= 72).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return X


def _coalesce_duplicate_columns(df: pd.DataFrame, logger=None, justif=120) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    collapsed = {}
    duplicate_names = []
    for column_name in pd.Index(df.columns).unique():
        selected = df.loc[:, df.columns == column_name]
        if isinstance(selected, pd.Series) or selected.shape[1] == 1:
            collapsed[column_name] = selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected
            continue
        duplicate_names.append(str(column_name))
        collapsed[column_name] = selected.bfill(axis=1).iloc[:, 0]

    if logger is not None:
        logger.warning(
            "AQUISTIL coalesced duplicate columns: {cols}".format(
                cols=", ".join(duplicate_names[:20]),
            ).ljust(justif - 2, ".") + "DEDUP"
        )
    return pd.DataFrame(collapsed, index=df.index)


def _target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    selected = df[target_column]
    if isinstance(selected, pd.Series):
        return pd.to_numeric(selected, errors="coerce")
    return pd.to_numeric(selected.bfill(axis=1).iloc[:, 0], errors="coerce")


class AQUISTILImputation(ImputationBase):
    def _parallel_workers(self, n_stations):
        configured = getattr(self.Configuration, "aquistil_parallel_workers", None)
        if configured is not None:
            try:
                workers = int(configured)
            except (TypeError, ValueError):
                workers = 1
        else:
            workers = max(1, min(n_stations, (os.cpu_count() or 1) // 2 or 1))
        return max(1, min(n_stations, workers))

    def _impute_station(self, station, station_df, target_column):
        return station, self.impute_frame(station_df, target_column=target_column)

    def _fallback_fill(self, df, target_column, reason):
        self.logger.warning(
            "AQUISTIL fallback fill for {target}: {reason}".format(
                target=target_column,
                reason=reason,
            ).ljust(self.justif - 2, ".") + "FALLBACK"
        )
        out = df.copy()
        series = pd.to_numeric(out[target_column], errors="coerce")
        filled = series.interpolate(method="time", limit_direction="both")
        filled = filled.ffill().bfill()
        if filled.isna().all():
            filled = filled.fillna(0.0)
        out[target_column] = filled
        return out

    def _select_target_columns(self, input_data_pd, target_column):
        prefixes = tuple("{prefix}_".format(prefix=prefix) for prefix in _target_prefixes(target_column))
        selected_columns = [col for col in input_data_pd.columns if str(col).startswith(prefixes)]
        if not selected_columns:
            # Fallback: maybe the input dataframe already contains a single column
            # for the target (e.g., 'O3' or 'OZONE') rather than wide-prefixed columns.
            resolved = _resolve_target_column(input_data_pd.columns, target_column)
            if resolved is not None:
                self.logger.info(
                    "AQUISTIL falling back to single-column target {t}".format(t=resolved).ljust(
                        self.justif - 2, "."
                    ) + "OK"
                )
                return input_data_pd.loc[:, [resolved]].copy()

            raise ValueError(
                "AQUISTIL could not find any wide input columns for target {target}. "
                "Available columns sample: {cols}".format(
                    target=target_column,
                    cols=", ".join(map(str, list(input_data_pd.columns)[:20])),
                )
            )
        self.logger.info(
            "AQUISTIL target-wide columns selected: {n} for {target}".format(
                n=len(selected_columns),
                target=target_column,
            ).ljust(self.justif - 2, ".") + "OK"
        )
        return input_data_pd.loc[:, selected_columns].copy()

    def impute_frame(self, input_data_pd, target_column=None):
        max_iter = getattr(self.Configuration, "aquistil_max_iter", 10)
        random_state = getattr(self.Configuration, "aquistil_random_state", 42)
        clamp = float(getattr(self.Configuration, "aquistil_clamp", 0.0))
        tol = float(getattr(self.Configuration, "aquistil_tol", 1e-4))

        if target_column is None:
            target_column = getattr(self.Configuration, "var_to_predict", [None])[0]
        if target_column is None:
            raise ValueError("AQUISTIL requires a target_column or Configuration.var_to_predict.")

        self.logger.info("".ljust(self.justif, "-"))
        self.logger.info("AQUISTIL Configuration:".ljust(self.justif, "|"))
        self.logger.info("  max_iter = {m}".format(m=max_iter).ljust(self.justif, "|"))
        self.logger.info("  random_state = {r}".format(r=random_state).ljust(self.justif, "|"))
        self.logger.info("  clamp = {c}".format(c=clamp).ljust(self.justif, "|"))
        self.logger.info("".ljust(self.justif, "-"))

        df = _coalesce_duplicate_columns(input_data_pd.copy(), logger=self.logger, justif=self.justif)
        resolved_target_column = _resolve_target_column(df.columns, target_column)
        if resolved_target_column is None:
            raise ValueError(
                "Target column {target} not found for AQUISTIL imputation. "
                "Available columns: {cols}".format(
                    target=target_column,
                    cols=", ".join(df.columns.astype(str).tolist()),
                )
            )
        if resolved_target_column != target_column:
            self.logger.info(
                "Resolved AQUISTIL target {src} -> {dst}".format(
                    src=target_column,
                    dst=resolved_target_column,
                ).ljust(self.justif - 2, ".") + "OK"
            )
        target_column = resolved_target_column

        y0 = _target_series(df, target_column)
        missing_mask = y0.isna().values
        if int(missing_mask.sum()) == 0:
            self.logger.info(
                "No missing values found for AQUISTIL".ljust(self.justif - 2, ".") + "SKIP"
            )
            return df

        observed_count = int((~missing_mask).sum())
        if observed_count == 0:
            return self._fallback_fill(
                df,
                target_column,
                "no observed target values available",
            )

        init_val = float(np.nanmedian(y0.values))
        if not np.isfinite(init_val):
            init_val = 0.0
        df.loc[missing_mask, target_column] = init_val
        prev = df.loc[missing_mask, target_column].values.copy()

        for it in range(1, max_iter + 1):
            X = _build_features(df, target_column)
            y = _target_series(df, target_column)

            obs_mask = ~missing_mask
            train_df = X.loc[obs_mask].copy()
            pred_df = X.loc[missing_mask].copy()
            y_train = y.loc[obs_mask]

            if target_column in train_df.columns:
                train_df = train_df.drop(columns=[target_column])
            if target_column in pred_df.columns:
                pred_df = pred_df.drop(columns=[target_column])

            train_df = train_df.select_dtypes(include=[np.number]).copy()
            if train_df.empty or train_df.shape[1] == 0 or y_train.empty:
                return self._fallback_fill(
                    df,
                    target_column,
                    "no usable training features for LightGBM",
                )
            pred_df = pred_df[train_df.columns].copy()
            if pred_df.empty:
                self.logger.info(
                    "AQUISTIL has no prediction rows after feature build".ljust(self.justif - 2, ".") + "SKIP"
                )
                return df

            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=30,
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
            )

            model.fit(train_df, y_train)
            y_new = model.predict(pred_df)

            if clamp > 0:
                y_new = np.clip(y_new, a_min=-clamp, a_max=clamp)

            df.loc[missing_mask, target_column] = y_new
            diff = float(np.mean(np.abs(y_new - prev)))
            self.logger.info(
                f"AQUISTIL iter {it}/{max_iter} mean_change={diff:.6f}".ljust(
                    self.justif - 2, "."
                ) + "RUN"
            )
            if diff < tol:
                self.logger.info(
                    f"AQUISTIL converged at iter {it}".ljust(self.justif - 2, ".") + "OK"
                )
                break
            prev = y_new.copy()

        return df

    def impute(self, input_data_pd, save_data=False):
        var_to_predict = getattr(self.Configuration, "var_to_predict", None)
        target_column = var_to_predict[0] if var_to_predict else None
        if target_column is None:
            raise ValueError("AQUISTIL requires Configuration.var_to_predict.")

        input_data_pd = self._select_target_columns(input_data_pd, target_column)
        station_dict = self.extract_station_data(input_data_pd, var_to_predict=var_to_predict)
        station_imputed_dict = {}
        station_items = list(station_dict.items())
        workers = self._parallel_workers(len(station_items))
        self.logger.info(
            "AQUISTIL station parallelism: {workers} worker(s)".format(workers=workers).ljust(
                self.justif - 2, "."
            ) + "SET"
        )

        if workers == 1:
            results = [
                self._impute_station(station, station_df, target_column)
                for station, station_df in station_items
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="aquistil") as executor:
                futures = [
                    executor.submit(self._impute_station, station, station_df, target_column)
                    for station, station_df in station_items
                ]
                results = [future.result() for future in futures]

        for station, imputed_station_df in results:
            station_imputed_dict[station] = imputed_station_df
            self.logger.info(
                "AQUISTIL Imputation for {s}".format(s=station).ljust(self.justif - 2, ".") + "OK"
            )
            if save_data:
                self.save_imputed_data(imputed_station_df, station_name=station)

        imputed_data_pd = self.combine_station_data(station_imputed_dict)
        self.logger.info(
            "Combined {n} stations".format(n=len(station_imputed_dict)).ljust(self.justif - 2, ".")
            + "OK"
        )
        self.logger.info("Imputed using AQUISTIL method".ljust(self.justif - 2, ".") + "OK")

        if save_data:
            self.save_imputed_data(imputed_data_pd, station_name=None)

        return imputed_data_pd, station_imputed_dict
