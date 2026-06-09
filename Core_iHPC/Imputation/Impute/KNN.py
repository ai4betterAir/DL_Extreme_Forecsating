import numpy as np
import pandas as pd
import sklearn.impute as SKI
import sklearn.preprocessing as SKP

from ._shared import ImputationBase


class KNNImputation(ImputationBase):
    def impute_frame(self, input_data_pd):
        scaler = SKP.MinMaxScaler(feature_range=(0, 1))
        df_knn = pd.DataFrame(
            scaler.fit_transform(input_data_pd),
            columns=input_data_pd.columns,
            index=input_data_pd.index,
        )

        knn_imputer = SKI.KNNImputer(
            missing_values=np.nan,
            n_neighbors=5,
            weights='uniform',
            metric='nan_euclidean',
        )
        imputed_data_pd = pd.DataFrame(
            knn_imputer.fit_transform(df_knn),
            columns=df_knn.columns,
            index=df_knn.index,
        )
        return imputed_data_pd

    def impute(self, input_data_pd, save_data=False):
        var_to_predict = getattr(self.Configuration, 'var_to_predict', None)
        station_dict = self.extract_station_data(input_data_pd, var_to_predict=var_to_predict)
        station_imputed_dict = {}

        for station, station_df in station_dict.items():
            self.logger.info(''.ljust(self.justif, '-'))
            self.logger.info("Imputing station: {s}".format(s=station).center(self.justif, '|'))
            self.logger.info(''.ljust(self.justif, '-'))
            imputed_station_df = self.impute_frame(station_df)
            station_imputed_dict[station] = imputed_station_df
            self.logger.info("KNN Imputation for {s}".format(s=station).ljust(self.justif - 2, '.') + 'OK')
            if save_data:
                self.save_imputed_data(imputed_station_df, station_name=station)

        imputed_data_pd = self.combine_station_data(station_imputed_dict)
        self.logger.info("Combined {n} stations".format(
            n=len(station_imputed_dict)).ljust(self.justif - 2, '.') + 'OK')

        if save_data:
            self.save_imputed_data(imputed_data_pd, station_name=None)

        return imputed_data_pd, station_imputed_dict
