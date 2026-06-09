from ._shared import ImputationBase


class NoneImputation(ImputationBase):
    def impute(self, input_data_pd, save_data=False):
        station_imputed_dict = {}
        if save_data:
            self.save_imputed_data(input_data_pd, station_name=None)
        return input_data_pd, station_imputed_dict
