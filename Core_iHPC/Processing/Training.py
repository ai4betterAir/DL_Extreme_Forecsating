"""
Training module for Sparse_LSTM_v1
Parallel to Forecast.py but for model training.
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Core_iHPC.Models.model import create_sparse_lstm_model
from Core_iHPC.Configuration.runtime_config import build_sparse_lstm_runtime

class Training_Class(object):

    def __init__(self, Configuration):
        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif, '-'))
        self.logger.info('Configuring Training'.center(self.justif, '|'))
        self.logger.info(''.ljust(self.justif, '-'))

        runtime = build_sparse_lstm_runtime(self.Configuration, "sparse_lstm_pm25_3")
        self.__dict__.update(runtime.__dict__)

        os.makedirs(self.model_dir,  exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)

        self.logger.info(f"Model output dir: {self.model_dir}".ljust(self.justif, '|'))
        self.logger.info(f"Scaler output dir: {self.scaler_dir}".ljust(self.justif, '|'))

###########################################################################################
    def _prepare_sequences(self, data, time_steps, lags):
        """Sliding window sequence creation"""
        X, y = [], []
        for i in range(len(data) - time_steps - lags):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps + lags - 1, :])  # predict lags ahead
        return np.array(X), np.array(y)

###########################################################################################
    def _train_single_imf(self, imf_name, station_decomposed_dict):
        """
        Train one model for one IMF across all stations.
        Mirrors train_single_imf() from original training.py.
        """
        station_names = list(station_decomposed_dict.keys())
        num_sites = len(station_names)

        # Stack IMF values across stations ? shape (n_timesteps, num_sites)
        imf_arrays = [station_decomposed_dict[s][imf_name].values for s in station_names]
        imf_data = np.column_stack(imf_arrays)

        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        imf_data_scaled = scaler.fit_transform(imf_data)

        # Sequences
        X, y = self._prepare_sequences(imf_data_scaled, self.time_steps, self.lags)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        self.logger.info(f"  {imf_name}: X_train={X_train.shape}, y_train={y_train.shape}".ljust(self.justif, '|'))

        # Build model
        n_epochs    = getattr(self.Configuration, 'n_epochs', 200)
        batch_size  = getattr(self.Configuration, 'batch_size', 32)

        model = create_sparse_lstm_model(
            input_shape=(self.time_steps, num_sites),
            num_sites=num_sites
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-6, verbose=0),
        ]

        model.fit(X_train, y_train,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  validation_split=0.2,
                  callbacks=callbacks,
                  verbose=1)

        # Save model � filename matches what Sparse_LSTM_v1.py expects
        model_path  = os.path.join(self.model_dir,  f"{imf_name}_model.keras")
        scaler_path = os.path.join(self.scaler_dir, f"{imf_name}_scaler.pkl")

        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        self.logger.info(f"  {imf_name} model saved".ljust(self.justif - 2, '.') + 'OK')

        # Quick test metrics
        test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
        test_actual = scaler.inverse_transform(y_test)

        return test_pred, test_actual

###########################################################################################
    def run_training(self, station_decomposed_dict, n_retrain):
        """
        Main entry point � mirrors Forecast.run_forecast() signature.
        Trains 21 IMF models sequentially and saves them.
        """
        import time
        start_time = time.time()

        self.logger.info(''.ljust(self.justif, '='))
        self.logger.info('Sparse_LSTM Training Workflow'.center(self.justif, '|'))
        self.logger.info(''.ljust(self.justif, '='))

        station_names = list(station_decomposed_dict.keys())
        self.logger.info(f"Training on {len(station_names)} stations, {len(self.imf_columns)} IMFs".ljust(self.justif, '|'))

        results = {}

        for idx, imf_name in enumerate(self.imf_columns, 1):
            self.logger.info(f"Training {imf_name} ({idx}/{len(self.imf_columns)})".ljust(self.justif - 2, '.') + 'RUN')
            try:
                test_pred, test_actual = self._train_single_imf(imf_name, station_decomposed_dict)
                results[imf_name] = {'success': True, 'test_pred': test_pred, 'test_actual': test_actual}
                self.logger.info(f"  {imf_name}".ljust(self.justif - 2, '.') + 'OK')
            except Exception as e:
                self.logger.error(f"  {imf_name}: {str(e)}".ljust(self.justif - 2, '.') + 'FAIL')
                results[imf_name] = {'success': False, 'error': str(e)}

        n_success = sum(1 for r in results.values() if r['success'])
        elapsed = time.time() - start_time

        self.logger.info(f"Trained {n_success}/{len(self.imf_columns)} IMFs".ljust(self.justif - 2, '.') + 'OK')
        self.logger.info(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)".ljust(self.justif, '|'))

        # Return None for list_output_pd during training (no forecast produced)
        return None, results
