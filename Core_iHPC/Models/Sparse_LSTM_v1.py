"""
..  module:: Sparse_LSTM_v1
    :platform: Unix
    :synopsis: Sparse Transformer-LSTM model for VMD-ensemble PM2.5 forecasting.
               Self-contained corporate-standard implementation: architecture,
               training, and inference all reside in this single file.
               model.py and the separate Training.py are no longer required.

    Architecture per IMF
    --------------------
    Input (batch, time_steps, num_sites)
        Conv1D (filters=64, kernel=3, padding=same, relu)
        Conv1D (filters=64, kernel=3, padding=same, relu)
        SparseTransformerBlock (local self-attention + FFN + residual + LayerNorm)
        LSTM   (units=100, return_sequences=False)
        Dense  (units=num_sites)
        AdaptiveLossLayer  [pass-through; holds trainable log_vars]
    Output (batch, num_sites)

    VMD Ensemble Workflow
    ---------------------
    21 independent models are trained, one per IMF (IMF_1 … IMF_20 + Residual).
    At inference time each model predicts its IMF contribution; the 21 outputs
    are summed to reconstruct the final PM2.5 signal.

    Corporate Standard Alignment (cf. CNN_LSTM_v1.py)
    --------------------------------------------------
    * Architecture defined inside this class (build_model flag: True=build, False=load)
    * fit()         – training loop with callbacks
    * configure_model() – switches between build/load mode, mirrors CNN_LSTM_v1
    * run_all()     – single entry point for both training and inference
    * Normalisation stored as a companion .pkl file per IMF (MinMaxScaler)
    * Metadata JSON written after training for version validation

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
.. moduleauthor:: Sagthitharan Karalasingham <d9630120@umail.usq.edu.au>
"""

import os
import sys
import stat
import json
import pickle
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as TFK
import tensorflow.keras.layers as TKL
import tensorflow.keras.models as TKM
import tensorflow.keras.callbacks as TKC
import tensorflow.keras.losses as TKLoss
import tensorflow.keras.optimizers as TKO
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

from Core_iHPC.Models.layers.sparse_transformer_block import SparseTransformerBlock
from Core_iHPC.Models.layers.adaptive_loss_layer import AdaptiveLossLayer
from Core_iHPC.Models.layers.sparse_attention import SparseAttention
from Core_iHPC.Configuration.runtime_config import build_sparse_lstm_runtime
from Core_iHPC.Processing import Data_preparation

# Reduce TensorFlow / Abseil log noise in training runs.
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# Fix random seed for reproducibility – mirrors CNN_LSTM_v1.py
tf.random.set_seed(1)


###########################################################################################
# Module-level adaptive loss factory
# ---------------------------------------------------------------------------
# Defined here at module level (not as a closure inside build_architecture)
# so that Keras can serialise/deserialise it when saving/loading .keras files.
# A default-argument binding to the AdaptiveLossLayer instance avoids any
# closure over a local variable, which cannot survive serialisation.
###########################################################################################

def make_adaptive_loss(adaptive_layer):
    """
    Return a Keras-compatible loss function bound to a specific
    AdaptiveLossLayer instance via a stable default argument.

    Parameters
    ----------
    adaptive_layer : AdaptiveLossLayer
        The last layer of the model that holds the trainable log_vars weights.

    Returns
    -------
    callable  (y_true, y_pred) → scalar loss
    """
    def adaptive_loss(y_true, y_pred, _layer=adaptive_layer):
        log_vars     = _layer.log_vars                           # shape (num_sites,)
        precision    = tf.exp(-log_vars)                         # 1 / variance
        mse          = tf.reduce_mean(
                           tf.square(y_true - y_pred), axis=0)  # shape (num_sites,)
        weighted     = tf.reduce_sum(precision * mse + log_vars)
        return weighted

    adaptive_loss.__name__ = 'adaptive_loss'   # stable name for Keras config
    return adaptive_loss


def _normalise_keras3_layer_config(node):
    """
    Patch legacy Keras 2 layer configs so they can be deserialised by Keras 3.

    Older saved .keras files may store ``batch_input_shape`` on layers such as
    Conv1D. Keras 3 only accepts that field on InputLayer, where it is now
    called ``batch_shape``.
    """
    if isinstance(node, dict):
        class_name = node.get('class_name')
        config = node.get('config')

        if isinstance(config, dict) and 'batch_input_shape' in config:
            batch_input_shape = config.pop('batch_input_shape')
            if class_name == 'InputLayer':
                config.setdefault('batch_shape', batch_input_shape)

        if isinstance(config, dict):
            config.pop('time_major', None)

        for value in node.values():
            _normalise_keras3_layer_config(value)

    elif isinstance(node, list):
        for value in node:
            _normalise_keras3_layer_config(value)


def _patched_legacy_keras_archive(model_path):
    """
    Return a temporary .keras archive with only config.json compatibility fixes.

    The original model file is not modified.
    """
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
        patched_path = tmp_file.name

    with zipfile.ZipFile(model_path, 'r') as src, zipfile.ZipFile(patched_path, 'w') as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == 'config.json':
                config = json.loads(data.decode('utf-8'))
                _normalise_keras3_layer_config(config)
                data = json.dumps(config).encode('utf-8')
            dst.writestr(info, data)

    return patched_path


###########################################################################################
class Sparse_LSTM_Class(object):
    MODEL_DISPLAY_NAME = "Sparse_LSTM_v1"
    MODEL_TAG = "v1"
    DEFAULT_MODEL_NAME = "sparse_lstm_v1"
    """
    Self-contained Sparse Transformer-LSTM class.

    Mirrors CNN_LSTM_v1.py corporate standard:
    - Architecture is defined and owned by this class (build_architecture flag).
    - configure_model() selects build-new vs load-from-disk mode.
    - fit() runs the training loop.
    - run_all() is the single entry point for both training and inference,
      called identically by Training.py / Forecast.py (or directly by main.py).

    Attributes
    ----------
    Configuration : Configuration_Class
        Shared configuration object carrying all run parameters.
    logger : logging.Logger
    justif : int
    models : dict  {imf_name: keras.Model}
    scalers : dict {imf_name: MinMaxScaler}
    model_save_dir : str
        Root directory for saved model weights (= Configuration.Main_model_data_full_dir).
    """

    def __init__(self, Configuration):
        self.Configuration = Configuration
        self.logger        = self.Configuration.logger
        self.justif        = self.Configuration.justif

        self.logger.info(''.ljust(self.justif, '-'))
        self.logger.info(f'Configuring {self.MODEL_DISPLAY_NAME}'.center(self.justif, '|'))
        self.logger.info(''.ljust(self.justif, '-'))

        runtime = build_sparse_lstm_runtime(self.Configuration, self.DEFAULT_MODEL_NAME)
        self.__dict__.update(runtime.__dict__)

        self.models  = {}    # {imf_name: keras.Model}
        self.scalers = {}    # {imf_name: MinMaxScaler}

        self.logger.info(f'{self.MODEL_DISPLAY_NAME} configuration'.ljust(self.justif - 2, '.') + 'OK')
        self.logger.info(f'  time_steps  : {self.time_steps}'.ljust(self.justif, '|'))
        self.logger.info(f'  lags        : {self.lags}'.ljust(self.justif, '|'))
        self.logger.info(f'  n_steps_in  : {self.n_steps_in}'.ljust(self.justif, '|'))
        self.logger.info(f'  n_steps_out : {self.n_steps_out}'.ljust(self.justif, '|'))
        self.logger.info(f'  vmd_n_imfs  : {self.vmd_n_imfs} → {len(self.imf_columns)} models'.ljust(self.justif, '|'))
        self.logger.info(f'  window_size : {self.window_size}'.ljust(self.justif, '|'))
        self.logger.info(f'  d_model     : {self.d_model}'.ljust(self.justif, '|'))
        self.logger.info(f'  model_dir   : {self.model_dir}'.ljust(self.justif, '|'))
        return

###########################################################################################
    def MakeDir(self, ddir):
        """Create a directory with group-writable permissions (corporate standard)."""
        if not os.path.exists(ddir):
            os.makedirs(ddir)
            mod775 = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                      stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                      stat.S_IROTH | stat.S_IXOTH)
            os.chmod(ddir, mod775)
            self.logger.info(
                f'Directory = {ddir}'.ljust(self.justif - 7, '.') + 'CREATED')
        return

###########################################################################################
    def build_architecture(self, num_sites, build_architecture):
        """
        Define or load the Sparse Transformer-LSTM architecture for one IMF.

        Mirrors CNN_LSTM_v1.cnn_lstm_model_v1() in role and structure:
        - build_architecture=True  → construct new model (training mode)
        - build_architecture=False → model will be loaded from disk by
                                     configure_model(); nothing built here.

        The method stores the result in self.model (singular) because it is
        called once per IMF inside the IMF training/inference loop.

        Parameters
        ----------
        num_sites : int
            Number of monitoring stations = input feature count = output size.
        build_architecture : bool
            True to build a new model; False when loading from disk.
        """
        self.model_save_filename_template = (
            "{imf_name}_model.keras"
        )

        if build_architecture:

            # ── Guard: window_size must not exceed sequence length ──────────
            if self.window_size > self.time_steps:
                raise ValueError(
                    f"window_size ({self.window_size}) > time_steps "
                    f"({self.time_steps}). Reduce sparse_lstm_window_size "
                    f"in the YAML or increase n_inputs/time_steps."
                )

            # ── Guard: d_model must equal cnn_filters for residual to work ──
            if self.d_model != self.cnn_filters:
                raise ValueError(
                    f"d_model ({self.d_model}) must equal cnn_filters "
                    f"({self.cnn_filters}) so the residual connection inside "
                    f"SparseTransformerBlock has matching tensor shapes."
                )

            # ── Build Sequential model ───────────────────────────────────────
            self.model = TKM.Sequential()
            self.model.add(TKL.Input(shape=(self.time_steps, num_sites)))

            # CNN feature extraction – both layers use padding='same' to
            # preserve sequence length (time_steps) through the CNN stack.
            self.model.add(TKL.Conv1D(
                filters     = self.cnn_filters,
                kernel_size = self.cnn_kernel_size,
                padding     = 'same',
                strides     = 1,
                activation  = 'relu',
            ))
            self.model.add(TKL.Conv1D(
                filters     = self.cnn_filters,
                kernel_size = self.cnn_kernel_size,
                padding     = 'same',   # preserves time_steps (was missing in original)
                activation  = 'relu',
            ))
            # Note: MaxPooling1D(pool_size=1) removed – it was a no-op.

            # Sparse Transformer block – local self-attention with residual.
            # All hyperparameters come from Configuration / YAML.
            self.model.add(SparseTransformerBlock(
                num_heads   = self.num_heads,
                window_size = self.window_size,
                ff_dim      = self.ff_dim,
                d_model     = self.d_model,
                dropout     = self.dropout_rate,
            ))

            # Temporal summarisation
            self.model.add(TKL.LSTM(units=self.lstm_units, return_sequences=False))

            # Per-station output – one scalar per station per prediction step.
            # Output shape: (batch, num_sites).  This matches the target shape
            # produced by _prepare_sequences().
            self.model.add(TKL.Dense(units=num_sites))

            # Adaptive uncertainty layer – pass-through that holds trainable
            # log_vars (shape: num_sites,).  num_outputs=num_sites so that
            # log_vars aligns with the Dense output and the training targets.
            adaptive_layer = AdaptiveLossLayer(num_outputs=num_sites)
            self.model.add(adaptive_layer)

            # Compile with module-level adaptive loss (serialisation-safe).
            loss_fn = make_adaptive_loss(adaptive_layer)
            self.model.compile(optimizer=TKO.Adam(), loss=loss_fn)

        self.logger.info(
            f'Sparse LSTM model {self.MODEL_TAG}'.ljust(self.justif - 10, '.') + 'CONFIGURED')
        return

###########################################################################################
    def configure_model(self, imf_name, num_sites, train_model):
        """
        Select build-new vs load-from-disk mode.  Mirrors CNN_LSTM_v1 pattern.

        Parameters
        ----------
        imf_name : str
            IMF identifier, e.g. 'IMF_1' or 'Residual'.
        num_sites : int
            Number of monitoring stations.
        train_model : bool
            True  → call build_architecture(build=True) to create a new model.
            False → load saved .keras weights from disk into self.model.
        """
        if train_model:
            self.build_architecture(num_sites, build_architecture=True)
        else:
            self.build_architecture(num_sites, build_architecture=False)
            model_path = os.path.join(self.model_dir, f"{imf_name}_model.keras")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Run in training mode first."
                )

            custom_objects = {
                'SparseAttention':        SparseAttention,
                'SparseTransformerBlock': SparseTransformerBlock,
                'AdaptiveLossLayer':      AdaptiveLossLayer,
                'adaptive_loss':          make_adaptive_loss,   # factory, not closure
            }

            try:
                self.model = keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
            except (TypeError, ValueError) as exc:
                if 'batch_input_shape' not in str(exc):
                    raise

                patched_model_path = _patched_legacy_keras_archive(model_path)
                try:
                    self.model = keras.models.load_model(
                        patched_model_path,
                        custom_objects=custom_objects,
                        compile=False,
                        safe_mode=False
                    )
                finally:
                    try:
                        os.remove(patched_model_path)
                    except OSError:
                        pass

            self.logger.info(
                f'{imf_name} model loaded'.ljust(self.justif - 10, '.') + 'OK')

        return

###########################################################################################
    def fit(self, X_train, y_train, n_epochs, save_model, imf_name):
        """
        Train self.model on one IMF's data.  Mirrors CNN_LSTM_v1.fit().

        Parameters
        ----------
        X_train : np.ndarray  shape (n_samples, time_steps, num_sites)
        y_train : np.ndarray  shape (n_samples, num_sites)
        n_epochs : int
        save_model : bool
            If True, save weights and scaler to disk after training.
        imf_name : str
            Used to name the saved files.

        Returns
        -------
        keras.callbacks.History
        """
        callbacks = [
            TKC.EarlyStopping(
                monitor             = 'val_loss',
                mode                = 'min',
                patience            = 20,
                restore_best_weights= True,
                verbose             = 1,
            ),
            TKC.ReduceLROnPlateau(
                monitor  = 'val_loss',
                factor   = 0.5,
                patience = 10,
                min_lr   = 1e-6,
                verbose  = 0,
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs           = n_epochs,
            batch_size       = self.batch_size,
            validation_split = 0.2,
            callbacks        = callbacks,
            verbose          = 1,
        )

        if save_model:
            self._save_model_and_scaler(imf_name)

        return history

###########################################################################################
    def _save_model_and_scaler(self, imf_name):
        """
        Persist model weights (.keras) and scaler (.pkl) for one IMF.

        Called by fit() when save_model=True.
        """
        os.makedirs(self.model_dir,  exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)

        model_path  = os.path.join(self.model_dir,  f"{imf_name}_model.keras")
        scaler_path = os.path.join(self.scaler_dir, f"{imf_name}_scaler.pkl")

        self.model.save(model_path)
        self.logger.info(
            f'  {imf_name} model saved'.ljust(self.justif - 2, '.') + 'OK')

        with open(scaler_path, 'wb') as fh:
            pickle.dump(self.scalers[imf_name], fh)
        self.logger.info(
            f'  {imf_name} scaler saved'.ljust(self.justif - 2, '.') + 'OK')

        return

###########################################################################################
    def save_metadata(self, station_names=None):
        """
        Write model_config.json after all 21 IMF models are trained.
        Enables version validation at inference time.
        """
        os.makedirs(self.metadata_dir, exist_ok=True)
        metadata = {
            'version':          self.model_version,
            'model_name':       self.model_name,
            'time_steps':       self.time_steps,
            'lags':             self.lags,
            'n_steps_in':       self.n_steps_in,
            'n_steps_out':      self.n_steps_out,
            'vmd_n_imfs':       self.vmd_n_imfs,
            'imf_columns':      self.imf_columns,
            'num_heads':        self.num_heads,
            'window_size':      self.window_size,
            'ff_dim':           self.ff_dim,
            'd_model':          self.d_model,
            'lstm_units':       self.lstm_units,
            'cnn_filters':      self.cnn_filters,
            'station_names':    list(station_names) if station_names is not None else None,
            'created_at':       datetime.now().isoformat(),
        }
        metadata_path = os.path.join(self.metadata_dir, 'model_config.json')
        with open(metadata_path, 'w') as fh:
            json.dump(metadata, fh, indent=2)
        self.logger.info(
            f'Metadata saved: {metadata_path}'.ljust(self.justif - 2, '.') + 'OK')
        return

###########################################################################################
    def _load_version_metadata(self):
        """
        Load and validate model_config.json.  Warns or raises on version mismatch
        depending on self.strict_version_check.
        """
        metadata_path = os.path.join(self.metadata_dir, 'model_config.json')

        if not os.path.exists(metadata_path):
            self.logger.warning(
                f'Metadata not found: {metadata_path}'.ljust(self.justif - 2, '.') + 'SKIP')
            return None

        try:
            with open(metadata_path, 'r') as fh:
                metadata = json.load(fh)

            file_version = metadata.get('version', 'unknown')
            if file_version != self.model_version:
                msg = (f"Version mismatch: expected {self.model_version}, "
                       f"found {file_version}")
                if self.strict_version_check:
                    self.logger.error(msg.ljust(self.justif - 2, '.') + 'FAIL')
                    raise ValueError(msg)
                else:
                    self.logger.warning(msg.ljust(self.justif - 2, '.') + 'WARN')
            else:
                self.logger.info(
                    f'Model version validated: {file_version}'.ljust(self.justif - 2, '.') + 'OK')

            return metadata

        except Exception as exc:
            self.logger.error(
                f'Failed to load metadata: {exc}'.ljust(self.justif - 2, '.') + 'FAIL')
            if self.strict_version_check:
                raise
            return None

###########################################################################################
    def _prepare_sequences(self, data, time_steps, lags):
        """
        Sliding-window sequence creation for TRAINING (produces X and y).

        Identical contract to the original Training.py._prepare_sequences()
        so existing saved scalers remain compatible.

        Parameters
        ----------
        data : np.ndarray  shape (n_timesteps, num_sites)  – already scaled
        time_steps : int   lookback window length
        lags : int         prediction horizon (steps ahead)

        Returns
        -------
        X : np.ndarray  shape (n_samples, time_steps, num_sites)
        y : np.ndarray  shape (n_samples, num_sites)
        """
        X, y = [], []
        for i in range(len(data) - time_steps - lags):
            X.append(data[i : i + time_steps])
            y.append(data[i + time_steps + lags - 1, :])   # one step, all stations
        return np.array(X), np.array(y)

###########################################################################################
    def _create_inference_sequences(self, data, time_steps, lags):
        """
        Sliding-window sequence creation for INFERENCE (X only, no y).

        Parameters
        ----------
        data : np.ndarray  shape (n_timesteps, num_sites)  – already scaled

        Returns
        -------
        X : np.ndarray  shape (n_samples, time_steps, num_sites)
        """
        min_length = time_steps + lags + 1
        if len(data) < min_length:
            raise ValueError(
                f"Insufficient data for sequences: need {min_length}, "
                f"have {len(data)}"
            )
        X = [data[i : i + time_steps]
             for i in range(len(data) - time_steps - lags)]
        return np.array(X)

###########################################################################################
    def _prepare_imf_data(self, station_decomposed_dict, imf_name):
        """
        Extract one IMF column from all stations and stack into a matrix.

        Parameters
        ----------
        station_decomposed_dict : dict
            {station_name: DataFrame with columns IMF_1…IMF_N, Residual, Original}
        imf_name : str

        Returns
        -------
        np.ndarray  shape (n_timesteps, n_stations)
        """
        station_names = list(station_decomposed_dict.keys())
        raw_signal_list = getattr(self.Configuration, 'var_data_list_from_input_pd', None) or [None]
        if isinstance(raw_signal_list, (list, tuple)) and len(raw_signal_list) > 0:
            raw_signal_name = raw_signal_list[0]
        else:
            raw_signal_name = None

        for station in station_names:
            if imf_name not in station_decomposed_dict[station].columns:
                fallback_column = None
                station_columns = station_decomposed_dict[station].columns
                if "Original" in station_columns:
                    fallback_column = "Original"
                elif raw_signal_name in station_columns:
                    fallback_column = raw_signal_name
                elif len(station_columns) >= 1:
                    fallback_column = station_columns[0]

                if fallback_column is None:
                    raise ValueError(
                        f"Column '{imf_name}' not found in station '{station}' and no fallback series is available"
                    )

                self.logger.warning(
                    f"  {station}: using fallback column '{fallback_column}' for missing '{imf_name}'".ljust(
                        self.justif - 2, '.') + 'WARN'
                )

        combined = np.column_stack([
            station_decomposed_dict[s][imf_name].values
            if imf_name in station_decomposed_dict[s].columns
            else station_decomposed_dict[s][
                "Original"
                if "Original" in station_decomposed_dict[s].columns
                else raw_signal_name
                if raw_signal_name in station_decomposed_dict[s].columns
                else station_decomposed_dict[s].columns[0]
            ].values
            for s in station_names
        ])

        if np.any(np.isnan(combined)):
            nan_count = int(np.sum(np.isnan(combined)))
            self.logger.warning(
                f'  {imf_name}: {nan_count} NaN values'.ljust(self.justif - 2, '.') + 'WARN')

        return combined

###########################################################################################
    def _sum_imf_predictions(self, imf_predictions):
        """
        Sum per-IMF prediction arrays to reconstruct the original signal.

        Parameters
        ----------
        imf_predictions : dict  {imf_name: np.ndarray shape (n_samples, n_stations)}

        Returns
        -------
        np.ndarray  shape (n_samples, n_stations)
        """
        shapes = {k: v.shape for k, v in imf_predictions.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"Shape mismatch across IMF predictions: {shapes}")

        summed = np.sum(list(imf_predictions.values()), axis=0)
        self.logger.info(
            f'Summed {len(imf_predictions)} IMF predictions'.ljust(self.justif - 2, '.') + 'OK')
        return summed

###########################################################################################
    def _format_output(self, predictions, station_names,
                       start_timestamp=None, forecast_segment=1):
        """
        Format final predictions as per-station DataFrames.
        Corporate standard output – mirrors CNN_LSTM_v1 list_output_pd convention.

        Parameters
        ----------
        predictions : np.ndarray  shape (n_samples, n_stations)
        station_names : list[str]
        start_timestamp : pd.Timestamp or None
        forecast_segment : int

        Returns
        -------
        list[pd.DataFrame]  one DataFrame per station, indexed by timestamp
        """
        list_output_pd = []
        n_samples      = predictions.shape[0]

        if start_timestamp is None:
            timestamps = pd.date_range(
                start=datetime.now(), periods=n_samples, freq='h')
            self.logger.warning(
                'No start timestamp – using current time'.ljust(self.justif - 2, '.') + 'WARN')
        else:
            timestamps = pd.date_range(
                start=start_timestamp, periods=n_samples, freq='h')

        forecast_hours = list(range(1, n_samples + 1))

        for idx, station in enumerate(station_names):
            station_df = pd.DataFrame({
                'timestamp':       timestamps,
                'forecast_hours':  forecast_hours,
                'forecast_number': forecast_segment,
                'station':         station,
                f'{self.Configuration.var_to_predict[0]}_{station}': predictions[:, idx],
                'model_version':   self.model_version,
                'generated_at':    datetime.now(),
            }).set_index('timestamp')

            list_output_pd.append(station_df)
            self.logger.info(
                f'  {station}: {len(station_df)} steps'.ljust(self.justif, '|'))

        self.logger.info(
            f'Output formatted for {len(station_names)} stations'.ljust(self.justif - 2, '.') + 'OK')
        return list_output_pd

###########################################################################################
    def _compute_metrics(self, y_true, y_pred):
        """
        Compute R-squared and RMSE.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray

        Returns
        -------
        dict  {'R2': float, 'RMSE': float}
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {'R2': round(r2, 4), 'RMSE': round(rmse, 4)}

###########################################################################################
    def _compute_dashboard_metrics(self, y_true, y_pred):
        """
        Match the legacy dashboard metric definitions from
        Evaluation/Evaluation_singlemodel.py.
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]

        if len(y_true) == 0:
            return {
                'MAE': np.nan,
                'RMSE': np.nan,
                'MAPE': np.nan,
                'PEARSON_R': np.nan,
                'R_SQUARE': np.nan,
                'MEAN_FC': np.nan,
                'MEAN_OBS': np.nan,
                'STD_FC': np.nan,
                'STD_OBS': np.nan,
                'MBE': np.nan,
            }

        epsilon = 0.01
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mape = np.mean(np.abs(y_true - y_pred) / (y_true + epsilon)) * 100

        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r = np.corrcoef(y_pred, y_true)[0, 1]
        else:
            pearson_r = np.nan

        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_pred - y_true) ** 2)
        r_square = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {
            'MAE': np.round(mae, 3),
            'RMSE': np.round(rmse, 3),
            'MAPE': np.round(mape, 3),
            'PEARSON_R': np.round(pearson_r, 3) if np.isfinite(pearson_r) else np.nan,
            'R_SQUARE': np.round(r_square, 3) if np.isfinite(r_square) else np.nan,
            'MEAN_FC': np.round(np.mean(y_pred), 3),
            'MEAN_OBS': np.round(np.mean(y_true), 3),
            'STD_FC': np.round(np.std(y_pred), 3),
            'STD_OBS': np.round(np.std(y_true), 3),
            'MBE': np.round(np.max(y_pred - y_true), 3),
        }

###########################################################################################
    def _write_dashboard_metrics(self, station_names, station_actuals,
                                 station_preds, iteration=0, save_file=True):
        """
        Write the legacy dashboard metrics table consumed by forecast mode.

        Output format:
        Stations,MAE,RMSE,MAPE,PEARSON_R,R_SQUARE,MEAN_FC,MEAN_OBS,STD_FC,STD_OBS,MBE
        """
        metrics_columns = [
            'MAE',
            'RMSE',
            'MAPE',
            'PEARSON_R',
            'R_SQUARE',
            'MEAN_FC',
            'MEAN_OBS',
            'STD_FC',
            'STD_OBS',
            'MBE',
        ]

        rows = []
        for station in station_names:
            if station_actuals.get(station) is None or station_preds.get(station) is None:
                continue

            metrics = self._compute_dashboard_metrics(
                station_actuals[station],
                station_preds[station],
            )
            metrics['Stations'] = station
            rows.append(metrics)

        if not rows:
            self.logger.warning(
                'Dashboard average metrics'.ljust(self.justif - 2, '.') + 'SKIPPED')
            return None

        metrics_pd = pd.DataFrame(rows)
        metrics_pd = metrics_pd[['Stations'] + metrics_columns]

        if save_file:
            metrics_filename = self.Configuration.metrics_output_filename_template.format(
                region=self.Configuration.selected_region,
                metrics='average_metric',
                var=self.Configuration.input_var_dir,
                inputs=self.Configuration.n_steps_in,
                outputs=self.Configuration.n_steps_out,
                additional_vars=self.Configuration.additional_var_dir,
                iteration=iteration,
                model=self.Configuration.Model_dir,
            )

            metrics_dir = self.Configuration.Main_Model_training_evaluation_full_dir
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, metrics_filename)
            metrics_pd.to_csv(metrics_path, index=False)

            self.logger.info(
                f'  {metrics_filename}'.ljust(self.justif - 2, '.') + 'SAVED')
        return metrics_pd

###########################################################################################
    def _generate_metrics_report(self, station_names, station_train_actuals,
                                  station_train_preds, station_test_actuals,
                                  station_test_preds):
        """
        Generate and save one metrics report CSV per station after training.

        File : {station}_metrics_report.csv
        Dir  : Configuration.Main_Model_training_full_dir

        Format
        ------
        Metric,Training,Testing
        R2,0.8616,0.8477
        RMSE,2.2972,2.4166
        """
        report_dir = self.Configuration.Main_Model_training_full_dir
        os.makedirs(report_dir, exist_ok=True)

        self.logger.info('''\u2550''' * self.justif)
        self.logger.info('Metrics Reports'.center(self.justif, '|'))
        self.logger.info('''\u2550''' * self.justif)

        for station in station_names:
            if station_train_actuals[station] is None:
                self.logger.warning(
                    f'  {station}: no data accumulated, skipping report'.ljust(
                        self.justif - 2, '.') + 'SKIP')
                continue

            train_m = self._compute_metrics(
                station_train_actuals[station], station_train_preds[station])
            test_m  = self._compute_metrics(
                station_test_actuals[station],  station_test_preds[station])

            report_df = pd.DataFrame({
                'Metric':   ['R2',          'RMSE'],
                'Training': [train_m['R2'],  train_m['RMSE']],
                'Testing':  [test_m['R2'],   test_m['RMSE']],
            })

            report_filename = f'{station}_metrics_report.csv'
            report_path     = os.path.join(report_dir, report_filename)
            report_df.to_csv(report_path, index=False)

            self.logger.info(f'  {station}'.ljust(self.justif, '|'))
            self.logger.info(f'    Training Metrics:'.ljust(self.justif, '|'))
            self.logger.info(f'      R-sq: {train_m["R2"]}'.ljust(self.justif, '|'))
            self.logger.info(f'      RMSE: {train_m["RMSE"]}'.ljust(self.justif, '|'))
            self.logger.info(f'    Testing Metrics:'.ljust(self.justif, '|'))
            self.logger.info(f'      R-sq: {test_m["R2"]}'.ljust(self.justif, '|'))
            self.logger.info(f'      RMSE: {test_m["RMSE"]}'.ljust(self.justif, '|'))
            self.logger.info(
                f'  {report_filename}'.ljust(self.justif - 2, '.') + 'SAVED')

        self._write_dashboard_metrics(
            station_names,
            station_test_actuals,
            station_test_preds,
        )

        return

###########################################################################################
    def run_all(self, station_decomposed_dict, n_retrain):
        """
        Unified entry point for BOTH training and inference.
        Called by main.py / Training wrapper / Forecast wrapper identically.

        Mirrors CNN_LSTM_v1.run_all() in role and signature.

        Training mode  (Configuration.train_model = True)
        --------------------------------------------------
        For each of the 21 IMFs:
          1. Extract and stack IMF data across stations
          2. Fit MinMaxScaler (all stations together)
          3. Build new model via configure_model()
          4. Create (X, y) sequences
          5. Train via fit() with early stopping
          6. Optionally save weights + scaler

        Inference mode (Configuration.train_model = False)
        ---------------------------------------------------
        For each of the 21 IMFs:
          1. Load scaler (.pkl) from disk
          2. Load model (.keras) via configure_model()
          3. Scale input data with loaded scaler
          4. Create X sequences (no y needed)
          5. Predict and inverse-transform
        Sum all 21 IMF predictions → final PM2.5 forecast per station.

        Parameters
        ----------
        station_decomposed_dict : dict
            {station_name: DataFrame(IMF_1 … IMF_N, Residual, [Original])}
        n_retrain : int
            Iteration number (used as forecast_segment identifier).

        Returns
        -------
        tuple : (list_output_pd, train_history)
            list_output_pd : list[pd.DataFrame] one per station
            train_history  : dict {imf_name: keras History} in training mode
                             None                           in inference mode
        """
        start_time    = time.time()
        train_model   = self.Configuration.train_model
        save_model    = self.Configuration.save_model
        station_names = list(station_decomposed_dict.keys())
        num_sites     = len(station_names)

        self.logger.info(''.ljust(self.justif, '='))
        mode_label = 'TRAINING' if train_model else 'INFERENCE'
        self.logger.info(
            f'{self.MODEL_DISPLAY_NAME}  {mode_label}  Workflow'.center(self.justif, '|'))
        self.logger.info(''.ljust(self.justif, '='))
        self.logger.info(
            f'Stations  : {len(station_names)} → {", ".join(station_names)}'.ljust(self.justif, '|'))
        self.logger.info(
            f'IMF models: {len(self.imf_columns)}'.ljust(self.justif, '|'))

        # Create output directories
        self.MakeDir(self.model_dir)
        self.MakeDir(self.scaler_dir)
        self.MakeDir(self.metadata_dir)

        # ── Initialise Data_preparation (corporate standard) ─────────────────
        try:
            self.Data_prep_class = Data_preparation.Data_Preparation_Class(
                self.Configuration)
        except Exception as exc:
            self.logger.warning(
                f'Data_preparation init: {exc}'.ljust(self.justif - 2, '.') + 'WARN')
            self.Data_prep_class = None

        # ── Validate input ────────────────────────────────────────────────────
        if not isinstance(station_decomposed_dict, dict):
            raise TypeError(
                f"Expected dict of VMD DataFrames, got {type(station_decomposed_dict)}")

        reference_cols = set(station_decomposed_dict[station_names[0]].columns)
        for station in station_names[1:]:
            if set(station_decomposed_dict[station].columns) != reference_cols:
                self.logger.warning(
                    f'  {station}: IMF column mismatch'.ljust(self.justif - 2, '.') + 'WARN')

        # ── Load version metadata (inference mode only) ───────────────────────
        model_metadata = None
        if not train_model:
            model_metadata = self._load_version_metadata()

        aligned_station_names = list(station_names)
        if not train_model:
            expected_station_names = None
            if isinstance(model_metadata, dict):
                metadata_station_names = model_metadata.get("station_names")
                if isinstance(metadata_station_names, list) and metadata_station_names:
                    expected_station_names = [
                        station for station in metadata_station_names
                        if station in station_decomposed_dict
                    ]

            if not expected_station_names and self.imf_columns:
                sample_scaler_path = os.path.join(
                    self.scaler_dir,
                    f"{self.imf_columns[0]}_scaler.pkl",
                )
                if os.path.exists(sample_scaler_path):
                    with open(sample_scaler_path, "rb") as fh:
                        sample_scaler = pickle.load(fh)
                    expected_n_features = getattr(sample_scaler, "n_features_in_", None)
                    if expected_n_features is not None and expected_n_features != len(aligned_station_names):
                        if len(aligned_station_names) < expected_n_features:
                            raise ValueError(
                                f"Scaler expects {expected_n_features} stations but only "
                                f"{len(aligned_station_names)} are available."
                            )
                        expected_station_names = aligned_station_names[:expected_n_features]
                        self.logger.warning(
                            f"Station count mismatch: model expects {expected_n_features} stations, "
                            f"current run has {len(aligned_station_names)}. "
                            f"Using the first {expected_n_features} stations: "
                            f"{', '.join(expected_station_names)}".ljust(self.justif - 2, '.') + 'WARN'
                        )

            if expected_station_names:
                aligned_station_names = expected_station_names
                station_decomposed_dict = {
                    station: station_decomposed_dict[station]
                    for station in aligned_station_names
                    if station in station_decomposed_dict
                }
                station_names = list(aligned_station_names)
                num_sites = len(station_names)

        # ═════════════════════════════════════════════════════════════════════
        # IMF loop – identical structure for training and inference
        # ═════════════════════════════════════════════════════════════════════
        imf_predictions = {}
        imf_hindcast_actuals = {}
        train_histories = {}
        test_index = None

        # Per-station accumulators for post-training metrics report.
        # Each IMF's inverse-transformed values are summed across all 21 IMFs,
        # reconstructing the full PM2.5 signal per station — matching the
        # same summation used at inference time.
        station_train_preds   = {s: None for s in station_names}
        station_train_actuals = {s: None for s in station_names}
        station_test_preds    = {s: None for s in station_names}
        station_test_actuals  = {s: None for s in station_names}

        for idx, imf_name in enumerate(self.imf_columns, 1):
            self.logger.info(''.ljust(self.justif, '-'))
            self.logger.info(
                f'{imf_name}  ({idx}/{len(self.imf_columns)})'.center(self.justif, '|'))
            self.logger.info(''.ljust(self.justif, '-'))

            try:
                # ── Step 1: extract IMF data (n_timesteps, num_sites) ─────────
                imf_data = self._prepare_imf_data(station_decomposed_dict, imf_name)

                # ── Step 2: scaler ────────────────────────────────────────────
                if train_model:
                    # Fit a fresh scaler on the training data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    imf_data_scaled = scaler.fit_transform(imf_data)
                    self.scalers[imf_name] = scaler
                    self.logger.info(
                        f'  {imf_name} scaler fitted'.ljust(self.justif - 2, '.') + 'OK')
                else:
                    # Load scaler that was fitted during training
                    scaler_path = os.path.join(
                        self.scaler_dir, f"{imf_name}_scaler.pkl")
                    if not os.path.exists(scaler_path):
                        raise FileNotFoundError(
                            f"Scaler not found: {scaler_path}. "
                            f"Run training first.")
                    with open(scaler_path, 'rb') as fh:
                        scaler = pickle.load(fh)
                    self.scalers[imf_name] = scaler
                    imf_data_scaled = scaler.transform(imf_data)
                    self.logger.info(
                        f'  {imf_name} scaler loaded'.ljust(self.justif - 2, '.') + 'OK')

                # ── Step 3: configure model (build or load) ───────────────────
                self.configure_model(imf_name, num_sites, train_model)

                # ── Step 4 + 5: training branch ───────────────────────────────
                if train_model:
                    X, y = self._prepare_sequences(
                        imf_data_scaled, self.time_steps, self.lags)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False)

                    self.logger.info(
                        f'  X_train {X_train.shape}  y_train {y_train.shape}'.ljust(self.justif, '|'))

                    history = self.fit(
                        X_train, y_train,
                        n_epochs   = self.n_epochs,
                        save_model = save_model,
                        imf_name   = imf_name,
                    )
                    train_histories[imf_name] = history

                    # Inverse-transform predictions and actuals for this IMF
                    train_pred   = scaler.inverse_transform(
                                       self.model.predict(X_train, verbose=0))
                    train_actual = scaler.inverse_transform(y_train)
                    test_pred    = scaler.inverse_transform(
                                       self.model.predict(X_test, verbose=0))
                    test_actual  = scaler.inverse_transform(y_test)

                    if test_index is None:
                        sequence_index = station_decomposed_dict[station_names[0]].index[
                            self.time_steps + self.lags - 1:
                        ]
                        sequence_index = pd.Index(sequence_index)
                        if len(sequence_index) >= len(test_actual):
                            test_index = sequence_index[-len(test_actual):]
                        else:
                            test_index = sequence_index

                    # Log per-IMF MAE
                    test_mae = float(np.mean(np.abs(test_pred - test_actual)))
                    self.logger.info(
                        f'  {imf_name} test MAE: {test_mae:.4f}'.ljust(self.justif, '|'))

                    # Accumulate per-station predictions across IMFs.
                    # Summing IMF contributions mirrors the VMD reconstruction
                    # used at inference, giving station-level metrics on the
                    # reconstructed PM2.5 signal rather than individual IMFs.
                    for s_idx, station in enumerate(station_names):
                        if imf_name == self.imf_columns[0]:
                            # First IMF — initialise arrays with correct length
                            station_train_preds[station]   = train_pred[:, s_idx].copy()
                            station_train_actuals[station] = train_actual[:, s_idx].copy()
                            station_test_preds[station]    = test_pred[:, s_idx].copy()
                            station_test_actuals[station]  = test_actual[:, s_idx].copy()
                        else:
                            # Subsequent IMFs — accumulate (sum) contributions
                            station_train_preds[station]   += train_pred[:, s_idx]
                            station_train_actuals[station] += train_actual[:, s_idx]
                            station_test_preds[station]    += test_pred[:, s_idx]
                            station_test_actuals[station]  += test_actual[:, s_idx]

                    self.logger.info(
                        f'  {imf_name}'.ljust(self.justif - 2, '.') + 'TRAINED')

                # ── Step 4 + 5: inference branch ──────────────────────────────
                else:
                    X            = self._create_inference_sequences(
                                       imf_data_scaled, self.time_steps, self.lags)
                    raw_pred     = self.model.predict(X, verbose=0)
                    predictions  = scaler.inverse_transform(raw_pred)

                    if np.any(np.isnan(predictions)):
                        self.logger.warning(
                            f'  {imf_name}: NaN in predictions'.ljust(self.justif - 2, '.') + 'WARN')
                    if np.any(np.isinf(predictions)):
                        self.logger.warning(
                            f'  {imf_name}: Inf in predictions'.ljust(self.justif - 2, '.') + 'WARN')

                    imf_predictions[imf_name] = predictions
                    target_start = self.time_steps + self.lags - 1
                    target_end = target_start + len(predictions)
                    imf_hindcast_actuals[imf_name] = imf_data[target_start:target_end]
                    self.logger.info(
                        f'  {imf_name}'.ljust(self.justif - 2, '.') + 'PREDICTED')

            except Exception as exc:
                self.logger.error(
                    f'  {imf_name}: {exc}'.ljust(self.justif - 2, '.') + 'FAIL')
                raise

        # ═════════════════════════════════════════════════════════════════════
        # Post-loop
        # ═════════════════════════════════════════════════════════════════════
        elapsed = time.time() - start_time
        self.logger.info(''.ljust(self.justif, '='))
        self.logger.info(
            f'Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)'.ljust(self.justif, '|'))

        if train_model:
            # Write metadata JSON so inference can validate model version
            if save_model:
                self.save_metadata(station_names)

            # Generate per-station metrics report (R2 + RMSE, train + test)
            self._generate_metrics_report(
                station_names,
                station_train_actuals,
                station_train_preds,
                station_test_actuals,
                station_test_preds,
            )

            self.latest_training_plot_data = {
                "timestamps": pd.DatetimeIndex(test_index)
                if isinstance(test_index, pd.DatetimeIndex)
                else test_index,
                "station_names": list(station_names),
                "predictions": {
                    station: np.asarray(values).copy()
                    for station, values in station_test_preds.items()
                    if values is not None
                },
                "actuals": {
                    station: np.asarray(values).copy()
                    for station, values in station_test_actuals.items()
                    if values is not None
                },
            }

            self.logger.info(
                f'Training complete: {len(self.imf_columns)} IMF models'.ljust(self.justif - 2, '.') + 'OK')
            # Training produces no forecast output – return None as per
            # corporate standard (mirrors CNN_LSTM_v1 training branch)
            return None, train_histories

        else:
            # Sum IMF contributions to reconstruct PM2.5 signal
            final_predictions = self._sum_imf_predictions(imf_predictions)
            final_hindcast_actuals = self._sum_imf_predictions(imf_hindcast_actuals)

            recent_hindcast_preds = {}
            recent_hindcast_actuals = {}
            for s_idx, station in enumerate(station_names):
                recent_hindcast_preds[station] = final_predictions[:, s_idx]
                recent_hindcast_actuals[station] = final_hindcast_actuals[:, s_idx]

            self.latest_dashboard_metrics_pd = self._write_dashboard_metrics(
                station_names,
                recent_hindcast_actuals,
                recent_hindcast_preds,
                save_file=False,
            )

            # Derive forecast start timestamp from last VMD timestep
            first_station = station_names[0]
            if isinstance(station_decomposed_dict[first_station].index,
                          pd.DatetimeIndex):
                last_ts         = station_decomposed_dict[first_station].index[-1]
                start_timestamp = last_ts + pd.Timedelta(hours=self.lags)
                self.logger.info(
                    f'Forecast start: {start_timestamp}'.ljust(self.justif, '|'))
            else:
                start_timestamp = None
                self.logger.warning(
                    'No DatetimeIndex in VMD data – timestamps approximated'.ljust(
                        self.justif - 2, '.') + 'WARN')

            forecast_segment = n_retrain + 1
            list_output_pd   = self._format_output(
                final_predictions,
                station_names,
                start_timestamp,
                forecast_segment = forecast_segment,
            )

            self.logger.info(
                f'{self.MODEL_DISPLAY_NAME} inference'.ljust(self.justif - 2, '.') + 'COMPLETE')
            return list_output_pd, None


###########################################################################################
if __name__ == '__main__':
    import logging
    import Core_iHPC.Tools.InitLogging as IL

    justif     = 102
    loggername = 'Sparse_LSTM_v1_Test'
    logger     = IL.Initialise_logging(loggername)

    class MockConfig:
        """Minimal configuration object for standalone testing."""
        def __init__(self):
            self.logger        = logger
            self.justif        = justif
            self.n_steps_in    = 12
            self.n_steps_out   = 3
            self.train_model   = False
            self.save_model    = False
            self.var_to_predict = ['PM2.5']
            # Sequence
            self.time_steps    = 12
            self.lags          = 3
            # VMD
            self.vmd_n_imfs    = 20
            # Sparse Transformer
            self.sparse_lstm_num_heads       = 4
            self.sparse_lstm_window_size     = 12   # must be ≤ time_steps
            self.sparse_lstm_ff_dim          = 256
            self.sparse_lstm_d_model         = 64
            self.sparse_lstm_dropout         = 0.1
            self.sparse_lstm_units           = 100
            self.sparse_lstm_cnn_filters     = 64
            self.sparse_lstm_cnn_kernel_size = 3
            # Training
            self.n_epochs      = 5
            self.batch_size    = 32
            # Paths
            self.model_base_path              = 'AI_Runs/Model_weights'
            self.model_name                   = 'sparse_lstm_pm25_3'
            self.model_version                = 'v1.0'
            self.strict_version_check         = False
            self.Main_model_data_full_dir     = 'AI_Runs/Model_weights/sparse_lstm_pm25_3_v1.0'

    Configuration = MockConfig()
    SLSTM = Sparse_LSTM_Class(Configuration)
    logger.info('Sparse_LSTM_v1 standalone test complete')
