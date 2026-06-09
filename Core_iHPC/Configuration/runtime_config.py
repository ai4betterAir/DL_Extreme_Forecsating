"""
Shared runtime configuration helpers.

This module collects model runtime defaults and derived paths so model classes
stay focused on architecture/training logic. It also hosts the default
target->model-module mapping used by the generic pipeline when YAML does not
explicitly specify a model module.
"""

from dataclasses import dataclass
import os


@dataclass
class SparseLSTMRuntimeConfig:
    n_steps_in: int
    n_steps_out: int
    time_steps: int
    lags: int
    use_vmd_decomposition: bool
    use_one_model_per_imf: bool
    vmd_n_imfs: int
    imf_columns: list
    num_heads: int
    window_size: int
    ff_dim: int
    d_model: int
    dropout_rate: float
    lstm_units: int
    cnn_filters: int
    cnn_kernel_size: int
    n_epochs: int
    batch_size: int
    model_save_dir: str
    model_base_path: str
    model_name: str
    model_version: str
    strict_version_check: bool
    model_root: str
    model_dir: str
    scaler_dir: str
    metadata_dir: str


def _as_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_sparse_lstm_runtime(configuration, default_model_name):
    """
    Build a normalized Sparse LSTM runtime bundle and push it back onto the
    shared configuration object.
    """
    n_steps_in = getattr(configuration, "n_steps_in", 12)
    n_steps_out = getattr(configuration, "n_steps_out", 3)
    time_steps = getattr(configuration, "time_steps", n_steps_in)
    lags = getattr(configuration, "lags", 3)

    use_vmd_decomposition = _as_bool(
        getattr(configuration, "use_decomposition", getattr(configuration, "use_vmd_decomposition", True)),
        True,
    )
    use_one_model_per_imf = _as_bool(getattr(configuration, "use_one_model_per_imf", True), True)
    vmd_n_imfs = max(_as_int(getattr(configuration, "vmd_n_imfs", 0), 0), 0)
    imf_columns = [f"IMF_{i}" for i in range(1, vmd_n_imfs + 1)] + ["Residual"] if use_vmd_decomposition else ["RAW"]

    num_heads = getattr(configuration, "sparse_lstm_num_heads", 4)
    window_size = getattr(configuration, "sparse_lstm_window_size", 32)
    ff_dim = getattr(configuration, "sparse_lstm_ff_dim", 256)
    d_model = getattr(configuration, "sparse_lstm_d_model", 64)
    dropout_rate = getattr(configuration, "sparse_lstm_dropout", 0.1)
    lstm_units = getattr(configuration, "sparse_lstm_units", 100)
    cnn_filters = getattr(configuration, "sparse_lstm_cnn_filters", 64)
    cnn_kernel_size = getattr(configuration, "sparse_lstm_cnn_kernel_size", 3)

    n_epochs = getattr(configuration, "n_epochs", 200)
    batch_size = getattr(configuration, "batch_size", 32)

    model_save_dir = getattr(configuration, "Main_model_data_full_dir", None)
    model_base_path = getattr(configuration, "model_base_path", "AI_Runs/Model_weights")
    model_name = getattr(configuration, "model_name", default_model_name)
    model_version = getattr(configuration, "model_version", "v1.0")
    strict_version_check = getattr(configuration, "strict_version_check", False)
    model_root = os.path.join(model_base_path, f"{model_name}_{model_version}")
    model_dir = os.path.join(model_root, "models")
    scaler_dir = os.path.join(model_root, "scalers")
    metadata_dir = os.path.join(model_root, "metadata")

    runtime = SparseLSTMRuntimeConfig(
        n_steps_in=n_steps_in,
        n_steps_out=n_steps_out,
        time_steps=time_steps,
        lags=lags,
        use_vmd_decomposition=use_vmd_decomposition,
        use_one_model_per_imf=use_one_model_per_imf,
        vmd_n_imfs=vmd_n_imfs,
        imf_columns=imf_columns,
        num_heads=num_heads,
        window_size=window_size,
        ff_dim=ff_dim,
        d_model=d_model,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units,
        cnn_filters=cnn_filters,
        cnn_kernel_size=cnn_kernel_size,
        n_epochs=n_epochs,
        batch_size=batch_size,
        model_save_dir=model_save_dir,
        model_base_path=model_base_path,
        model_name=model_name,
        model_version=model_version,
        strict_version_check=strict_version_check,
        model_root=model_root,
        model_dir=model_dir,
        scaler_dir=scaler_dir,
        metadata_dir=metadata_dir,
    )

    # Push the normalized values back to the shared configuration object so the
    # rest of the pipeline can continue using attribute access.
    configuration.n_steps_in = n_steps_in
    configuration.n_steps_out = n_steps_out
    configuration.time_steps = time_steps
    configuration.lags = lags
    configuration.use_decomposition = use_vmd_decomposition
    configuration.use_vmd_decomposition = use_vmd_decomposition
    configuration.use_one_model_per_imf = use_one_model_per_imf
    configuration.vmd_n_imfs = vmd_n_imfs
    configuration.imf_columns = imf_columns
    configuration.sparse_lstm_num_heads = num_heads
    configuration.sparse_lstm_window_size = window_size
    configuration.sparse_lstm_ff_dim = ff_dim
    configuration.sparse_lstm_d_model = d_model
    configuration.sparse_lstm_dropout = dropout_rate
    configuration.sparse_lstm_units = lstm_units
    configuration.sparse_lstm_cnn_filters = cnn_filters
    configuration.sparse_lstm_cnn_kernel_size = cnn_kernel_size
    configuration.n_epochs = n_epochs
    configuration.batch_size = batch_size
    configuration.model_save_dir = model_save_dir
    configuration.model_base_path = model_base_path
    configuration.model_name = model_name
    configuration.model_version = model_version
    configuration.strict_version_check = strict_version_check
    configuration.model_root = model_root
    configuration.model_dir = model_dir
    configuration.scaler_dir = scaler_dir
    configuration.metadata_dir = metadata_dir

    return runtime


# --------------------------------------------------------------------------------------
# Generic model selection defaults
# --------------------------------------------------------------------------------------

# Default model module name when YAML does not specify `forecast_method`.
DEFAULT_FORECAST_METHOD = "Sparse_LSTM_v1"

# Optional target->model-module mapping used by the pipeline when YAML does not
# explicitly request a model module.
TARGET_MODEL_CONFIG = {
    "O3": {"module_name": "Sparse_LSTM_v1", "weights_name": "sparse_lstm_o3_{lags}"},
    "PM2.5": {"module_name": "Sparse_LSTM_v2", "weights_name": "sparse_lstm_pm25_{lags}"},
    "PM10": {"module_name": "Sparse_LSTM_v1", "weights_name": "sparse_lstm_pm10_{lags}"},
}
