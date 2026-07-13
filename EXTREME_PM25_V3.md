# Extreme PM2.5 forecasting v3

This branch adds a leakage-safe model family for predicting rare high PM2.5 concentrations. Existing operational models and the production configuration tree are unchanged.

## Stage 1 — corrected existing model

`Core_iHPC/Models/Sparse_LSTM_v3.py`

The v3 workflow corrects the main methodological limitations of the previous VMD ensemble:

- chronological train, validation and test boundaries are created before scaling;
- `RobustScaler` is fitted using the training period only;
- station-specific extreme thresholds are calculated from training data only;
- the model uses the raw `Original` PM2.5 signal when legacy decomposition frames are supplied, avoiding full-series VMD leakage;
- one model directly predicts the complete forecast trajectory;
- upper quantiles and event-balanced sample weights increase attention to rare peaks.

## Stage 2 — core models

| Module | Purpose |
|---|---|
| `GraphWaveNet_Quantile_PM25.py` | Learns adaptive links among monitoring stations and predicts PM2.5 quantiles for every future hour. |
| `TFT_Extreme_PM25.py` | Uses variable selection, recurrent encoding and causal attention for an interpretable multi-horizon forecast. |
| `TCN_Extreme_PM25.py` | Fast causal-dilated convolutional benchmark with direct quantile output. |

## Stage 3 — specialised tail models

| Module | Purpose |
|---|---|
| `GraphWaveNet_Hurdle_GPD_PM25.py` | Jointly predicts concentration, exceedance probability and GPD tail magnitude. |
| `DeepExtrema_GEV_PM25.py` | Predicts a GEV distribution for the maximum PM2.5 during the future block. |
| `PatchTST_TailQuantile_PM25.py` | Uses long temporal patches and transformer blocks to predict upper-tail quantiles. |

## Shared training behaviour

`Core_iHPC/Models/extreme_common.py` and `extreme_distributions.py` provide:

- leakage-safe chronological splitting;
- training-only robust scaling;
- direct multi-horizon sequence generation;
- station-specific percentile thresholds;
- event-balanced sample weights;
- multi-quantile pinball loss;
- numerically stable focal exceedance, GPD and GEV losses;
- station-level and overall tail metrics;
- saved station ordering, thresholds, scalers and metrics for safe inference;
- dashboard-compatible training metrics and plotting data.

The default quantiles are `0.50, 0.75, 0.90, 0.95, 0.99`. The default extreme definition is the station-specific training-period 95th percentile.

## Configurations

The experiment configurations are deliberately isolated from the production auto-discovery directory:

```text
experiments/extreme_v3/configs/
```

This prevents a standard dashboard run from unintentionally launching all seven training experiments.

Available configurations:

```text
PM25_Sparse_LSTM_v3.yaml
PM25_GraphWaveNet_Quantile.yaml
PM25_TFT_Extreme.yaml
PM25_TCN_Extreme.yaml
PM25_GraphWaveNet_Hurdle_GPD.yaml
PM25_DeepExtrema_GEV.yaml
PM25_PatchTST_TailQuantile.yaml
```

List the available experiments:

```bash
python experiments/extreme_v3/run_experiment.py --list
```

Run one selected model:

```bash
python experiments/extreme_v3/run_experiment.py PM25_GraphWaveNet_Quantile
```

The runner passes the selected absolute YAML path into the existing pipeline. Set `train_model: false` in that YAML after training to load the saved weights and generate forecasts.

## Recommended experiment order

1. Train `Sparse_LSTM_v3` and `TCN_Extreme` as corrected baselines.
2. Compare `GraphWaveNet_Quantile` and `TFT_Extreme` using the same chronological split.
3. Train `GraphWaveNet_Hurdle_GPD` after selecting the threshold percentile.
4. Use `DeepExtrema_GEV` only for future-block maximum prediction, not as a replacement for the hourly trajectory model.
5. Test `PatchTST_TailQuantile` with 72, 168 and 336-hour lookbacks.

## Required evaluation

In addition to ordinary RMSE and MAE, report:

- tail RMSE and tail MAE above the training-derived threshold;
- exceedance precision, recall and F1;
- peak magnitude error and peak timing error;
- performance by station, season, forecast horizon and episode type;
- quantile coverage and pinball loss for probabilistic models.

## Validation status

All new Python modules and the isolated runner were syntax-checked before committing. A TensorFlow architecture smoke test is included at `tests/test_extreme_models_v3.py`. Full training and data-level validation must be run in the project TensorFlow/HPC environment because repository data and a TensorFlow runtime are not available in the connector execution environment.
