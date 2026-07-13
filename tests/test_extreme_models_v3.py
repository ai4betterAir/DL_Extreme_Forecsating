"""Architecture smoke tests for the extreme PM2.5 v3 model family."""
import unittest
from types import SimpleNamespace

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - CI without TensorFlow
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class ExtremeModelShapeTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            n_steps_in=24,
            n_steps_out=3,
            n_epochs=1,
            batch_size=2,
            var_to_predict=["PM2.5"],
            model_parameters_dict={},
            model_base_path="/tmp/extreme_pm25_test",
            model_name="shape_test",
            model_version="v3.0",
            sparse_lstm_window_size=6,
            sparse_lstm_cnn_filters=32,
            sparse_lstm_d_model=32,
            sparse_lstm_num_heads=4,
            sparse_lstm_ff_dim=64,
            sparse_lstm_units=32,
            sparse_lstm_cnn_kernel_size=3,
            sparse_lstm_dropout=0.1,
        )
        self.batch = tf.random.normal((2, 24, 4))

    def test_quantile_models(self):
        from Core_iHPC.Models.Sparse_LSTM_v3 import Sparse_LSTM_v3_Class
        from Core_iHPC.Models.GraphWaveNet_Quantile_PM25 import GraphWaveNet_Quantile_Class
        from Core_iHPC.Models.TFT_Extreme_PM25 import TFT_Extreme_Class
        from Core_iHPC.Models.TCN_Extreme_PM25 import TCN_Extreme_Class
        from Core_iHPC.Models.PatchTST_TailQuantile_PM25 import PatchTST_TailQuantile_Class

        for model_class in (
            Sparse_LSTM_v3_Class,
            GraphWaveNet_Quantile_Class,
            TFT_Extreme_Class,
            TCN_Extreme_Class,
            PatchTST_TailQuantile_Class,
        ):
            forecaster = model_class(self.config)
            output = forecaster.build_model(4)(self.batch)
            self.assertEqual(tuple(output.shape), (2, 3, 4, 5), model_class.__name__)

    def test_hurdle_gpd_shape(self):
        from Core_iHPC.Models.GraphWaveNet_Hurdle_GPD_PM25 import GraphWaveNet_Hurdle_GPD_Class

        forecaster = GraphWaveNet_Hurdle_GPD_Class(self.config)
        output = forecaster.build_model(4)(self.batch)
        self.assertEqual(tuple(output.shape), (2, 3, 4, 4))

    def test_gev_shape(self):
        from Core_iHPC.Models.DeepExtrema_GEV_PM25 import DeepExtrema_GEV_Class

        forecaster = DeepExtrema_GEV_Class(self.config)
        output = forecaster.build_model(4)(self.batch)
        self.assertEqual(tuple(output.shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
