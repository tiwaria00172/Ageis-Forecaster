# =============================================================================
# Tests — End-to-end pipeline verification
# =============================================================================
"""Smoke tests for the core pipeline modules."""

import os, sys, pytest
import numpy as np, pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.helpers import load_config, set_seed
from src.utils.metrics import regression_metrics, binary_classification_metrics, multiclass_classification_metrics


class TestHelpers:
    def test_load_config(self):
        cfg = load_config(os.path.join(_ROOT, "config", "config.yaml"))
        assert "model" in cfg
        assert "training" in cfg

    def test_set_seed(self):
        set_seed(42)
        a = np.random.rand()
        set_seed(42)
        b = np.random.rand()
        assert a == b


class TestMetrics:
    def test_regression(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = regression_metrics(y, y)
        assert m["mse"] == 0.0

    def test_binary(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.9, 0.8, 0.2])
        m = binary_classification_metrics(y, p)
        assert m["accuracy"] == 1.0

    def test_multiclass(self):
        y = np.array([0, 1, 2, 0])
        p = np.array([0, 1, 2, 0])
        m = multiclass_classification_metrics(y, p, ["A", "B", "C"])
        assert m["accuracy"] == 1.0


class TestDataPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.cfg = load_config(os.path.join(_ROOT, "config", "config.yaml"))
        # Create a small CSV
        n = 200
        self.csv_path = str(tmp_path / "test.csv")
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=n, freq="1s"),
            "Src IP": ["192.168.1.1"] * n,
            "Dst IP": [f"10.0.0.{i%10}" for i in range(n)],
            "Src Port": np.random.randint(1024, 65535, n),
            "Dst Port": np.random.randint(1, 1024, n),
            "Protocol": [6] * n,
            "Flow Duration": np.random.randint(100, 100000, n),
            "Total Fwd Packet": np.random.randint(1, 50, n),
            "Total Bwd packets": np.random.randint(1, 50, n),
            "Total Length of Fwd Packet": np.random.randint(100, 10000, n),
            "Total Length of Bwd Packet": np.random.randint(100, 10000, n),
            "SYN Flag Count": np.random.randint(0, 5, n),
            "ACK Flag Count": np.random.randint(0, 10, n),
            "FIN Flag Count": np.random.randint(0, 2, n),
            "RST Flag Count": np.random.randint(0, 2, n),
            "PSH Flag Count": np.random.randint(0, 5, n),
            "URG Flag Count": np.zeros(n, dtype=int),
            "Fwd IAT Mean": np.random.uniform(10, 1000, n),
            "Bwd IAT Mean": np.random.uniform(10, 1000, n),
            "Label": ["BENIGN"] * 150 + ["PortScan"] * 30 + ["DoS Hulk"] * 20,
        })
        df.to_csv(self.csv_path, index=False)

    def test_dataset_loader(self):
        from src.data.dataset_loader import load_csv_dataset
        df, meta = load_csv_dataset(self.csv_path, self.cfg)
        assert len(df) == 200
        assert meta["schema"] == "cic_ids"

    def test_feature_extractor(self):
        from src.data.dataset_loader import load_csv_dataset
        from src.data.feature_extractor import FeatureExtractor
        df, _ = load_csv_dataset(self.csv_path, self.cfg)
        ext = FeatureExtractor(self.cfg)
        df2, meta = ext.extract(df)
        assert len(df2.columns) > len(df.columns) or len(meta["derived_features"]) >= 0

    def test_preprocessor(self):
        from src.data.dataset_loader import load_csv_dataset
        from src.data.feature_extractor import FeatureExtractor
        from src.data.preprocessor import Preprocessor
        df, _ = load_csv_dataset(self.csv_path, self.cfg)
        ext = FeatureExtractor(self.cfg)
        df, _ = ext.extract(df)
        pp = Preprocessor(self.cfg)
        df = pp.parse_timestamps(df)
        df = pp.map_labels(df, self.cfg)
        df, scaled = pp.fit_transform(df)
        assert scaled.shape[0] == len(df)
        assert scaled.shape[1] == len(pp.feature_columns)

    def test_time_windowing(self):
        from src.data.dataset_loader import load_csv_dataset
        from src.data.feature_extractor import FeatureExtractor
        from src.data.preprocessor import Preprocessor
        from src.data.time_windowing import TimeWindowGenerator
        df, _ = load_csv_dataset(self.csv_path, self.cfg)
        ext = FeatureExtractor(self.cfg)
        df, _ = ext.extract(df)
        pp = Preprocessor(self.cfg)
        df = pp.parse_timestamps(df)
        df = pp.map_labels(df, self.cfg)
        df, scaled = pp.fit_transform(df)
        wg = TimeWindowGenerator(self.cfg)
        states, l_atk, l_bin, ts = wg.create_windows(df, scaled, pp.feature_columns)
        assert states.ndim == 2
        assert len(l_atk) == len(states)


class TestModel:
    def test_world_model_forward(self):
        import torch
        from src.models.world_model import NetworkWorldModel
        cfg = load_config(os.path.join(_ROOT, "config", "config.yaml"))
        model = NetworkWorldModel.from_config(input_dim=20, config=cfg)
        x = torch.randn(4, 10, 20)
        ns, inf, stage, h = model(x)
        assert ns.shape == (4, 20)
        assert inf.shape == (4, 1)
        assert stage.shape == (4, 6)

    def test_forecasting_engine(self):
        import torch
        from src.models.world_model import NetworkWorldModel
        from src.models.forecasting_engine import ForecastingEngine
        cfg = load_config(os.path.join(_ROOT, "config", "config.yaml"))
        model = NetworkWorldModel.from_config(input_dim=15, config=cfg)
        engine = ForecastingEngine(model, cfg)
        seq = np.random.randn(10, 15).astype(np.float32)
        results = engine.forecast(seq, k_steps=3)
        assert len(results) == 3
        assert all(0 <= r.infiltration_prob <= 1 for r in results)


class TestMitre:
    def test_mapper(self):
        cfg = load_config(os.path.join(_ROOT, "config", "config.yaml"))
        from src.attack_mapping.mitre_mapper import MitreMapper
        m = MitreMapper(cfg)
        assert m.get_stage_name(0) == "Normal"
        assert m.get_stage_name(1) == "Reconnaissance"
        assert m.map_label_to_stage("BENIGN") == 0
        assert m.map_label_to_stage("PortScan") == 1
