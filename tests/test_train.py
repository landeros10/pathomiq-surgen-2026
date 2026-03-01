"""Smoke test for the training loop.

Runs 2 epochs on 6 tiny synthetic slides and checks:
  - loss values printed to stdout are finite (not NaN / inf)
  - best_model.pt is written to disk
  - at least one MLflow run is created in the local SQLite store
"""
import math
import re

import pandas as pd
import pytest
import torch
import yaml


@pytest.fixture
def synthetic_run(tmp_path):
    """Write 6 synthetic .pt slides + split CSVs + a minimal config yaml."""
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()

    slides = [f"S{i:02d}" for i in range(6)]
    labels = [0, 0, 0, 0, 1, 1]
    for sid in slides:
        torch.save(torch.randn(20, 1024), emb_dir / f"{sid}.pt")

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    df = pd.DataFrame({"case_id": slides, "slide_id": slides, "label": labels})
    df.iloc[:4].to_csv(split_dir / "train.csv", index=False)
    df.iloc[4:5].to_csv(split_dir / "val.csv",   index=False)
    df.iloc[5:].to_csv( split_dir / "test.csv",  index=False)

    cfg = {
        "paths": {
            "embeddings_dir": str(emb_dir),
            "data_dir":       str(split_dir),
            "models_dir":     str(tmp_path / "models"),
            "logs_dir":       str(tmp_path / "logs"),
            "results_dir":    str(tmp_path / "results"),
        },
        "data": {
            "train_split":    "train.csv",
            "val_split":      "val.csv",
            "label_column":   "label",
            "slide_id_column": "case_id",
        },
        "model": {
            "input_dim": 1024, "hidden_dim": 512, "transformer_layers": 2,
            "num_heads": 2, "ffn_dim": 2048, "dropout": 0.15, "layer_norm_eps": 1e-5,
        },
        "training": {
            "epochs": 2, "lr": 1e-4, "batch_size": 1,
            "optimizer": "adam", "random_seed": 1, "save_every": 2,
        },
        "mlflow": {
            "experiment_name": "test-smoke",
            "tracking_uri":    f"sqlite:///{tmp_path}/mlflow.db",
        },
        "evaluation": {"thresholds": [0.5]},
    }

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    return cfg_path, tmp_path


class TestTrainSmoke:

    def test_loss_is_finite(self, synthetic_run, capsys):
        from scripts.train import main
        cfg_path, _ = synthetic_run
        main(str(cfg_path))
        losses = re.findall(r"train_loss=([\d.eE+\-]+)", capsys.readouterr().out)
        assert losses, "No train_loss values found in output"
        for v in losses:
            assert math.isfinite(float(v)), f"Non-finite loss: {v}"

    def test_best_model_saved(self, synthetic_run):
        from scripts.train import main
        cfg_path, tmp_path = synthetic_run
        main(str(cfg_path))
        assert (tmp_path / "models" / "best_model.pt").exists()

    def test_mlflow_run_created(self, synthetic_run):
        import mlflow
        cfg_path, tmp_path = synthetic_run
        from scripts.train import main
        main(str(cfg_path))
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
        runs = mlflow.search_runs(experiment_names=["test-smoke"])
        assert len(runs) >= 1
