"""End-to-end smoke test."""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil

from faircare.config import ExperimentConfig
from faircare.core.trainer import run_experiment


class TestEndToEnd:
    
    def test_smoke_fedavg(self):
        """Smoke test for FedAvg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig()
            config.name = "smoke_test"
            config.logdir = tmpdir
            config.seed = 42
            
            # Small scale for testing
            config.data.dataset = "synth_health"
            config.data.n_clients = 3
            config.model.input_dim = 20
            config.model.hidden_dims = [16, 8]
            config.training.algo = "fedavg"
            config.training.rounds = 2
            config.training.local_epochs = 1
            config.training.eval_every = 1
            
            # Run experiment
            results = run_experiment(config)
            
            # Check results structure
            assert "config" in results
            assert "history" in results
            assert "final_metrics" in results
            
            # Check metrics exist
            assert "final_accuracy" in results["final_metrics"]
            
            # Check files created
            log_path = Path(tmpdir)
            assert (log_path / "config.json").exists()
            assert (log_path / "metrics.jsonl").exists()
            assert (log_path / "results.json").exists()
    
    def test_smoke_faircare(self):
        """Smoke test for FairCare-FL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig()
            config.name = "smoke_faircare"
            config.logdir = tmpdir
            config.seed = 42
            
            # Small scale for testing
            config.data.dataset = "synth_health"
            config.data.n_clients = 3
            config.model.input_dim = 20
            config.model.hidden_dims = [16, 8]
            config.training.algo = "faircare_fl"
            config.training.rounds = 2
            config.training.local_epochs = 1
            config.training.eval_every = 1
            
            # Fairness parameters
            config.fairness.alpha = 1.0
            config.fairness.tau = 1.0
            config.fairness.mu = 0.9
            
            # Run experiment
            results = run_experiment(config)
            
            # Check fairness metrics exist
            final_metrics = results["final_metrics"]
            assert "final_EO_gap" in final_metrics
            assert "final_FPR_gap" in final_metrics
            assert "final_SP_gap" in final_metrics
            assert "final_worst_group_F1" in final_metrics
    
    def test_smoke_all_algorithms(self):
        """Quick smoke test for all algorithms."""
        algos = ["fedavg", "fedprox", "qffl", "afl", "fairfate", "faircare_fl"]
        
        for algo in algos:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = ExperimentConfig()
                config.name = f"smoke_{algo}"
                config.logdir = tmpdir
                config.seed = 42
                
                # Minimal config
                config.data.dataset = "synth_health"
                config.data.n_clients = 2
                config.model.input_dim = 20
                config.model.hidden_dims = [8]
                config.training.algo = algo
                config.training.rounds = 1
                config.training.local_epochs = 1
                config.training.eval_every = 1
                
                # Run experiment
                try:
                    results = run_experiment(config)
                    assert "final_metrics" in results
                except Exception as e:
                    pytest.fail(f"Algorithm {algo} failed: {e}")
