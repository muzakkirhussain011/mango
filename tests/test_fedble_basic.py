# tests/test_fedble_basic.py
"""Basic smoke tests for FedBLE algorithm."""
import pytest
import torch
import numpy as np
from faircare.algos import make_aggregator
from faircare.fairness.detector import BiasDetector, BiasState
from faircare.fairness.mitigation import MitigationPolicy
from faircare.fairness.losses import (
    combined_fairness_loss,
    AdversaryNetwork,
    gradient_reversal
)
from faircare.data.synth_health import generate_synthetic_health
from faircare.models.classifier import create_model
from faircare.core.client import Client
from torch.utils.data import TensorDataset


class TestFedBLE:
    """Test suite for FedBLE components."""
    
    def test_aggregator_initialization(self):
        """Test FedBLE aggregator initialization."""
        aggregator = make_aggregator(
            "faircare_fl",  # FedBLE uses faircare_fl name
            n_clients=5,
            gate_mode="heuristic",
            lambda_fair=0.1,
            tau=1.0
        )
        
        assert aggregator is not None
        assert aggregator.n_clients == 5
        assert aggregator.gate_mode == "heuristic"
        assert aggregator.lambda_fair == 0.1
    
    def test_component_weights_computation(self):
        """Test computation of component algorithm weights."""
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=3,
            gate_mode="heuristic"
        )
        
        # Create mock client summaries
        client_summaries = [
            {
                "client_id": 0,
                "n_samples": 100,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "eo_gap": 0.1,
                "fpr_gap": 0.05,
                "sp_gap": 0.08,
                "worst_group_f1": 0.7
            },
            {
                "client_id": 1,
                "n_samples": 200,
                "train_loss": 0.6,
                "val_loss": 0.5,
                "eo_gap": 0.2,
                "fpr_gap": 0.15,
                "sp_gap": 0.12,
                "worst_group_f1": 0.6
            },
            {
                "client_id": 2,
                "n_samples": 150,
                "train_loss": 0.4,
                "val_loss": 0.35,
                "eo_gap": 0.05,
                "fpr_gap": 0.03,
                "sp_gap": 0.04,
                "worst_group_f1": 0.8
            }
        ]
        
        # Compute component weights
        component_weights = aggregator._compute_component_weights(client_summaries)
        
        # Check all components computed
        assert component_weights.fedavg is not None
        assert component_weights.fedprox is not None
        assert component_weights.qffl is not None
        assert component_weights.afl is not None
        assert component_weights.fairfate is not None
        
        # Check weights sum to 1
        assert torch.allclose(component_weights.fedavg.sum(), torch.tensor(1.0))
        assert torch.allclose(component_weights.qffl.sum(), torch.tensor(1.0))
    
    def test_ensemble_weights(self):
        """Test ensemble weight computation."""
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=3,
            gate_mode="heuristic"
        )
        
        client_summaries = [
            {"client_id": i, "n_samples": 100, "eo_gap": 0.1 * i, "val_loss": 0.5}
            for i in range(3)
        ]
        
        # Compute weights
        weights = aggregator.compute_weights(client_summaries)
        
        # Check basic properties
        assert len(weights) == 3
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
        assert (weights >= 0).all()
        assert (weights <= 1).all()
    
    def test_bias_detector(self):
        """Test bias detection functionality."""
        detector = BiasDetector(
            thresholds={
                'eo_gap': 0.15,
                'fpr_gap': 0.15,
                'sp_gap': 0.10,
                'worst_group_f1_min': 0.6
            },
            patience=2,
            hysteresis=0.02
        )
        
        # Test with low bias
        state1 = detector.update({
            'eo_gap': 0.05,
            'fpr_gap': 0.05,
            'sp_gap': 0.05,
            'worst_group_f1': 0.8
        })
        assert not state1.is_biased
        assert len(state1.triggered_metrics) == 0
        
        # Test with high bias (but need patience)
        state2 = detector.update({
            'eo_gap': 0.2,  # Above threshold
            'fpr_gap': 0.05,
            'sp_gap': 0.05,
            'worst_group_f1': 0.8
        })
        assert not state2.is_biased  # Not yet, need patience
        assert 'eo_gap' in state2.triggered_metrics
        
        # Test with sustained bias
        state3 = detector.update({
            'eo_gap': 0.2,
            'fpr_gap': 0.2,  # Also above threshold
            'sp_gap': 0.05,
            'worst_group_f1': 0.8
        })
        assert state3.is_biased  # Now biased after patience
        assert 'eo_gap' in state3.triggered_metrics
        assert 'fpr_gap' in state3.triggered_metrics
    
    def test_mitigation_policy(self):
        """Test mitigation policy actions."""
        policy = MitigationPolicy(
            lambda_fair_init=0.1,
            lambda_fair_max=2.0,
            delta_acc_init=0.2,
            tau_init=1.0
        )
        
        # Create bias state
        bias_state = BiasState(
            is_biased=True,
            triggered_metrics=['eo_gap', 'fpr_gap'],
            eo_gap=0.2,
            fpr_gap=0.18,
            sp_gap=0.08,
            worst_group_f1=0.55,
            trend="increasing",
            confidence=0.8
        )
        
        # Get mitigation action
        action = policy.compute_action(bias_state)
        
        # Check action properties
        assert action.lambda_fair > 0.1  # Should increase
        assert action.delta_acc < 0.2  # Should decrease
        assert action.tau < 1.0  # Should decrease
        assert action.w_eo > 0  # Should be positive
        
        # Test with resolved bias
        no_bias_state = BiasState(
            is_biased=False,
            triggered_metrics=[],
            eo_gap=0.05,
            fpr_gap=0.05,
            sp_gap=0.05,
            worst_group_f1=0.8,
            trend="stable",
            confidence=0.9
        )
        
        action2 = policy.compute_action(no_bias_state)
        assert action2.lambda_fair < action.lambda_fair  # Should decrease
    
    def test_fairness_losses(self):
        """Test differentiable fairness loss functions."""
        batch_size = 32
        n_features = 10
        
        # Create dummy data
        logits = torch.randn(batch_size, 1)
        labels = torch.randint(0, 2, (batch_size,))
        sensitive = torch.randint(0, 2, (batch_size,))
        
        # Test combined fairness loss
        loss = combined_fairness_loss(
            logits,
            labels,
            sensitive,
            w_eo=1.0,
            w_fpr=0.5,
            w_sp=0.5
        )
        
        assert loss is not None
        assert loss.requires_grad
        assert loss.item() >= 0
        
        # Test with no sensitive attribute
        loss_no_sensitive = combined_fairness_loss(
            logits,
            labels,
            None
        )
        assert loss_no_sensitive.item() == 0.0
    
    def test_adversary_network(self):
        """Test adversary network for debiasing."""
        input_dim = 1
        adversary = AdversaryNetwork(
            input_dim=input_dim,
            hidden_dims=[16, 8],
            n_sensitive_classes=2
        )
        
        # Test forward pass
        batch_size = 16
        x = torch.randn(batch_size, input_dim)
        output = adversary(x)
        
        assert output.shape == (batch_size, 2)
        
        # Test gradient reversal
        x = torch.randn(batch_size, input_dim, requires_grad=True)  # Add requires_grad=True
        x_rev = gradient_reversal(x, alpha=1.0)
        assert x_rev.shape == x.shape
        
        # Check gradients are reversed
        loss = x_rev.sum()
        loss.backward()
        assert torch.allclose(x.grad, -torch.ones_like(x))
    
    def test_fairness_config_broadcast(self):
        """Test fairness configuration broadcasting."""
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=3,
            lambda_fair=0.2,
            use_adversary=True,
            bias_threshold_eo=0.1
        )
        
        # Get fairness config
        config = aggregator.get_fairness_config()
        
        # Check required fields
        assert 'lambda_fair' in config
        assert config['lambda_fair'] == 0.2
        assert 'use_adversary' in config
        assert config['use_adversary'] == True
        assert 'w_eo' in config
        assert 'w_fpr' in config
        assert 'w_sp' in config
        assert 'bias_mitigation_mode' in config
    
    def test_client_training_with_fairness(self):
        """Test client training with fairness components."""
        # Create synthetic data
        n_samples = 100
        n_features = 10
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, 2, (n_samples,))
        a = torch.randint(0, 2, (n_samples,))
        
        dataset = TensorDataset(X, y, a)
        
        # Create model and client
        model = create_model(
            model_type="mlp",
            input_dim=n_features,
            hidden_dims=[16, 8],
            output_dim=1
        )
        
        client = Client(
            client_id=0,
            model=model,
            train_dataset=dataset,
            batch_size=32,
            device="cpu"
        )
        
        # Create fairness config
        fairness_config = {
            'lambda_fair': 0.1,
            'use_adversary': True,
            'w_eo': 1.0,
            'w_fpr': 0.5,
            'w_sp': 0.5,
            'bias_mitigation_mode': True,
            'extra_epoch': True
        }
        
        # Train
        global_weights = model.state_dict()
        delta, n_samples_trained, stats = client.train(
            global_weights=global_weights,
            epochs=1,
            lr=0.01,
            fairness_config=fairness_config
        )
        
        # Check outputs
        assert delta is not None
        assert n_samples_trained == n_samples
        assert 'train_loss' in stats
        assert 'fairness_loss' in stats
        assert 'adversary_loss' in stats
        assert stats['adversary_loss'] >= 0  # Should be computed
    
    def test_single_round_smoke(self):
        """Smoke test for single training round."""
        # Create minimal dataset
        dataset = generate_synthetic_health(
            n_samples=200,
            n_features=10,
            bias_level=0.3,
            seed=42
        )
        
        # Create aggregator
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=2,
            gate_mode="heuristic",
            lambda_fair=0.1
        )
        
        # Create mock client summaries
        client_summaries = [
            {
                "client_id": 0,
                "n_samples": 100,
                "train_loss": 0.5,
                "eo_gap": 0.15,
                "fpr_gap": 0.1,
                "sp_gap": 0.08,
                "worst_group_f1": 0.65
            },
            {
                "client_id": 1,
                "n_samples": 100,
                "train_loss": 0.6,
                "eo_gap": 0.2,
                "fpr_gap": 0.15,
                "sp_gap": 0.12,
                "worst_group_f1": 0.6
            }
        ]
        
        # Test weight computation doesn't crash
        weights = aggregator.compute_weights(client_summaries)
        
        # Basic checks
        assert weights is not None
        assert len(weights) == 2
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert (weights >= 0).all()
        
        # Check statistics
        stats = aggregator.get_statistics()
        assert 'round' in stats
        assert stats['round'] == 1
        assert 'bias_mitigation_mode' in stats
        assert 'lambda_fair' in stats
    
    def test_learned_gating(self):
        """Test learned gating mode."""
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=3,
            gate_mode="learned"
        )
        
        # Check gate network created
        assert hasattr(aggregator, 'gate_network')
        assert aggregator.gate_network is not None
        
        # Create client summaries with enough rounds for training
        for round_num in range(10):
            client_summaries = [
                {
                    "client_id": i,
                    "n_samples": 100,
                    "train_loss": 0.5 + 0.1 * i,
                    "val_loss": 0.4 + 0.1 * i,
                    "eo_gap": 0.1 * (i + 1),
                    "fpr_gap": 0.05 * (i + 1),
                    "sp_gap": 0.08 * (i + 1),
                    "worst_group_f1": 0.7 - 0.05 * i,
                    "delta_norm": 1.0,
                    "grad_norm": 0.5
                }
                for i in range(3)
            ]
            
            weights = aggregator.compute_weights(client_summaries)
            
            # Should not crash and produce valid weights
            assert len(weights) == 3
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_detector_integration(self):
        """Test detector integration with aggregator."""
        from faircare.fairness.detector import BiasDetector
        
        detector = BiasDetector(
            thresholds={'eo_gap': 0.15, 'fpr_gap': 0.15, 'sp_gap': 0.10},
            patience=1
        )
        
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=3,
            detector_patience=1
        )
        
        # Simulate rounds with increasing bias
        for bias_level in [0.05, 0.1, 0.2, 0.25]:
            global_metrics = {
                'eo_gap': bias_level,
                'fpr_gap': bias_level * 0.8,
                'sp_gap': bias_level * 0.6,
                'worst_group_f1': max(0.5, 0.8 - bias_level)
            }
            
            bias_state = detector.update(global_metrics)
            
            # Create client summaries
            client_summaries = [
                {
                    "client_id": i,
                    "n_samples": 100,
                    "eo_gap": bias_level + 0.02 * i,
                    "fpr_gap": bias_level * 0.8,
                    "sp_gap": bias_level * 0.6,
                    "worst_group_f1": max(0.5, 0.8 - bias_level)
                }
                for i in range(3)
            ]
            
            weights = aggregator.compute_weights(client_summaries)
            
            # Check aggregator responds to bias
            if bias_state.is_biased:
                assert aggregator.bias_mitigation_mode
                assert aggregator.lambda_fair > aggregator.lambda_fair_init

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
