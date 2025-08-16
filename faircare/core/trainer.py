"""Main training orchestrator with FairCare-FL++ support."""
from typing import Dict, Any, Optional, List
import torch
from pathlib import Path
import json

from faircare.config import ExperimentConfig
from faircare.core.utils import set_seed, Logger
from faircare.core.client import Client
from faircare.core.server import Server
from faircare.data.partition import make_federated_splits
from faircare.models.classifier import create_model
from faircare.algos.aggregator import make_aggregator
from faircare.core.evaluation import Evaluator


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a FairCare-FL++ federated learning experiment."""
    # Set seed
    set_seed(config.seed)
    
    # Setup logger
    logger = Logger(config.logdir, config.name)
    logger.info(f"Starting FairCare-FL++ experiment: {config.name}")
    logger.info(f"Algorithm: {config.training.algo}")
    logger.log_config(config.to_dict())
    
    # Load data
    from faircare.data import load_dataset
    dataset = load_dataset(
        config.data.dataset,
        sensitive_attribute=config.data.sensitive_attribute
    )
    
    # Update model input dimension based on actual dataset
    actual_input_dim = dataset.get("n_features", config.model.input_dim)
    if actual_input_dim != config.model.input_dim:
        logger.info(f"Updating model input dimension from {config.model.input_dim} to {actual_input_dim}")
        config.model.input_dim = actual_input_dim
    
    # Create federated splits
    fed_data = make_federated_splits(
        dataset=dataset,
        n_clients=config.data.n_clients,
        partition=config.data.partition,
        alpha=config.data.alpha,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed
    )
    
    # Create model
    model = create_model(
        model_type=config.model.model_type,
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.output_dim,
        dropout=config.model.dropout
    )
    
    # Log FairCare-FL++ configuration
    if config.training.algo == "faircare_fl":
        logger.info("=" * 60)
        logger.info("FairCare-FL++ Configuration:")
        logger.info(f"  Fairness weights: α={config.fairness.alpha}, β={config.fairness.beta}, γ={config.fairness.gamma}")
        logger.info(f"  Fairness penalty: λ={config.fairness.lambda_fair} (range: {config.fairness.lambda_fair_min}-{config.fairness.lambda_fair_max})")
        logger.info(f"  Temperature: τ={config.fairness.tau} (min: {config.fairness.tau_min})")
        logger.info(f"  Momentum: client={config.fairness.mu_client}, server={config.fairness.theta_server}")
        logger.info(f"  Bias thresholds: EO={config.fairness.bias_threshold_eo}, FPR={config.fairness.bias_threshold_fpr}, SP={config.fairness.bias_threshold_sp}")
        logger.info(f"  Bias detection: {config.fairness.enable_bias_detection}")
        logger.info(f"  Multi-metric fairness: {config.fairness.enable_multi_metric}")
        logger.info("=" * 60)
    
    # Create clients
    clients = []
    for i in range(config.data.n_clients):
        client = Client(
            client_id=i,
            model=create_model(
                model_type=config.model.model_type,
                input_dim=config.model.input_dim,
                hidden_dims=config.model.hidden_dims,
                output_dim=config.model.output_dim,
                dropout=config.model.dropout
            ),
            train_dataset=fed_data["client_data"][i]["train"],
            val_dataset=fed_data["client_data"][i].get("val"),
            batch_size=config.data.batch_size,
            device=config.training.device
        )
        clients.append(client)
    
    # Create aggregator with full fairness configuration
    aggregator = make_aggregator(
        config.training.algo,
        n_clients=config.data.n_clients,
        fairness_config=config.fairness,
        **config.algo.to_dict()
    )
    
    # Create server with enhanced functionality
    server = Server(
        model=model,
        clients=clients,
        aggregator=aggregator,
        val_data=fed_data.get("server_val"),
        test_data=fed_data.get("test"),
        secure_agg_config=config.secure_agg.to_dict() if config.secure_agg else None,
        logger=logger,
        device=config.training.device
    )
    
    # Run training
    logger.info("Starting federated training...")
    history = server.run(
        rounds=config.training.rounds,
        n_clients_per_round=min(
            config.data.n_clients,
            max(1, int(config.data.n_clients * 0.4))  # 40% participation
        ),
        local_epochs=config.training.local_epochs,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        server_lr=config.training.server_lr,
        eval_every=config.training.eval_every,
        checkpoint_every=config.training.checkpoint_every
    )
    
    # Evaluate final model
    evaluator = Evaluator(logger=logger)
    final_metrics = evaluator.evaluate_model(
        model=server.model,
        test_data=fed_data["test"],
        prefix="final"
    )
    
    # Log final performance
    logger.info("=" * 60)
    logger.info("FINAL PERFORMANCE:")
    logger.info(f"  Accuracy: {final_metrics.get('final_accuracy', 0):.4f}")
    logger.info(f"  Worst Group F1: {final_metrics.get('final_worst_group_F1', 0):.4f}")
    logger.info(f"  EO Gap: {final_metrics.get('final_EO_gap', 0):.4f}")
    logger.info(f"  FPR Gap: {final_metrics.get('final_FPR_gap', 0):.4f}")
    logger.info(f"  SP Gap: {final_metrics.get('final_SP_gap', 0):.4f}")
    logger.info(f"  Max Gap: {final_metrics.get('final_max_group_gap', 0):.4f}")
    
    # Check if FairCare-FL++ achieved targets
    if config.training.algo == "faircare_fl":
        logger.info("-" * 60)
        logger.info("TARGET ACHIEVEMENT:")
        
        # Target: Better worst_group_F1 than qFFL (>0.46)
        wgf1 = final_metrics.get('final_worst_group_F1', 0)
        if wgf1 > 0.46:
            logger.info(f"✓ Worst Group F1 ({wgf1:.4f}) > 0.46 target")
        else:
            logger.info(f"✗ Worst Group F1 ({wgf1:.4f}) < 0.46 target")
        
        # Target: Lower EO gap than baseline (<0.20)
        eo_gap = final_metrics.get('final_EO_gap', 1)
        if eo_gap < 0.20:
            logger.info(f"✓ EO Gap ({eo_gap:.4f}) < 0.20 target")
        else:
            logger.info(f"✗ EO Gap ({eo_gap:.4f}) > 0.20 target")
        
        # Target: Maintain reasonable accuracy (>0.65)
        acc = final_metrics.get('final_accuracy', 0)
        if acc > 0.65:
            logger.info(f"✓ Accuracy ({acc:.4f}) > 0.65 target")
        else:
            logger.info(f"✗ Accuracy ({acc:.4f}) < 0.65 target")
    
    logger.info("=" * 60)
    
    # Get aggregator statistics if available
    if hasattr(server.aggregator, 'get_statistics'):
        agg_stats = server.aggregator.get_statistics()
        logger.info(f"Aggregator final statistics: {agg_stats}")
    
    # Save results
    results = {
        "config": config.to_dict(),
        "history": history,
        "final_metrics": final_metrics
    }
    
    # Add aggregator statistics to results
    if hasattr(server.aggregator, 'get_statistics'):
        results["aggregator_statistics"] = server.aggregator.get_statistics()
    
    # Add bias mitigation summary
    if 'bias_mitigation_rounds' in history:
        n_bias_rounds = sum(history['bias_mitigation_rounds'])
        results["bias_mitigation_summary"] = {
            "total_rounds_in_bias_mode": n_bias_rounds,
            "percentage_rounds_in_bias_mode": n_bias_rounds / config.training.rounds * 100
        }
    
    results_path = Path(config.logdir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Experiment completed")
    logger.info(f"Results saved to: {results_path}")
    
    return results
