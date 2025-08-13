"""Main training orchestrator."""

import torch
from typing import Dict, Any, Optional, List
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
    """Run a federated learning experiment."""
    # Set seed
    set_seed(config.seed)
    
    # Setup logger
    logger = Logger(config.logdir, config.name)
    logger.info(f"Starting experiment: {config.name}")
    logger.log_config(config.to_dict())
    
    # Load data
    from faircare.data import load_dataset
    dataset = load_dataset(
        config.data.dataset,
        sensitive_attribute=config.data.sensitive_attribute
    )
    
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
    
    # Create aggregator
    aggregator = make_aggregator(
        config.training.algo,
        n_clients=config.data.n_clients,
        fairness_config=config.fairness,
        algo_config=config.algo
    )
    
    # Create server
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
    
    # Save results
    results = {
        "config": config.to_dict(),
        "history": history,
        "final_metrics": final_metrics
    }
    
    results_path = Path(config.logdir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Experiment completed")
    logger.info(f"Final metrics: {final_metrics}")
    
    return results
