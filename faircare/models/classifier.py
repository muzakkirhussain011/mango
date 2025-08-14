"""Neural network classifiers."""
from typing import List, Optional

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """Multi-layer perceptron classifier."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class LogisticRegression(nn.Module):
    """Logistic regression classifier."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dims: Optional[List[int]] = None,
    output_dim: int = 1,
    dropout: float = 0.2,
    activation: str = "relu"
) -> nn.Module:
    """Create a classifier model."""
    if model_type == "mlp":
        if hidden_dims is None:
            hidden_dims = [64, 32]
        return MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation
        )
    elif model_type == "logistic":
        return LogisticRegression(
            input_dim=input_dim,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
