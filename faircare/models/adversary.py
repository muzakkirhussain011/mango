"""Adversarial debiaser model (optional)."""
from typing import List

import torch
import torch.nn as nn
from typing import List


class AdversarialDebiaser(nn.Module):
    """
    Adversarial debiasing model.
    
    Consists of a predictor and an adversary that tries to predict
    the sensitive attribute from the predictor's representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        adversary_dims: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if adversary_dims is None:
            adversary_dims = [32]
        
        # Build predictor
        predictor_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            predictor_layers.append(nn.Linear(prev_dim, hidden_dim))
            predictor_layers.append(nn.ReLU())
            if dropout > 0:
                predictor_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
            # Store representation dimension (last hidden layer)
            if i == len(hidden_dims) - 1:
                self.repr_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*predictor_layers)
        self.predictor = nn.Linear(self.repr_dim, output_dim)
        
        # Build adversary
        adversary_layers = []
        prev_dim = self.repr_dim
        
        for adv_dim in adversary_dims:
            adversary_layers.append(nn.Linear(prev_dim, adv_dim))
            adversary_layers.append(nn.ReLU())
            if dropout > 0:
                adversary_layers.append(nn.Dropout(dropout))
            prev_dim = adv_dim
        
        adversary_layers.append(nn.Linear(prev_dim, 1))  # Binary sensitive attribute
        self.adversary = nn.Sequential(*adversary_layers)
        
        # Gradient reversal layer
        self.grad_reverse = GradientReversal()
    
    def forward(self, x, alpha=1.0):
        """
        Forward pass.
        
        Args:
            x: Input features
            alpha: Gradient reversal strength
        
        Returns:
            predictions, adversary_predictions
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Main prediction
        y_pred = self.predictor(features)
        
        # Adversary prediction (with gradient reversal)
        reversed_features = self.grad_reverse(features, alpha)
        a_pred = self.adversary(reversed_features)
        
        return y_pred, a_pred
    
    def predict(self, x):
        """Get predictions without adversary."""
        features = self.feature_extractor(x)
        return self.predictor(features)


class GradientReversal(nn.Module):
    """Gradient reversal layer."""
    
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal function."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
