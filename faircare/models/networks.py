# ============================================================================
# faircare/models/networks.py
# ============================================================================

import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for tabular data."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    """Convolutional neural network for image data."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create a model based on type.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', etc.)
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    if model_type == 'mlp':
        return MLP(
            input_dim=kwargs.get('input_dim', 100),
            hidden_dims=kwargs.get('hidden_dims', [256, 128]),
            output_dim=kwargs.get('output_dim', 2)
        )
    elif model_type == 'cnn':
        return CNN(num_classes=kwargs.get('num_classes', 2))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
