import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

class NeuralEnsemble(nn.Module):
    """
    Neural ensemble model that combines multiple architectures
    with learnable weighting
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize member models
        self.member_models = nn.ModuleDict({
            'transformer': SoccerTransformer(config['transformer']),
            'lstm': LSTMPredictor(config['lstm']),
            'simple_nn': SimpleNeuralNetwork(config['simple_nn'])
        })
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(self.member_models)) / len(self.member_models)
        )
        
        # Meta-learner for combining predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(
                len(self.member_models) * 3,  # 3 outcomes per model
                config['meta_learner']['hidden_size']
            ),
            nn.ReLU(),
            nn.Dropout(config['meta_learner']['dropout']),
            nn.Linear(
                config['meta_learner']['hidden_size'],
                config['meta_learner']['hidden_size'] // 2
            ),
            nn.ReLU(),
            nn.Linear(config['meta_learner']['hidden_size'] // 2, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        member_predictions = {}
        member_features = []
        
        # Get predictions from all member models
        for name, model in self.member_models.items():
            prediction = model(batch[name + '_features'])
            member_predictions[name] = prediction
            member_features.append(prediction['score_probabilities'])
        
        # Stack predictions for meta-learning
        stacked_predictions = torch.stack(member_features, dim=1)
        batch_size, num_models, num_classes = stacked_predictions.shape
        
        # Apply learnable weights
        weighted_predictions = stacked_predictions * self.ensemble_weights.view(1, -1, 1)
        
        # Simple weighted average
        simple_ensemble = weighted_predictions.sum(dim=1)
        
        # Meta-learner combination
        flattened_predictions = stacked_predictions.view(batch_size, -1)
        meta_ensemble = self.meta_learner(flattened_predictions)
        
        # Combine simple and meta ensembles
        final_ensemble = 0.7 * simple_ensemble + 0.3 * meta_ensemble
        
        return {
            'score_probabilities': final_ensemble,
            'member_predictions': member_predictions,
            'ensemble_weights': self.ensemble_weights,
            'simple_ensemble': simple_ensemble,
            'meta_ensemble': meta_ensemble
        }
    
    def get_member_contributions(self) -> Dict[str, float]:
        """Get contribution of each member model"""
        weights = self.ensemble_weights.detach().cpu().numpy()
        contributions = {}
        
        for i, name in enumerate(self.member_models.keys()):
            contributions[name] = float(weights[i])
            
        return contributions

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network as ensemble member"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        layers = []
        input_size = config['input_size']
        
        for hidden_size in config['hidden_sizes']:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 3))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = self.network(x)
        
        return {
            'score_probabilities': probabilities
        }
