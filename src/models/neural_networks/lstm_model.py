import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class LSTMPredictor(nn.Module):
    """
    LSTM-based model for soccer predictions with sequence modeling
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # LSTM layers for temporal sequences
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            bidirectional=config['bidirectional']
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config['hidden_size'] * (2 if config['bidirectional'] else 1),
            num_heads=config['attention_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Feature embeddings
        self.team_embedding = nn.Embedding(config['num_teams'], config['team_embedding_dim'])
        self.formation_embedding = nn.Embedding(config['num_formations'], config['formation_dim'])
        
        # Static feature processing
        self.static_processor = nn.Sequential(
            nn.Linear(config['static_feature_dim'], config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], config['hidden_size'] // 2)
        )
        
        # Output heads
        lstm_output_size = config['hidden_size'] * (2 if config['bidirectional'] else 1)
        combined_size = lstm_output_size + (config['hidden_size'] // 2)
        
        self.score_predictor = nn.Sequential(
            nn.Linear(combined_size, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], 3),
            nn.Softmax(dim=-1)
        )
        
        self.goal_predictor = nn.Sequential(
            nn.Linear(combined_size, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], 2),
            nn.ReLU()
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Process sequence data with LSTM
        sequence_output, (hidden, cell) = self.lstm(batch['sequence_data'])
        
        # Apply attention to LSTM outputs
        attended_output, attention_weights = self.attention(
            sequence_output, sequence_output, sequence_output
        )
        
        # Use last hidden state and attended output
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        # Combine with static features
        static_features = self.static_processor(batch['static_features'])
        combined_features = torch.cat([hidden, static_features], dim=-1)
        
        # Generate predictions
        score_probs = self.score_predictor(combined_features)
        goal_preds = self.goal_predictor(combined_features)
        
        return {
            'score_probabilities': score_probs,
            'goal_predictions': goal_preds,
            'attention_weights': attention_weights,
            'sequence_representation': attended_output
        }
