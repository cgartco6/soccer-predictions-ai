import torch
import torch.nn as nn
import torch.nn.functional as F

class SoccerTransformer(nn.Module):
    """
    Transformer-based model for soccer predictions with multi-head attention
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.team_embedding_dim = config['team_embedding_dim']
        self.player_embedding_dim = config['player_embedding_dim']
        self.context_dim = config['context_dim']
        
        # Embedding layers
        self.team_embedding = nn.Embedding(
            config['num_teams'], self.team_embedding_dim
        )
        self.formation_embedding = nn.Embedding(
            config['num_formations'], config['formation_dim']
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['ff_dim'],
            dropout=config['dropout']
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config['num_layers']
        )
        
        # Attention mechanisms
        self.temporal_attention = TemporalAttention(config)
        self.feature_attention = FeatureAttention(config)
        
        # Output heads
        self.score_predictor = nn.Linear(config['hidden_dim'], 3)  # Home, Draw, Away
        self.goal_predictor = nn.Linear(config['hidden_dim'], 2)   # Home goals, Away goals
        
    def forward(self, batch: Dict) -> Dict:
        # Embed teams and formations
        team1_emb = self.team_embedding(batch['team1_ids'])
        team2_emb = self.team_embedding(batch['team2_ids'])
        
        # Combine features
        features = torch.cat([
            team1_emb, team2_emb,
            batch['formation_features'],
            batch['context_features']
        ], dim=-1)
        
        # Apply transformer with attention
        encoded = self.transformer(features)
        
        # Apply specialized attention mechanisms
        temporal_features = self.temporal_attention(encoded, batch['sequence_mask'])
        attended_features = self.feature_attention(temporal_features)
        
        # Generate predictions
        score_probs = F.softmax(self.score_predictor(attended_features), dim=-1)
        goal_preds = F.relu(self.goal_predictor(attended_features))
        
        return {
            'score_probabilities': score_probs,
            'goal_predictions': goal_preds,
            'attention_weights': attended_features.attn_weights
        }
