import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(output)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self attention
        src2, attention_weights = self.self_attention(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        
        return src, attention_weights

class SoccerTransformer(nn.Module):
    """
    Transformer-based model for soccer predictions with multi-head attention
    and temporal modeling
    """
    
    def __init__(self, config: Dict[str, Any]):
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
        self.league_embedding = nn.Embedding(
            config['num_leagues'], config['league_dim']
        )
        
        # Input projection
        self.input_projection = nn.Linear(
            self.team_embedding_dim * 2 + config['formation_dim'] * 2 + 
            config['league_dim'] + self.context_dim,
            config['hidden_dim']
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config['hidden_dim'])
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config['hidden_dim'],
                config['num_heads'],
                config['dim_feedforward'],
                config['dropout']
            ) for _ in range(config['num_layers'])
        ])
        
        # Output heads
        self.score_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, 3),  # Home, Draw, Away
            nn.Softmax(dim=-1)
        )
        
        self.goals_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, 2),  # Home goals, Away goals
            nn.ReLU()  # Goals can't be negative
        )
        
        self.attention_weights = None
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Embed categorical features
        team1_emb = self.team_embedding(batch['team1_ids'])
        team2_emb = self.team_embedding(batch['team2_ids'])
        formation1_emb = self.formation_embedding(batch['formation1_ids'])
        formation2_emb = self.formation_embedding(batch['formation2_ids'])
        league_emb = self.league_embedding(batch['league_ids'])
        
        # Combine all features
        features = torch.cat([
            team1_emb, team2_emb,
            formation1_emb, formation2_emb,
            league_emb,
            batch['context_features'],
            batch['historical_features']
        ], dim=-1)
        
        # Project to hidden dimension
        x = self.input_projection(features)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, batch.get('attention_mask', None))
            attention_weights.append(attn_weights)
        
        self.attention_weights = attention_weights
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Generate predictions
        score_probs = self.score_head(x)
        goal_preds = self.goals_head(x)
        
        return {
            'score_probabilities': score_probs,
            'goal_predictions': goal_preds,
            'attention_weights': attention_weights
        }
    
    def get_attention_scores(self, team1: str, team2: str) -> Dict[str, float]:
        """Get attention scores for model interpretability"""
        if self.attention_weights is None:
            return {}
        
        # Analyze which features the model is focusing on
        last_layer_weights = self.attention_weights[-1]
        avg_attention = last_layer_weights.mean(dim=1)  # Average over heads
        
        feature_importance = {
            'team_strength': float(avg_attention[0, 1].mean()),
            'recent_form': float(avg_attention[0, 2].mean()),
            'head_to_head': float(avg_attention[0, 3].mean()),
            'player_performance': float(avg_attention[0, 4].mean()),
            'tactical_matchup': float(avg_attention[0, 5].mean())
        }
        
        return feature_importance
