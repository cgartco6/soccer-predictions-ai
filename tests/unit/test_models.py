import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import Mock, patch

from src.models.neural_networks.transformer_model import SoccerTransformer
from src.models.neural_networks.lstm_model import LSTMPredictor
from src.models.statistical_models.poisson_model import PoissonPredictor
from src.models.statistical_models.elo_system import ELOPredictor
from src.models.feature_engineering import FeatureEngineer

class TestSoccerTransformer:
    """Test cases for Soccer Transformer model"""
    
    @pytest.fixture
    def transformer_config(self):
        return {
            'team_embedding_dim': 64,
            'player_embedding_dim': 32,
            'context_dim': 128,
            'num_teams': 100,
            'num_formations': 20,
            'num_leagues': 10,
            'formation_dim': 16,
            'league_dim': 8,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def transformer(self, transformer_config):
        return SoccerTransformer(transformer_config)
    
    def test_forward_pass(self, transformer):
        """Test transformer forward pass"""
        batch_size = 2
        seq_len = 10
        
        batch = {
            'team1_ids': torch.randint(0, 100, (batch_size,)),
            'team2_ids': torch.randint(0, 100, (batch_size,)),
            'formation1_ids': torch.randint(0, 20, (batch_size,)),
            'formation2_ids': torch.randint(0, 20, (batch_size,)),
            'league_ids': torch.randint(0, 10, (batch_size,)),
            'context_features': torch.randn(batch_size, 128),
            'historical_features': torch.randn(batch_size, seq_len, 50)
        }
        
        output = transformer(batch)
        
        assert 'score_probabilities' in output
        assert 'goal_predictions' in output
        assert 'attention_weights' in output
        
        probs = output['score_probabilities']
        goals = output['goal_predictions']
        
        assert probs.shape == (batch_size, 3)
        assert goals.shape == (batch_size, 2)
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones(batch_size))
        assert torch.all(goals >= 0)

class TestLSTMPredictor:
    """Test cases for LSTM Predictor"""
    
    @pytest.fixture
    def lstm_config(self):
        return {
            'input_size': 100,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True,
            'attention_heads': 4,
            'num_teams': 50,
            'team_embedding_dim': 32,
            'num_formations': 20,
            'formation_dim': 16,
            'static_feature_dim': 64
        }
    
    @pytest.fixture
    def lstm_predictor(self, lstm_config):
        return LSTMPredictor(lstm_config)
    
    def test_forward_pass(self, lstm_predictor):
        """Test LSTM forward pass"""
        batch_size = 4
        seq_len = 5
        
        batch = {
            'sequence_data': torch.randn(batch_size, seq_len, 100),
            'static_features': torch.randn(batch_size, 64)
        }
        
        output = lstm_predictor(batch)
        
        assert 'score_probabilities' in output
        assert 'goal_predictions' in output
        assert 'attention_weights' in output
        
        probs = output['score_probabilities']
        goals = output['goal_predictions']
        
        assert probs.shape == (batch_size, 3)
        assert goals.shape == (batch_size, 2)
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones(batch_size))

class TestPoissonPredictor:
    """Test cases for Poisson Predictor"""
    
    @pytest.fixture
    def poisson_predictor(self):
        config = {
            'home_advantage': 0.3,
            'alpha': 1.0
        }
        return PoissonPredictor(config)
    
    def test_goal_expectancy_calculation(self, poisson_predictor):
        """Test goal expectancy calculation"""
        match = {
            'home_team': 'Team A',
            'away_team': 'Team B',
            'home_team_strength': 1.2,
            'away_team_strength': 1.0,
            'home_form': 1.1,
            'away_form': 0.9
        }
        
        # Mock team strengths
        poisson_predictor.home_attack = {'Team A': 1.2}
        poisson_predictor.away_defense = {'Team B': 1.0}
        poisson_predictor.away_attack = {'Team B': 1.0}
        poisson_predictor.home_defense = {'Team A': 1.0}
        
        lambda_home = poisson_predictor._calculate_expected_goals('Team A', 'Team B', 'home')
        lambda_away = poisson_predictor._calculate_expected_goals('Team A', 'Team B', 'away')
        
        assert lambda_home > 0
        assert lambda_away > 0
        assert lambda_home > lambda_away  # Home advantage
    
    def test_prediction(self, poisson_predictor):
        """Test Poisson prediction"""
        match = {
            'home_team': 'Team A',
            'away_team': 'Team B'
        }
        
        # Mock team strengths
        poisson_predictor.home_attack = {'Team A': 1.2}
        poisson_predictor.away_defense = {'Team B': 1.0}
        poisson_predictor.away_attack = {'Team B': 1.0}
        poisson_predictor.home_defense = {'Team A': 1.0}
        
        prediction = poisson_predictor.predict(match)
        
        assert 'expected_home_goals' in prediction
        assert 'expected_away_goals' in prediction
        assert 'home_win_prob' in prediction
        assert 'draw_prob' in prediction
        assert 'away_win_prob' in prediction
        
        # Probabilities should sum to approximately 1
        total_prob = (prediction['home_win_prob'] + 
                     prediction['draw_prob'] + 
                     prediction['away_win_prob'])
        assert abs(total_prob - 1.0) < 0.01

class TestELOPredictor:
    """Test cases for ELO Predictor"""
    
    @pytest.fixture
    def elo_predictor(self):
        config = {
            'k_factor': 20,
            'home_advantage': 100,
            'goal_difference_factor': 0.1
        }
        return ELOPredictor(config)
    
    def test_rating_update(self, elo_predictor):
        """Test ELO rating updates"""
        matches = pd.DataFrame({
            'home_team': ['Team A', 'Team B'],
            'away_team': ['Team B', 'Team A'],
            'home_goals': [2, 1],
            'away_goals': [1, 2],
            'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
        })
        
        elo_predictor.update_ratings(matches)
        
        assert 'Team A' in elo_predictor.ratings
        assert 'Team B' in elo_predictor.ratings
        
        # Team A should have higher rating (won first match)
        assert elo_predictor.ratings['Team A'] > elo_predictor.ratings['Team B']
    
    def test_prediction(self, elo_predictor):
        """Test ELO prediction"""
        # Set up ratings
        elo_predictor.ratings = {
            'Team A': 1600,
            'Team B': 1400
        }
        
        match = {
            'home_team': 'Team A',
            'away_team': 'Team B'
        }
        
        prediction = elo_predictor.predict(match)
        
        assert 'home_win_prob' in prediction
        assert 'draw_prob' in prediction
        assert 'away_win_prob' in prediction
        assert 'home_rating' in prediction
        assert 'away_rating' in prediction
        
        # Team A should have higher win probability
        assert prediction['home_win_prob'] > prediction['away_win_prob']

class TestFeatureEngineer:
    """Test cases for Feature Engineer"""
    
    @pytest.fixture
    def feature_engineer(self):
        config = {}
        return FeatureEngineer(config)
    
    @pytest.fixture
    def sample_matches(self):
        return pd.DataFrame({
            'home_team': ['Team A', 'Team B', 'Team A', 'Team C'],
            'away_team': ['Team B', 'Team C', 'Team C', 'Team B'],
            'home_goals': [2, 1, 3, 0],
            'away_goals': [1, 2, 1, 2],
            'league': ['Premier League'] * 4,
            'date': pd.date_range('2024-01-01', periods=4),
            'attendance': [40000, 35000, 42000, 38000],
            'home_shots': [15, 12, 18, 8],
            'away_shots': [10, 14, 9, 16]
        })
    
    def test_feature_creation(self, feature_engineer, sample_matches):
        """Test feature creation"""
        features = feature_engineer.create_features(
            sample_matches, 
            feature_types=['basic', 'team_strength', 'form']
        )
        
        assert not features.empty
        assert len(features) == len(sample_matches)
        
        # Check for expected features
        expected_features = [
            'league_encoded', 'day_of_week', 'month', 
            'home_elo', 'away_elo', 'elo_difference',
            'home_recent_form', 'away_recent_form'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_team_strength_features(self, feature_engineer, sample_matches):
        """Test team strength feature calculation"""
        features = feature_engineer._create_team_strength_features(
            pd.DataFrame(index=sample_matches.index),
            sample_matches
        )
        
        assert 'home_elo' in features.columns
        assert 'away_elo' in features.columns
        assert 'elo_difference' in features.columns
        
        # ELO ratings should be calculated for all teams
        assert features['home_elo'].notna().all()
        assert features['away_elo'].notna().all()
    
    def test_form_features(self, feature_engineer, sample_matches):
        """Test form feature calculation"""
        features = feature_engineer._create_form_features(
            pd.DataFrame(index=sample_matches.index),
            sample_matches
        )
        
        assert 'home_recent_form' in features.columns
        assert 'away_recent_form' in features.columns
        assert 'home_goals_scored_form' in features.columns
        assert 'away_goals_scored_form' in features.columns
        
        # Form features should be between 0 and 1
        assert (features['home_recent_form'] >= 0).all()
        assert (features['home_recent_form'] <= 1).all()
