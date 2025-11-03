import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.agents import (
    SyntheticIntelligenceAgent, 
    StrategicIntelligenceAgent,
    DataCollectorAgent,
    PredictionEnsembleAgent
)

class TestSyntheticIntelligenceAgent:
    """Test cases for Synthetic Intelligence Agent"""
    
    @pytest.fixture
    def synthetic_agent(self):
        config = {
            'gan': {
                'generator': {},
                'discriminator': {},
                'batch_size': 32,
                'latent_dim': 100
            },
            'transformer': {
                'model_name': 'microsoft/deberta-v3-base'
            },
            'simulation': {}
        }
        return SyntheticIntelligenceAgent(config)
    
    def test_initialization(self, synthetic_agent):
        """Test agent initialization"""
        assert synthetic_agent.name == "synthetic_intelligence"
        assert synthetic_agent.initialized == False
    
    def test_setup(self, synthetic_agent):
        """Test agent setup"""
        with patch.object(synthetic_agent, '_setup_gan') as mock_gan, \
             patch.object(synthetic_agent, '_setup_transformer') as mock_transformer, \
             patch.object(synthetic_agent, '_setup_simulation_engine') as mock_simulator:
            
            synthetic_agent.initialize()
            
            mock_gan.assert_called_once()
            mock_transformer.assert_called_once()
            mock_simulator.assert_called_once()
            assert synthetic_agent.initialized == True
    
    def test_generate_synthetic_matches(self, synthetic_agent):
        """Test synthetic match generation"""
        # Mock historical data
        historical_data = pd.DataFrame({
            'home_team': ['Team A', 'Team B', 'Team C'],
            'away_team': ['Team X', 'Team Y', 'Team Z'],
            'home_goals': [2, 1, 3],
            'away_goals': [1, 2, 0],
            'league': ['Premier League'] * 3
        })
        
        with patch.object(synthetic_agent, 'generator') as mock_generator, \
             patch.object(synthetic_agent, '_generate_match_sequences') as mock_sequences:
            
            # Mock generator output
            mock_features = Mock()
            mock_generator.return_value = mock_features
            mock_sequences.return_value = ['sequence1', 'sequence2']
            
            synthetic_matches = synthetic_agent.generate_synthetic_matches(
                historical_data, num_matches=2
            )
            
            assert len(synthetic_matches) == 2
            assert all(match['synthetic'] for match in synthetic_matches)
    
    def test_simulate_match_scenarios(self, synthetic_agent):
        """Test match scenario simulation"""
        matches = [
            {
                'home_team': 'Team A',
                'away_team': 'Team B',
                'home_team_strength': 1.2,
                'away_team_strength': 1.0,
                'home_form': 1.1,
                'away_form': 0.9
            }
        ]
        
        simulations = synthetic_agent.simulate_match_scenarios(matches, num_simulations=100)
        
        assert 'Team A_Team B' in simulations
        sim_result = simulations['Team A_Team B']
        
        assert 'home_win_prob' in sim_result
        assert 'draw_prob' in sim_result
        assert 'away_win_prob' in sim_result
        assert 0 <= sim_result['home_win_prob'] <= 1
        assert 0 <= sim_result['draw_prob'] <= 1
        assert 0 <= sim_result['away_win_prob'] <= 1

class TestStrategicIntelligenceAgent:
    """Test cases for Strategic Intelligence Agent"""
    
    @pytest.fixture
    def strategic_agent(self):
        config = {
            'strategy': {},
            'context': {},
            'tactics': {}
        }
        return StrategicIntelligenceAgent(config)
    
    def test_tactical_analysis(self, strategic_agent):
        """Test tactical analysis"""
        match_data = {
            'home_team': 'Team A',
            'away_team': 'Team B'
        }
        
        team_data = {
            'home_team': {
                'preferred_formation': '4-3-3',
                'playing_style': 'possession'
            },
            'away_team': {
                'preferred_formation': '4-4-2',
                'playing_style': 'counter_attack'
            }
        }
        
        with patch.object(strategic_agent, '_extract_tactical_patterns') as mock_extract:
            mock_extract.side_effect = [
                {'formation': '4-3-3', 'style': 'possession'},
                {'formation': '4-4-2', 'style': 'counter_attack'}
            ]
            
            tactical_analysis = strategic_agent._analyze_tactics(match_data, team_data)
            
            assert 'formation_matchup' in tactical_analysis
            assert 'style_clash' in tactical_analysis
            assert 'key_battles' in tactical_analysis
    
    def test_motivational_analysis(self, strategic_agent):
        """Test motivational analysis"""
        match_data = {
            'home_team': 'Team A',
            'away_team': 'Team B'
        }
        
        context_data = {
            'home_motivation_factors': {
                'league_position': 1,
                'rivalry': True
            },
            'away_motivation_factors': {
                'league_position': 5,
                'relegation_battle': True
            }
        }
        
        motivational_analysis = strategic_agent._analyze_motivation(match_data, context_data)
        
        assert 'home_motivation' in motivational_analysis
        assert 'away_motivation' in motivational_analysis
        assert 'motivation_differential' in motivational_analysis

class TestDataCollectorAgent:
    """Test cases for Data Collector Agent"""
    
    @pytest.fixture
    def data_agent(self):
        config = {
            'sources': {
                'hollywoodbets': {'enabled': True},
                'betway': {'enabled': True},
                'football_data': {'enabled': True}
            }
        }
        return DataCollectorAgent(config)
    
    @pytest.mark.asyncio
    async def test_data_collection(self, data_agent):
        """Test data collection from multiple sources"""
        with patch.object(data_agent, '_collect_from_scraper') as mock_scraper, \
             patch.object(data_agent, '_collect_from_api') as mock_api:
            
            mock_scraper.return_value = {
                'source': 'hollywoodbets',
                'type': 'scraper',
                'data': [{'match': 'test'}],
                'success': True
            }
            
            mock_api.return_value = {
                'source': 'football_data',
                'type': 'api',
                'data': {'matches': [{'match': 'test'}]},
                'success': True
            }
            
            result = await data_agent.process({})
            
            assert 'collected_data' in result
            assert 'collection_metadata' in result
            assert 'timestamp' in result

class TestPredictionEnsembleAgent:
    """Test cases for Prediction Ensemble Agent"""
    
    @pytest.fixture
    def ensemble_agent(self):
        config = {
            'models': {
                'transformer': {},
                'lstm': {},
                'poisson': {}
            },
            'weights': {
                'transformer': 0.4,
                'lstm': 0.3,
                'poisson': 0.3
            }
        }
        return PredictionEnsembleAgent(config)
    
    def test_ensemble_prediction(self, ensemble_agent):
        """Test ensemble prediction combination"""
        # Mock individual model predictions
        individual_predictions = {
            'transformer': {
                'probabilities': np.array([[0.6, 0.2, 0.2]]),
                'prediction': np.array([0])
            },
            'lstm': {
                'probabilities': np.array([[0.5, 0.3, 0.2]]),
                'prediction': np.array([0])
            },
            'poisson': {
                'probabilities': np.array([[0.7, 0.2, 0.1]]),
                'prediction': np.array([0])
            }
        }
        
        with patch.object(ensemble_agent, '_get_individual_predictions') as mock_predictions:
            mock_predictions.return_value = individual_predictions
            
            result = ensemble_agent.process({
                'match_data': {},
                'features': np.array([[1, 2, 3]])
            })
            
            assert 'ensemble_prediction' in result
            assert 'individual_predictions' in result
            assert 'model_weights' in result
            assert 'confidence_metrics' in result
            
            ensemble_probs = result['ensemble_prediction']['probabilities']
            assert ensemble_probs.shape == (1, 3)
            assert np.allclose(np.sum(ensemble_probs, axis=1), 1.0)
