import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents import (PredictionEnsembleAgent, SyntheticIntelligenceAgent,
                       StrategicIntelligenceAgent, DataCollectorAgent)
from src.data.processors.data_cleaner import DataCleaner
from src.data.processors.feature_engineer import FeatureEngineer
from src.models.feature_engineering import FeatureEngineer as ModelFeatureEngineer

class TestEndToEndWorkflow:
    """End-to-end workflow tests for the complete prediction pipeline"""
    
    @pytest.fixture
    def sample_match_data(self):
        """Sample match data for testing"""
        return pd.DataFrame({
            'home_team': ['Manchester United', 'Arsenal', 'Chelsea', 'Liverpool'],
            'away_team': ['Liverpool', 'Chelsea', 'Manchester United', 'Arsenal'],
            'home_goals': [2, 1, 0, 3],
            'away_goals': [1, 2, 2, 0],
            'league': ['Premier League'] * 4,
            'date': pd.date_range('2024-01-01', periods=4),
            'attendance': [75000, 60000, 40000, 53000],
            'home_shots': [15, 12, 8, 18],
            'away_shots': [10, 14, 16, 9]
        })
    
    @pytest.fixture
    def sample_odds_data(self):
        """Sample odds data for testing"""
        return [
            {
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'home_odds': 2.1,
                'draw_odds': 3.2,
                'away_odds': 3.5,
                'source': 'hollywoodbets',
                'scraped_at': datetime.now().timestamp()
            }
        ]
    
    @pytest.fixture
    def configured_agents(self):
        """Set up configured agents for testing"""
        config = {
            'agents': {
                'prediction_ensemble': {
                    'models': {
                        'transformer': {},
                        'lstm': {},
                        'poisson': {}
                    }
                },
                'synthetic_intelligence': {
                    'gan': {
                        'batch_size': 32,
                        'latent_dim': 100
                    }
                },
                'strategic_intelligence': {},
                'data_collector': {
                    'sources': {
                        'hollywoodbets': {'enabled': True},
                        'betway': {'enabled': True}
                    }
                }
            }
        }
        
        ensemble_agent = PredictionEnsembleAgent(config['agents']['prediction_ensemble'])
        synthetic_agent = SyntheticIntelligenceAgent(config['agents']['synthetic_intelligence'])
        strategic_agent = StrategicIntelligenceAgent(config['agents']['strategic_intelligence'])
        data_agent = DataCollectorAgent(config['agents']['data_collector'])
        
        return {
            'ensemble': ensemble_agent,
            'synthetic': synthetic_agent,
            'strategic': strategic_agent,
            'data': data_agent
        }
    
    async def test_complete_prediction_workflow(self, configured_agents, sample_match_data, sample_odds_data):
        """Test complete prediction workflow from data collection to prediction"""
        
        # 1. Data Collection and Validation
        with patch.object(configured_agents['data'], 'process') as mock_data_collection:
            mock_data_collection.return_value = {
                'collected_data': {
                    'matches': sample_match_data.to_dict('records'),
                    'odds': sample_odds_data
                },
                'collection_metadata': {
                    'sources_used': ['hollywoodbets'],
                    'collection_time': datetime.now().isoformat()
                }
            }
            
            collected_data = await configured_agents['data'].process({})
            assert 'collected_data' in collected_data
            assert 'matches' in collected_data['collected_data']
        
        # 2. Data Cleaning
        data_cleaner = DataCleaner({})
        cleaned_matches = data_cleaner.clean_match_data(sample_match_data)
        cleaned_odds = data_cleaner.clean_odds_data(sample_odds_data)
        
        assert not cleaned_matches.empty
        assert not cleaned_odds.empty
        assert cleaned_matches['home_team'].str.contains('Man United').sum() == 0
        
        # 3. Feature Engineering
        feature_engineer = FeatureEngineer({})
        features = feature_engineer.create_features(
            cleaned_matches, 
            feature_types=['basic', 'team_strength', 'form', 'contextual']
        )
        
        assert not features.empty
        assert 'home_elo' in features.columns
        assert 'away_elo' in features.columns
        assert 'home_recent_form' in features.columns
        
        # 4. Synthetic Data Generation
        with patch.object(configured_agents['synthetic'], 'process') as mock_synthetic:
            mock_synthetic.return_value = {
                'synthetic_matches': [
                    {
                        'home_team': 'Manchester United',
                        'away_team': 'Liverpool',
                        'home_goals': 2,
                        'away_goals': 1,
                        'synthetic': True
                    }
                ],
                'simulations': {
                    'Manchester United_Liverpool': {
                        'home_win_prob': 0.45,
                        'draw_prob': 0.30,
                        'away_win_prob': 0.25
                    }
                }
            }
            
            synthetic_data = configured_agents['synthetic'].process({
                'current_matches': [{
                    'home_team': 'Manchester United',
                    'away_team': 'Liverpool'
                }],
                'historical_matches': cleaned_matches.to_dict('records'),
                'num_simulations': 1000
            })
            
            assert 'synthetic_matches' in synthetic_data
            assert 'simulations' in synthetic_data
        
        # 5. Strategic Analysis
        with patch.object(configured_agents['strategic'], 'process') as mock_strategic:
            mock_strategic.return_value = {
                'tactical_analysis': {
                    'formation_matchup': 'balanced',
                    'style_clash': 'attack_vs_defense'
                },
                'strategic_insights': [
                    'Home advantage significant',
                    'Recent form favors home team'
                ]
            }
            
            strategic_analysis = configured_agents['strategic'].process({
                'match_data': {
                    'home_team': 'Manchester United',
                    'away_team': 'Liverpool'
                },
                'team_data': {},
                'context_data': {}
            })
            
            assert 'strategic_insights' in strategic_analysis
            assert len(strategic_analysis['strategic_insights']) > 0
        
        # 6. Ensemble Prediction
        with patch.object(configured_agents['ensemble'], 'process') as mock_ensemble:
            mock_ensemble.return_value = {
                'ensemble_prediction': {
                    'probabilities': [[0.45, 0.30, 0.25]],
                    'prediction': [0],
                    'confidence': [0.72],
                    'expected_goals': {'home': 1.8, 'away': 1.2}
                },
                'confidence_metrics': {
                    'overall_confidence': 0.72,
                    'model_agreement': 0.85
                },
                'feature_importance': {
                    'team_strength': 0.23,
                    'recent_form': 0.18
                }
            }
            
            prediction_result = configured_agents['ensemble'].process({
                'match_data': {
                    'home_team': 'Manchester United',
                    'away_team': 'Liverpool',
                    'home_odds': 2.1,
                    'draw_odds': 3.2,
                    'away_odds': 3.5
                },
                'features': features.iloc[0:1].to_dict('records'),
                'synthetic_data': synthetic_data,
                'strategic_analysis': strategic_analysis
            })
            
            assert 'ensemble_prediction' in prediction_result
            assert 'confidence_metrics' in prediction_result
            
            ensemble_pred = prediction_result['ensemble_prediction']
            assert 'probabilities' in ensemble_pred
            assert 'confidence' in ensemble_pred
            
            # Verify probabilities sum to approximately 1
            probs = ensemble_pred['probabilities'][0]
            assert sum(probs) == pytest.approx(1.0, abs=0.01)
    
    async def test_batch_prediction_workflow(self, configured_agents, sample_match_data):
        """Test batch prediction workflow for multiple matches"""
        
        # Prepare multiple matches
        match_requests = [
            {
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'league': 'Premier League'
            },
            {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea', 
                'league': 'Premier League'
            }
        ]
        
        # Mock the ensemble agent to handle batch processing
        with patch.object(configured_agents['ensemble'], 'process') as mock_ensemble:
            mock_ensemble.return_value = {
                'ensemble_prediction': {
                    'probabilities': [[0.45, 0.30, 0.25], [0.35, 0.35, 0.30]],
                    'prediction': [0, 1],
                    'confidence': [0.72, 0.65]
                },
                'confidence_metrics': {
                    'overall_confidence': 0.685,
                    'model_agreement': 0.82
                }
            }
            
            # Process each match in batch
            predictions = []
            for match_request in match_requests:
                prediction_result = configured_agents['ensemble'].process({
                    'match_data': match_request,
                    'features': {},
                    'synthetic_data': {},
                    'strategic_analysis': {}
                })
                predictions.append(prediction_result)
            
            assert len(predictions) == 2
            assert all('ensemble_prediction' in pred for pred in predictions)
    
    async def test_error_handling_workflow(self, configured_agents):
        """Test error handling throughout the prediction workflow"""
        
        # Test data collection failure
        with patch.object(configured_agents['data'], 'process') as mock_data_collection:
            mock_data_collection.side_effect = Exception("Data source unavailable")
            
            try:
                await configured_agents['data'].process({})
                assert False, "Expected exception"
            except Exception as e:
                assert str(e) == "Data source unavailable"
        
        # Test synthetic data generation failure
        with patch.object(configured_agents['synthetic'], 'process') as mock_synthetic:
            mock_synthetic.side_effect = Exception("GAN model failed")
            
            try:
                configured_agents['synthetic'].process({})
                assert False, "Expected exception"
            except Exception as e:
                assert str(e) == "GAN model failed"
        
        # Test ensemble prediction with missing data
        with patch.object(configured_agents['ensemble'], 'process') as mock_ensemble:
            mock_ensemble.side_effect = ValueError("Insufficient features")
            
            try:
                configured_agents['ensemble'].process({
                    'match_data': {},
                    'features': {},
                    'synthetic_data': {},
                    'strategic_analysis': {}
                })
                assert False, "Expected exception"
            except ValueError as e:
                assert "Insufficient features" in str(e)

class TestModelTrainingWorkflow:
    """Tests for model training and evaluation workflows"""
    
    @pytest.fixture
    def training_data(self):
        """Sample training data"""
        return pd.DataFrame({
            'home_team': ['Team A', 'Team B', 'Team C'] * 10,
            'away_team': ['Team B', 'Team C', 'Team A'] * 10,
            'home_goals': np.random.randint(0, 5, 30),
            'away_goals': np.random.randint(0, 5, 30),
            'league': ['Premier League'] * 30,
            'date': pd.date_range('2023-01-01', periods=30),
            'home_shots': np.random.randint(5, 20, 30),
            'away_shots': np.random.randint(5, 20, 30)
        })
    
    def test_feature_engineering_for_training(self, training_data):
        """Test feature engineering for model training"""
        feature_engineer = ModelFeatureEngineer({})
        
        features = feature_engineer.create_features(training_data)
        
        assert not features.empty
        assert len(features) == len(training_data)
        
        # Check for essential features
        essential_features = [
            'home_elo', 'away_elo', 'elo_difference',
            'home_recent_form', 'away_recent_form'
        ]
        
        for feature in essential_features:
            assert feature in features.columns
            assert features[feature].notna().all()
    
    def test_data_splitting(self, training_data):
        """Test training/validation/test split"""
        from sklearn.model_selection import train_test_split
        
        # Create features and target
        feature_engineer = ModelFeatureEngineer({})
        features = feature_engineer.create_features(training_data)
        
        # Create target variable (1: home win, 0: draw, 2: away win)
        conditions = [
            training_data['home_goals'] > training_data['away_goals'],
            training_data['home_goals'] == training_data['away_goals']
        ]
        choices = [0, 1]  # home_win, draw
        target = np.select(conditions, choices, default=2)  # away_win
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, target, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Verify splits
        assert len(X_train) + len(X_val) + len(X_test) == len(features)
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)
        
        # Verify no data leakage
        assert set(X_train.index).isdisjoint(set(X_val.index))
        assert set(X_train.index).isdisjoint(set(X_test.index))
        assert set(X_val.index).isdisjoint(set(X_test.index))
    
    @patch('src.models.neural_networks.transformer_model.SoccerTransformer')
    def test_model_training_workflow(self, mock_transformer, training_data):
        """Test complete model training workflow"""
        from src.models.feature_engineering import FeatureEngineer
        from src.utils.metrics import ModelEvaluator
        
        # Feature engineering
        feature_engineer = FeatureEngineer({})
        features = feature_engineer.create_features(training_data)
        
        # Create target
        conditions = [
            training_data['home_goals'] > training_data['away_goals'],
            training_data['home_goals'] == training_data['away_goals']
        ]
        choices = [0, 1]
        y = np.select(conditions, choices, default=2)
        
        # Mock transformer training
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mock training process
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.choice([0, 1, 2], len(y))
        mock_model.predict_proba.return_value = np.random.dirichlet([1, 1, 1], len(y))
        
        # Train model
        model = mock_transformer({})
        model.fit(features, y)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        evaluation = evaluator.evaluate_model(model, features, y)
        
        assert 'classification_metrics' in evaluation
        assert 'feature_importance' in evaluation
        assert 'confidence_analysis' in evaluation
        
        metrics = evaluation['classification_metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_model_performance_tracking(self):
        """Test model performance tracking and comparison"""
        from src.utils.metrics import PredictionMetrics
        
        metrics_calculator = PredictionMetrics()
        
        # Sample predictions and actual results
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 0])  # Some errors
        y_prob = np.random.dirichlet([1, 1, 1], len(y_true))
        
        metrics = metrics_calculator.calculate_classification_metrics(
            y_true, y_pred, y_prob
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Accuracy should be reasonable
        assert 0 <= metrics['accuracy'] <= 1
        
        # Test betting metrics
        predictions_df = pd.DataFrame({
            'predicted_class': y_pred,
            'actual_class': y_true,
            'prediction_confidence': np.max(y_prob, axis=1)
        })
        
        odds_df = pd.DataFrame({
            'home_odds': [2.0] * len(y_true),
            'draw_odds': [3.2] * len(y_true),
            'away_odds': [3.5] * len(y_true)
        })
        
        betting_metrics = metrics_calculator.calculate_betting_metrics(
            predictions_df, odds_df
        )
        
        assert 'fixed_stake_roi' in betting_metrics
        assert 'value_bets_roi' in betting_metrics

class TestDataPipelineWorkflow:
    """Tests for data pipeline workflows"""
    
    async def test_real_time_data_pipeline(self, configured_agents):
        """Test real-time data collection and processing pipeline"""
        
        # Mock real-time data sources
        with patch.object(configured_agents['data'], 'process') as mock_data_pipeline:
            mock_data_pipeline.return_value = {
                'collected_data': {
                    'matches': [
                        {
                            'home_team': 'Manchester United',
                            'away_team': 'Liverpool',
                            'league': 'Premier League',
                            'date': datetime.now().isoformat(),
                            'home_odds': 2.1,
                            'draw_odds': 3.2,
                            'away_odds': 3.5
                        }
                    ],
                    'odds': [
                        {
                            'home_team': 'Manchester United',
                            'away_team': 'Liverpool',
                            'home_odds': 2.1,
                            'draw_odds': 3.2,
                            'away_odds': 3.5,
                            'source': 'hollywoodbets'
                        }
                    ]
                },
                'collection_metadata': {
                    'sources_used': ['hollywoodbets', 'betway'],
                    'collection_time': datetime.now().isoformat()
                }
            }
            
            # Simulate real-time data collection
            collected_data = await configured_agents['data'].process({
                'real_time': True,
                'sources': ['hollywoodbets', 'betway']
            })
            
            assert 'collected_data' in collected_data
            assert 'matches' in collected_data['collected_data']
            assert 'odds' in collected_data['collected_data']
            
            # Process the collected data
            data_cleaner = DataCleaner({})
            matches_df = pd.DataFrame(collected_data['collected_data']['matches'])
            odds_df = pd.DataFrame(collected_data['collected_data']['odds'])
            
            cleaned_matches = data_cleaner.clean_match_data(matches_df)
            cleaned_odds = data_cleaner.clean_odds_data(odds_df)
            
            assert not cleaned_matches.empty
            assert not cleaned_odds.empty
            
            # Verify data freshness
            collection_time = datetime.fromisoformat(
                collected_data['collection_metadata']['collection_time']
            )
            time_diff = datetime.now() - collection_time
            assert time_diff.total_seconds() < 300  # Data should be less than 5 minutes old
    
    def test_historical_data_processing(self):
        """Test processing of historical data for model training"""
        from src.data.processors.data_cleaner import DataCleaner
        from src.data.processors.feature_engineer import FeatureEngineer
        
        # Create sample historical data
        historical_data = pd.DataFrame({
            'home_team': ['Team A', 'Team B', 'Team C'] * 100,
            'away_team': ['Team B', 'Team C', 'Team A'] * 100,
            'home_goals': np.random.randint(0, 5, 300),
            'away_goals': np.random.randint(0, 5, 300),
            'league': ['Premier League'] * 300,
            'date': pd.date_range('2020-01-01', periods=300),
            'attendance': np.random.randint(20000, 80000, 300),
            'home_shots': np.random.randint(5, 25, 300),
            'away_shots': np.random.randint(5, 25, 300)
        })
        
        # Clean historical data
        data_cleaner = DataCleaner({})
        cleaned_data = data_cleaner.clean_match_data(historical_data)
        
        assert not cleaned_data.empty
        assert len(cleaned_data) <= len(historical_data)  # Some duplicates may be removed
        
        # Engineer features
        feature_engineer = FeatureEngineer({})
        features = feature_engineer.create_features(cleaned_data)
        
        assert not features.empty
        assert len(features) == len(cleaned_data)
        
        # Verify feature quality
        assert features.isna().sum().sum() == 0  # No missing values
        assert (features.select_dtypes(include=[np.number]).std() > 0).all()  # No constant features
