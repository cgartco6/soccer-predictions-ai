import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.data.collectors.data_validator import DataValidator
from src.data.processors.data_cleaner import DataCleaner
from src.data.processors.feature_engineer import FeatureEngineer
from src.data.schemas.match_schema import MatchSchema, MatchOddsSchema

class TestDataValidator:
    """Test cases for Data Validator"""
    
    @pytest.fixture
    def match_validator(self):
        return DataValidator('match')
    
    @pytest.fixture
    def sample_match_data(self):
        return [
            {
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'league': 'Premier League',
                'date': '2024-01-15',
                'home_goals': 2,
                'away_goals': 1,
                'attendance': 75000
            },
            {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea', 
                'league': 'Premier League',
                'date': '2024-01-14',
                'home_goals': 1,
                'away_goals': 1,
                'attendance': 60000
            }
        ]
    
    def test_validation_success(self, match_validator, sample_match_data):
        """Test successful data validation"""
        result = match_validator.validate(sample_match_data)
        
        assert len(result['valid_data']) == 2
        assert result['validation_report']['valid_records'] == 2
        assert result['validation_report']['validity_rate'] == 1.0
    
    def test_validation_failure(self, match_validator):
        """Test data validation with errors"""
        invalid_data = [
            {
                'home_team': 'Team A',
                'away_team': 'Team A',  # Same team
                'league': 'Premier League',
                'date': 'invalid_date',  # Invalid date
                'home_goals': -1,  # Negative goals
                'away_goals': 1
            }
        ]
        
        result = match_validator.validate(invalid_data)
        
        assert len(result['valid_data']) == 0
        assert result['validation_report']['invalid_records'] == 1
        assert len(result['validation_report']['errors']) > 0
    
    def test_odds_validation(self):
        """Test odds data validation"""
        odds_validator = DataValidator('odds')
        
        odds_data = [
            {
                'home_team': 'Team A',
                'away_team': 'Team B',
                'home_odds': 2.0,
                'away_odds': 3.0,
                'source': 'test'
            },
            {
                'home_team': 'Team C',
                'away_team': 'Team D', 
                'home_odds': 0.5,  # Invalid odds
                'away_odds': 3.0,
                'source': 'test'
            }
        ]
        
        result = odds_validator.validate(odds_data)
        
        assert len(result['valid_data']) == 1
        assert result['validation_report']['invalid_records'] == 1

class TestDataCleaner:
    """Test cases for Data Cleaner"""
    
    @pytest.fixture
    def data_cleaner(self):
        config = {}
        return DataCleaner(config)
    
    @pytest.fixture
    def dirty_match_data(self):
        return pd.DataFrame({
            'home_team': ['Man United', 'man city', 'Arsenal  ', 'Chelsea', 'Man United'],
            'away_team': ['Liverpool', 'Tottenham', 'Chelsea', 'Arsenal', 'Liverpool'],
            'home_goals': [2, 1, np.nan, 0, 2],
            'away_goals': [1, 2, 1, 2, 1],
            'league': ['Premier League', 'premier league', 'Premier League', np.nan, 'Premier League'],
            'date': ['2024-01-15', '2024-01-14', 'invalid', '2024-01-13', '2024-01-15'],
            'attendance': [75000, 55000, 60000, 40000, 75000]
        })
    
    def test_team_name_standardization(self, data_cleaner):
        """Test team name standardization"""
        test_names = ['Man United', 'man city', 'Spurs', 'Real Mad.']
        
        standardized = [data_cleaner._standardize_team_name(name) for name in test_names]
        
        assert 'Manchester United' in standardized
        assert 'Manchester City' in standardized
        assert 'Tottenham Hotspur' in standardized
        assert 'Real Madrid' in standardized
    
    def test_match_data_cleaning(self, data_cleaner, dirty_match_data):
        """Test comprehensive match data cleaning"""
        cleaned_data = data_cleaner.clean_match_data(dirty_match_data)
        
        # Check for duplicates removed
        assert len(cleaned_data) == 4  # One duplicate removed
        
        # Check team name standardization
        assert cleaned_data['home_team'].str.contains('Man United').sum() == 0  # Should be standardized
        assert 'Manchester United' in cleaned_data['home_team'].values
        
        # Check missing value handling
        assert cleaned_data['home_goals'].isna().sum() == 0
        assert cleaned_data['league'].isna().sum() == 0
        
        # Check date parsing
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['date'])
    
    def test_odds_data_cleaning(self, data_cleaner):
        """Test odds data cleaning"""
        odds_data = [
            {
                'home_team': 'Man United',
                'away_team': 'Liverpool',
                'home_odds': 2.0,
                'draw_odds': 3.2,
                'away_odds': 3.5,
                'source': 'test',
                'scraped_at': 1234567890
            },
            {
                'home_team': 'Man United',  # Duplicate
                'away_team': 'Liverpool',
                'home_odds': 2.1,
                'draw_odds': 3.1,
                'away_odds': 3.6,
                'source': 'test',
                'scraped_at': 1234567891
            }
        ]
        
        cleaned_odds = data_cleaner.clean_odds_data(odds_data)
        
        # Should remove duplicates and keep the latest
        assert len(cleaned_odds) == 1
        assert cleaned_odds.iloc[0]['home_odds'] == 2.1

class TestFeatureEngineer:
    """Test cases for Feature Engineer"""
    
    @pytest.fixture
    def feature_engineer(self):
        config = {}
        return FeatureEngineer(config)
    
    @pytest.fixture
    def historical_matches(self):
        return pd.DataFrame({
            'home_team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B', 'Team C'],
            'away_team': ['Team B', 'Team C', 'Team C', 'Team B', 'Team A', 'Team A'],
            'home_goals': [2, 1, 3, 0, 2, 1],
            'away_goals': [1, 2, 1, 2, 1, 2],
            'league': ['Premier League'] * 6,
            'date': pd.date_range('2024-01-01', periods=6)
        })
    
    def test_elo_calculation(self, feature_engineer, historical_matches):
        """Test ELO rating calculation"""
        elo_ratings = feature_engineer._calculate_elo_ratings(historical_matches)
        
        assert 'Team A' in elo_ratings
        assert 'Team B' in elo_ratings
        assert 'Team C' in elo_ratings
        
        # Team A should have highest ELO (won most matches)
        team_a_elo = elo_ratings['Team A']
        team_b_elo = elo_ratings['Team B'] 
        team_c_elo = elo_ratings['Team C']
        
        assert team_a_elo > team_b_elo
        assert team_a_elo > team_c_elo
    
    def test_form_calculation(self, feature_engineer, historical_matches):
        """Test recent form calculation"""
        form = feature_engineer._calculate_recent_form(
            historical_matches, 'Team A', 
            pd.Timestamp('2024-01-07'), 'home'
        )
        
        assert 'points_per_game' in form
        assert 'goals_scored_avg' in form
        assert 'goals_conceded_avg' in form
        
        # Team A should have good form (won both home matches)
        assert form['points_per_game'] > 0.5
    
    def test_head_to_head_calculation(self, feature_engineer, historical_matches):
        """Test head-to-head statistics calculation"""
        current_match = historical_matches.iloc[0]  # Team A vs Team B
        h2h_stats = feature_engineer._calculate_head_to_head(
            historical_matches, current_match, 0
        )
        
        assert 'home_win_rate' in h2h_stats
        assert 'away_win_rate' in h2h_stats
        assert 'draw_rate' in h2h_stats
        assert 'avg_goals' in h2h_stats

class TestMatchSchema:
    """Test cases for Match Schema validation"""
    
    def test_valid_match_schema(self):
        """Test valid match data"""
        valid_match = {
            'home_team': 'Manchester United',
            'away_team': 'Liverpool',
            'league': 'Premier League',
            'date': '2024-01-15T15:00:00',
            'home_goals': 2,
            'away_goals': 1,
            'attendance': 75000,
            'home_possession': 55.0,
            'away_possession': 45.0
        }
        
        match = MatchSchema(**valid_match)
        
        assert match.home_team == 'Manchester United'
        assert match.away_team == 'Liverpool'
        assert match.result.value == 'home_win'
    
    def test_invalid_match_schema(self):
        """Test invalid match data"""
        invalid_match = {
            'home_team': 'Manchester United',
            'away_team': 'Manchester United',  # Same team
            'league': 'Premier League',
            'date': '2024-01-15T15:00:00',
            'home_goals': -1,  # Negative goals
            'away_goals': 1
        }
        
        with pytest.raises(ValueError):
            MatchSchema(**invalid_match)
    
    def test_match_odds_schema(self):
        """Test match odds schema"""
        valid_odds = {
            'home_team': 'Manchester United',
            'away_team': 'Liverpool',
            'league': 'Premier League',
            'date': '2024-01-15T15:00:00',
            'home_odds': 2.0,
            'draw_odds': 3.2,
            'away_odds': 3.5,
            'bookmaker': 'test',
            'collected_at': '2024-01-14T10:00:00',
            'source': 'api'
        }
        
        odds = MatchOddsSchema(**valid_odds)
        
        assert odds.home_odds == 2.0
        assert odds.draw_odds == 3.2
        assert odds.away_odds == 3.5
