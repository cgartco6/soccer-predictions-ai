import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class ELOPredictor:
    """
    ELO rating system for soccer predictions
    Enhanced with goal difference and home advantage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ratings = {}
        self.k_factor = config.get('k_factor', 20)
        self.home_advantage = config.get('home_advantage', 100)
        self.goal_difference_factor = config.get('goal_difference_factor', 0.1)
        
    def update_ratings(self, matches: pd.DataFrame):
        """Update ELO ratings based on match results"""
        for _, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            # Initialize ratings if needed
            if home_team not in self.ratings:
                self.ratings[home_team] = 1500
            if away_team not in self.ratings:
                self.ratings[away_team] = 1500
            
            # Get current ratings with home advantage
            home_rating = self.ratings[home_team] + self.home_advantage
            away_rating = self.ratings[away_team]
            
            # Calculate expected scores
            expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
            expected_away = 1 - expected_home
            
            # Calculate actual scores
            if home_goals > away_goals:
                actual_home = 1.0
                actual_away = 0.0
            elif home_goals == away_goals:
                actual_home = 0.5
                actual_away = 0.5
            else:
                actual_home = 0.0
                actual_away = 1.0
            
            # Adjust for goal difference
            goal_difference = abs(home_goals - away_goals)
            margin_multiplier = np.log(goal_difference + 1) * self.goal_difference_factor
            
            # Update ratings
            self.ratings[home_team] += self.k_factor * margin_multiplier * (actual_home - expected_home)
            self.ratings[away_team] += self.k_factor * margin_multiplier * (actual_away - expected_away)
    
    def predict(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Predict match outcome using ELO ratings"""
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Get ratings with home advantage
        home_rating = self.ratings.get(home_team, 1500) + self.home_advantage
        away_rating = self.ratings.get(away_team, 1500)
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        expected_away = 1 - expected_home
        
        # Convert to probabilities (draw adjustment)
        home_win_prob = expected_home ** 2 / (expected_home ** 2 + expected_away ** 2 + (expected_home * expected_away))
        away_win_prob = expected_away ** 2 / (expected_home ** 2 + expected_away ** 2 + (expected_home * expected_away))
        draw_prob = 1 - home_win_prob - away_win_prob
        
        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'home_rating': home_rating - self.home_advantage,
            'away_rating': away_rating,
            'rating_difference': home_rating - away_rating
        }
    
    def get_team_ratings(self) -> Dict[str, float]:
        """Get current ELO ratings for all teams"""
        return self.ratings.copy()
