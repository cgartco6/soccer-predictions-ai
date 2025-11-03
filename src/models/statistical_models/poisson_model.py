import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from typing import Dict, Any, Tuple

class PoissonPredictor:
    """
    Poisson regression model for soccer score prediction
    Based on the work of Maher (1982) and Dixon & Coles (1997)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.home_attack = {}
        self.home_defense = {}
        self.away_attack = {}
        self.away_defense = {}
        self.home_advantage = config.get('home_advantage', 0.3)
        self.regressor = PoissonRegressor(alpha=config.get('alpha', 1.0))
        
    def fit(self, matches: pd.DataFrame):
        """Fit Poisson model to historical match data"""
        # Prepare features and targets
        X, y_home, y_away = self._prepare_features(matches)
        
        # Fit home goals model
        self.home_model = PoissonRegressor(alpha=self.config.get('alpha', 1.0))
        self.home_model.fit(X, y_home)
        
        # Fit away goals model
        self.away_model = PoissonRegressor(alpha=self.config.get('alpha', 1.0))
        self.away_model.fit(X, y_away)
        
        # Calculate team strengths
        self._calculate_team_strengths(matches)
        
    def predict(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Predict match outcome using Poisson distribution"""
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Get expected goals
        lambda_home = self._calculate_expected_goals(home_team, away_team, 'home')
        lambda_away = self._calculate_expected_goals(home_team, away_team, 'away')
        
        # Calculate probabilities using Poisson distribution
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        # Consider realistic goal ranges (0-10 goals)
        max_goals = 10
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Normalize probabilities
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        
        return {
            'expected_home_goals': lambda_home,
            'expected_away_goals': lambda_away,
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'most_likely_score': self._find_most_likely_score(lambda_home, lambda_away)
        }
    
    def _calculate_expected_goals(self, home_team: str, away_team: str, 
                                side: str) -> float:
        """Calculate expected goals for a team"""
        if side == 'home':
            attack = self.home_attack.get(home_team, 1.0)
            defense = self.away_defense.get(away_team, 1.0)
            advantage = self.home_advantage
        else:
            attack = self.away_attack.get(away_team, 1.0)
            defense = self.home_defense.get(home_team, 1.0)
            advantage = 0.0  # No home advantage for away team
        
        base_rate = self.config.get('base_goal_rate', 1.5)
        return base_rate * attack * defense + advantage
    
    def _calculate_team_strengths(self, matches: pd.DataFrame):
        """Calculate team attacking and defensive strengths"""
        # Average goals per game
        avg_home_goals = matches['home_goals'].mean()
        avg_away_goals = matches['away_goals'].mean()
        
        # Calculate team strengths
        for team in set(matches['home_team']).union(set(matches['away_team'])):
            home_matches = matches[matches['home_team'] == team]
            away_matches = matches[matches['away_team'] == team]
            
            # Home attacking strength
            if len(home_matches) > 0:
                home_goals_scored = home_matches['home_goals'].mean()
                home_goals_conceded = home_matches['away_goals'].mean()
                
                self.home_attack[team] = home_goals_scored / avg_home_goals
                self.home_defense[team] = home_goals_conceded / avg_away_goals
            
            # Away attacking strength
            if len(away_matches) > 0:
                away_goals_scored = away_matches['away_goals'].mean()
                away_goals_conceded = away_matches['home_goals'].mean()
                
                self.away_attack[team] = away_goals_scored / avg_away_goals
                self.away_defense[team] = away_goals_conceded / avg_home_goals
    
    def _find_most_likely_score(self, lambda_home: float, 
                              lambda_away: float) -> Tuple[int, int]:
        """Find the most likely scoreline"""
        max_prob = 0
        most_likely = (0, 0)
        
        for i in range(8):  # 0-7 goals
            for j in range(8):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if prob > max_prob:
                    max_prob = prob
                    most_likely = (i, j)
                    
        return most_likely
    
    def _prepare_features(self, matches: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare features for Poisson regression"""
        # This would create features for team strengths, form, etc.
        # Simplified implementation
        X = pd.get_dummies(matches[['home_team', 'away_team']])
        y_home = matches['home_goals']
        y_away = matches['away_goals']
        
        return X, y_home, y_away
