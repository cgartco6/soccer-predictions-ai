import pandas as pd
import numpy as np
from typing import Dict, List

class PlayerPerformanceAnalyzer:
    def __init__(self):
        self.performance_metrics = [
            'goals_scored', 'assists', 'pass_accuracy', 'tackles', 
            'interceptions', 'shots_on_target', 'distance_covered'
        ]
    
    def calculate_player_form(self, player_data: pd.DataFrame, window: int = 5) -> Dict:
        """Calculate player form over recent matches"""
        form_scores = {}
        
        for player_id, matches in player_data.groupby('player_id'):
            recent_matches = matches.tail(window)
            
            form_score = self._compute_weighted_form(recent_matches)
            consistency = self._calculate_consistency(recent_matches)
            momentum = self._calculate_momentum(recent_matches)
            
            form_scores[player_id] = {
                'form_score': form_score,
                'consistency': consistency,
                'momentum': momentum,
                'recent_goals': recent_matches['goals'].sum(),
                'recent_assists': recent_matches['assists'].sum()
            }
            
        return form_scores
    
    def _compute_weighted_form(self, matches: pd.DataFrame) -> float:
        """Compute weighted form with recent matches having more importance"""
        weights = np.linspace(0.5, 1.0, len(matches))
        performance_score = (
            matches['rating'] * 0.3 +
            matches['goals'] * 0.25 +
            matches['assists'] * 0.2 +
            matches['pass_accuracy'] * 0.15 +
            matches['successful_tackles'] * 0.1
        )
        
        return np.average(performance_score, weights=weights)
