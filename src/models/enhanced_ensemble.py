import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, List

class EnhancedEnsembleModel:
    def __init__(self):
        self.models = {
            'transformer': SoccerTransformer(),
            'xgboost': xgb.XGBClassifier(),
            'random_forest': RandomForestClassifier(),
            'neural_network': EnhancedNeuralNetwork()
        }
        self.feature_weights = self._initialize_feature_weights()
    
    def predict_match(self, match_features: Dict) -> Dict:
        """Generate predictions using ensemble approach"""
        # Get bookmaker predictions
        bookmaker_preds = self._get_bookmaker_predictions(match_features)
        
        # Get AI predictions
        ai_predictions = {}
        
        for model_name, model in self.models.items():
            prediction = model.predict(match_features)
            ai_predictions[model_name] = prediction
            
        # Ensemble predictions
        ensemble_pred = self._ensemble_predictions(ai_predictions, bookmaker_preds)
        
        return {
            'bookmaker_predictions': {
                'hollywoodbets': bookmaker_preds['hollywoodbets'],
                'betway': bookmaker_preds['betway']
            },
            'ai_predictions': ai_predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence_scores': self._calculate_confidence(ensemble_pred),
            'key_factors': self._explain_prediction(match_features)
        }
    
    def _get_bookmaker_predictions(self, match_features: Dict) -> Dict:
        """Convert odds to probabilities"""
        def odds_to_probability(odds):
            return 1 / odds if odds > 1 else 0
            
        return {
            'hollywoodbets': {
                'home_win': odds_to_probability(match_features.get('hb_home_odds', 0)),
                'draw': odds_to_probability(match_features.get('hb_draw_odds', 0)),
                'away_win': odds_to_probability(match_features.get('hb_away_odds', 0))
            },
            'betway': {
                'home_win': odds_to_probability(match_features.get('bw_home_odds', 0)),
                'draw': odds_to_probability(match_features.get('bw_draw_odds', 0)),
                'away_win': odds_to_probability(match_features.get('bw_away_odds', 0))
            }
        }
