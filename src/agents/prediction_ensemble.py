import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import torch
import torch.nn as nn
from .base_agent import BaseAgent

class PredictionEnsembleAgent(BaseAgent):
    """
    AI Agent for combining predictions from multiple models
    using advanced ensemble techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("prediction_ensemble", config)
        self.models = {}
        self.ensemble_weights = {}
        self.calibrators = {}
        
    def _setup(self):
        """Initialize ensemble models and weights"""
        self._load_models()
        self._calculate_ensemble_weights()
        self._initialize_calibrators()
        
    def _load_models(self):
        """Load all individual prediction models"""
        model_configs = self.config['models']
        
        # Load neural network models
        if 'transformer' in model_configs:
            from ..models.neural_networks.transformer_model import SoccerTransformer
            self.models['transformer'] = SoccerTransformer(model_configs['transformer'])
            
        if 'lstm' in model_configs:
            from ..models.neural_networks.lstm_model import LSTMPredictor
            self.models['lstm'] = LSTMPredictor(model_configs['lstm'])
            
        # Load statistical models
        if 'poisson' in model_configs:
            from ..models.statistical_models.poisson_model import PoissonPredictor
            self.models['poisson'] = PoissonPredictor(model_configs['poisson'])
            
        if 'elo' in model_configs:
            from ..models.statistical_models.elo_system import ELOPredictor
            self.models['elo'] = ELOPredictor(model_configs['elo'])
            
        if 'bayesian' in model_configs:
            from ..models.statistical_models.bayesian_inference import BayesianPredictor
            self.models['bayesian'] = BayesianPredictor(model_configs['bayesian'])
    
    def _calculate_ensemble_weights(self):
        """Calculate weights for ensemble based on model performance"""
        # Use configured weights or calculate based on historical performance
        if 'weights' in self.config:
            self.ensemble_weights = self.config['weights']
        else:
            # Calculate weights based on model performance metrics
            performance_metrics = self._get_model_performance()
            total_performance = sum(performance_metrics.values())
            
            for model_name, performance in performance_metrics.items():
                self.ensemble_weights[model_name] = performance / total_performance
    
    def _initialize_calibrators(self):
        """Initialize probability calibrators for each model"""
        for model_name, model in self.models.items():
            self.calibrators[model_name] = CalibratedClassifierCV(
                model, method='isotonic', cv=3
            )
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble predictions combining all models
        """
        match_data = data['match_data']
        features = data['features']
        
        # Get predictions from all models
        individual_predictions = self._get_individual_predictions(features)
        
        # Apply ensemble combination
        ensemble_prediction = self._combine_predictions(individual_predictions)
        
        # Calculate confidence and uncertainty
        confidence_metrics = self._calculate_confidence(individual_predictions)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': individual_predictions,
            'model_weights': self.ensemble_weights,
            'confidence_metrics': confidence_metrics,
            'uncertainty_estimation': self._estimate_uncertainty(individual_predictions),
            'explanation': self._explain_prediction(ensemble_prediction, individual_predictions)
        }
    
    def _get_individual_predictions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get predictions from all individual models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # For scikit-learn models
                    proba = model.predict_proba(features)
                    predictions[model_name] = {
                        'probabilities': proba,
                        'prediction': model.predict(features)
                    }
                elif isinstance(model, nn.Module):
                    # For PyTorch models
                    with torch.no_grad():
                        model_output = model(features)
                        predictions[model_name] = {
                            'probabilities': torch.softmax(model_output, dim=1).numpy(),
                            'prediction': torch.argmax(model_output, dim=1).numpy()
                        }
            except Exception as e:
                self.logger.error(f"Error getting prediction from {model_name}: {e}")
                predictions[model_name] = None
                
        return predictions
    
    def _combine_predictions(self, individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions using weighted ensemble"""
        valid_predictions = {
            k: v for k, v in individual_predictions.items() 
            if v is not None
        }
        
        if not valid_predictions:
            raise ValueError("No valid predictions available for ensemble")
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(
            next(iter(valid_predictions.values()))['probabilities']
        )
        
        total_weight = 0
        for model_name, prediction in valid_predictions.items():
            weight = self.ensemble_weights.get(model_name, 0.1)
            ensemble_proba += prediction['probabilities'] * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_proba /= total_weight
        
        # Determine final prediction
        final_prediction = np.argmax(ensemble_proba, axis=1)
        
        return {
            'probabilities': ensemble_proba,
            'prediction': final_prediction,
            'confidence': np.max(ensemble_proba, axis=1)
        }
    
    def _calculate_confidence(self, individual_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence metrics for the ensemble prediction"""
        valid_predictions = [
            p for p in individual_predictions.values() 
            if p is not None
        ]
        
        if not valid_predictions:
            return {'overall_confidence': 0.0}
        
        # Agreement between models
        predictions = [p['prediction'] for p in valid_predictions]
        agreement = self._calculate_agreement(predictions)
        
        # Probability consistency
        prob_std = np.std([p['probabilities'] for p in valid_predictions], axis=0)
        consistency = 1.0 - np.mean(prob_std)
        
        return {
            'overall_confidence': (agreement + consistency) / 2,
            'model_agreement': agreement,
            'probability_consistency': consistency,
            'models_used': len(valid_predictions)
        }
    
    def _calculate_agreement(self, predictions: List[np.ndarray]) -> float:
        """Calculate agreement between model predictions"""
        if len(predictions) == 1:
            return 1.0
            
        agreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                agreement = np.mean(predictions[i] == predictions[j])
                agreements.append(agreement)
                
        return np.mean(agreements) if agreements else 0.0
    
    def _estimate_uncertainty(self, individual_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Estimate prediction uncertainty"""
        valid_probas = [
            p['probabilities'] for p in individual_predictions.values() 
            if p is not None
        ]
        
        if not valid_probas:
            return {'total_uncertainty': 1.0}
        
        # Calculate epistemic uncertainty (model uncertainty)
        proba_std = np.std(valid_probas, axis=0)
        epistemic_uncertainty = np.mean(proba_std)
        
        # Calculate aleatoric uncertainty (data uncertainty)
        mean_proba = np.mean(valid_probas, axis=0)
        aleatoric_uncertainty = -np.sum(mean_proba * np.log(mean_proba + 1e-8), axis=1).mean()
        
        return {
            'total_uncertainty': (epistemic_uncertainty + aleatoric_uncertainty) / 2,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty
        }
    
    def _explain_prediction(self, ensemble_pred: Dict[str, Any], 
                          individual_preds: Dict[str, Any]) -> List[str]:
        """Generate explanation for the ensemble prediction"""
        explanations = []
        
        # Model agreement explanation
        agreement = self._calculate_agreement([
            p['prediction'] for p in individual_preds.values() 
            if p is not None
        ])
        
        if agreement > 0.8:
            explanations.append("High model agreement indicates confident prediction")
        elif agreement < 0.5:
            explanations.append("Low model agreement suggests uncertain outcome")
            
        # Confidence level explanation
        avg_confidence = np.mean(ensemble_pred['confidence'])
        if avg_confidence > 0.7:
            explanations.append("High prediction confidence based on feature strength")
        elif avg_confidence < 0.4:
            explanations.append("Low prediction confidence suggests close match")
            
        # Model contribution explanation
        top_models = sorted(
            self.ensemble_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        
        if top_models:
            explanations.append(
                f"Primary models: {top_models[0][0]} ({top_models[0][1]:.2f})"
            )
            
        return explanations
    
    def _get_model_performance(self) -> Dict[str, float]:
        """Get historical performance metrics for all models"""
        # This would typically load from a model registry or database
        # For now, return default weights
        return {
            'transformer': 0.35,
            'lstm': 0.25,
            'poisson': 0.15,
            'elo': 0.15,
            'bayesian': 0.10
        }
