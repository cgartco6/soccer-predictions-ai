import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import logging

class PredictionMetrics:
    """
    Calculation and tracking of prediction performance metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate classification metrics for predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Probability-based metrics
            if y_prob is not None:
                try:
                    metrics['log_loss'] = log_loss(y_true, y_prob)
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except Exception as e:
                    self.logger.warning(f"Could not calculate probability metrics: {e}")
            
            # Confidence metrics
            if y_prob is not None:
                max_probs = np.max(y_prob, axis=1)
                metrics['avg_confidence'] = float(np.mean(max_probs))
                metrics['confidence_std'] = float(np.std(max_probs))
                
                # Calibration metrics (simplified)
                correct_predictions = (y_true == y_pred)
                if len(correct_predictions) > 0:
                    metrics['calibration'] = float(
                        np.mean(max_probs[correct_predictions]) - 
                        np.mean(max_probs[~correct_predictions])
                    )
            
            # Additional custom metrics
            metrics = self._calculate_custom_metrics(metrics, y_true, y_pred, y_prob)
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            metrics = {'error': str(e)}
            
        return metrics
    
    def calculate_betting_metrics(self, predictions: pd.DataFrame, 
                                odds_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate betting performance metrics
        
        Args:
            predictions: DataFrame with predictions
            odds_data: DataFrame with odds information
            
        Returns:
            Dictionary of betting metrics
        """
        metrics = {}
        
        try:
            # Merge predictions with odds
            merged_data = pd.merge(
                predictions, odds_data, 
                on=['home_team', 'away_team', 'date'],
                how='inner'
            )
            
            if merged_data.empty:
                return {'error': 'No matching odds data found'}
            
            # Calculate returns for different strategies
            metrics.update(self._calculate_kelly_returns(merged_data))
            metrics.update(self._calculate_value_betting_returns(merged_data))
            metrics.update(self._calculate_fixed_stake_returns(merged_data))
            
            # Risk metrics
            metrics.update(self._calculate_risk_metrics(merged_data))
            
        except Exception as e:
            self.logger.error(f"Error calculating betting metrics: {e}")
            metrics = {'error': str(e)}
            
        return metrics
    
    def _calculate_custom_metrics(self, metrics: Dict[str, float],
                                y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate custom soccer prediction metrics"""
        # Match-specific metrics
        if len(y_true) > 0:
            # Draw prediction accuracy (often hardest to predict)
            draw_mask = y_true == 1  # Assuming draw is class 1
            if draw_mask.any():
                metrics['draw_accuracy'] = accuracy_score(
                    y_true[draw_mask], y_pred[draw_mask]
                )
            
            # Home win prediction accuracy
            home_win_mask = y_true == 0  # Assuming home win is class 0
            if home_win_mask.any():
                metrics['home_win_accuracy'] = accuracy_score(
                    y_true[home_win_mask], y_pred[home_win_mask]
                )
            
            # Away win prediction accuracy
            away_win_mask = y_true == 2  # Assuming away win is class 2
            if away_win_mask.any():
                metrics['away_win_accuracy'] = accuracy_score(
                    y_true[away_win_mask], y_pred[away_win_mask]
                )
        
        return metrics
    
    def _calculate_kelly_returns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate returns using Kelly Criterion"""
        # Simplified Kelly Criterion implementation
        returns = {}
        
        try:
            # Calculate implied probabilities from odds
            data['implied_home'] = 1 / data['home_odds']
            data['implied_draw'] = 1 / data['draw_odds']
            data['implied_away'] = 1 / data['away_odds']
            
            # Calculate Kelly fraction for each bet
            kelly_fractions = []
            actual_returns = []
            
            for _, row in data.iterrows():
                pred_class = row['predicted_class']
                actual_class = row['actual_class']
                confidence = row['prediction_confidence']
                
                if pred_class == 0:  # Home win
                    implied_prob = row['implied_home']
                    odds = row['home_odds']
                elif pred_class == 1:  # Draw
                    implied_prob = row['implied_draw']
                    odds = row['draw_odds']
                else:  # Away win
                    implied_prob = row['implied_away']
                    odds = row['away_odds']
                
                # Kelly fraction: (p * odds - 1) / (odds - 1)
                kelly_fraction = (confidence * odds - 1) / (odds - 1) if odds > 1 else 0
                kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
                
                kelly_fractions.append(kelly_fraction)
                
                # Calculate actual return
                if pred_class == actual_class:
                    actual_returns.append(kelly_fraction * (odds - 1))
                else:
                    actual_returns.append(-kelly_fraction)
            
            if kelly_fractions:
                returns['avg_kelly_fraction'] = float(np.mean(kelly_fractions))
                returns['kelly_total_return'] = float(np.sum(actual_returns))
                returns['kelly_roi'] = returns['kelly_total_return'] / np.sum(kelly_fractions) if np.sum(kelly_fractions) > 0 else 0
        
        except Exception as e:
            self.logger.warning(f"Error in Kelly calculation: {e}")
            
        return returns
    
    def _calculate_value_betting_returns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate returns for value betting strategy"""
        returns = {}
        
        try:
            # Bet when predicted probability > implied probability + margin
            value_threshold = 0.05  # 5% value threshold
            
            value_bets = []
            bet_returns = []
            
            for _, row in data.iterrows():
                home_value = row['predicted_home_win'] - row.get('implied_home', 0)
                draw_value = row['predicted_draw'] - row.get('implied_draw', 0)
                away_value = row['predicted_away_win'] - row.get('implied_away', 0)
                
                max_value = max(home_value, draw_value, away_value)
                
                if max_value > value_threshold:
                    if max_value == home_value:
                        odds = row['home_odds']
                        won = (row['actual_class'] == 0)
                    elif max_value == draw_value:
                        odds = row['draw_odds']
                        won = (row['actual_class'] == 1)
                    else:
                        odds = row['away_odds']
                        won = (row['actual_class'] == 2)
                    
                    value_bets.append(1)  # 1 unit stake
                    if won:
                        bet_returns.append(odds - 1)
                    else:
                        bet_returns.append(-1)
            
            if value_bets:
                returns['value_bets_count'] = len(value_bets)
                returns['value_bets_roi'] = sum(bet_returns) / len(value_bets) if value_bets else 0
                returns['value_bets_total_return'] = sum(bet_returns)
        
        except Exception as e:
            self.logger.warning(f"Error in value betting calculation: {e}")
            
        return returns
    
    def _calculate_fixed_stake_returns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate returns for fixed stake betting"""
        returns = {}
        
        try:
            stake = 1  # Fixed 1 unit stake
            total_stake = len(data) * stake
            total_return = 0
            
            for _, row in data.iterrows():
                pred_class = row['predicted_class']
                actual_class = row['actual_class']
                
                if pred_class == 0:  # Home win
                    odds = row['home_odds']
                elif pred_class == 1:  # Draw
                    odds = row['draw_odds']
                else:  # Away win
                    odds = row['away_odds']
                
                if pred_class == actual_class:
                    total_return += (odds - 1) * stake
                else:
                    total_return -= stake
            
            returns['fixed_stake_total_return'] = total_return
            returns['fixed_stake_roi'] = total_return / total_stake if total_stake > 0 else 0
        
        except Exception as e:
            self.logger.warning(f"Error in fixed stake calculation: {e}")
            
        return returns
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for betting strategy"""
        risk_metrics = {}
        
        try:
            # Calculate returns series for risk analysis
            returns = []
            
            for _, row in data.iterrows():
                pred_class = row['predicted_class']
                actual_class = row['actual_class']
                
                if pred_class == 0:
                    odds = row['home_odds']
                elif pred_class == 1:
                    odds = row['draw_odds']
                else:
                    odds = row['away_odds']
                
                if pred_class == actual_class:
                    returns.append(odds - 1)
                else:
                    returns.append(-1)
            
            if returns:
                returns_series = np.array(returns)
                risk_metrics['max_drawdown'] = float(self._calculate_max_drawdown(returns_series))
                risk_metrics['sharpe_ratio'] = float(self._calculate_sharpe_ratio(returns_series))
                risk_metrics['volatility'] = float(np.std(returns_series))
                risk_metrics['win_rate'] = float(np.mean(np.array(returns) > 0))
        
        except Exception as e:
            self.logger.warning(f"Error in risk calculation: {e}")
            
        return risk_metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-8)
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns)

class ModelEvaluator:
    """
    Comprehensive model evaluation for soccer predictions
    """
    
    def __init__(self):
        self.metrics_calculator = PredictionMetrics()
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features (for interpretability)
            
        Returns:
            Comprehensive evaluation results
        """
        evaluation = {}
        
        try:
            # Generate predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                y_pred = np.argmax(y_prob, axis=1)
            else:
                y_pred = model.predict(X_test)
                y_prob = None
            
            # Calculate metrics
            evaluation['classification_metrics'] = self.metrics_calculator.calculate_classification_metrics(
                y_test, y_pred, y_prob
            )
            
            # Feature importance (if available)
            if feature_names is not None:
                evaluation['feature_importance'] = self._get_feature_importance(
                    model, feature_names
                )
            
            # Confidence analysis
            if y_prob is not None:
                evaluation['confidence_analysis'] = self._analyze_confidence(y_prob, y_test, y_pred)
            
            # Error analysis
            evaluation['error_analysis'] = self._analyze_errors(y_test, y_pred)
            
        except Exception as e:
            evaluation['error'] = str(e)
            
        return evaluation
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for name, imp in zip(feature_names, importances):
                    importance[name] = float(imp)
            
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) > 1:
                    # Multi-class
                    for i, coefs in enumerate(model.coef_):
                        for name, coef in zip(feature_names, coefs):
                            importance[f'{name}_class_{i}'] = float(coef)
                else:
                    # Binary
                    for name, coef in zip(feature_names, model.coef_):
                        importance[name] = float(coef)
        
        except Exception as e:
            importance['error'] = str(e)
            
        return importance
    
    def _analyze_confidence(self, y_prob: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence patterns"""
        analysis = {}
        
        try:
            max_probs = np.max(y_prob, axis=1)
            correct = (y_true == y_pred)
            
            analysis['confidence_vs_accuracy'] = {
                'high_confidence_accuracy': float(np.mean(correct[max_probs > 0.7])),
                'medium_confidence_accuracy': float(np.mean(correct[(max_probs > 0.5) & (max_probs <= 0.7)])),
                'low_confidence_accuracy': float(np.mean(correct[max_probs <= 0.5]))
            }
            
            analysis['calibration'] = {
                'expected_vs_actual_high': float(np.mean(max_probs[max_probs > 0.7]) - np.mean(correct[max_probs > 0.7])),
                'expected_vs_actual_medium': float(np.mean(max_probs[(max_probs > 0.5) & (max_probs <= 0.7)]) - np.mean(correct[(max_probs > 0.5) & (max_probs <= 0.7)])),
                'expected_vs_actual_low': float(np.mean(max_probs[max_probs <= 0.5]) - np.mean(correct[max_probs <= 0.5]))
            }
        
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors"""
        analysis = {}
        
        try:
            # Confusion matrix analysis
            cm = confusion_matrix(y_true, y_pred)
            analysis['confusion_matrix'] = cm.tolist()
            
            # Error patterns by class
            errors = (y_true != y_pred)
            if errors.any():
                analysis['error_rates_by_class'] = {
                    'class_0_errors': float(np.mean(errors[y_true == 0])),
                    'class_1_errors': float(np.mean(errors[y_true == 1])),
                    'class_2_errors': float(np.mean(errors[y_true == 2]))
                }
            
            # Most common error types
            error_pairs = []
            for true, pred in zip(y_true[errors], y_pred[errors]):
                error_pairs.append((int(true), int(pred)))
            
            from collections import Counter
            common_errors = Counter(error_pairs).most_common(5)
            analysis['common_errors'] = [f"True: {t}, Pred: {p} (count: {c})" 
                                       for (t, p), c in common_errors]
        
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis
