import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import re

def format_odds(odds: float) -> str:
    """
    Format odds to standardized string representation
    
    Args:
        odds: Decimal odds value
        
    Returns:
        Formatted odds string
    """
    if odds is None or np.isnan(odds):
        return "N/A"
    
    if odds < 2.0:
        return f"{odds:.2f}"
    elif odds < 10.0:
        return f"{odds:.2f}"
    else:
        return f"{odds:.1f}"

def calculate_implied_probability(odds: float, method: str = 'basic') -> float:
    """
    Calculate implied probability from odds
    
    Args:
        odds: Decimal odds
        method: Probability calculation method ('basic', 'margin_adjusted')
        
    Returns:
        Implied probability
    """
    if odds is None or odds < 1.0:
        return 0.0
    
    if method == 'basic':
        return 1.0 / odds
    elif method == 'margin_adjusted':
        # Adjust for bookmaker margin using basic method
        return 1.0 / odds
    else:
        raise ValueError(f"Unknown probability calculation method: {method}")

def validate_match_data(match_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate match data for prediction
    
    Args:
        match_data: Dictionary containing match data
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['home_team', 'away_team', 'league']
    for field in required_fields:
        if field not in match_data or not match_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Team validation
    if 'home_team' in match_data and 'away_team' in match_data:
        if match_data['home_team'] == match_data['away_team']:
            errors.append("Home and away teams cannot be the same")
    
    # Date validation
    if 'date' in match_data:
        try:
            if isinstance(match_data['date'], str):
                pd.to_datetime(match_data['date'])
            elif not isinstance(match_data['date'], (datetime, pd.Timestamp)):
                errors.append("Invalid date format")
        except (ValueError, TypeError):
            errors.append("Invalid date format")
    
    # Odds validation (if present)
    odds_fields = ['home_odds', 'draw_odds', 'away_odds']
    for field in odds_fields:
        if field in match_data and match_data[field] is not None:
            if match_data[field] < 1.0:
                errors.append(f"Invalid {field}: must be >= 1.0")
    
    return len(errors) == 0, errors

def generate_prediction_explanation(prediction: Dict[str, Any], 
                                  feature_importance: Dict[str, float],
                                  match_data: Dict[str, Any]) -> List[str]:
    """
    Generate human-readable explanation for prediction
    
    Args:
        prediction: Prediction results
        feature_importance: Feature importance scores
        match_data: Original match data
        
    Returns:
        List of explanation strings
    """
    explanations = []
    
    try:
        # Get prediction details
        predicted_result = prediction.get('predicted_result', 'Unknown')
        confidence = prediction.get('confidence', 0.5)
        
        # Base explanation
        if confidence > 0.7:
            explanations.append(f"High confidence prediction: {predicted_result}")
        elif confidence > 0.5:
            explanations.append(f"Moderate confidence prediction: {predicted_result}")
        else:
            explanations.append(f"Low confidence prediction: {predicted_result}")
        
        # Feature-based explanations
        top_features = sorted(feature_importance.items(), 
                             key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for feature, importance in top_features:
            if abs(importance) > 0.1:  # Significant feature
                if 'elo' in feature.lower():
                    explanations.append(f"Team strength (ELO rating) strongly influences this prediction")
                elif 'form' in feature.lower():
                    explanations.append(f"Recent team form is a key factor")
                elif 'home_advantage' in feature.lower():
                    explanations.append(f"Home advantage plays significant role")
                elif 'head_to_head' in feature.lower():
                    explanations.append(f"Historical performance between teams is important")
        
        # Contextual explanations
        if 'home_team' in match_data and 'away_team' in match_data:
            home_team = match_data['home_team']
            away_team = match_data['away_team']
            
            # Add team-specific context if available
            explanations.append(f"Analysis considers {home_team} vs {away_team} dynamics")
        
        # Odds-based explanations (if available)
        if all(field in match_data for field in ['home_odds', 'draw_odds', 'away_odds']):
            bookmaker_favorite = None
            min_odds = min(match_data['home_odds'], match_data['draw_odds'], match_data['away_odds'])
            
            if match_data['home_odds'] == min_odds:
                bookmaker_favorite = "home win"
            elif match_data['draw_odds'] == min_odds:
                bookmaker_favorite = "draw"
            else:
                bookmaker_favorite = "away win"
            
            if bookmaker_favorite:
                explanations.append(f"Bookmakers favor {bookmaker_favorite}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error generating explanation: {e}")
        explanations.append("Prediction based on comprehensive AI analysis")
    
    return explanations

def calculate_expected_value(probability: float, odds: float, stake: float = 1.0) -> float:
    """
    Calculate expected value for a bet
    
    Args:
        probability: True probability of outcome
        odds: Decimal odds offered
        stake: Bet stake (default 1 unit)
        
    Returns:
        Expected value
    """
    if probability <= 0 or odds <= 1.0:
        return 0.0
    
    win_return = (odds - 1) * stake
    loss_return = -stake
    
    expected_value = (probability * win_return) + ((1 - probability) * loss_return)
    return expected_value

def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name to standard format
    
    Args:
        team_name: Raw team name
        
    Returns:
        Normalized team name
    """
    if not team_name or not isinstance(team_name, str):
        return "Unknown"
    
    # Common replacements and standardizations
    replacements = {
        r'\bFC\b': 'FC',
        r'\bAFC\b': 'AFC',
        r'\bUnited\b': 'United',
        r'\bCity\b': 'City',
        r'\bHotspur\b': 'Hotspur',
        r'\bSpurs\b': 'Spurs',
        r'\bWanderers\b': 'Wanderers',
        r'\bAthletic\b': 'Athletic',
        r'\bRovers\b': 'Rovers',
    }
    
    # Remove extra spaces and title case
    normalized = re.sub(r'\s+', ' ', team_name.strip()).title()
    
    # Apply replacements
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)
    
    # Specific team name standardizations
    team_mapping = {
        'Man United': 'Manchester United',
        'Man Utd': 'Manchester United',
        'Man City': 'Manchester City',
        'Spurs': 'Tottenham Hotspur',
        'Real Mad.': 'Real Madrid',
        'Barca': 'Barcelona',
        'PSG': 'Paris Saint-Germain',
        'Bayern': 'Bayern Munich',
        'Inter': 'Inter Milan',
        'Milan': 'AC Milan',
    }
    
    return team_mapping.get(normalized, normalized)

def calculate_confidence_interval(predictions: List[float], 
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for predictions
    
    Args:
        predictions: List of prediction probabilities
        confidence_level: Confidence level (0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not predictions:
        return 0.0, 0.0
    
    predictions_array = np.array(predictions)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array)
    
    # Z-score for confidence level
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    margin_of_error = z_score * (std / np.sqrt(len(predictions_array)))
    
    lower_bound = max(0.0, mean - margin_of_error)
    upper_bound = min(1.0, mean + margin_of_error)
    
    return float(lower_bound), float(upper_bound)

def format_prediction_output(prediction: Dict[str, Any], 
                           include_explanation: bool = True) -> Dict[str, Any]:
    """
    Format prediction output for API response
    
    Args:
        prediction: Raw prediction dictionary
        include_explanation: Whether to include explanation
        
    Returns:
        Formatted prediction output
    """
    formatted = {
        'prediction_id': prediction.get('prediction_id', ''),
        'home_team': prediction.get('home_team', ''),
        'away_team': prediction.get('away_team', ''),
        'league': prediction.get('league', ''),
        'match_date': prediction.get('match_date', ''),
        'predicted_result': prediction.get('predicted_result', ''),
        'confidence': round(prediction.get('confidence', 0.0), 3),
        'probabilities': {
            'home_win': round(prediction.get('home_win_probability', 0.0), 3),
            'draw': round(prediction.get('draw_probability', 0.0), 3),
            'away_win': round(prediction.get('away_win_probability', 0.0), 3)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Add expected goals if available
    if 'expected_goals' in prediction:
        formatted['expected_goals'] = {
            'home': round(prediction['expected_goals']['home'], 2),
            'away': round(prediction['expected_goals']['away'], 2)
        }
    
    # Add explanation if requested
    if include_explanation and 'explanation' in prediction:
        formatted['explanation'] = prediction['explanation']
    
    # Add model information
    if 'model_used' in prediction:
        formatted['model_used'] = prediction['model_used']
    
    return formatted

def calculate_performance_metrics(predictions: pd.DataFrame, 
                                actual_results: pd.Series) -> Dict[str, float]:
    """
    Calculate performance metrics for a set of predictions
    
    Args:
        predictions: DataFrame with predictions
        actual_results: Series with actual results
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    try:
        # Ensure alignment
        aligned_data = pd.concat([predictions, actual_results], axis=1, join='inner')
        
        if aligned_data.empty:
            return {'error': 'No aligned data for metric calculation'}
        
        # Basic accuracy
        correct_predictions = (aligned_data['predicted_result'] == aligned_data['actual_result'])
        metrics['accuracy'] = float(correct_predictions.mean())
        
        # Confidence-weighted accuracy
        if 'confidence' in aligned_data.columns:
            weighted_accuracy = (correct_predictions * aligned_data['confidence']).sum() / aligned_data['confidence'].sum()
            metrics['confidence_weighted_accuracy'] = float(weighted_accuracy)
        
        # Brier score (for probability calibration)
        if all(col in aligned_data.columns for col in ['home_win_probability', 'draw_probability', 'away_win_probability']):
            # Convert actual results to one-hot encoding
            actual_probs = pd.get_dummies(aligned_data['actual_result'])
            
            # Get predicted probabilities
            pred_probs = aligned_data[['home_win_probability', 'draw_probability', 'away_win_probability']]
            
            # Calculate Brier score
            brier_score = ((pred_probs - actual_probs) ** 2).mean().mean()
            metrics['brier_score'] = float(brier_score)
        
        # ROI calculation (if odds are available)
        if all(col in aligned_data.columns for col in ['home_odds', 'draw_odds', 'away_odds']):
            # Simplified ROI calculation
            stake = 1.0
            total_return = 0.0
            
            for _, row in aligned_data.iterrows():
                if row['predicted_result'] == 0:  # Home win
                    odds = row['home_odds']
                elif row['predicted_result'] == 1:  # Draw
                    odds = row['draw_odds']
                else:  # Away win
                    odds = row['away_odds']
                
                if row['predicted_result'] == row['actual_result']:
                    total_return += (odds - 1) * stake
                else:
                    total_return -= stake
            
            metrics['roi'] = float(total_return / (len(aligned_data) * stake))
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics
