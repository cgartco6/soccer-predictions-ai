from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, validator
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ....agents import PredictionEnsembleAgent, SyntheticIntelligenceAgent, StrategicIntelligenceAgent
from ....agents import DataCollectorAgent
from ....utils.config import config
from ....utils.helpers import validate_match_data, format_prediction_output
from ....utils.logger import prediction_logger

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize agents (in a real app, these would be properly managed)
try:
    ensemble_agent = PredictionEnsembleAgent(config.get('agents.prediction_ensemble', {}))
    synthetic_agent = SyntheticIntelligenceAgent(config.get('agents.synthetic_intelligence', {}))
    strategic_agent = StrategicIntelligenceAgent(config.get('agents.strategic_intelligence', {}))
    data_agent = DataCollectorAgent(config.get('agents.data_collector', {}))
    
    # Initialize agents
    ensemble_agent.initialize()
    synthetic_agent.initialize()
    strategic_agent.initialize()
    data_agent.initialize()
    
except Exception as e:
    logger.error(f"Failed to initialize agents: {e}")
    # In production, you might want to fail fast or use fallbacks

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    
    home_team: str
    away_team: str
    league: str
    match_date: Optional[str] = None
    venue: Optional[str] = None
    
    # Optional context data
    home_team_form: Optional[Dict[str, Any]] = None
    away_team_form: Optional[Dict[str, Any]] = None
    head_to_head: Optional[Dict[str, Any]] = None
    weather_conditions: Optional[Dict[str, Any]] = None
    player_availability: Optional[Dict[str, Any]] = None
    
    # Optional odds data
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    
    @validator('home_team', 'away_team', 'league')
    def validate_required_fields(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('home_odds', 'draw_odds', 'away_odds')
    def validate_odds(cls, v):
        if v is not None and v < 1.0:
            raise ValueError('Odds must be greater than or equal to 1.0')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    
    prediction_id: str
    home_team: str
    away_team: str
    league: str
    match_date: Optional[str]
    predicted_result: str
    confidence: float
    probabilities: Dict[str, float]
    expected_goals: Optional[Dict[str, float]] = None
    explanation: List[str]
    model_used: str
    timestamp: str
    bookmaker_odds: Optional[Dict[str, float]] = None
    ai_analysis: Optional[Dict[str, Any]] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    matches: List[PredictionRequest]
    include_analysis: bool = False

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

@router.post("/predict", response_model=PredictionResponse)
async def predict_match(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    include_analysis: bool = False
):
    """
    Generate AI prediction for a soccer match
    
    Uses synthetic intelligence, strategic intelligence, and ensemble models
    to provide accurate predictions with confidence scores and explanations.
    """
    try:
        # Validate input data
        match_data = request.dict()
        is_valid, errors = validate_match_data(match_data)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})
        
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(match_data))}"
        prediction_logger.set_prediction_id(prediction_id)
        
        logger.info(f"Generating prediction for {match_data['home_team']} vs {match_data['away_team']}")
        
        # Collect additional data if needed
        if include_analysis:
            additional_data = await data_agent.process({
                'home_team': match_data['home_team'],
                'away_team': match_data['away_team'],
                'league': match_data['league']
            })
            match_data.update(additional_data.get('collected_data', {}))
        
        # Generate synthetic data and simulations
        synthetic_data = synthetic_agent.process({
            'current_matches': [match_data],
            'historical_matches': [],  # Would be loaded from database
            'num_simulations': 10000
        })
        
        # Perform strategic analysis
        strategic_analysis = strategic_agent.process({
            'match_data': match_data,
            'team_data': {},  # Would be loaded from database
            'context_data': match_data.get('weather_conditions', {})
        })
        
        # Generate ensemble prediction
        ensemble_prediction = ensemble_agent.process({
            'match_data': match_data,
            'features': {},  # Would be engineered from match_data
            'synthetic_data': synthetic_data,
            'strategic_analysis': strategic_analysis
        })
        
        # Format response
        prediction_result = ensemble_prediction['ensemble_prediction']
        confidence = prediction_result['confidence_metrics']['overall_confidence']
        
        # Determine predicted result
        prob_home = prediction_result['probabilities'][0]
        prob_draw = prediction_result['probabilities'][1]
        prob_away = prediction_result['probabilities'][2]
        
        if prob_home >= prob_draw and prob_home >= prob_away:
            predicted_result = "home_win"
        elif prob_draw >= prob_home and prob_draw >= prob_away:
            predicted_result = "draw"
        else:
            predicted_result = "away_win"
        
        # Generate explanation
        explanation = [
            f"AI ensemble prediction with {confidence:.1%} confidence",
            f"Based on {synthetic_data.get('num_simulations', 0)} match simulations",
            f"Strategic analysis: {len(strategic_analysis.get('strategic_insights', []))} key factors identified"
        ]
        
        # Add feature-based explanations if available
        if 'feature_importance' in ensemble_prediction:
            top_features = sorted(
                ensemble_prediction['feature_importance'].items(),
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:2]
            
            for feature, importance in top_features:
                if abs(importance) > 0.1:
                    if 'elo' in feature:
                        explanation.append("Team strength analysis heavily weighted")
                    elif 'form' in feature:
                        explanation.append("Recent performance strongly considered")
        
        response_data = {
            'prediction_id': prediction_id,
            'home_team': match_data['home_team'],
            'away_team': match_data['away_team'],
            'league': match_data['league'],
            'match_date': match_data.get('match_date'),
            'predicted_result': predicted_result,
            'confidence': confidence,
            'probabilities': {
                'home_win': prob_home,
                'draw': prob_draw,
                'away_win': prob_away
            },
            'explanation': explanation,
            'model_used': 'ensemble_ai',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add expected goals if available
        if 'expected_goals' in prediction_result:
            response_data['expected_goals'] = {
                'home': prediction_result['expected_goals']['home'],
                'away': prediction_result['expected_goals']['away']
            }
        
        # Add bookmaker odds if provided
        if any([match_data.get('home_odds'), match_data.get('draw_odds'), match_data.get('away_odds')]):
            response_data['bookmaker_odds'] = {
                'home': match_data.get('home_odds'),
                'draw': match_data.get('draw_odds'),
                'away': match_data.get('away_odds')
            }
        
        # Add AI analysis if requested
        if include_analysis:
            response_data['ai_analysis'] = {
                'synthetic_simulations': synthetic_data.get('simulations', {}),
                'strategic_insights': strategic_analysis.get('strategic_insights', []),
                'model_confidence': ensemble_prediction['confidence_metrics']
            }
        
        # Log prediction
        prediction_logger.log_prediction(
            match_data, 
            predicted_result, 
            confidence, 
            'ensemble_ai'
        )
        
        # Schedule background task to track prediction outcome
        background_tasks.add_task(
            track_prediction_outcome,
            prediction_id,
            match_data,
            predicted_result,
            confidence
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_matches(request: BatchPredictionRequest):
    """
    Generate predictions for multiple matches in batch
    """
    try:
        predictions = []
        successful = 0
        failed = 0
        
        for match_request in request.matches:
            try:
                # Use the single prediction endpoint logic for each match
                prediction = await predict_match(match_request, BackgroundTasks(), request.include_analysis)
                predictions.append(prediction)
                successful += 1
            except Exception as e:
                logger.error(f"Batch prediction failed for {match_request.home_team} vs {match_request.away_team}: {e}")
                failed += 1
        
        summary = {
            "total_matches": len(request.matches),
            "successful_predictions": successful,
            "failed_predictions": failed,
            "success_rate": successful / len(request.matches) if request.matches else 0
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/predict/odds-comparison")
async def get_odds_comparison(home_team: str, away_team: str, league: str):
    """
    Get odds comparison from multiple bookmakers for a match
    """
    try:
        # This would integrate with the data collector to get real-time odds
        odds_data = {
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "odds": {
                "hollywoodbets": {
                    "home_odds": 2.10,
                    "draw_odds": 3.25,
                    "away_odds": 3.50
                },
                "betway": {
                    "home_odds": 2.05,
                    "draw_odds": 3.30,
                    "away_odds": 3.60
                }
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return odds_data
        
    except Exception as e:
        logger.error(f"Odds comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Odds comparison failed: {str(e)}")

async def track_prediction_outcome(prediction_id: str, match_data: Dict[str, Any],
                                 predicted_result: str, confidence: float):
    """
    Background task to track prediction outcomes for model improvement
    
    In a real implementation, this would:
    1. Store the prediction in a database
    2. Schedule a job to check the actual result after the match
    3. Update model performance metrics
    """
    try:
        # This would be implemented with proper database storage
        logger.info(f"Tracking prediction {prediction_id} for outcome analysis")
        
        # Simulate storing prediction for later evaluation
        # In production, this would use a proper database
        prediction_record = {
            'prediction_id': prediction_id,
            'match_data': match_data,
            'predicted_result': predicted_result,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # Here you would store to database
        # await database.store_prediction(prediction_record)
        
    except Exception as e:
        logger.error(f"Error tracking prediction outcome: {e}")

@router.get("/predict/history")
async def get_prediction_history(limit: int = 10, offset: int = 0):
    """
    Get prediction history (would connect to database in production)
    """
    try:
        # This would query a database in production
        # For now, return mock data
        return {
            "predictions": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve prediction history")
