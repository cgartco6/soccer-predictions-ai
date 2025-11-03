from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ....agents import PredictionEnsembleAgent
from ....utils.config import config
from ....utils.metrics import ModelEvaluator

router = APIRouter()
logger = logging.getLogger(__name__)

class ModelInfo(BaseModel):
    """Model information response"""
    
    model_name: str
    model_type: str
    version: str
    status: str
    performance_metrics: Dict[str, float]
    last_trained: Optional[str]
    features_used: List[str]

class TrainingRequest(BaseModel):
    """Request model for model training"""
    
    model_type: str
    retrain_all: bool = False
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingResponse(BaseModel):
    """Response model for training endpoint"""
    
    training_id: str
    model_type: str
    status: str
    message: str
    started_at: str
    estimated_completion: Optional[str] = None

@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """
    Get information about all available prediction models
    """
    try:
        models_info = []
        
        # Ensemble model info
        models_info.append(ModelInfo(
            model_name="ensemble_ai",
            model_type="ensemble",
            version="1.0.0",
            status="active",
            performance_metrics={
                "accuracy": 0.65,
                "precision": 0.64,
                "recall": 0.65,
                "f1_score": 0.64
            },
            last_trained="2024-01-15T10:30:00Z",
            features_used=["team_strength", "recent_form", "head_to_head", "tactical_analysis"]
        ))
        
        # Transformer model info
        models_info.append(ModelInfo(
            model_name="transformer_v1",
            model_type="neural_network",
            version="1.0.0",
            status="active",
            performance_metrics={
                "accuracy": 0.63,
                "precision": 0.62,
                "recall": 0.63,
                "f1_score": 0.62
            },
            last_trained="2024-01-10T14:20:00Z",
            features_used=["sequence_data", "team_embeddings", "temporal_patterns"]
        ))
        
        # LSTM model info
        models_info.append(ModelInfo(
            model_name="lstm_v1",
            model_type="neural_network",
            version="1.0.0",
            status="active",
            performance_metrics={
                "accuracy": 0.62,
                "precision": 0.61,
                "recall": 0.62,
                "f1_score": 0.61
            },
            last_trained="2024-01-08T09:15:00Z",
            features_used=["time_series", "form_sequences", "match_patterns"]
        ))
        
        return models_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_details(model_name: str):
    """
    Get detailed information about a specific model
    """
    try:
        # This would fetch detailed model information from model registry
        if model_name == "ensemble_ai":
            return ModelInfo(
                model_name="ensemble_ai",
                model_type="ensemble",
                version="1.0.0",
                status="active",
                performance_metrics={
                    "accuracy": 0.65,
                    "precision": 0.64,
                    "recall": 0.65,
                    "f1_score": 0.64,
                    "log_loss": 0.95,
                    "roc_auc": 0.72
                },
                last_trained="2024-01-15T10:30:00Z",
                features_used=[
                    "team_strength", "recent_form", "head_to_head", 
                    "tactical_analysis", "player_performance", "contextual_factors"
                ]
            )
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model details")

@router.post("/models/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training for a specific model type
    """
    try:
        training_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.model_type}"
        
        # Validate model type
        valid_model_types = ["ensemble", "transformer", "lstm", "poisson", "elo", "bayesian"]
        if request.model_type not in valid_model_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model type. Must be one of: {valid_model_types}"
            )
        
        # Start training in background
        background_tasks.add_task(
            train_model_background,
            training_id,
            request.model_type,
            request.retrain_all,
            request.hyperparameters or {}
        )
        
        return TrainingResponse(
            training_id=training_id,
            model_type=request.model_type,
            status="started",
            message=f"Training initiated for {request.model_type} model",
            started_at=datetime.now().isoformat(),
            estimated_completion=(
                datetime.now().replace(hour=datetime.now().hour + 1).isoformat()
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start model training")

@router.get("/models/performance")
async def get_model_performance(days: int = 30):
    """
    Get model performance metrics over time
    """
    try:
        # This would query performance metrics from database
        # For now, return mock data
        performance_data = {
            "time_period": f"last_{days}_days",
            "models": {
                "ensemble_ai": {
                    "accuracy": 0.65,
                    "precision": 0.64,
                    "recall": 0.65,
                    "f1_score": 0.64,
                    "confidence": 0.72,
                    "predictions_count": 1250
                },
                "transformer_v1": {
                    "accuracy": 0.63,
                    "precision": 0.62,
                    "recall": 0.63,
                    "f1_score": 0.62,
                    "confidence": 0.68,
                    "predictions_count": 980
                }
            },
            "trends": {
                "overall_accuracy_trend": "stable",
                "best_performing_model": "ensemble_ai",
                "improvement_since_last_period": 0.02
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model performance")

@router.post("/models/evaluate")
async def evaluate_model(model_name: str, test_data: Optional[Dict[str, Any]] = None):
    """
    Evaluate model performance on test data
    """
    try:
        # This would perform comprehensive model evaluation
        evaluator = ModelEvaluator()
        
        # In a real implementation, you would:
        # 1. Load the model
        # 2. Prepare test data
        # 3. Run evaluation
        # 4. Return results
        
        evaluation_results = {
            "model_name": model_name,
            "evaluation_date": datetime.now().isoformat(),
            "metrics": {
                "accuracy": 0.65,
                "precision": 0.64,
                "recall": 0.65,
                "f1_score": 0.64,
                "log_loss": 0.95,
                "roc_auc": 0.72
            },
            "confusion_matrix": [
                [45, 15, 10],
                [12, 38, 20],
                [8, 18, 34]
            ],
            "feature_importance": {
                "team_strength": 0.23,
                "recent_form": 0.18,
                "head_to_head": 0.15,
                "home_advantage": 0.12,
                "player_performance": 0.10
            }
        }
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")

@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    Delete a specific model version
    """
    try:
        # This would remove the model from the model registry
        # In production, you might want to archive instead of delete
        
        logger.info(f"Deleting model: {model_name}")
        
        return {
            "message": f"Model {model_name} scheduled for deletion",
            "deletion_scheduled_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

async def train_model_background(training_id: str, model_type: str, 
                               retrain_all: bool, hyperparameters: Dict[str, Any]):
    """
    Background task for model training
    """
    try:
        logger.info(f"Starting background training {training_id} for {model_type}")
        
        # Simulate training process
        # In production, this would:
        # 1. Load training data
        # 2. Preprocess features
        # 3. Train model
        # 4. Evaluate performance
        # 5. Save to model registry
        
        # Simulate training time
        import asyncio
        await asyncio.sleep(10)  # Simulate training process
        
        logger.info(f"Completed background training {training_id} for {model_type}")
        
        # Here you would update training status in database
        # await database.update_training_status(training_id, "completed")
        
    except Exception as e:
        logger.error(f"Background training failed {training_id}: {e}")
        # Update status to failed
        # await database.update_training_status(training_id, "failed")
