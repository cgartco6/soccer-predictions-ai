import logging
import sys
from typing import Optional
import json
from datetime import datetime
import os

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'props') and isinstance(record.props, dict):
            log_entry.update(record.props)
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging(level: str = 'INFO', 
                 json_format: bool = False,
                 log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON formatting
        log_file: Optional file to write logs to
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(module)s:%(funcName)s:%(lineno)d]'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get logger with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

class PredictionLogger:
    """
    Specialized logger for prediction-related events
    """
    
    def __init__(self, name: str = 'predictions'):
        self.logger = get_logger(name)
        self.prediction_id = None
    
    def set_prediction_id(self, prediction_id: str):
        """Set current prediction ID for correlation"""
        self.prediction_id = prediction_id
    
    def log_prediction(self, match_data: dict, prediction: dict, 
                      confidence: float, model_used: str):
        """Log prediction event"""
        extra = {
            'props': {
                'prediction_id': self.prediction_id,
                'home_team': match_data.get('home_team'),
                'away_team': match_data.get('away_team'),
                'prediction': prediction,
                'confidence': confidence,
                'model_used': model_used,
                'event_type': 'prediction'
            }
        }
        
        self.logger.info('Prediction generated', extra=extra)
    
    def log_prediction_outcome(self, prediction_id: str, actual_result: str, 
                             was_correct: bool, confidence: float):
        """Log prediction outcome for model evaluation"""
        extra = {
            'props': {
                'prediction_id': prediction_id,
                'actual_result': actual_result,
                'was_correct': was_correct,
                'confidence': confidence,
                'event_type': 'prediction_outcome'
            }
        }
        
        self.logger.info('Prediction outcome recorded', extra=extra)
    
    def log_model_performance(self, model_name: str, metrics: dict):
        """Log model performance metrics"""
        extra = {
            'props': {
                'model_name': model_name,
                'metrics': metrics,
                'event_type': 'model_performance'
            }
        }
        
        self.logger.info('Model performance recorded', extra=extra)

# Global prediction logger instance
prediction_logger = PredictionLogger()
