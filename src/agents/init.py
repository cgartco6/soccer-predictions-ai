"""
AI Agents for Soccer Predictions
"""

from .base_agent import BaseAgent
from .synthetic_intelligence import SyntheticIntelligenceAgent
from .strategic_intelligence import StrategicIntelligenceAgent
from .data_collector import DataCollectorAgent
from .prediction_ensemble import PredictionEnsembleAgent

__all__ = [
    'BaseAgent',
    'SyntheticIntelligenceAgent',
    'StrategicIntelligenceAgent',
    'DataCollectorAgent',
    'PredictionEnsembleAgent'
]
