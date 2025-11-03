from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from datetime import datetime

class BaseAgent(ABC):
    """
    Base class for all AI agents in the soccer prediction system
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"agent.{name}")
        self.initialized = False
        self.metrics = {}
    
    def initialize(self) -> bool:
        """
        Initialize the agent with required resources
        """
        try:
            self._setup()
            self.initialized = True
            self.logger.info(f"{self.name} agent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name} agent: {e}")
            return False
    
    @abstractmethod
    def _setup(self):
        """
        Agent-specific setup logic
        """
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results
        """
        pass
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing
        """
        required_fields = self.config.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
        return True
    
    def update_metrics(self, metric_name: str, value: float):
        """
        Update agent performance metrics
        """
        self.metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current agent metrics
        """
        return self.metrics
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent
        """
        return {
            'agent': self.name,
            'initialized': self.initialized,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
