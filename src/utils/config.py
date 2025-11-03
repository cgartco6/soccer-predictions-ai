import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """
    Configuration management for the soccer predictions system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_path()
        self._config = self._load_config()
        
    def _find_config_path(self) -> str:
        """Find the configuration file path"""
        possible_paths = [
            'config/default.yaml',
            '../config/default.yaml',
            './config/default.yaml'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        raise FileNotFoundError("Could not find configuration file")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Override with environment-specific config
            env = os.getenv('ENVIRONMENT', 'development')
            env_config_path = self.config_path.replace('default', env)
            
            if Path(env_config_path).exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    config = self._deep_merge(config, env_config)
            
            # Override with environment variables
            config = self._override_with_env_vars(config)
            
            return config
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _override_with_env_vars(self, config: Dict[str, Any], 
                              prefix: str = "") -> Dict[str, Any]:
        """Override configuration with environment variables"""
        for key, value in config.items():
            env_key = f"{prefix}_{key}".upper().strip('_')
            
            if isinstance(value, dict):
                config[key] = self._override_with_env_vars(value, env_key)
            else:
                env_value = os.getenv(env_key)
                if env_value is not None:
                    # Convert environment variable to appropriate type
                    if isinstance(value, bool):
                        config[key] = env_value.lower() in ('true', '1', 'yes')
                    elif isinstance(value, int):
                        config[key] = int(env_value)
                    elif isinstance(value, float):
                        config[key] = float(env_value)
                    else:
                        config[key] = env_value
                        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type"""
        model_config_path = f"config/model_configs/{model_type}_config.yaml"
        
        if Path(model_config_path).exists():
            with open(model_config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self.get(f'models.{model_type}', {})
    
    @property
    def environment(self) -> str:
        """Get current environment"""
        return os.getenv('ENVIRONMENT', 'development')
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == 'development'
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == 'production'
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == 'testing'

# Global configuration instance
config = Config()
