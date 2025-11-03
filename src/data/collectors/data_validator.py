import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

class DataValidator:
    """
    Validates and cleans collected data
    """
    
    def __init__(self, data_type: str):
        self.data_type = data_type
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._get_validation_rules()
        
    def validate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data and return cleaned version with validation report
        """
        if not data:
            return {
                'valid_data': [],
                'validation_report': {
                    'total_records': 0,
                    'valid_records': 0,
                    'errors': ['No data provided']
                }
            }
        
        validated_data = []
        errors = []
        
        for record in data:
            is_valid, record_errors = self._validate_record(record)
            if is_valid:
                validated_data.append(record)
            else:
                errors.extend(record_errors)
        
        return {
            'valid_data': validated_data,
            'validation_report': {
                'total_records': len(data),
                'valid_records': len(validated_data),
                'invalid_records': len(data) - len(validated_data),
                'errors': errors,
                'validity_rate': len(validated_data) / len(data) if data else 0
            }
        }
    
    def _validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single record"""
        errors = []
        
        if self.data_type == 'match':
            errors.extend(self._validate_match_record(record))
        elif self.data_type == 'odds':
            errors.extend(self._validate_odds_record(record))
        elif self.data_type == 'player':
            errors.extend(self._validate_player_record(record))
        
        return len(errors) == 0, errors
    
    def _validate_match_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate match record"""
        errors = []
        
        # Required fields
        required_fields = ['home_team', 'away_team', 'league', 'date']
        for field in required_fields:
            if field not in record or not record[field]:
                errors.append(f"Missing required field: {field}")
        
        # Team names should be strings
        if 'home_team' in record and not isinstance(record['home_team'], str):
            errors.append("home_team must be a string")
        if 'away_team' in record and not isinstance(record['away_team'], str):
            errors.append("away_team must be a string")
            
        # Goals should be integers if present
        if 'home_goals' in record and record['home_goals'] is not None:
            if not isinstance(record['home_goals'], (int, float)) or record['home_goals'] < 0:
                errors.append("home_goals must be non-negative number")
                
        if 'away_goals' in record and record['away_goals'] is not None:
            if not isinstance(record['away_goals'], (int, float)) or record['away_goals'] < 0:
                errors.append("away_goals must be non-negative number")
        
        # Date validation
        if 'date' in record and record['date']:
            try:
                pd.to_datetime(record['date'])
            except (ValueError, TypeError):
                errors.append("Invalid date format")
        
        return errors
    
    def _validate_odds_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate odds record"""
        errors = []
        
        required_fields = ['home_team', 'away_team', 'home_odds', 'away_odds', 'source']
        for field in required_fields:
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Odds validation
        if 'home_odds' in record and record['home_odds'] is not None:
            if record['home_odds'] < 1.0:
                errors.append("home_odds must be >= 1.0")
                
        if 'away_odds' in record and record['away_odds'] is not None:
            if record['away_odds'] < 1.0:
                errors.append("away_odds must be >= 1.0")
                
        if 'draw_odds' in record and record['draw_odds'] is not None:
            if record['draw_odds'] < 1.0:
                errors.append("draw_odds must be >= 1.0")
        
        return errors
    
    def _validate_player_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate player record"""
        errors = []
        
        required_fields = ['player_name', 'team', 'position']
        for field in required_fields:
            if field not in record or not record[field]:
                errors.append(f"Missing required field: {field}")
        
        # Performance metrics validation
        performance_fields = ['rating', 'goals', 'assists', 'minutes_played']
        for field in performance_fields:
            if field in record and record[field] is not None:
                if not isinstance(record[field], (int, float)):
                    errors.append(f"{field} must be a number")
                elif record[field] < 0:
                    errors.append(f"{field} must be non-negative")
        
        return errors
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for data type"""
        rules = {
            'match': {
                'required_fields': ['home_team', 'away_team', 'league', 'date'],
                'numeric_fields': ['home_goals', 'away_goals', 'attendance'],
                'string_fields': ['home_team', 'away_team', 'league', 'venue'],
                'date_fields': ['date']
            },
            'odds': {
                'required_fields': ['home_team', 'away_team', 'home_odds', 'away_odds', 'source'],
                'numeric_fields': ['home_odds', 'draw_odds', 'away_odds'],
                'string_fields': ['home_team', 'away_team', 'source', 'league']
            },
            'player': {
                'required_fields': ['player_name', 'team', 'position'],
                'numeric_fields': ['rating', 'goals', 'assists', 'minutes_played'],
                'string_fields': ['player_name', 'team', 'position', 'nationality']
            }
        }
        
        return rules.get(self.data_type, {})
