from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class TeamSchema(BaseModel):
    """Schema for team data validation"""
    
    # Basic team information
    team_id: Optional[str] = None
    team_name: str
    short_name: Optional[str] = None
    country: str
    league: str
    
    # Team characteristics
    founded: Optional[int] = None
    stadium: Optional[str] = None
    stadium_capacity: Optional[int] = None
    manager: Optional[str] = None
    owner: Optional[str] = None
    
    # Financial information
    market_value: Optional[float] = None
    transfer_budget: Optional[float] = None
    wage_bill: Optional[float] = None
    
    # Performance metrics (current season)
    league_position: Optional[int] = None
    matches_played: Optional[int] = None
    wins: Optional[int] = None
    draws: Optional[int] = None
    losses: Optional[int] = None
    points: Optional[int] = None
    goals_for: Optional[int] = None
    goals_against: Optional[int] = None
    goal_difference: Optional[int] = None
    
    # Advanced metrics
    expected_goals_for: Optional[float] = None
    expected_goals_against: Optional[float] = None
    expected_points: Optional[float] = None
    
    # Style of play
    preferred_formation: Optional[str] = None
    playing_style: Optional[str] = None  # e.g., 'possession', 'counter_attack'
    attacking_rating: Optional[float] = None
    defensive_rating: Optional[float] = None
    
    # Historical performance
    historical_elo: Optional[float] = None
    form_rating: Optional[float] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    data_source: Optional[str] = None
    
    @validator('league_position')
    def validate_league_position(cls, v):
        if v is not None and v <= 0:
            raise ValueError('League position must be positive')
        return v
    
    @validator('market_value', 'transfer_budget', 'wage_bill')
    def validate_financials(cls, v):
        if v is not None and v < 0:
            raise ValueError('Financial values cannot be negative')
        return v
    
    @validator('attacking_rating', 'defensive_rating', 'form_rating')
    def validate_ratings(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Ratings must be between 0 and 10')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TeamFormationSchema(BaseModel):
    """Schema for team formation and tactics"""
    
    team_id: str
    match_id: str
    formation: str  # e.g., '4-3-3', '4-4-2'
    formation_variant: Optional[str] = None  # e.g., 'diamond', 'flat'
    
    # Tactical preferences
    pressing_intensity: Optional[float] = None  # 0-1 scale
    defensive_line: Optional[str] = None  # 'high', 'medium', 'low'
    attacking_width: Optional[str] = None  # 'narrow', 'balanced', 'wide'
    
    # Player positions
    starting_xi: List[Dict[str, Any]]  # List of player positions and roles
    
    @validator('pressing_intensity')
    def validate_pressing(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Pressing intensity must be between 0 and 1')
        return v
    
    class Config:
        arbitrary_types_allowed = True
