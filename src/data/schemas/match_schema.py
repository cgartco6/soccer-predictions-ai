from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class MatchResult(str, Enum):
    HOME_WIN = "home_win"
    AWAY_WIN = "away_win"
    DRAW = "draw"

class MatchSchema(BaseModel):
    """Schema for match data validation"""
    
    # Basic match information
    match_id: Optional[str] = None
    home_team: str
    away_team: str
    league: str
    competition: Optional[str] = None
    season: Optional[str] = None
    date: datetime
    venue: Optional[str] = None
    
    # Match results
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    result: Optional[MatchResult] = None
    
    # Match statistics
    attendance: Optional[int] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    
    # Advanced metrics
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_deep: Optional[int] = None  # Deep completions
    away_deep: Optional[int] = None
    
    # Contextual data
    weather_conditions: Optional[str] = None
    temperature: Optional[float] = None
    pitch_condition: Optional[str] = None
    
    # Metadata
    source: Optional[str] = None
    last_updated: Optional[datetime] = None
    data_quality: Optional[float] = None
    
    @validator('home_goals', 'away_goals')
    def validate_goals(cls, v):
        if v is not None and v < 0:
            raise ValueError('Goals cannot be negative')
        if v is not None and v > 20:  # Reasonable upper limit
            raise ValueError('Unrealistic number of goals')
        return v
    
    @validator('home_possession', 'away_possession')
    def validate_possession(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Possession must be between 0 and 100')
        return v
    
    @validator('home_xg', 'away_xg')
    def validate_xg(cls, v):
        if v is not None and v < 0:
            raise ValueError('Expected goals cannot be negative')
        if v is not None and v > 10:  # Very high xG
            raise ValueError('Unrealistic expected goals value')
        return v
    
    @validator('result', always=True)
    def set_result(cls, v, values):
        if v is None and 'home_goals' in values and 'away_goals' in values:
            if values['home_goals'] > values['away_goals']:
                return MatchResult.HOME_WIN
            elif values['home_goals'] < values['away_goals']:
                return MatchResult.AWAY_WIN
            else:
                return MatchResult.DRAW
        return v
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MatchOddsSchema(BaseModel):
    """Schema for match odds data"""
    
    match_id: Optional[str] = None
    home_team: str
    away_team: str
    league: str
    date: datetime
    
    # Odds from different bookmakers
    home_odds: float
    draw_odds: float
    away_odds: float
    
    # Additional betting markets
    over_under_odds: Optional[Dict[str, float]] = None
    both_teams_score_odds: Optional[Dict[str, float]] = None
    
    # Source and timing
    bookmaker: str
    collected_at: datetime
    source: str
    
    @validator('home_odds', 'draw_odds', 'away_odds')
    def validate_odds(cls, v):
        if v < 1.0:
            raise ValueError('Odds must be greater than or equal to 1.0')
        if v > 1000:  # Very high odds
            raise ValueError('Unrealistic odds value')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
