from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum

class PlayerPosition(str, Enum):
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"

class PlayerStatus(str, Enum):
    ACTIVE = "active"
    INJURED = "injured"
    SUSPENDED = "suspended"
    BENCH = "bench"

class PlayerSchema(BaseModel):
    """Schema for player data validation"""
    
    # Basic player information
    player_id: Optional[str] = None
    player_name: str
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    nationality: Optional[str] = None
    country_of_birth: Optional[str] = None
    
    # Physical attributes
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    preferred_foot: Optional[str] = None  # 'left', 'right', 'both'
    
    # Player characteristics
    position: PlayerPosition
    secondary_positions: Optional[List[str]] = None
    shirt_number: Optional[int] = None
    current_team: str
    team_join_date: Optional[date] = None
    contract_expiry: Optional[date] = None
    
    # Market information
    market_value: Optional[float] = None
    weekly_wage: Optional[float] = None
    
    # Performance metrics (current season)
    appearances: Optional[int] = None
    starts: Optional[int] = None
    minutes_played: Optional[int] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    clean_sheets: Optional[int] = None  # for goalkeepers
    
    # Advanced metrics
    expected_goals: Optional[float] = None
    expected_assists: Optional[float] = None
    goals_per_90: Optional[float] = None
    assists_per_90: Optional[float] = None
    pass_accuracy: Optional[float] = None
    tackles_success_rate: Optional[float] = None
    aerial_duels_won: Optional[float] = None
    
    # Player ratings
    average_rating: Optional[float] = None  # 0-10 scale
    form_rating: Optional[float] = None  # 0-10 scale
    potential_rating: Optional[float] = None  # 0-100 scale
    
    # Status and availability
    status: PlayerStatus = PlayerStatus.ACTIVE
    injury_status: Optional[str] = None
    expected_return: Optional[date] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    data_source: Optional[str] = None
    
    @validator('height', 'weight')
    def validate_physical_attributes(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Height and weight must be positive')
        return v
    
    @validator('market_value', 'weekly_wage')
    def validate_financials(cls, v):
        if v is not None and v < 0:
            raise ValueError('Financial values cannot be negative')
        return v
    
    @validator('average_rating', 'form_rating')
    def validate_ratings(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Ratings must be between 0 and 10')
        return v
    
    @validator('potential_rating')
    def validate_potential(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Potential rating must be between 0 and 100')
        return v
    
    @validator('pass_accuracy', 'tackles_success_rate', 'aerial_duels_won')
    def validate_percentages(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Percentage metrics must be between 0 and 100')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class PlayerPerformanceSchema(BaseModel):
    """Schema for individual player performance in a match"""
    
    player_id: str
    match_id: str
    team: str
    
    # Basic performance
    minutes_played: int
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    
    # Offensive metrics
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    key_passes: Optional[int] = None
    dribbles: Optional[int] = None
    dribbles_successful: Optional[int] = None
    
    # Defensive metrics
    tackles: Optional[int] = None
    tackles_successful: Optional[int] = None
    interceptions: Optional[int] = None
    clearances: Optional[int] = None
    blocks: Optional[int] = None
    
    # Passing metrics
    passes: Optional[int] = None
    passes_completed: Optional[int] = None
    pass_accuracy: Optional[float] = None
    crosses: Optional[int] = None
    long_passes: Optional[int] = None
    
    # Goalkeeping metrics
    saves: Optional[int] = None
    goals_conceded: Optional[int] = None
    clean_sheet: Optional[bool] = None
    
    # Physical metrics
    distance_covered: Optional[float] = None  # in km
    sprints: Optional[int] = None
    
    # Advanced metrics
    expected_goals: Optional[float] = None
    expected_assists: Optional[float] = None
    player_rating: Optional[float] = None  # 0-10 scale
    
    @validator('minutes_played')
    def validate_minutes(cls, v):
        if v < 0 or v > 120:  # Including extra time
            raise ValueError('Minutes played must be between 0 and 120')
        return v
    
    @validator('player_rating')
    def validate_rating(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Player rating must be between 0 and 10')
        return v
    
    class Config:
        arbitrary_types_allowed = True
