import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import re

class DataCleaner:
    """
    Cleans and preprocesses raw football data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.team_name_mapping = self._load_team_name_mapping()
        
    def clean_match_data(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Clean raw match data"""
        if matches.empty:
            return matches
            
        df = matches.copy()
        
        # Standardize team names
        df['home_team'] = df['home_team'].apply(self._standardize_team_name)
        df['away_team'] = df['away_team'].apply(self._standardize_team_name)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Standardize date format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['home_goals', 'away_goals', 'attendance', 'home_shots', 'away_shots']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Validate data consistency
        df = self._validate_consistency(df)
        
        return df
    
    def clean_odds_data(self, odds_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Clean raw odds data"""
        if not odds_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(odds_data)
        
        if df.empty:
            return df
            
        # Standardize team names
        df['home_team'] = df['home_team'].apply(self._standardize_team_name)
        df['away_team'] = df['away_team'].apply(self._standardize_team_name)
        
        # Clean odds values
        odds_columns = ['home_odds', 'draw_odds', 'away_odds']
        for col in odds_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remove unrealistic odds
                df = df[(df[col] >= 1.0) | (df[col].isna())]
        
        # Handle missing odds
        df = self._handle_missing_odds(df)
        
        # Remove duplicates (same match from same source)
        df = df.drop_duplicates(
            subset=['home_team', 'away_team', 'source', 'scraped_at'], 
            keep='last'
        )
        
        return df
    
    def clean_player_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Clean player performance data"""
        if player_data.empty:
            return player_data
            
        df = player_data.copy()
        
        # Standardize names
        df['player_name'] = df['player_name'].apply(self._standardize_player_name)
        df['team'] = df['team'].apply(self._standardize_team_name)
        
        # Clean performance metrics
        performance_columns = ['goals', 'assists', 'minutes_played', 'rating', 'pass_accuracy']
        for col in performance_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle outliers
                if col == 'rating':
                    df[col] = df[col].clip(0, 10)
                elif col == 'pass_accuracy':
                    df[col] = df[col].clip(0, 100)
                elif col in ['goals', 'assists']:
                    df[col] = df[col].clip(0, 20)  # Reasonable upper limit
        
        return df
    
    def _standardize_team_name(self, name: str) -> str:
        """Standardize team name format"""
        if not isinstance(name, str):
            return "Unknown"
            
        # Apply mapping for common variations
        if name in self.team_name_mapping:
            return self.team_name_mapping[name]
        
        # Standardize formatting
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
        name = name.title()  # Title case
        
        # Common replacements
        replacements = {
            ' Fc': ' FC',
            ' Fc ': ' FC ',
            'Afc ': 'AFC ',
            'Utd': 'United',
            'Spurs': 'Tottenham Hotspur'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
            
        return name
    
    def _standardize_player_name(self, name: str) -> str:
        """Standardize player name format"""
        if not isinstance(name, str):
            return "Unknown"
            
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)
        
        # Convert to title case but preserve certain parts
        parts = name.split()
        standardized_parts = []
        
        for part in parts:
            if part.upper() in ['FC', 'AFC', 'UEFA']:
                standardized_parts.append(part.upper())
            else:
                standardized_parts.append(part.title())
                
        return ' '.join(standardized_parts)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in match data"""
        # For goals, use 0 if missing (assume no goals)
        if 'home_goals' in df.columns:
            df['home_goals'] = df['home_goals'].fillna(0)
        if 'away_goals' in df.columns:
            df['away_goals'] = df['away_goals'].fillna(0)
        
        # For other numeric columns, use median
        numeric_columns = ['attendance', 'home_shots', 'away_shots']
        for col in numeric_columns:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # For categorical columns, use mode
        categorical_columns = ['league', 'venue']
        for col in categorical_columns:
            if col in df.columns:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df
    
    def _handle_missing_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing odds values"""
        # If only some odds are missing, we can calculate implied probabilities
        # and back-calculate missing odds
        odds_columns = ['home_odds', 'draw_odds', 'away_odds']
        
        for col in odds_columns:
            if col in df.columns:
                # Remove rows where all odds are missing
                all_odds_missing = df[odds_columns].isna().all(axis=1)
                df = df[~all_odds_missing]
                
                # For individual missing odds, we can estimate based on market margin
                # This is a simplified approach
                margin = 1.05  # Assume 5% margin
                
                for idx, row in df.iterrows():
                    if pd.isna(row[col]):
                        available_odds = [row[oc] for oc in odds_columns 
                                        if oc != col and not pd.isna(row[oc])]
                        if len(available_odds) >= 2:
                            # Simple estimation - in practice, use more sophisticated method
                            estimated_odd = sum(available_odds) / len(available_odds) * margin
                            df.at[idx, col] = estimated_odd
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate matches"""
        # Identify duplicates based on key fields
        key_columns = ['home_team', 'away_team', 'date']
        if all(col in df.columns for col in key_columns):
            df = df.drop_duplicates(subset=key_columns, keep='last')
        
        return df
    
    def _validate_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data consistency"""
        # Remove matches where team plays against itself
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df = df[df['home_team'] != df['away_team']]
        
        # Remove matches with unrealistic scores
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            df = df[(df['home_goals'] <= 20) & (df['away_goals'] <= 20)]
        
        # Remove future matches without odds (if applicable)
        if 'date' in df.columns:
            current_date = pd.Timestamp.now()
            future_matches = df[df['date'] > current_date]
            
            # If we have odds columns, keep future matches with odds
            odds_columns = ['home_odds', 'draw_odds', 'away_odds']
            if all(col in df.columns for col in odds_columns):
                future_with_odds = future_matches[~future_matches[odds_columns].isna().all(axis=1)]
                df = pd.concat([df[df['date'] <= current_date], future_with_odds])
        
        return df
    
    def _load_team_name_mapping(self) -> Dict[str, str]:
        """Load team name standardization mapping"""
        return {
            'Man United': 'Manchester United',
            'Man Utd': 'Manchester United',
            'Man City': 'Manchester City',
            'Spurs': 'Tottenham Hotspur',
            'Real Mad.': 'Real Madrid',
            'Barca': 'Barcelona',
            'PSG': 'Paris Saint-Germain',
            'Bayern': 'Bayern Munich',
            'Inter': 'Inter Milan',
            'Milan': 'AC Milan'
        }
