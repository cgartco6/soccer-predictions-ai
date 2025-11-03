import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    """
    Feature engineering for soccer prediction models
    Creates advanced features from raw match data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def create_features(self, matches: pd.DataFrame, 
                       feature_types: List[str] = None) -> pd.DataFrame:
        """
        Create comprehensive features for model training
        """
        if matches.empty:
            return matches
            
        if feature_types is None:
            feature_types = ['basic', 'team_strength', 'form', 'contextual']
            
        df = matches.copy()
        features = pd.DataFrame(index=df.index)
        
        # Create different types of features
        if 'basic' in feature_types:
            features = self._create_basic_features(features, df)
            
        if 'team_strength' in feature_types:
            features = self._create_team_strength_features(features, df)
            
        if 'form' in feature_types:
            features = self._create_form_features(features, df)
            
        if 'contextual' in feature_types:
            features = self._create_contextual_features(features, df)
            
        if 'advanced' in feature_types:
            features = self._create_advanced_metrics(features, df)
        
        # Handle missing values in features
        features = self._handle_feature_missing_values(features)
        
        return features
    
    def _create_basic_features(self, features: pd.DataFrame, 
                             matches: pd.DataFrame) -> pd.DataFrame:
        """Create basic match features"""
        # League features
        if 'league' in matches.columns:
            features['league_encoded'] = self._encode_categorical(matches['league'])
            
        # Time-based features
        if 'date' in matches.columns:
            features['day_of_week'] = matches['date'].dt.dayofweek
            features['month'] = matches['date'].dt.month
            features['season_week'] = matches['date'].dt.isocalendar().week
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            
        # Venue features
        if 'venue' in matches.columns:
            features['venue_encoded'] = self._encode_categorical(matches['venue'])
            
        return features
    
    def _create_team_strength_features(self, features: pd.DataFrame,
                                     matches: pd.DataFrame) -> pd.DataFrame:
        """Create team strength features"""
        # Calculate ELO-like ratings
        elo_ratings = self._calculate_elo_ratings(matches)
        
        for idx, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            date = match['date']
            
            # ELO ratings
            home_elo = elo_ratings.get(home_team, 1500)
            away_elo = elo_ratings.get(away_team, 1500)
            features.loc[idx, 'home_elo'] = home_elo
            features.loc[idx, 'away_elo'] = away_elo
            features.loc[idx, 'elo_difference'] = home_elo - away_elo
            
            # Historical performance
            home_stats = self._get_team_stats(matches, home_team, date, 'home')
            away_stats = self._get_team_stats(matches, away_team, date, 'away')
            
            for stat, value in home_stats.items():
                features.loc[idx, f'home_{stat}'] = value
            for stat, value in away_stats.items():
                features.loc[idx, f'away_{stat}'] = value
        
        return features
    
    def _create_form_features(self, features: pd.DataFrame,
                            matches: pd.DataFrame) -> pd.DataFrame:
        """Create team form features"""
        for idx, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            date = match['date']
            
            # Recent form (last 5 matches)
            home_form = self._calculate_recent_form(matches, home_team, date, 'home')
            away_form = self._calculate_recent_form(matches, away_team, date, 'away')
            
            features.loc[idx, 'home_recent_form'] = home_form['points_per_game']
            features.loc[idx, 'away_recent_form'] = away_form['points_per_game']
            features.loc[idx, 'home_goals_scored_form'] = home_form['goals_scored_avg']
            features.loc[idx, 'away_goals_scored_form'] = away_form['goals_scored_avg']
            features.loc[idx, 'home_goals_conceded_form'] = home_form['goals_conceded_avg']
            features.loc[idx, 'away_goals_conceded_form'] = away_form['goals_conceded_avg']
            
            # Home/Away specific form
            home_home_form = self._calculate_venue_form(matches, home_team, date, 'home')
            away_away_form = self._calculate_venue_form(matches, away_team, date, 'away')
            
            features.loc[idx, 'home_home_form'] = home_home_form
            features.loc[idx, 'away_away_form'] = away_away_form
        
        return features
    
    def _create_contextual_features(self, features: pd.DataFrame,
                                  matches: pd.DataFrame) -> pd.DataFrame:
        """Create contextual features"""
        # Match importance
        features['match_importance'] = self._calculate_match_importance(matches)
        
        # Fatigue factors
        features['home_days_rest'] = self._calculate_days_rest(matches, 'home')
        features['away_days_rest'] = self._calculate_days_rest(matches, 'away')
        
        # Head-to-head history
        for idx, match in matches.iterrows():
            h2h_stats = self._calculate_head_to_head(matches, match, idx)
            for stat, value in h2h_stats.items():
                features.loc[idx, f'h2h_{stat}'] = value
        
        return features
    
    def _create_advanced_metrics(self, features: pd.DataFrame,
                               matches: pd.DataFrame) -> pd.DataFrame:
        """Create advanced soccer metrics"""
        # Expected Goals (xG) metrics
        if all(col in matches.columns for col in ['home_xg', 'away_xg']):
            features['home_xg_per_shot'] = matches['home_xg'] / (matches['home_shots'] + 1)
            features['away_xg_per_shot'] = matches['away_xg'] / (matches['away_shots'] + 1)
            features['xg_difference'] = matches['home_xg'] - matches['away_xg']
            
        # Possession efficiency
        if 'home_possession' in matches.columns:
            features['home_possession_efficiency'] = (
                matches['home_goals'] / (matches['home_possession'] + 1)
            )
            features['away_possession_efficiency'] = (
                matches['away_goals'] / (100 - matches['home_possession'] + 1)
            )
            
        # Defensive solidity
        if all(col in matches.columns for col in ['home_shots', 'away_shots_on_target']):
            features['home_defense_efficiency'] = (
                matches['away_shots_on_target'] / (matches['away_shots'] + 1)
            )
            features['away_defense_efficiency'] = (
                matches['home_shots_on_target'] / (matches['home_shots'] + 1)
            )
        
        return features
    
    def _calculate_elo_ratings(self, matches: pd.DataFrame, 
                             k_factor: int = 20, home_advantage: int = 100) -> Dict[str, float]:
        """Calculate ELO ratings for teams"""
        ratings = {}
        
        for _, match in matches.sort_values('date').iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Initialize ratings
            if home_team not in ratings:
                ratings[home_team] = 1500
            if away_team not in ratings:
                ratings[away_team] = 1500
                
            # Get ratings with home advantage
            home_rating = ratings[home_team] + home_advantage
            away_rating = ratings[away_team]
            
            # Calculate expected result
            expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
            
            # Calculate actual result
            if match['home_goals'] > match['away_goals']:
                actual_home = 1.0
            elif match['home_goals'] == match['away_goals']:
                actual_home = 0.5
            else:
                actual_home = 0.0
                
            # Update ratings
            ratings[home_team] += k_factor * (actual_home - expected_home)
            ratings[away_team] += k_factor * ((1 - actual_home) - (1 - expected_home))
            
        return ratings
    
    def _get_team_stats(self, matches: pd.DataFrame, team: str, 
                       current_date: pd.Timestamp, venue: str) -> Dict[str, float]:
        """Get team historical statistics"""
        if venue == 'home':
            team_matches = matches[
                (matches['home_team'] == team) & 
                (matches['date'] < current_date)
            ]
        else:
            team_matches = matches[
                (matches['away_team'] == team) & 
                (matches['date'] < current_date)
            ]
            
        if team_matches.empty:
            return {
                'win_rate': 0.33,
                'goals_scored_avg': 1.5,
                'goals_conceded_avg': 1.5,
                'clean_sheet_rate': 0.2
            }
            
        recent_matches = team_matches.tail(10)
        
        if venue == 'home':
            wins = (recent_matches['home_goals'] > recent_matches['away_goals']).sum()
            goals_scored = recent_matches['home_goals'].sum()
            goals_conceded = recent_matches['away_goals'].sum()
            clean_sheets = (recent_matches['away_goals'] == 0).sum()
        else:
            wins = (recent_matches['away_goals'] > recent_matches['home_goals']).sum()
            goals_scored = recent_matches['away_goals'].sum()
            goals_conceded = recent_matches['home_goals'].sum()
            clean_sheets = (recent_matches['home_goals'] == 0).sum()
            
        total_matches = len(recent_matches)
        
        return {
            'win_rate': wins / total_matches if total_matches > 0 else 0,
            'goals_scored_avg': goals_scored / total_matches if total_matches > 0 else 0,
            'goals_conceded_avg': goals_conceded / total_matches if total_matches > 0 else 0,
            'clean_sheet_rate': clean_sheets / total_matches if total_matches > 0 else 0
        }
    
    def _calculate_recent_form(self, matches: pd.DataFrame, team: str,
                             current_date: pd.Timestamp, venue: str) -> Dict[str, float]:
        """Calculate recent form for a team"""
        if venue == 'home':
            team_matches = matches[
                (matches['home_team'] == team) & 
                (matches['date'] < current_date)
            ].tail(5)
        else:
            team_matches = matches[
                (matches['away_team'] == team) & 
                (matches['date'] < current_date)
            ].tail(5)
            
        if team_matches.empty:
            return {
                'points_per_game': 1.5,
                'goals_scored_avg': 1.5,
                'goals_conceded_avg': 1.5
            }
            
        points = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, match in team_matches.iterrows():
            if venue == 'home':
                home_goals = match['home_goals']
                away_goals = match['away_goals']
            else:
                home_goals = match['away_goals']
                away_goals = match['home_goals']
                
            goals_scored += home_goals
            goals_conceded += away_goals
            
            if home_goals > away_goals:
                points += 3
            elif home_goals == away_goals:
                points += 1
                
        total_matches = len(team_matches)
        
        return {
            'points_per_game': points / total_matches,
            'goals_scored_avg': goals_scored / total_matches,
            'goals_conceded_avg': goals_conceded / total_matches
        }
    
    def _calculate_venue_form(self, matches: pd.DataFrame, team: str,
                            current_date: pd.Timestamp, venue: str) -> float:
        """Calculate form at specific venue"""
        if venue == 'home':
            team_matches = matches[
                (matches['home_team'] == team) & 
                (matches['date'] < current_date)
            ].tail(5)
        else:
            team_matches = matches[
                (matches['away_team'] == team) & 
                (matches['date'] < current_date)
            ].tail(5)
            
        if team_matches.empty:
            return 0.5
            
        points = 0
        for _, match in team_matches.iterrows():
            if venue == 'home':
                if match['home_goals'] > match['away_goals']:
                    points += 3
                elif match['home_goals'] == match['away_goals']:
                    points += 1
            else:
                if match['away_goals'] > match['home_goals']:
                    points += 3
                elif match['away_goals'] == match['home_goals']:
                    points += 1
                    
        max_points = len(team_matches) * 3
        return points / max_points if max_points > 0 else 0
    
    def _calculate_match_importance(self, matches: pd.DataFrame) -> pd.Series:
        """Calculate match importance based on competition and timing"""
        importance = pd.Series(0.5, index=matches.index)
        
        # Competition importance
        if 'league' in matches.columns:
            competition_importance = {
                'Champions League': 1.0,
                'Europa League': 0.9,
                'Premier League': 0.8,
                'La Liga': 0.8,
                'Bundesliga': 0.8,
                'Serie A': 0.8,
                'Ligue 1': 0.8,
                'FA Cup': 0.7,
                'League Cup': 0.6,
                'Friendly': 0.3
            }
            importance = matches['league'].map(
                lambda x: competition_importance.get(x, 0.5)
            )
        
        # End of season importance
        if 'date' in matches.columns:
            # Increase importance for matches in last 5 weeks of season
            season_end = matches['date'].max()
            weeks_to_end = (season_end - matches['date']).dt.days / 7
            end_season_boost = np.where(weeks_to_end <= 5, 0.2, 0)
            importance += end_season_boost
            
        return importance.clip(0, 1)
    
    def _calculate_days_rest(self, matches: pd.DataFrame, side: str) -> pd.Series:
        """Calculate days rest since last match"""
        days_rest = pd.Series(7, index=matches.index)  # Default 7 days
        
        for idx, match in matches.iterrows():
            team = match[f'{side}_team']
            current_date = match['date']
            
            # Find previous match for this team
            if side == 'home':
                previous_matches = matches[
                    ((matches['home_team'] == team) | (matches['away_team'] == team)) &
                    (matches['date'] < current_date)
                ]
            else:
                previous_matches = matches[
                    ((matches['home_team'] == team) | (matches['away_team'] == team)) &
                    (matches['date'] < current_date)
                ]
                
            if not previous_matches.empty:
                last_match_date = previous_matches['date'].max()
                rest_days = (current_date - last_match_date).days
                days_rest[idx] = rest_days
                
        return days_rest
    
    def _calculate_head_to_head(self, matches: pd.DataFrame, 
                              current_match: pd.Series, 
                              current_idx: int) -> Dict[str, float]:
        """Calculate head-to-head statistics"""
        home_team = current_match['home_team']
        away_team = current_match['away_team']
        current_date = current_match['date']
        
        # Get historical matches between these teams
        historical = matches[
            ((matches['home_team'] == home_team) & (matches['away_team'] == away_team) |
             (matches['home_team'] == away_team) & (matches['away_team'] == home_team)) &
            (matches['date'] < current_date) &
            (matches.index != current_idx)
        ].tail(10)
        
        if historical.empty:
            return {
                'home_win_rate': 0.33,
                'away_win_rate': 0.33,
                'draw_rate': 0.34,
                'avg_goals': 2.5
            }
            
        home_wins = 0
        away_wins = 0
        draws = 0
        total_goals = 0
        
        for _, match in historical.iterrows():
            if match['home_team'] == home_team:
                if match['home_goals'] > match['away_goals']:
                    home_wins += 1
                elif match['home_goals'] < match['away_goals']:
                    away_wins += 1
                else:
                    draws += 1
                total_goals += match['home_goals'] + match['away_goals']
            else:
                if match['away_goals'] > match['home_goals']:
                    home_wins += 1
                elif match['away_goals'] < match['home_goals']:
                    away_wins += 1
                else:
                    draws += 1
                total_goals += match['home_goals'] + match['away_goals']
                
        total_matches = len(historical)
        
        return {
            'home_win_rate': home_wins / total_matches,
            'away_win_rate': away_wins / total_matches,
            'draw_rate': draws / total_matches,
            'avg_goals': total_goals / total_matches
        }
    
    def _encode_categorical(self, series: pd.Series) -> pd.Series:
        """Encode categorical variables"""
        if series.name not in self.encoders:
            self.encoders[series.name] = LabelEncoder()
            encoded = self.encoders[series.name].fit_transform(series.fillna('Unknown'))
        else:
            encoded = self.encoders[series.name].transform(series.fillna('Unknown'))
            
        return pd.Series(encoded, index=series.index)
    
    def _handle_feature_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        for column in features.columns:
            if features[column].isna().any():
                if column not in self.imputers:
                    self.imputers[column] = SimpleImputer(strategy='median')
                    features[column] = self.imputers[column].fit_transform(
                        features[[column]]
                    )[:, 0]
                else:
                    features[column] = self.imputers[column].transform(
                        features[[column]]
                    )[:, 0]
                    
        return features
