import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

class DataAugmentor:
    """
    Data augmentation for training data
    Creates synthetic matches to improve model robustness
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def augment_training_data(self, matches: pd.DataFrame, 
                            augmentation_factor: float = 0.5) -> pd.DataFrame:
        """
        Augment training data with synthetic matches
        """
        if matches.empty:
            return matches
            
        num_synthetic = int(len(matches) * augmentation_factor)
        if num_synthetic == 0:
            return matches
            
        synthetic_matches = []
        
        # Different augmentation strategies
        strategies = [
            self._create_similar_matches,
            self._create_opposite_matches,
            self._create_noise_augmented_matches
        ]
        
        matches_per_strategy = max(1, num_synthetic // len(strategies))
        
        for strategy in strategies:
            synthetic = strategy(matches, matches_per_strategy)
            synthetic_matches.extend(synthetic)
            
        # Combine original and synthetic matches
        augmented_data = pd.concat([
            matches, 
            pd.DataFrame(synthetic_matches)
        ], ignore_index=True)
        
        self.logger.info(f"Augmented data: {len(matches)} -> {len(augmented_data)} matches")
        
        return augmented_data
    
    def _create_similar_matches(self, matches: pd.DataFrame, 
                              num_matches: int) -> List[Dict[str, Any]]:
        """Create matches with similar characteristics"""
        synthetic = []
        
        for _ in range(num_matches):
            # Sample a random match
            base_match = matches.sample(1).iloc[0]
            
            # Create similar match with small variations
            synthetic_match = base_match.to_dict()
            
            # Add small noise to numeric features
            numeric_columns = ['home_goals', 'away_goals', 'home_shots', 'away_shots']
            for col in numeric_columns:
                if col in synthetic_match:
                    synthetic_match[col] = self._add_noise(synthetic_match[col], variation=0.1)
                    
            # Slightly modify team strengths
            if 'home_elo' in synthetic_match and 'away_elo' in synthetic_match:
                elo_diff = synthetic_match['home_elo'] - synthetic_match['away_elo']
                synthetic_match['home_elo'] = self._add_noise(synthetic_match['home_elo'], 0.05)
                synthetic_match['away_elo'] = synthetic_match['home_elo'] - elo_diff
                
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
    
    def _create_opposite_matches(self, matches: pd.DataFrame,
                               num_matches: int) -> List[Dict[str, Any]]:
        """Create matches with opposite outcomes"""
        synthetic = []
        
        for _ in range(num_matches):
            base_match = matches.sample(1).iloc[0]
            synthetic_match = base_match.to_dict()
            
            # Swap teams and reverse outcome
            synthetic_match['home_team'], synthetic_match['away_team'] = (
                synthetic_match['away_team'], synthetic_match['home_team']
            )
            synthetic_match['home_goals'], synthetic_match['away_goals'] = (
                synthetic_match['away_goals'], synthetic_match['home_goals']
            )
            
            # Swap other team-specific features
            team_specific_features = ['home_elo', 'away_elo', 'home_shots', 'away_shots']
            for feature in team_specific_features:
                if feature in synthetic_match:
                    opposite_feature = feature.replace('home', 'temp').replace('away', 'home').replace('temp', 'away')
                    if opposite_feature in synthetic_match:
                        synthetic_match[feature], synthetic_match[opposite_feature] = (
                            synthetic_match[opposite_feature], synthetic_match[feature]
                        )
                        
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
    
    def _create_noise_augmented_matches(self, matches: pd.DataFrame,
                                      num_matches: int) -> List[Dict[str, Any]]:
        """Create matches with added noise"""
        synthetic = []
        
        for _ in range(num_matches):
            base_match = matches.sample(1).iloc[0]
            synthetic_match = base_match.to_dict()
            
            # Add Gaussian noise to all numeric features
            for key, value in synthetic_match.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    synthetic_match[key] = self._add_noise(value, variation=0.15)
                    
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
    
    def _add_noise(self, value: float, variation: float = 0.1) -> float:
        """Add controlled noise to a value"""
        if isinstance(value, int):
            noise = np.random.normal(0, variation * value)
            return max(0, int(value + noise))
        else:
            noise = np.random.normal(0, variation * abs(value))
            return max(0, value + noise)
    
    def augment_with_domain_knowledge(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Augment data using domain knowledge about soccer"""
        synthetic = []
        
        # Common soccer scenarios to augment
        scenarios = [
            self._create_high_scoring_draw,
            self._create_narrow_win,
            self._create_upset_victory,
            self._create_defensive_masterclass
        ]
        
        for scenario in scenarios:
            synthetic_matches = scenario(matches, num_matches=len(matches) // 10)
            synthetic.extend(synthetic_matches)
            
        if synthetic:
            return pd.concat([matches, pd.DataFrame(synthetic)], ignore_index=True)
        return matches
    
    def _create_high_scoring_draw(self, matches: pd.DataFrame, 
                                num_matches: int) -> List[Dict[str, Any]]:
        """Create high-scoring draw matches"""
        synthetic = []
        
        for _ in range(num_matches):
            base_match = matches.sample(1).iloc[0]
            synthetic_match = base_match.to_dict()
            
            # Set as high-scoring draw
            goals = np.random.randint(2, 5)  # 2-4 goals each
            synthetic_match['home_goals'] = goals
            synthetic_match['away_goals'] = goals
            
            # Increase shots and other offensive metrics
            offensive_metrics = ['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']
            for metric in offensive_metrics:
                if metric in synthetic_match:
                    synthetic_match[metric] = int(synthetic_match[metric] * 1.5)
                    
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
    
    def _create_narrow_win(self, matches: pd.DataFrame,
                          num_matches: int) -> List[Dict[str, Any]]:
        """Create narrow win matches (1-0, 2-1, etc.)"""
        synthetic = []
        
        for _ in range(num_matches):
            base_match = matches.sample(1).iloc[0]
            synthetic_match = base_match.to_dict()
            
            # Create narrow win scenario
            win_margins = [(1, 0), (2, 1), (2, 0), (1, 0)]
            home_goals, away_goals = win_margins[np.random.randint(0, len(win_margins))]
            
            # Randomly assign to home or away
            if np.random.random() > 0.5:
                synthetic_match['home_goals'] = home_goals
                synthetic_match['away_goals'] = away_goals
            else:
                synthetic_match['home_goals'] = away_goals
                synthetic_match['away_goals'] = home_goals
                
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
    
    def _create_upset_victory(self, matches: pd.DataFrame,
                            num_matches: int) -> List[Dict[str, Any]]:
        """Create upset victory matches"""
        synthetic = []
        
        for _ in range(num_matches):
            base_match = matches.sample(1).iloc[0]
            synthetic_match = base_match.to_dict()
            
            # Reverse the expected outcome (weaker team wins)
            if synthetic_match.get('home_elo', 0) > synthetic_match.get('away_elo', 0):
                # Home was stronger, so away wins
                synthetic_match['home_goals'] = np.random.randint(0, 2)
                synthetic_match['away_goals'] = np.random.randint(2, 4)
            else:
                # Away was stronger, so home wins
                synthetic_match['home_goals'] = np.random.randint(2, 4)
                synthetic_match['away_goals'] = np.random.randint(0, 2)
                
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
    
    def _create_defensive_masterclass(self, matches: pd.DataFrame,
                                    num_matches: int) -> List[Dict[str, Any]]:
        """Create defensive masterclass matches (clean sheets)"""
        synthetic = []
        
        for _ in range(num_matches):
            base_match = matches.sample(1).iloc[0]
            synthetic_match = base_match.to_dict()
            
            # One team keeps clean sheet
            if np.random.random() > 0.5:
                synthetic_match['home_goals'] = np.random.randint(1, 3)
                synthetic_match['away_goals'] = 0
            else:
                synthetic_match['home_goals'] = 0
                synthetic_match['away_goals'] = np.random.randint(1, 3)
                
            # Reduce offensive metrics for both teams
            offensive_metrics = ['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']
            for metric in offensive_metrics:
                if metric in synthetic_match:
                    synthetic_match[metric] = max(1, int(synthetic_match[metric] * 0.7))
                    
            synthetic_match['synthetic'] = True
            synthetic.append(synthetic_match)
            
        return synthetic
