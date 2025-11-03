import numpy as np
import pymc3 as pm
import pandas as pd
from typing import Dict, Any, List

class BayesianPredictor:
    """
    Bayesian inference model for soccer predictions
    Using hierarchical modeling for team strengths
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.trace = None
        self.team_indices = {}
        self.teams = []
        
    def build_model(self, matches: pd.DataFrame):
        """Build Bayesian hierarchical model"""
        # Create team indices
        all_teams = sorted(set(matches['home_team']).union(set(matches['away_team'])))
        self.teams = all_teams
        self.team_indices = {team: i for i, team in enumerate(all_teams)}
        
        # Prepare data
        home_team_idx = [self.team_indices[team] for team in matches['home_team']]
        away_team_idx = [self.team_indices[team] for team in matches['away_team']]
        home_goals = matches['home_goals'].values
        away_goals = matches['away_goals'].values
        
        with pm.Model() as self.model:
            # Global parameters
            home_advantage = pm.Normal('home_advantage', mu=0.3, sigma=0.1)
            base_goals = pm.Gamma('base_goals', alpha=2, beta=1)
            
            # Team-specific parameters (hierarchical)
            attack_sd = pm.HalfNormal('attack_sd', sigma=0.5)
            defense_sd = pm.HalfNormal('defense_sd', sigma=0.5)
            
            attack = pm.Normal('attack', mu=0, sigma=attack_sd, shape=len(all_teams))
            defense = pm.Normal('defense', mu=0, sigma=defense_sd, shape=len(all_teams))
            
            # Expected goals
            home_goal_rate = pm.Deterministic(
                'home_goal_rate',
                base_goals * pm.math.exp(
                    attack[home_team_idx] - defense[away_team_idx] + home_advantage
                )
            )
            
            away_goal_rate = pm.Deterministic(
                'away_goal_rate',
                base_goals * pm.math.exp(
                    attack[away_team_idx] - defense[home_team_idx]
                )
            )
            
            # Likelihood
            home_goals_obs = pm.Poisson('home_goals', mu=home_goal_rate, observed=home_goals)
            away_goals_obs = pm.Poisson('away_goals', mu=away_goal_rate, observed=away_goals)
    
    def fit(self, matches: pd.DataFrame, samples: int = 2000, tune: int = 1000):
        """Fit the Bayesian model"""
        if self.model is None:
            self.build_model(matches)
        
        with self.model:
            self.trace = pm.sample(
                samples, 
                tune=tune, 
                cores=2, 
                chains=2,
                target_accept=0.9
            )
    
    def predict(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Predict match outcome using Bayesian inference"""
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction")
        
        home_team = match['home_team']
        away_team = match['away_team']
        
        if home_team not in self.team_indices or away_team not in self.team_indices:
            # Use default probabilities for unknown teams
            return {
                'home_win_prob': 0.33,
                'draw_prob': 0.34,
                'away_win_prob': 0.33,
                'expected_home_goals': 1.5,
                'expected_away_goals': 1.5
            }
        
        home_idx = self.team_indices[home_team]
        away_idx = self.team_indices[away_team]
        
        # Get posterior samples
        home_attack = self.trace['attack'][:, home_idx]
        away_attack = self.trace['attack'][:, away_idx]
        home_defense = self.trace['defense'][:, home_idx]
        away_defense = self.trace['defense'][:, away_idx]
        home_advantage = self.trace['home_advantage']
        base_goals = self.trace['base_goals']
        
        # Calculate expected goals
        home_goal_rate = base_goals * np.exp(
            home_attack - away_defense + home_advantage
        )
        away_goal_rate = base_goals * np.exp(
            away_attack - home_defense
        )
        
        # Simulate outcomes
        n_simulations = 10000
        home_goals_sim = np.random.poisson(
            np.mean(home_goal_rate), n_simulations
        )
        away_goals_sim = np.random.poisson(
            np.mean(away_goal_rate), n_simulations
        )
        
        # Calculate probabilities
        home_wins = np.sum(home_goals_sim > away_goals_sim) / n_simulations
        draws = np.sum(home_goals_sim == away_goals_sim) / n_simulations
        away_wins = np.sum(home_goals_sim < away_goals_sim) / n_simulations
        
        return {
            'home_win_prob': home_wins,
            'draw_prob': draws,
            'away_win_prob': away_wins,
            'expected_home_goals': np.mean(home_goal_rate),
            'expected_away_goals': np.mean(away_goal_rate),
            'home_goal_credible_interval': np.percentile(home_goal_rate, [2.5, 97.5]),
            'away_goal_credible_interval': np.percentile(away_goal_rate, [2.5, 97.5])
        }
    
    def get_team_strengths(self) -> Dict[str, Dict[str, float]]:
        """Get posterior estimates of team strengths"""
        if self.trace is None:
            return {}
        
        strengths = {}
        for team, idx in self.team_indices.items():
            attack_mean = np.mean(self.trace['attack'][:, idx])
            defense_mean = np.mean(self.trace['defense'][:, idx])
            
            strengths[team] = {
                'attack': float(attack_mean),
                'defense': float(defense_mean),
                'overall': float(attack_mean - defense_mean)
            }
            
        return strengths
