class InjuryImpactAnalyzer:
    def __init__(self):
        self.injury_weights = {
            'goalkeeper': 1.2,
            'defender': 1.0,
            'midfielder': 1.1,
            'forward': 1.0
        }
    
    def analyze_injury_impact(self, team_squad: Dict, injuries: List) -> Dict:
        """Analyze impact of injuries on team performance"""
        impact_scores = {
            'total_impact': 0,
            'key_players_missing': [],
            'position_weaknesses': {},
            'replacement_quality': 0
        }
        
        for injury in injuries:
            player_impact = self._calculate_player_impact(injury)
            impact_scores['total_impact'] += player_impact
            
            if injury['is_key_player']:
                impact_scores['key_players_missing'].append(injury['player_name'])
                
            # Update position impact
            position = injury['position']
            if position not in impact_scores['position_weaknesses']:
                impact_scores['position_weaknesses'][position] = 0
            impact_scores['position_weaknesses'][position] += player_impact
            
        impact_scores['replacement_quality'] = self._assess_replacements(team_squad, injuries)
        
        return impact_scores
    
    def _calculate_player_impact(self, injury: Dict) -> float:
        """Calculate individual player injury impact"""
        base_impact = injury['player_rating'] / 10
        position_weight = self.injury_weights.get(injury['position'], 1.0)
        duration_factor = self._get_duration_factor(injury['expected_return'])
        
        return base_impact * position_weight * duration_factor
