class StrategicIntelligenceAgent:
    """
    AI Agent for strategic analysis and contextual understanding
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.knowledge_graph = self._build_knowledge_graph()
        self.strategy_analyzer = StrategyAnalyzer(config)
        
    def analyze_match_context(self, match_data: Dict) -> Dict:
        """
        Perform deep strategic analysis of match context
        """
        analysis = {
            'tactical_analysis': self._analyze_tactics(match_data),
            'motivational_factors': self._analyze_motivation(match_data),
            'external_factors': self._analyze_external_factors(match_data),
            'historical_patterns': self._analyze_historical_patterns(match_data)
        }
        
        return self._synthesize_analysis(analysis)
    
    def _analyze_tactics(self, match_data: Dict) -> Dict:
        """Analyze team tactics and formations"""
        team1_tactics = self._extract_tactical_patterns(match_data['team1'])
        team2_tactics = self._extract_tactical_patterns(match_data['team2'])
        
        return {
            'formation_matchup': self._analyze_formation_matchup(
                team1_tactics, team2_tactics
            ),
            'style_clash': self._analyze_playing_styles(team1_tactics, team2_tactics),
            'key_battles': self._identify_key_battles(match_data)
        }
    
    def _analyze_motivation(self, match_data: Dict) -> Dict:
        """Analyze motivational factors for both teams"""
        motivation_factors = {
            'team1': self._calculate_motivation_score(match_data['team1'], match_data),
            'team2': self._calculate_motivation_score(match_data['team2'], match_data)
        }
        
        return motivation_factors
    
    def generate_strategic_insights(self, match_data: Dict) -> List[str]:
        """
        Generate human-readable strategic insights
        """
        context_analysis = self.analyze_match_context(match_data)
        insights = []
        
        # Generate insights based on analysis
        if context_analysis['tactical_analysis']['style_clash'] == 'attack_vs_defense':
            insights.append("Classic attack vs defense matchup expected")
            
        if context_analysis['motivational_factors']['team1'] > 0.7:
            insights.append("Home team shows high motivation for this fixture")
            
        return insights
