import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
from .base_agent import BaseAgent

class DataCollectorAgent(BaseAgent):
    """
    AI Agent for collecting and managing soccer data from multiple sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("data_collector", config)
        self.sources = config.get('sources', {})
        self.session = None
        
    def _setup(self):
        """Initialize data collection components"""
        self.scrapers = self._initialize_scrapers()
        self.apis = self._initialize_apis()
        self.validators = self._initialize_validators()
        
    def _initialize_scrapers(self) -> Dict[str, Any]:
        """Initialize web scrapers for different sources"""
        scrapers = {}
        
        if 'hollywoodbets' in self.sources:
            from ..data.collectors.web_scraper import HollywoodBetsScraper
            scrapers['hollywoodbets'] = HollywoodBetsScraper(
                self.sources['hollywoodbets']
            )
            
        if 'betway' in self.sources:
            from ..data.collectors.web_scraper import BetwayScraper
            scrapers['betway'] = BetwayScraper(self.sources['betway'])
            
        return scrapers
    
    def _initialize_apis(self) -> Dict[str, Any]:
        """Initialize API clients"""
        apis = {}
        
        if 'football_data' in self.sources:
            from ..data.collectors.api_collector import FootballDataAPI
            apis['football_data'] = FootballDataAPI(
                self.sources['football_data']
            )
            
        if 'weather' in self.sources:
            from ..data.collectors.api_collector import WeatherAPI
            apis['weather'] = WeatherAPI(self.sources['weather'])
            
        return apis
    
    def _initialize_validators(self) -> Dict[str, Any]:
        """Initialize data validators"""
        from ..data.collectors.data_validator import DataValidator
        return {
            'match_data': DataValidator('match'),
            'odds_data': DataValidator('odds'),
            'player_data': DataValidator('player')
        }
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and process data from all sources
        """
        collection_tasks = []
        
        # Schedule data collection tasks
        for source, scraper in self.scrapers.items():
            task = self._collect_from_scraper(scraper, source)
            collection_tasks.append(task)
            
        for api_name, api_client in self.apis.items():
            task = self._collect_from_api(api_client, api_name)
            collection_tasks.append(task)
        
        # Execute all collection tasks concurrently
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Process and validate results
        processed_data = self._process_collected_data(results)
        validated_data = self._validate_data(processed_data)
        
        return {
            'collected_data': validated_data,
            'collection_metadata': self._get_collection_metadata(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _collect_from_scraper(self, scraper, source: str) -> Dict[str, Any]:
        """Collect data from a web scraper"""
        try:
            data = await scraper.get_todays_matches()
            return {
                'source': source,
                'type': 'scraper',
                'data': data,
                'success': True
            }
        except Exception as e:
            self.logger.error(f"Error collecting from {source}: {e}")
            return {
                'source': source,
                'type': 'scraper',
                'data': [],
                'success': False,
                'error': str(e)
            }
    
    async def _collect_from_api(self, api_client, api_name: str) -> Dict[str, Any]:
        """Collect data from an API"""
        try:
            data = await api_client.get_data()
            return {
                'source': api_name,
                'type': 'api',
                'data': data,
                'success': True
            }
        except Exception as e:
            self.logger.error(f"Error collecting from {api_name}: {e}")
            return {
                'source': api_name,
                'type': 'api',
                'data': [],
                'success': False,
                'error': str(e)
            }
    
    def _process_collected_data(self, results: List[Dict]) -> Dict[str, Any]:
        """Process and merge collected data"""
        processed_data = {
            'matches': [],
            'odds': {},
            'player_stats': [],
            'team_stats': [],
            'weather_data': {}
        }
        
        for result in results:
            if result.get('success', False):
                data = result['data']
                source = result['source']
                
                if result['type'] == 'scraper':
                    # Process scraper data (odds)
                    processed_data['odds'][source] = data
                elif result['type'] == 'api':
                    # Process API data
                    if 'matches' in data:
                        processed_data['matches'].extend(data['matches'])
                    if 'player_stats' in data:
                        processed_data['player_stats'].extend(data['player_stats'])
                    if 'weather' in data:
                        processed_data['weather_data'] = data['weather']
        
        return processed_data
    
    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate collected data"""
        validated_data = {}
        
        for data_type, values in data.items():
            validator = self.validators.get(data_type)
            if validator:
                validated_data[data_type] = validator.validate(values)
            else:
                validated_data[data_type] = values
                
        return validated_data
    
    def _get_collection_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data collection process"""
        return {
            'sources_used': list(self.scrapers.keys()) + list(self.apis.keys()),
            'collection_time': datetime.now().isoformat(),
            'data_types_collected': ['matches', 'odds', 'player_stats', 'weather_data']
        }
