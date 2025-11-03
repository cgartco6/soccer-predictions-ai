import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

class APICollector:
    """
    Collector for various football data APIs
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.rate_limits = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_football_data(self, endpoint: str, 
                                  params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Collect data from football-data API
        """
        base_url = self.config['football_data']['base_url']
        api_key = self.config['football_data']['api_key']
        
        headers = {
            'X-Auth-Token': api_key,
            'User-Agent': 'SoccerPredictionsAI/1.0'
        }
        
        try:
            async with self.session.get(
                f"{base_url}/{endpoint}",
                headers=headers,
                params=params
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'data': data,
                        'endpoint': endpoint
                    }
                else:
                    self.logger.error(f"API error: {response.status}")
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}",
                        'endpoint': endpoint
                    }
                    
        except Exception as e:
            self.logger.error(f"Error collecting from {endpoint}: {e}")
            return {
                'success': False,
                'error': str(e),
                'endpoint': endpoint
            }
    
    async def collect_weather_data(self, location: str, 
                                 date: datetime) -> Dict[str, Any]:
        """
        Collect weather data for match location
        """
        base_url = self.config['weather']['base_url']
        api_key = self.config['weather']['api_key']
        
        params = {
            'q': location,
            'dt': date.strftime('%Y-%m-%d'),
            'appid': api_key,
            'units': 'metric'
        }
        
        try:
            async with self.session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_weather_data(data)
                else:
                    return {'success': False, 'error': f"Weather API error: {response.status}"}
                    
        except Exception as e:
            self.logger.error(f"Weather API error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def collect_injury_data(self, team: str) -> Dict[str, Any]:
        """
        Collect player injury data
        """
        # Implementation for injury data collection
        return {}
    
    async def collect_transfer_data(self, team: str) -> Dict[str, Any]:
        """
        Collect recent transfer data
        """
        # Implementation for transfer data collection
        return {}
    
    def _parse_weather_data(self, data: Dict) -> Dict[str, Any]:
        """Parse weather API response"""
        try:
            weather = data.get('weather', [{}])[0]
            main = data.get('main', {})
            wind = data.get('wind', {})
            
            return {
                'success': True,
                'temperature': main.get('temp'),
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'wind_speed': wind.get('speed'),
                'wind_direction': wind.get('deg'),
                'conditions': weather.get('main'),
                'description': weather.get('description'),
                'precipitation': data.get('rain', {}).get('1h', 0)
            }
        except Exception as e:
            self.logger.error(f"Error parsing weather data: {e}")
            return {'success': False, 'error': str(e)}
