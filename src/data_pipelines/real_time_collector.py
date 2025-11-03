import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class RealTimeDataCollector:
    def __init__(self):
        self.scrapers = {
            'hollywoodbets': HollywoodBetsScraper(),
            'betway': BetwayScraper()
        }
        self.data_sources = {
            'player_stats': 'https://api.football-data.org/v4',
            'weather': 'https://api.openweathermap.org/data/2.5',
            'injury_data': 'https://api.sports-injury-data.org/v1'
        }
    
    async def collect_todays_data(self) -> Dict:
        """Collect all data for today's predictions"""
        tasks = [
            self.get_bookmaker_odds(),
            self.get_player_performance(),
            self.get_team_news(),
            self.get_weather_data(),
            self.get_injury_updates()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'bookmaker_odds': results[0],
            'player_performance': results[1],
            'team_news': results[2],
            'weather_conditions': results[3],
            'injury_data': results[4],
            'collected_at': datetime.now().isoformat()
        }
    
    async def get_bookmaker_odds(self) -> Dict:
        """Get odds from all bookmakers"""
        all_odds = {}
        
        for source, scraper in self.scrapers.items():
            try:
                matches = await scraper.get_todays_matches()
                all_odds[source] = matches
            except Exception as e:
                print(f"Error getting odds from {source}: {e}")
                all_odds[source] = []
                
        return all_odds
