import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict

class BetwayScraper:
    def __init__(self):
        self.base_url = "https://www.betway.co.za"
        self.api_endpoints = {
            'soccer': '/api/sports/soccer/matches',
            'live': '/api/sports/live-events'
        }
    
    async def get_todays_matches(self) -> List[Dict]:
        """Get today's matches from Betway using API and web scraping"""
        matches = []
        
        try:
            # Try API first
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                
                async with session.get(
                    f"{self.base_url}{self.api_endpoints['soccer']}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches.extend(self._parse_api_response(data))
                    else:
                        # Fallback to web scraping
                        matches.extend(await self._scrape_web_matches())
                        
        except Exception as e:
            print(f"Error scraping Betway: {e}")
            matches.extend(await self._scrape_web_matches())
            
        return matches
    
    def _parse_api_response(self, data: Dict) -> List[Dict]:
        """Parse Betway API response"""
        matches = []
        
        for event in data.get('events', []):
            if event.get('sportName') == 'Soccer':
                match_data = {
                    'home_team': event.get('homeTeam'),
                    'away_team': event.get('awayTeam'),
                    'home_odds': event.get('homeOdds'),
                    'draw_odds': event.get('drawOdds'),
                    'away_odds': event.get('awayOdds'),
                    'start_time': event.get('startTime'),
                    'league': event.get('leagueName'),
                    'timestamp': time.time(),
                    'source': 'betway'
                }
                matches.append(match_data)
                
        return matches
    
    async def _scrape_web_matches(self) -> List[Dict]:
        """Fallback web scraping method"""
        # Implementation for web scraping fallback
        return []
