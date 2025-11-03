import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from anti_detection import StealthBrowser

class HollywoodBetsScraper:
    def __init__(self):
        self.base_url = "https://www.hollywoodbets.net"
        self.browser = self._setup_browser()
        
    def _setup_browser(self):
        """Configure stealth browser to avoid detection"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    
    async def get_todays_matches(self) -> List[Dict]:
        """Get today's soccer matches from Hollywoodbets"""
        matches = []
        try:
            self.browser.get(f"{self.base_url}/sports/soccer")
            await asyncio.sleep(3)  # Wait for page load
            
            # Parse match cards
            soup = BeautifulSoup(self.browser.page_source, 'html.parser')
            match_cards = soup.find_all('div', class_=['event-row', 'match-card'])
            
            for card in match_cards:
                match_data = self._parse_match_card(card)
                if match_data:
                    matches.append(match_data)
                    
        except Exception as e:
            print(f"Error scraping Hollywoodbets: {e}")
            
        return matches
    
    def _parse_match_card(self, card) -> Dict:
        """Parse individual match card"""
        try:
            teams = card.find_all('span', class_='team-name')
            odds_elements = card.find_all('button', class_='odds-button')
            
            if len(teams) >= 2 and len(odds_elements) >= 3:
                return {
                    'home_team': teams[0].text.strip(),
                    'away_team': teams[1].text.strip(),
                    'home_odds': float(odds_elements[0].text.strip()),
                    'draw_odds': float(odds_elements[1].text.strip()),
                    'away_odds': float(odds_elements[2].text.strip()),
                    'timestamp': time.time(),
                    'source': 'hollywoodbets'
                }
        except Exception as e:
            print(f"Error parsing match card: {e}")
        return None
