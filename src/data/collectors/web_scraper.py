import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging

class WebScraper:
    """
    Base web scraper for collecting odds and match data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.driver = None
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.driver:
            self.driver.quit()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with anti-detection measures"""
        options = Options()
        
        # Anti-detection options
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        
        return self.driver

class HollywoodBetsScraper(WebScraper):
    """Scraper for Hollywoodbets odds"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://www.hollywoodbets.net"
        
    async def get_todays_matches(self) -> List[Dict[str, Any]]:
        """Get today's matches from Hollywoodbets"""
        matches = []
        
        try:
            if not self.driver:
                self._setup_selenium()
                
            self.driver.get(f"{self.base_url}/sports/soccer")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event-row"))
            )
            
            # Parse page
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            match_elements = soup.find_all('div', class_='event-row')
            
            for element in match_elements:
                match_data = self._parse_match_element(element)
                if match_data:
                    matches.append(match_data)
                    
        except Exception as e:
            self.logger.error(f"Error scraping Hollywoodbets: {e}")
            
        return matches
    
    def _parse_match_element(self, element) -> Optional[Dict[str, Any]]:
        """Parse individual match element"""
        try:
            # Extract team names
            teams = element.find_all('span', class_='team-name')
            if len(teams) < 2:
                return None
                
            home_team = teams[0].text.strip()
            away_team = teams[1].text.strip()
            
            # Extract odds
            odds_elements = element.find_all('button', class_='odds-button')
            if len(odds_elements) < 3:
                return None
                
            home_odds = float(odds_elements[0].text.strip())
            draw_odds = float(odds_elements[1].text.strip())
            away_odds = float(odds_elements[2].text.strip())
            
            # Extract match time
            time_element = element.find('span', class_='event-time')
            match_time = time_element.text.strip() if time_element else "Unknown"
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                'match_time': match_time,
                'source': 'hollywoodbets',
                'scraped_at': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing match element: {e}")
            return None

class BetwayScraper(WebScraper):
    """Scraper for Betway odds"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://www.betway.co.za"
        
    async def get_todays_matches(self) -> List[Dict[str, Any]]:
        """Get today's matches from Betway"""
        matches = []
        
        try:
            # Try API first
            api_matches = await self._get_api_matches()
            if api_matches:
                return api_matches
                
            # Fallback to web scraping
            return await self._scrape_web_matches()
            
        except Exception as e:
            self.logger.error(f"Error scraping Betway: {e}")
            return []
    
    async def _get_api_matches(self) -> List[Dict[str, Any]]:
        """Get matches from Betway API"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            async with self.session.get(
                f"{self.base_url}/api/sports/soccer/matches",
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return self._parse_api_response(data)
                else:
                    self.logger.warning(f"Betway API returned {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.warning(f"Betway API failed: {e}")
            return []
    
    async def _scrape_web_matches(self) -> List[Dict[str, Any]]:
        """Scrape matches from Betway website"""
        matches = []
        
        try:
            if not self.driver:
                self._setup_selenium()
                
            self.driver.get(f"{self.base_url}/sports/soccer")
            
            # Wait for page load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "sport-event"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            match_elements = soup.find_all('div', class_='sport-event')
            
            for element in match_elements:
                match_data = self._parse_web_element(element)
                if match_data:
                    matches.append(match_data)
                    
        except Exception as e:
            self.logger.error(f"Error scraping Betway web: {e}")
            
        return matches
    
    def _parse_api_response(self, data: Dict) -> List[Dict[str, Any]]:
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
                    'source': 'betway',
                    'scraped_at': time.time()
                }
                matches.append(match_data)
                
        return matches
    
    def _parse_web_element(self, element) -> Optional[Dict[str, Any]]:
        """Parse web element for match data"""
        try:
            # Implementation for web parsing
            # This would extract data from HTML elements
            return None
        except Exception as e:
            self.logger.error(f"Error parsing web element: {e}")
            return None
