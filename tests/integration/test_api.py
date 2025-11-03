import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json

from src.api.main import app
from src.agents import (PredictionEnsembleAgent, SyntheticIntelligenceAgent,
                       StrategicIntelligenceAgent, DataCollectorAgent)

class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def sample_prediction_request(self):
        return {
            "home_team": "Manchester United",
            "away_team": "Liverpool", 
            "league": "Premier League",
            "match_date": "2024-01-15T15:00:00",
            "home_odds": 2.1,
            "draw_odds": 3.2,
            "away_odds": 3.5
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AI Soccer Predictions API"
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    @patch('src.api.routes.predictions.PredictionEnsembleAgent')
    @patch('src.api.routes.predictions.SyntheticIntelligenceAgent') 
    @patch('src.api.routes.predictions.StrategicIntelligenceAgent')
    @patch('src.api.routes.predictions.DataCollectorAgent')
    def test_prediction_endpoint(self, mock_data_agent, mock_strategic_agent,
                               mock_synthetic_agent, mock_ensemble_agent, 
                               client, sample_prediction_request):
        """Test prediction endpoint with mocked agents"""
        
        # Mock agent responses
        mock_ensemble_instance = AsyncMock()
        mock_ensemble_instance.process.return_value = {
            'ensemble_prediction': {
                'probabilities': [[0.45, 0.30, 0.25]],
                'confidence_metrics': {'overall_confidence': 0.72},
                'expected_goals': {'home': 1.8, 'away': 1.2}
            },
            'feature_importance': {
                'team_strength': 0.23,
                'recent_form': 0.18
            }
        }
        mock_ensemble_agent.return_value = mock_ensemble_instance
        
        mock_synthetic_instance = AsyncMock()
        mock_synthetic_instance.process.return_value = {
            'simulations': {'num_simulations': 10000},
            'synthetic_matches': []
        }
        mock_synthetic_agent.return_value = mock_synthetic_instance
        
        mock_strategic_instance = AsyncMock()
        mock_strategic_instance.process.return_value = {
            'strategic_insights': ['Home advantage', 'Strong recent form']
        }
        mock_strategic_agent.return_value = mock_strategic_instance
        
        mock_data_instance = AsyncMock()
        mock_data_instance.process.return_value = {
            'collected_data': {'additional_data': 'test'}
        }
        mock_data_agent.return_value = mock_data_instance
        
        response = client.post("/api/v1/predict", json=sample_prediction_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction_id" in data
        assert "predicted_result" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "explanation" in data
        
        # Verify probabilities
        probs = data["probabilities"]
        assert "home_win" in probs
        assert "draw" in probs
        assert "away_win" in probs
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
    
    def test_prediction_validation(self, client):
        """Test prediction request validation"""
        invalid_request = {
            "home_team": "",  # Empty team name
            "away_team": "Liverpool",
            "league": "Premier League"
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data
    
    @patch('src.api.routes.predictions.PredictionEnsembleAgent')
    def test_batch_prediction(self, mock_ensemble_agent, client):
        """Test batch prediction endpoint"""
        mock_ensemble_instance = AsyncMock()
        mock_ensemble_instance.process.return_value = {
            'ensemble_prediction': {
                'probabilities': [[0.45, 0.30, 0.25]],
                'confidence_metrics': {'overall_confidence': 0.72}
            }
        }
        mock_ensemble_agent.return_value = mock_ensemble_instance
        
        batch_request = {
            "matches": [
                {
                    "home_team": "Manchester United",
                    "away_team": "Liverpool",
                    "league": "Premier League"
                },
                {
                    "home_team": "Arsenal", 
                    "away_team": "Chelsea",
                    "league": "Premier League"
                }
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 2
        assert data["summary"]["total_matches"] == 2
    
    def test_models_endpoint(self, client):
        """Test models information endpoint"""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        model = data[0]
        assert "model_name" in model
        assert "model_type" in model
        assert "performance_metrics" in model
    
    def test_odds_comparison_endpoint(self, client):
        """Test odds comparison endpoint"""
        response = client.get(
            "/api/v1/predict/odds-comparison",
            params={
                "home_team": "Manchester United",
                "away_team": "Liverpool", 
                "league": "Premier League"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "match" in data
        assert "odds" in data
        assert "hollywoodbets" in data["odds"]
        assert "betway" in data["odds"]
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Make multiple rapid requests to trigger rate limiting
        requests = []
        for i in range(10):
            request = {
                "home_team": f"Team {i}",
                "away_team": "Liverpool",
                "league": "Premier League"
            }
            response = client.post("/api/v1/predict", json=request)
            requests.append(response.status_code)
        
        # At least some requests should be rate limited
        assert 429 in requests
    
    def test_authentication(self, client):
        """Test API authentication"""
        # Test without API key
        response = client.get("/api/v1/models")
        assert response.status_code == 401
        
        # Test with invalid API key
        response = client.get(
            "/api/v1/models", 
            headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code == 401

class TestErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_invalid_endpoint(self, client):
        """Test response for invalid endpoint"""
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404
    
    @patch('src.api.routes.predictions.PredictionEnsembleAgent')
    def test_prediction_server_error(self, mock_ensemble_agent, client):
        """Test prediction endpoint with server error"""
        mock_ensemble_instance = AsyncMock()
        mock_ensemble_instance.process.side_effect = Exception("Model error")
        mock_ensemble_agent.return_value = mock_ensemble_instance
        
        request = {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "league": "Premier League"
        }
        
        response = client.post("/api/v1/predict", json=request)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON"""
        response = client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

class TestWebSocketIntegration:
    """Test WebSocket functionality (if implemented)"""
    
    # Note: WebSocket testing would require additional setup
    # This is a placeholder for future WebSocket integration tests
    
    def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        # This would test real-time prediction updates via WebSocket
        pass
    
    def test_websocket_prediction_updates(self):
        """Test WebSocket prediction update messages"""
        # This would test sending prediction updates via WebSocket
        pass
