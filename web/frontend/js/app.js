class SoccerPredictionsApp {
    constructor() {
        this.apiBaseUrl = '/api/v1';
        this.currentPredictions = [];
        this.init();
    }

    init() {
        this.loadTodaysPredictions();
        this.setupEventListeners();
        this.startRealTimeUpdates();
    }

    async loadTodaysPredictions() {
        try {
            this.showLoading('oddsContainer');
            this.showLoading('predictionContainer');

            const response = await fetch(`${this.apiBaseUrl}/predictions/today`);
            const data = await response.json();

            this.displayBookmakerOdds(data.bookmaker_odds);
            this.displayAIPredictions(data.ai_predictions);
            
        } catch (error) {
            console.error('Error loading predictions:', error);
            this.showError('Failed to load predictions');
        }
    }

    displayBookmakerOdds(oddsData) {
        const container = document.getElementById('oddsContainer');
        container.innerHTML = '';

        Object.entries(oddsData).forEach(([source, matches]) => {
            matches.forEach(match => {
                const oddsCard = this.createOddsCard(match, source);
                container.appendChild(oddsCard);
            });
        });
    }

    createOddsCard(match, source) {
        const card = document.createElement('div');
        card.className = `prediction-card bookmaker ${source}`;
        
        card.innerHTML = `
            <div class="match-teams">
                <h3>${match.home_team} vs ${match.away_team}</h3>
                <span class="source-badge">${source}</span>
            </div>
            <div class="odds-comparison">
                <div class="odds-source">
                    <div class="odds-label">Home</div>
                    <div class="odds-value">${match.home_odds}</div>
                </div>
                <div class="odds-source">
                    <div class="odds-label">Draw</div>
                    <div class="odds-value">${match.draw_odds}</div>
                </div>
                <div class="odds-source">
                    <div class="odds-label">Away</div>
                    <div class="odds-value">${match.away_odds}</div>
                </div>
            </div>
            <div class="match-time">
                ${new Date(match.start_time).toLocaleString()}
            </div>
        `;

        return card;
    }

    displayAIPredictions(predictions) {
        const container = document.getElementById('predictionContainer');
        container.innerHTML = '';

        predictions.forEach(prediction => {
            const predictionCard = this.createAIPredictionCard(prediction);
            container.appendChild(predictionCard);
        });
    }

    createAIPredictionCard(prediction) {
        const card = document.createElement('div');
        card.className = 'prediction-card ai-prediction';
        
        const confidencePercent = Math.round(prediction.confidence * 100);
        
        card.innerHTML = `
            <div class="match-header">
                <h3>${prediction.home_team} vs ${prediction.away_team}</h3>
                <div class="prediction-result">
                    <span class="predicted-result">${prediction.predicted_result}</span>
                    <span class="confidence">${confidencePercent}% confidence</span>
                </div>
            </div>
            
            <div class="confidence-meter">
                <div class="confidence-level" style="width: ${confidencePercent}%"></div>
            </div>
            
            <div class="probability-breakdown">
                <div class="prob-item">
                    <span>Home Win: ${Math.round(prediction.probabilities.home_win * 100)}%</span>
                </div>
                <div class="prob-item">
                    <span>Draw: ${Math.round(prediction.probabilities.draw * 100)}%</span>
                </div>
                <div class="prob-item">
                    <span>Away Win: ${Math.round(prediction.probabilities.away_win * 100)}%</span>
                </div>
            </div>
            
            <div class="key-factors">
                <h4>Key Factors:</h4>
                <ul>
                    ${prediction.key_factors.map(factor => 
                        `<li>${factor}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="ai-sources">
                <div class="source-comparison">
                    <strong>AI vs Bookmakers:</strong>
                    <span class="comparison ${prediction.ai_vs_bookmakers}">
                        ${prediction.ai_vs_bookmakers}
                    </span>
                </div>
            </div>
        `;

        return card;
    }

    async generateCustomPrediction(formData) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/predictions/custom`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const prediction = await response.json();
            this.displayCustomPrediction(prediction);
            
        } catch (error) {
            console.error('Error generating custom prediction:', error);
            this.showError('Failed to generate prediction');
        }
    }

    startRealTimeUpdates() {
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws/predictions`);
        
        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            this.handleRealTimeUpdate(update);
        };

        // Polling fallback
        setInterval(() => {
            this.loadTodaysPredictions();
        }, 300000); // Update every 5 minutes
    }

    setupEventListeners() {
        // Custom prediction form
        document.getElementById('customPredictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleCustomPrediction(e);
        });

        // Real-time updates toggle
        document.getElementById('realTimeToggle').addEventListener('change', (e) => {
            this.toggleRealTimeUpdates(e.target.checked);
        });
    }

    handleCustomPrediction(event) {
        const formData = new FormData(event.target);
        const predictionData = {
            home_team: formData.get('homeTeam'),
            away_team: formData.get('awayTeam'),
            league: formData.get('league'),
            venue: formData.get('venue')
        };

        this.generateCustomPrediction(predictionData);
    }

    showLoading(containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '<div class="spinner"></div>';
    }

    showError(message) {
        // Implement error display logic
        console.error(message);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SoccerPredictionsApp();
});
