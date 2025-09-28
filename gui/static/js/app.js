/**
 * Main Application Controller
 * Connects frontend interface with backend API and coordinates all functionality
 */

class SentimentAnalysisApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.isModelLoaded = false;
        this.analysisHistory = [];
        this.statistics = {
            positive: 0,
            negative: 0,
            neutral: 0,
            total: 0
        };
        
        // DOM elements
        this.elements = {};
        this.bindElements();
        this.bindEvents();
        
        // Voice input manager
        this.voiceManager = null;
        
        // Initialize app
        this.init();
    }
    
    bindElements() {
        const elementIds = [
            'loadingScreen', 'textInput', 'analyzeBtn', 'clearBtn', 'modelSelect',
            'resultsSection', 'sentimentResult', 'confidenceBars', 'exampleChips',
            'charCount', 'historyList', 'positiveCount', 'negativeCount', 'neutralCount',
            'totalAnalyzed', 'responseTime', 'modelStatus', 'settingsModal',
            'settingsBtn', 'closeModal', 'fullscreenBtn', 'resetCamera',
            'toggleParticles', 'toggleWireframe', 'uploadBtn', 'fileInput', 'batchBtn',
            'voiceBtn', 'testMicBtn', 'sentimentIndicator', 'sentimentIcon', 
            'sentimentLabel', 'confidenceLevel'
        ];
        
        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }
    
    bindEvents() {
        // Main functionality
        this.elements.analyzeBtn?.addEventListener('click', () => this.analyzeText());
        this.elements.clearBtn?.addEventListener('click', () => this.clearText());
        this.elements.textInput?.addEventListener('input', (e) => this.updateCharCount(e.target.value.length));
        this.elements.textInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.analyzeText();
            }
        });
        
        // Header controls
        this.elements.settingsBtn?.addEventListener('click', () => this.showSettings());
        this.elements.closeModal?.addEventListener('click', () => this.hideSettings());
        this.elements.fullscreenBtn?.addEventListener('click', () => this.toggleFullscreen());
        
        // Visualization controls
        this.elements.resetCamera?.addEventListener('click', () => this.resetCamera());
        this.elements.toggleParticles?.addEventListener('click', () => this.toggleParticles());
        this.elements.toggleWireframe?.addEventListener('click', () => this.toggleWireframe());
        
        // File upload
        this.elements.uploadBtn?.addEventListener('click', () => this.elements.fileInput?.click());
        this.elements.fileInput?.addEventListener('change', (e) => this.handleFileUpload(e));
        this.elements.batchBtn?.addEventListener('click', () => this.analyzeBatch());
        
        // Voice input controls
        this.elements.voiceBtn?.addEventListener('click', () => this.toggleVoiceRecording());
        this.elements.testMicBtn?.addEventListener('click', () => this.testMicrophone());
        
        // Settings modal
        this.elements.settingsModal?.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.hideSettings();
            }
        });
        
        // Settings controls
        this.setupSettingsControls();
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }
    
    setupSettingsControls() {
        const particleCount = document.getElementById('particleCount');
        const particleCountValue = document.getElementById('particleCountValue');
        const animationSpeed = document.getElementById('animationSpeed');
        const animationSpeedValue = document.getElementById('animationSpeedValue');
        const autoRotate = document.getElementById('autoRotate');
        const particleAnimation = document.getElementById('particleAnimation');
        
        particleCount?.addEventListener('input', (e) => {
            const value = e.target.value;
            if (particleCountValue) particleCountValue.textContent = value;
            if (window.sentimentVisualization) {
                window.sentimentVisualization.updateConfig({ particleCount: parseInt(value) });
            }
        });
        
        animationSpeed?.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (animationSpeedValue) animationSpeedValue.textContent = value + 'x';
            if (window.sentimentVisualization) {
                window.sentimentVisualization.updateConfig({ animationSpeed: value });
            }
        });
        
        autoRotate?.addEventListener('change', (e) => {
            if (window.sentimentVisualization) {
                window.sentimentVisualization.updateConfig({ autoRotate: e.target.checked });
            }
        });
        
        particleAnimation?.addEventListener('change', (e) => {
            if (window.sentimentVisualization) {
                window.sentimentVisualization.updateConfig({ particleAnimation: e.target.checked });
            }
        });
    }
    
    async init() {
        console.log('Initializing Sentiment Analysis App...');
        
        // Show loading screen
        this.showLoading();
        
        // Check model status
        await this.checkModelStatus();
        
        // Load examples
        await this.loadExamples();
        
        // Initialize UI
        this.initializeUI();
        
        // Hide loading screen after a delay
        setTimeout(() => {
            this.hideLoading();
        }, 2000);
        
        console.log('App initialized successfully');
    }
    
    showLoading() {
        if (window.animations) {
            window.animations.showLoadingScreen();
        }
    }
    
    hideLoading() {
        if (window.animations) {
            window.animations.hideLoadingScreen();
        }
    }
    
    async checkModelStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/model_status`);
            const data = await response.json();
            
            this.isModelLoaded = data.status === 'ready';
            this.updateModelStatus(data.status);
            
            if (!this.isModelLoaded) {
                // Poll for model loading
                this.pollModelStatus();
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            this.showNotification('Error connecting to server', 'error');
        }
    }
    
    pollModelStatus() {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/model_status`);
                const data = await response.json();
                
                if (data.status === 'ready') {
                    this.isModelLoaded = true;
                    this.updateModelStatus('ready');
                    this.showNotification('Models loaded successfully!', 'success');
                    clearInterval(interval);
                }
            } catch (error) {
                console.error('Error polling model status:', error);
            }
        }, 2000);
    }
    
    updateModelStatus(status) {
        if (this.elements.modelStatus) {
            const statusText = status === 'ready' ? '✅ Ready' : '⏳ Loading...';
            this.elements.modelStatus.textContent = statusText;
            
            if (window.animations) {
                window.animations.updateMetric('modelStatus', statusText);
            }
        }
        
        // Update analyze button state
        if (this.elements.analyzeBtn) {
            this.elements.analyzeBtn.disabled = status !== 'ready';
        }
    }
    
    async loadExamples() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/examples`);
            const data = await response.json();
            
            if (data.examples && this.elements.exampleChips) {
                this.renderExamples(data.examples);
            }
        } catch (error) {
            console.error('Error loading examples:', error);
        }
    }
    
    renderExamples(examples) {
        const container = this.elements.exampleChips;
        if (!container) return;
        
        container.innerHTML = '';
        
        examples.slice(0, 6).forEach(example => {
            const chip = document.createElement('div');
            chip.className = 'example-chip';
            chip.textContent = example.slice(0, 30) + (example.length > 30 ? '...' : '');
            chip.title = example;
            chip.addEventListener('click', () => {
                if (this.elements.textInput) {
                    this.elements.textInput.value = example;
                    this.updateCharCount(example.length);
                }
            });
            
            container.appendChild(chip);
        });
    }
    
    initializeUI() {
        // Set initial character count
        if (this.elements.textInput) {
            this.updateCharCount(this.elements.textInput.value.length);
        }
        
        // Update statistics display
        this.updateStatisticsDisplay();
    }
    
    updateCharCount(count) {
        if (this.elements.charCount) {
            this.elements.charCount.textContent = count;
            
            if (window.animations) {
                window.animations.animateCharCount(count, 500);
            }
        }
    }
    
    async analyzeText() {
        const text = this.elements.textInput?.value?.trim();
        if (!text) {
            this.showNotification('Please enter some text to analyze', 'error');
            return;
        }
        
        if (!this.isModelLoaded) {
            this.showNotification('Models are still loading, please wait...', 'info');
            return;
        }
        
        const modelType = this.elements.modelSelect?.value || 'transformer';
        
        // Disable button and show loading
        this.setAnalyzeButtonLoading(true);
        
        const startTime = Date.now();
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    model_type: modelType
                })
            });
            
            const data = await response.json();
            const responseTime = Date.now() - startTime;
            
            if (data.status === 'success') {
                this.displayResults(data);
                this.addToHistory(text, data.primary_sentiment, data.confidence);
                this.updateResponseTime(responseTime);
                this.showNotification('Analysis complete!', 'success');
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Error analyzing text:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.setAnalyzeButtonLoading(false);
        }
    }
    
    setAnalyzeButtonLoading(loading) {
        if (!this.elements.analyzeBtn) return;
        
        if (loading) {
            this.elements.analyzeBtn.disabled = true;
            this.elements.analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        } else {
            this.elements.analyzeBtn.disabled = false;
            this.elements.analyzeBtn.innerHTML = '<i class="fas fa-analytics"></i> Analyze Sentiment';
        }
    }
    
    displayResults(data) {
        if (!this.elements.resultsSection) return;
        
        // Create sentiment result HTML
        const sentimentHTML = `
            <div class="sentiment-badge ${data.primary_sentiment}">
                <i class="fas fa-${this.getSentimentIcon(data.primary_sentiment)}"></i>
                ${data.primary_sentiment.toUpperCase()}
            </div>
            <div class="confidence-text">
                Confidence: ${Math.round(data.confidence * 100)}%
            </div>
        `;
        
        if (this.elements.sentimentResult) {
            this.elements.sentimentResult.innerHTML = sentimentHTML;
        }
        
        // Create confidence bars
        if (this.elements.confidenceBars && data.predictions) {
            this.renderConfidenceBars(data.predictions);
        }
        
        // Show results with animation
        if (window.animations) {
            window.animations.showSentimentResults(data.primary_sentiment, data.confidence);
        }
    }
    
    renderConfidenceBars(predictions) {
        const container = this.elements.confidenceBars;
        if (!container) return;
        
        container.innerHTML = '';
        
        predictions.forEach(pred => {
            const percentage = Math.round(pred.score * 100);
            const barHTML = `
                <div class="confidence-bar">
                    <div class="confidence-label">${pred.label}</div>
                    <div class="confidence-track">
                        <div class="confidence-fill ${pred.label}" style="width: ${percentage}%"></div>
                    </div>
                    <div class="confidence-value">${percentage}%</div>
                </div>
            `;
            
            container.insertAdjacentHTML('beforeend', barHTML);
        });
    }
    
    getSentimentIcon(sentiment) {
        const icons = {
            positive: 'smile',
            negative: 'frown',
            neutral: 'meh'
        };
        return icons[sentiment] || 'meh';
    }
    
    addToHistory(text, sentiment, confidence) {
        const historyItem = {
            text: text.slice(0, 100) + (text.length > 100 ? '...' : ''),
            sentiment,
            confidence,
            time: new Date().toLocaleTimeString()
        };
        
        this.analysisHistory.unshift(historyItem);
        
        // Update statistics
        this.statistics[sentiment]++;
        this.statistics.total++;
        
        // Animate history and stats
        if (window.animations) {
            window.animations.addHistoryItem(historyItem);
        }
        
        this.updateStatisticsDisplay();
    }
    
    updateStatisticsDisplay() {
        const stats = ['positive', 'negative', 'neutral', 'total'];
        
        stats.forEach(stat => {
            const element = document.getElementById(`${stat}Count`) || 
                           document.getElementById(`totalAnalyzed`);
            if (element) {
                const key = stat === 'total' ? 'total' : stat;
                element.textContent = this.statistics[key];
            }
        });
    }
    
    updateResponseTime(time) {
        if (this.elements.responseTime) {
            this.elements.responseTime.textContent = `${time}ms`;
            
            if (window.animations) {
                window.animations.updateMetric('responseTime', `${time}ms`);
            }
        }
    }
    
    clearText() {
        if (this.elements.textInput) {
            this.elements.textInput.value = '';
            this.updateCharCount(0);
        }
        
        if (this.elements.resultsSection) {
            this.elements.resultsSection.style.display = 'none';
        }
        
        // Reset 3D visualization
        if (window.sentimentVisualization) {
            window.sentimentVisualization.updateSentiment('neutral', 0);
        }
    }
    
    // File upload functionality
    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const lines = content.split('\n').filter(line => line.trim());
            
            if (lines.length === 1) {
                // Single text - put in main input
                if (this.elements.textInput) {
                    this.elements.textInput.value = lines[0];
                    this.updateCharCount(lines[0].length);
                }
            } else {
                // Multiple lines - prepare for batch analysis
                this.batchTexts = lines.slice(0, 10); // Limit to 10
                this.elements.batchBtn.disabled = false;
                this.showNotification(`Loaded ${this.batchTexts.length} texts for batch analysis`, 'info');
            }
        };
        
        reader.readAsText(file);
    }
    
    async analyzeBatch() {
        if (!this.batchTexts || this.batchTexts.length === 0) {
            this.showNotification('No texts loaded for batch analysis', 'error');
            return;
        }
        
        this.elements.batchBtn.disabled = true;
        this.elements.batchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/batch_analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    texts: this.batchTexts,
                    model_type: this.elements.modelSelect?.value || 'transformer'
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Add all results to history
                data.results.forEach(result => {
                    this.addToHistory(result.text, result.primary_sentiment, result.confidence);
                });
                
                this.showNotification(`Batch analysis complete! Processed ${data.results.length} texts`, 'success');
            } else {
                throw new Error(data.error || 'Batch analysis failed');
            }
            
        } catch (error) {
            console.error('Error in batch analysis:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.elements.batchBtn.disabled = false;
            this.elements.batchBtn.innerHTML = '<i class="fas fa-play"></i> Analyze Batch';
        }
    }
    
    // UI Controls
    showSettings() {
        if (this.elements.settingsModal && window.animations) {
            window.animations.showModal(this.elements.settingsModal);
        }
    }
    
    hideSettings() {
        if (this.elements.settingsModal && window.animations) {
            window.animations.hideModal(this.elements.settingsModal);
        }
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    // 3D Visualization Controls
    resetCamera() {
        if (window.sentimentVisualization) {
            window.sentimentVisualization.resetCamera();
        }
    }
    
    toggleParticles() {
        if (window.sentimentVisualization) {
            window.sentimentVisualization.toggleParticles();
        }
    }
    
    toggleWireframe() {
        if (window.sentimentVisualization) {
            window.sentimentVisualization.toggleWireframe();
        }
    }
    
    // Keyboard shortcuts
    handleKeyboard(event) {
        // Ctrl/Cmd + Enter: Analyze
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            this.analyzeText();
        }
        
        // Escape: Close modal
        if (event.key === 'Escape') {
            if (this.elements.settingsModal && this.elements.settingsModal.style.display !== 'none') {
                this.hideSettings();
            }
        }
        
        // Ctrl/Cmd + L: Clear
        if ((event.ctrlKey || event.metaKey) && event.key === 'l') {
            event.preventDefault();
            this.clearText();
        }
    }
    
    // Voice input functionality
    initializeVoiceManager() {
        if (!window.VoiceInputManager) {
            console.error('VoiceInputManager not available');
            return;
        }
        
        this.voiceManager = new window.VoiceInputManager(this.apiBaseUrl);
        
        // Set up callbacks
        this.voiceManager.setCallbacks({
            onTextUpdate: (text) => this.onVoiceTextUpdate(text),
            onSentimentUpdate: (sentiment) => this.onVoiceSentimentUpdate(sentiment),
            onStatusChange: (isRecording, currentSentiment) => this.onVoiceStatusChange(isRecording, currentSentiment)
        });
        
        console.log('Voice manager initialized');
    }
    
    async testMicrophone() {
        if (!this.voiceManager) {
            this.initializeVoiceManager();
        }
        
        if (this.voiceManager) {
            await this.voiceManager.testMicrophone();
        }
    }
    
    async toggleVoiceRecording() {
        if (!this.isModelLoaded) {
            this.showNotification('Models are still loading, please wait...', 'info');
            return;
        }
        
        if (!this.voiceManager) {
            this.initializeVoiceManager();
        }
        
        if (!this.voiceManager) {
            this.showNotification('Voice input not available', 'error');
            return;
        }
        
        const modelType = this.elements.modelSelect?.value || 'transformer';
        
        if (this.voiceManager.isRecording) {
            await this.voiceManager.stopRecording();
        } else {
            await this.voiceManager.startRecording(modelType);
        }
    }
    
    // Voice callbacks
    onVoiceTextUpdate(text) {
        // Optionally update the main text input
        // this.elements.textInput.value = text;
        // this.updateCharCount(text.length);
    }
    
    onVoiceSentimentUpdate(sentiment) {
        if (!sentiment) return;
        
        // Update 3D visualization
        if (window.sentimentVisualization) {
            window.sentimentVisualization.updateSentiment(sentiment.primary_sentiment, sentiment.confidence);
        }
        
        // Update the main sentiment indicator
        this.updateSentimentIndicator(sentiment.primary_sentiment, sentiment.confidence);
        
        // Add to history and statistics
        this.addToHistory(
            this.voiceManager.recognizedText, 
            sentiment.primary_sentiment, 
            sentiment.confidence
        );
    }
    
    onVoiceStatusChange(isRecording, currentSentiment) {
        // Additional UI updates based on recording status
        if (isRecording && this.elements.resultsSection) {
            this.elements.resultsSection.style.display = 'none';
        }
    }
    
    updateSentimentIndicator(sentiment, confidence) {
        if (this.elements.sentimentIcon) {
            const icon = this.elements.sentimentIcon.querySelector('i');
            if (icon) {
                // Remove existing classes
                icon.classList.remove('fa-smile', 'fa-frown', 'fa-meh');
                // Add new icon class
                icon.classList.add(`fa-${this.getSentimentIcon(sentiment)}`);
            }
        }
        
        if (this.elements.sentimentLabel) {
            this.elements.sentimentLabel.textContent = sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
        }
        
        if (this.elements.confidenceLevel) {
            this.elements.confidenceLevel.textContent = `${Math.round(confidence * 100)}%`;
        }
    }
    
    // Notification helper
    showNotification(message, type = 'info') {
        if (window.animations) {
            window.animations.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Wait for other scripts to load
    setTimeout(() => {
        window.sentimentApp = new SentimentAnalysisApp();
        console.log('Sentiment Analysis App started');
    }, 200);
});
