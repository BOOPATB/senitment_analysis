/**
 * Voice Input Manager
 * Handles real-time voice recording, transcription, and sentiment analysis
 */

class VoiceInputManager {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.isRecording = false;
        this.sessionId = null;
        this.pollInterval = null;
        this.updateInterval = null;
        this.recognizedText = '';
        this.currentSentiment = null;
        
        // DOM elements
        this.elements = {};
        this.bindElements();
        
        // Callbacks
        this.onTextUpdate = null;
        this.onSentimentUpdate = null;
        this.onStatusChange = null;
    }
    
    bindElements() {
        const elementIds = [
            'voiceBtn', 'voiceBtnText', 'testMicBtn', 'voiceStatus',
            'voiceTextDisplay', 'voiceText', 'voiceSentimentDisplay',
            'voiceSentiment', 'audioLevelBar'
        ];
        
        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }
    
    // Test microphone functionality
    async testMicrophone() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/voice/test_microphone`, {
                method: 'GET'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Microphone test successful!', 'success');
                return true;
            } else {
                throw new Error(data.error || 'Microphone test failed');
            }
        } catch (error) {
            this.showNotification(`Microphone test failed: ${error.message}`, 'error');
            return false;
        }
    }
    
    // Start voice recording
    async startRecording(modelType = 'transformer') {
        if (this.isRecording) {
            this.stopRecording();
            return;
        }
        
        try {
            // Generate a unique session ID
            this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2);
            
            const response = await fetch(`${this.apiBaseUrl}/api/voice/start_recording`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    model_type: modelType
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRecording = true;
                this.updateUI();
                this.startPolling();
                this.showNotification('Voice recording started!', 'success');
                
                // Add active class to voice section
                const voiceSection = document.querySelector('.voice-section');
                if (voiceSection) {
                    voiceSection.classList.add('active');
                }
            } else {
                throw new Error(data.error || 'Failed to start recording');
            }
        } catch (error) {
            this.showNotification(`Error starting recording: ${error.message}`, 'error');
        }
    }
    
    // Stop voice recording
    async stopRecording() {
        if (!this.isRecording || !this.sessionId) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/voice/stop_recording`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            
            this.isRecording = false;
            this.stopPolling();
            this.updateUI();
            
            // Remove active class from voice section
            const voiceSection = document.querySelector('.voice-section');
            if (voiceSection) {
                voiceSection.classList.remove('active');
            }
            
            if (data.status === 'success') {
                this.showNotification('Recording stopped. Final analysis complete!', 'success');
                
                // Display final summary if available
                if (data.summary) {
                    this.displayFinalSummary(data.summary);
                }
            } else {
                this.showNotification(`Warning: ${data.error || 'Recording stopped with errors'}`, 'warning');
            }
            
        } catch (error) {
            this.isRecording = false;
            this.stopPolling();
            this.updateUI();
            this.showNotification(`Error stopping recording: ${error.message}`, 'error');
        }
        
        this.sessionId = null;
    }
    
    // Start polling for updates
    startPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
        
        this.pollInterval = setInterval(async () => {
            if (!this.isRecording || !this.sessionId) {
                this.stopPolling();
                return;
            }
            
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/voice/status?session_id=${this.sessionId}`);
                const data = await response.json();
                
                if (data.status === 'recording') {
                    // Use the actual status data from the server
                    const statusData = {
                        text: data.latest_text || `Recording... (${Math.floor(data.duration)}s)`,
                        sentiment: data.latest_sentiment,
                        audio_level: Math.random() * 0.8 + 0.2  // Still simulated for visual feedback
                    };
                    this.updateFromStatus(statusData);
                }
            } catch (error) {
                console.error('Error polling voice status:', error);
            }
        }, 1000); // Poll every second
    }
    
    // Stop polling
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    // Update from status data
    updateFromStatus(data) {
        // Update recognized text
        if (data.text && data.text !== this.recognizedText) {
            this.recognizedText = data.text;
            this.updateRecognizedText(data.text);
            
            // Trigger callback
            if (this.onTextUpdate) {
                this.onTextUpdate(data.text);
            }
        }
        
        // Update sentiment analysis
        if (data.sentiment) {
            this.currentSentiment = data.sentiment;
            this.updateSentimentDisplay(data.sentiment);
            
            // Trigger callback
            if (this.onSentimentUpdate) {
                this.onSentimentUpdate(data.sentiment);
            }
        }
        
        // Update audio level (simulated)
        if (data.audio_level !== undefined) {
            this.updateAudioLevel(data.audio_level);
        } else {
            // Simulate audio level for visual feedback
            this.simulateAudioLevel();
        }
    }
    
    // Update recognized text display
    updateRecognizedText(text) {
        if (this.elements.voiceText) {
            this.elements.voiceText.textContent = text || 'Listening...';
            
            // Add typewriter animation
            this.elements.voiceText.style.animation = 'none';
            setTimeout(() => {
                this.elements.voiceText.style.animation = 'typewriter 0.3s ease-in-out';
            }, 10);
        }
    }
    
    // Update sentiment display
    updateSentimentDisplay(sentiment) {
        if (!this.elements.voiceSentiment || !sentiment) return;
        
        const badge = this.elements.voiceSentiment.querySelector('.sentiment-badge');
        const confidence = this.elements.voiceSentiment.querySelector('.confidence-text');
        
        // Extract sentiment data - could be from different structures
        let sentimentLabel, sentimentConfidence;
        
        if (sentiment.sentiment) {
            // Backend sentiment data structure
            sentimentLabel = sentiment.sentiment;
            sentimentConfidence = sentiment.confidence;
        } else if (sentiment.primary_sentiment) {
            // Frontend sentiment data structure
            sentimentLabel = sentiment.primary_sentiment;
            sentimentConfidence = sentiment.confidence;
        } else {
            // Simple string sentiment
            sentimentLabel = sentiment;
            sentimentConfidence = 0.5;
        }
        
        if (badge && sentimentLabel) {
            // Remove existing classes
            badge.classList.remove('positive', 'negative', 'neutral');
            
            // Add new sentiment class
            badge.classList.add(sentimentLabel);
            badge.textContent = sentimentLabel.toUpperCase();
            
            // Add animation
            badge.style.animation = 'none';
            setTimeout(() => {
                badge.style.animation = 'sentimentPop 0.4s ease-out';
            }, 10);
        }
        
        if (confidence && sentimentConfidence !== undefined) {
            const confidencePercent = Math.round(sentimentConfidence * 100);
            confidence.textContent = `${confidencePercent}%`;
        }
    }
    
    // Update audio level visualization
    updateAudioLevel(level) {
        if (this.elements.audioLevelBar) {
            const percentage = Math.max(0, Math.min(100, level * 100));
            this.elements.audioLevelBar.style.width = `${percentage}%`;
        }
    }
    
    // Simulate audio level for visual feedback
    simulateAudioLevel() {
        if (!this.isRecording) return;
        
        // Generate a realistic audio level simulation
        const baseLevel = 0.2 + Math.random() * 0.3;
        const spike = Math.random() > 0.8 ? Math.random() * 0.5 : 0;
        const level = Math.min(1, baseLevel + spike);
        
        this.updateAudioLevel(level);
    }
    
    // Update UI state
    updateUI() {
        if (!this.elements.voiceBtn || !this.elements.voiceBtnText) return;
        
        if (this.isRecording) {
            // Recording state
            this.elements.voiceBtn.classList.add('recording');
            this.elements.voiceBtnText.textContent = 'Stop Recording';
            
            // Show status displays
            if (this.elements.voiceStatus) {
                this.elements.voiceStatus.style.display = 'block';
            }
            if (this.elements.voiceTextDisplay) {
                this.elements.voiceTextDisplay.style.display = 'block';
            }
            if (this.elements.voiceSentimentDisplay) {
                this.elements.voiceSentimentDisplay.style.display = 'block';
            }
            
            // Start audio level simulation
            this.updateInterval = setInterval(() => {
                this.simulateAudioLevel();
            }, 200);
            
        } else {
            // Stopped state
            this.elements.voiceBtn.classList.remove('recording');
            this.elements.voiceBtnText.textContent = 'Start Recording';
            
            // Hide status displays
            if (this.elements.voiceStatus) {
                this.elements.voiceStatus.style.display = 'none';
            }
            if (this.elements.voiceTextDisplay) {
                this.elements.voiceTextDisplay.style.display = 'none';
            }
            if (this.elements.voiceSentimentDisplay) {
                this.elements.voiceSentimentDisplay.style.display = 'none';
            }
            
            // Clear audio level
            if (this.elements.audioLevelBar) {
                this.elements.audioLevelBar.style.width = '0%';
            }
        }
        
        // Trigger status change callback
        if (this.onStatusChange) {
            this.onStatusChange(this.isRecording, this.currentSentiment);
        }
    }
    
    // Display final summary
    displayFinalSummary(summary) {
        let message = `Final Analysis:\n`;
        message += `Sentiment: ${summary.primary_sentiment}\n`;
        message += `Confidence: ${Math.round(summary.confidence * 100)}%\n`;
        message += `Words processed: ${summary.total_words || 'N/A'}`;
        
        this.showNotification(message, 'info');
        
        // Also update the main text input if available
        const textInput = document.getElementById('textInput');
        if (textInput && this.recognizedText) {
            textInput.value = this.recognizedText;
            
            // Update character count
            const charCount = document.getElementById('charCount');
            if (charCount) {
                charCount.textContent = this.recognizedText.length;
            }
        }
    }
    
    // Show notification (depends on external notification system)
    showNotification(message, type = 'info') {
        if (window.sentimentApp && window.sentimentApp.showNotification) {
            window.sentimentApp.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
    
    // Clean up resources
    destroy() {
        this.stopPolling();
        if (this.isRecording) {
            this.stopRecording();
        }
    }
    
    // Get current status
    getStatus() {
        return {
            isRecording: this.isRecording,
            sessionId: this.sessionId,
            recognizedText: this.recognizedText,
            currentSentiment: this.currentSentiment
        };
    }
    
    // Set callbacks
    setCallbacks(callbacks) {
        this.onTextUpdate = callbacks.onTextUpdate || null;
        this.onSentimentUpdate = callbacks.onSentimentUpdate || null;
        this.onStatusChange = callbacks.onStatusChange || null;
    }
}

// Export for use in other scripts
window.VoiceInputManager = VoiceInputManager;
