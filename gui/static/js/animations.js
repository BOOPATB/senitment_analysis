/**
 * Anime.js Animation Controller
 * Handles smooth UI transitions and sentiment feedback animations
 */

class AnimationController {
    constructor() {
        this.isInitialized = false;
        this.activeAnimations = new Map();
        this.defaultEasing = 'easeOutCubic';
        this.fastDuration = 300;
        this.normalDuration = 600;
        this.slowDuration = 1000;
    }
    
    init() {
        if (this.isInitialized) return;
        
        this.setupPageAnimations();
        this.setupElementObserver();
        this.isInitialized = true;
        
        console.log('Animation Controller initialized');
    }
    
    setupPageAnimations() {
        // Initial page load animations
        this.fadeInElements();
        this.staggerElements();
    }
    
    setupElementObserver() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateElementIn(entry.target);
                }
            });
        }, observerOptions);
        
        // Observe elements with animation classes
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    }
    
    fadeInElements() {
        // Fade in main sections
        const sections = [
            '.left-panel',
            '.center-panel',
            '.right-panel'
        ];
        
        sections.forEach((selector, index) => {
            const element = document.querySelector(selector);
            if (element) {
                element.style.opacity = '0';
                element.style.transform = 'translateY(30px)';
                
                anime({
                    targets: element,
                    opacity: 1,
                    translateY: 0,
                    duration: this.normalDuration,
                    delay: index * 100,
                    easing: this.defaultEasing
                });
            }
        });
    }
    
    staggerElements() {
        // Stagger animation for cards and buttons
        const staggerGroups = [
            { selector: '.stat-card', delay: 50 },
            { selector: '.example-chip', delay: 30 },
            { selector: '.viz-btn', delay: 40 }
        ];
        
        staggerGroups.forEach(group => {
            const elements = document.querySelectorAll(group.selector);
            if (elements.length > 0) {
                elements.forEach(el => {
                    el.style.opacity = '0';
                    el.style.transform = 'scale(0.8)';
                });
                
                anime({
                    targets: elements,
                    opacity: 1,
                    scale: 1,
                    duration: this.fastDuration,
                    delay: anime.stagger(group.delay),
                    easing: 'easeOutElastic(1, .5)'
                });
            }
        });
    }
    
    animateElementIn(element) {
        if (element.classList.contains('animated')) return;
        
        element.classList.add('animated');
        
        anime({
            targets: element,
            opacity: [0, 1],
            translateY: [30, 0],
            duration: this.normalDuration,
            easing: this.defaultEasing
        });
    }
    
    // Loading animations
    showLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        if (!loadingScreen) return;
        
        loadingScreen.style.display = 'flex';
        loadingScreen.style.opacity = '0';
        
        anime({
            targets: loadingScreen,
            opacity: 1,
            duration: this.fastDuration,
            easing: this.defaultEasing
        });
        
        this.animateProgressBar();
    }
    
    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        if (!loadingScreen) return;
        
        anime({
            targets: loadingScreen,
            opacity: 0,
            duration: this.fastDuration,
            easing: this.defaultEasing,
            complete: () => {
                loadingScreen.style.display = 'none';
            }
        });
    }
    
    animateProgressBar() {
        const progressBar = document.getElementById('progressBar');
        if (!progressBar) return;
        
        anime({
            targets: progressBar,
            width: ['0%', '100%'],
            duration: 3000,
            easing: 'easeInOutQuart'
        });
    }
    
    // Sentiment result animations
    showSentimentResults(sentiment, confidence) {
        const resultsSection = document.getElementById('resultsSection');
        const sentimentResult = document.getElementById('sentimentResult');
        const confidenceBars = document.getElementById('confidenceBars');
        
        if (!resultsSection) return;
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Animate sentiment badge
        this.animateSentimentBadge(sentiment, confidence);
        
        // Animate confidence bars
        this.animateConfidenceBars();
        
        // Update 3D visualization
        if (window.sentimentVisualization) {
            window.sentimentVisualization.updateSentiment(sentiment, confidence);
        }
        
        // Update sentiment indicator
        this.updateSentimentIndicator(sentiment, confidence);
        
        // Animate statistics
        this.updateStatistics(sentiment);
    }
    
    animateSentimentBadge(sentiment, confidence) {
        const badge = document.querySelector('.sentiment-badge');
        if (!badge) return;
        
        // Scale and glow animation
        anime({
            targets: badge,
            scale: [0.5, 1.2, 1],
            duration: this.normalDuration,
            easing: 'easeOutElastic(1, .8)'
        });
        
        // Add glow effect
        badge.classList.add('glow');
        setTimeout(() => {
            badge.classList.remove('glow');
        }, 1000);
    }
    
    animateConfidenceBars() {
        const confidenceFills = document.querySelectorAll('.confidence-fill');
        
        confidenceFills.forEach((fill, index) => {
            const targetWidth = fill.style.width || '0%';
            fill.style.width = '0%';
            
            anime({
                targets: fill,
                width: targetWidth,
                duration: this.slowDuration,
                delay: index * 100,
                easing: 'easeOutQuart'
            });
        });
    }
    
    updateSentimentIndicator(sentiment, confidence) {
        const indicator = document.getElementById('sentimentIndicator');
        const icon = document.getElementById('sentimentIcon');
        const label = document.getElementById('sentimentLabel');
        const level = document.getElementById('confidenceLevel');
        
        if (!indicator) return;
        
        // Update content
        const sentimentData = {
            positive: { icon: 'fa-smile', color: '#4facfe' },
            negative: { icon: 'fa-frown', color: '#fa709a' },
            neutral: { icon: 'fa-meh', color: '#43e97b' }
        };
        
        const data = sentimentData[sentiment] || sentimentData.neutral;
        
        if (icon) {
            icon.innerHTML = `<i class="fas ${data.icon}"></i>`;
        }
        
        if (label) {
            label.textContent = sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
        }
        
        if (level) {
            level.textContent = `${Math.round(confidence * 100)}%`;
        }
        
        // Animate indicator
        anime({
            targets: indicator,
            scale: [1, 1.1, 1],
            duration: this.normalDuration,
            easing: 'easeOutElastic(1, .8)'
        });
        
        // Color animation
        anime({
            targets: indicator,
            backgroundColor: [data.color + '20', data.color + '10'],
            borderColor: [data.color + '60', data.color + '30'],
            duration: this.slowDuration,
            easing: this.defaultEasing
        });
    }
    
    updateStatistics(sentiment) {
        const countElement = document.getElementById(`${sentiment}Count`);
        const totalElement = document.getElementById('totalAnalyzed');
        
        if (countElement) {
            const currentValue = parseInt(countElement.textContent) || 0;
            const newValue = currentValue + 1;
            
            this.animateCounter(countElement, currentValue, newValue);
        }
        
        if (totalElement) {
            const currentTotal = parseInt(totalElement.textContent) || 0;
            const newTotal = currentTotal + 1;
            
            this.animateCounter(totalElement, currentTotal, newTotal);
        }
    }
    
    animateCounter(element, from, to) {
        const counter = { value: from };
        
        anime({
            targets: counter,
            value: to,
            duration: this.normalDuration,
            easing: this.defaultEasing,
            round: 1,
            update: () => {
                element.textContent = Math.round(counter.value);
            }
        });
        
        // Pulse animation
        anime({
            targets: element.parentElement,
            scale: [1, 1.05, 1],
            duration: this.fastDuration,
            easing: 'easeOutQuad'
        });
    }
    
    // Button animations
    animateButtonPress(button) {
        anime({
            targets: button,
            scale: [1, 0.95, 1],
            duration: 150,
            easing: 'easeOutQuad'
        });
    }
    
    animateButtonHover(button, isHover) {
        anime({
            targets: button,
            scale: isHover ? 1.05 : 1,
            duration: this.fastDuration,
            easing: this.defaultEasing
        });
    }
    
    // Notification animations
    showNotification(message, type = 'info', duration = 3000) {
        const container = document.getElementById('notifications');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Initial state
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        
        container.appendChild(notification);
        
        // Slide in animation
        anime({
            targets: notification,
            opacity: 1,
            translateX: 0,
            duration: this.fastDuration,
            easing: 'easeOutCubic'
        });
        
        // Auto remove
        setTimeout(() => {
            this.hideNotification(notification);
        }, duration);
        
        return notification;
    }
    
    hideNotification(notification) {
        anime({
            targets: notification,
            opacity: 0,
            translateX: '100%',
            duration: this.fastDuration,
            easing: 'easeInCubic',
            complete: () => {
                notification.remove();
            }
        });
    }
    
    // Modal animations
    showModal(modal) {
        modal.style.display = 'flex';
        const content = modal.querySelector('.modal-content');
        
        // Background fade in
        anime({
            targets: modal,
            opacity: [0, 1],
            duration: this.fastDuration,
            easing: this.defaultEasing
        });
        
        // Content scale in
        if (content) {
            anime({
                targets: content,
                scale: [0.8, 1],
                opacity: [0, 1],
                duration: this.fastDuration,
                easing: 'easeOutBack'
            });
        }
    }
    
    hideModal(modal) {
        const content = modal.querySelector('.modal-content');
        
        // Content scale out
        if (content) {
            anime({
                targets: content,
                scale: 0.8,
                opacity: 0,
                duration: this.fastDuration,
                easing: 'easeInBack'
            });
        }
        
        // Background fade out
        anime({
            targets: modal,
            opacity: 0,
            duration: this.fastDuration,
            easing: this.defaultEasing,
            complete: () => {
                modal.style.display = 'none';
            }
        });
    }
    
    // History animations
    addHistoryItem(item) {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;
        
        // Remove empty state
        const emptyState = historyList.querySelector('.empty-history');
        if (emptyState) {
            anime({
                targets: emptyState,
                opacity: 0,
                scale: 0.8,
                duration: this.fastDuration,
                complete: () => emptyState.remove()
            });
        }
        
        // Add new item
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-text">${item.text}</div>
            <div class="history-meta">
                <span class="history-sentiment ${item.sentiment}">${item.sentiment}</span>
                <span class="history-time">${item.time}</span>
            </div>
        `;
        
        // Initial state
        historyItem.style.opacity = '0';
        historyItem.style.transform = 'translateY(-20px)';
        
        historyList.insertBefore(historyItem, historyList.firstChild);
        
        // Animate in
        anime({
            targets: historyItem,
            opacity: 1,
            translateY: 0,
            duration: this.fastDuration,
            easing: this.defaultEasing
        });
        
        // Limit history items
        const items = historyList.querySelectorAll('.history-item');
        if (items.length > 10) {
            const oldItem = items[items.length - 1];
            anime({
                targets: oldItem,
                opacity: 0,
                translateY: 20,
                duration: this.fastDuration,
                complete: () => oldItem.remove()
            });
        }
    }
    
    // Performance metrics animations
    updateMetric(metricId, newValue) {
        const element = document.getElementById(metricId);
        if (!element) return;
        
        element.textContent = newValue;
        
        // Highlight animation
        anime({
            targets: element,
            color: ['#00f5ff', '#ffffff'],
            duration: this.slowDuration,
            easing: this.defaultEasing
        });
    }
    
    // Character count animation
    animateCharCount(count, maxCount) {
        const charCount = document.getElementById('charCount');
        if (!charCount) return;
        
        const percentage = count / maxCount;
        let color = '#6c757d'; // default
        
        if (percentage > 0.8) color = '#fa709a'; // warning
        else if (percentage > 0.6) color = '#fee140'; // caution
        
        anime({
            targets: charCount,
            color: color,
            duration: this.fastDuration,
            easing: this.defaultEasing
        });
        
        if (percentage > 0.9) {
            anime({
                targets: charCount.parentElement,
                scale: [1, 1.1, 1],
                duration: this.fastDuration,
                easing: 'easeOutElastic(1, .8)'
            });
        }
    }
    
    // Utility methods
    stopAnimation(animationId) {
        const animation = this.activeAnimations.get(animationId);
        if (animation) {
            animation.pause();
            this.activeAnimations.delete(animationId);
        }
    }
    
    stopAllAnimations() {
        this.activeAnimations.forEach(animation => {
            animation.pause();
        });
        this.activeAnimations.clear();
    }
}

// Global instance
const animationController = new AnimationController();

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        animationController.init();
        
        // Expose to global scope
        window.animations = animationController;
        
        console.log('Animation system ready');
    }, 50);
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationController;
}
