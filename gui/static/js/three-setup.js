/**
 * Three.js 3D Visualization Setup
 * Creates interactive 3D sentiment visualization with particles and effects
 */

class SentimentVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, this.canvas.clientWidth / this.canvas.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas, 
            antialias: true, 
            alpha: true 
        });
        
        // Configuration
        this.config = {
            particleCount: 1000,
            animationSpeed: 1.0,
            autoRotate: true,
            particleAnimation: true,
            wireframe: false
        };
        
        // Visualization objects
        this.particles = null;
        this.particleGeometry = null;
        this.particleMaterial = null;
        this.sentimentSphere = null;
        this.sentimentLight = null;
        this.controls = null;
        
        // Animation properties
        this.time = 0;
        this.currentSentiment = 'neutral';
        this.targetSentimentColor = new THREE.Color(0x00f5ff);
        this.currentSentimentColor = new THREE.Color(0x00f5ff);
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.setupCamera();
        this.setupLights();
        this.setupControls();
        this.createParticles();
        this.createSentimentSphere();
        this.setupEventListeners();
        this.animate();
    }
    
    setupRenderer() {
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }
    
    setupCamera() {
        this.camera.position.set(0, 0, 5);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Sentiment-reactive light
        this.sentimentLight = new THREE.PointLight(0x00f5ff, 1, 10);
        this.sentimentLight.position.set(0, 0, 3);
        this.scene.add(this.sentimentLight);
    }
    
    setupControls() {
        // Simple mouse controls for rotation
        this.mouseX = 0;
        this.mouseY = 0;
        this.targetRotationX = 0;
        this.targetRotationY = 0;
        
        this.canvas.addEventListener('mousemove', (event) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = (event.clientX - rect.left) / rect.width * 2 - 1;
            this.mouseY = -(event.clientY - rect.top) / rect.height * 2 + 1;
            
            if (!this.config.autoRotate) {
                this.targetRotationX = this.mouseY * 0.5;
                this.targetRotationY = this.mouseX * 0.5;
            }
        });
        
        // Touch support for mobile
        this.canvas.addEventListener('touchmove', (event) => {
            event.preventDefault();
            const touch = event.touches[0];
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = (touch.clientX - rect.left) / rect.width * 2 - 1;
            this.mouseY = -(touch.clientY - rect.top) / rect.height * 2 + 1;
        });
    }
    
    createParticles() {
        this.particleGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.config.particleCount * 3);
        const colors = new Float32Array(this.config.particleCount * 3);
        const sizes = new Float32Array(this.config.particleCount);
        
        // Create particle positions in a sphere
        for (let i = 0; i < this.config.particleCount; i++) {
            const i3 = i * 3;
            
            // Spherical distribution
            const radius = Math.random() * 4 + 1;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            
            positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = radius * Math.cos(phi);
            
            // Initial colors (cyan)
            colors[i3] = 0.0;     // R
            colors[i3 + 1] = 0.96; // G
            colors[i3 + 2] = 1.0;  // B
            
            // Random sizes
            sizes[i] = Math.random() * 2 + 1;
        }
        
        this.particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        this.particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Particle material with custom shader
        this.particleMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                pixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
            },
            vertexShader: `
                attribute float size;
                varying vec3 vColor;
                uniform float time;
                
                void main() {
                    vColor = color;
                    
                    vec3 pos = position;
                    
                    // Add some movement
                    pos.x += sin(time * 0.5 + position.y * 0.01) * 0.1;
                    pos.y += cos(time * 0.3 + position.z * 0.01) * 0.1;
                    pos.z += sin(time * 0.7 + position.x * 0.01) * 0.1;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                    gl_PointSize = size * (300.0 / -mvPosition.z);
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                
                void main() {
                    float distanceToCenter = distance(gl_PointCoord, vec2(0.5));
                    float alpha = 1.0 - smoothstep(0.0, 0.5, distanceToCenter);
                    
                    gl_FragColor = vec4(vColor, alpha * 0.8);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            vertexColors: true
        });
        
        this.particles = new THREE.Points(this.particleGeometry, this.particleMaterial);
        this.scene.add(this.particles);
    }
    
    createSentimentSphere() {
        const geometry = new THREE.SphereGeometry(0.5, 32, 32);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00f5ff,
            transparent: true,
            opacity: 0.3,
            wireframe: this.config.wireframe
        });
        
        this.sentimentSphere = new THREE.Mesh(geometry, material);
        this.sentimentSphere.position.set(0, 0, 0);
        this.scene.add(this.sentimentSphere);
        
        // Add glow effect
        const glowGeometry = new THREE.SphereGeometry(0.7, 32, 32);
        const glowMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(0x00f5ff) }
            },
            vertexShader: `
                varying vec3 vNormal;
                varying vec3 vPosition;
                
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                varying vec3 vNormal;
                varying vec3 vPosition;
                
                void main() {
                    float intensity = pow(0.4 - dot(vNormal, vec3(0, 0, 1.0)), 2.0);
                    intensity += sin(time * 2.0) * 0.1;
                    gl_FragColor = vec4(color, intensity * 0.6);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide
        });
        
        const glowSphere = new THREE.Mesh(glowGeometry, glowMaterial);
        this.sentimentSphere.add(glowSphere);
        this.glowMaterial = glowMaterial;
    }
    
    setupEventListeners() {
        // Resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Fullscreen handler
        document.addEventListener('fullscreenchange', () => {
            setTimeout(() => this.handleResize(), 100);
        });
    }
    
    handleResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    updateSentiment(sentiment, confidence = 0.5) {
        this.currentSentiment = sentiment;
        
        // Color mapping for different sentiments
        const sentimentColors = {
            positive: 0x4facfe,   // Blue
            negative: 0xfa709a,   // Pink
            neutral: 0x43e97b     // Green
        };
        
        const targetColor = sentimentColors[sentiment] || sentimentColors.neutral;
        this.targetSentimentColor.setHex(targetColor);
        
        // Update particle colors based on sentiment
        this.updateParticleColors(sentiment, confidence);
        
        // Update sphere and light colors
        this.animateSentimentChange(confidence);
    }
    
    updateParticleColors(sentiment, confidence) {
        const colors = this.particleGeometry.attributes.color.array;
        const sentimentColors = {
            positive: { r: 0.29, g: 0.67, b: 1.0 },   // Blue
            negative: { r: 0.98, g: 0.44, b: 0.60 },  // Pink
            neutral: { r: 0.26, g: 0.91, b: 0.48 }    // Green
        };
        
        const color = sentimentColors[sentiment] || sentimentColors.neutral;
        
        for (let i = 0; i < this.config.particleCount; i++) {
            const i3 = i * 3;
            
            // Add some randomness based on confidence
            const variation = (1 - confidence) * 0.3;
            colors[i3] = color.r + (Math.random() - 0.5) * variation;
            colors[i3 + 1] = color.g + (Math.random() - 0.5) * variation;
            colors[i3 + 2] = color.b + (Math.random() - 0.5) * variation;
        }
        
        this.particleGeometry.attributes.color.needsUpdate = true;
    }
    
    animateSentimentChange(confidence) {
        // Animate sphere pulsing based on confidence
        const targetScale = 1 + confidence * 0.5;
        
        anime({
            targets: this.sentimentSphere.scale,
            x: targetScale,
            y: targetScale,
            z: targetScale,
            duration: 1000,
            easing: 'easeOutElastic(1, .8)'
        });
        
        // Animate color transition
        anime({
            targets: this.currentSentimentColor,
            r: this.targetSentimentColor.r,
            g: this.targetSentimentColor.g,
            b: this.targetSentimentColor.b,
            duration: 2000,
            easing: 'easeInOutCubic',
            update: () => {
                this.sentimentSphere.material.color.copy(this.currentSentimentColor);
                this.sentimentLight.color.copy(this.currentSentimentColor);
                this.glowMaterial.uniforms.color.value.copy(this.currentSentimentColor);
            }
        });
    }
    
    // Control methods
    resetCamera() {
        anime({
            targets: this.camera.position,
            x: 0,
            y: 0,
            z: 5,
            duration: 1000,
            easing: 'easeInOutCubic'
        });
        
        anime({
            targets: this.camera.rotation,
            x: 0,
            y: 0,
            z: 0,
            duration: 1000,
            easing: 'easeInOutCubic'
        });
    }
    
    toggleParticles() {
        this.particles.visible = !this.particles.visible;
    }
    
    toggleWireframe() {
        this.config.wireframe = !this.config.wireframe;
        this.sentimentSphere.material.wireframe = this.config.wireframe;
    }
    
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        // Rebuild particles if count changed
        if (newConfig.particleCount && newConfig.particleCount !== this.config.particleCount) {
            this.scene.remove(this.particles);
            this.createParticles();
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.time += 0.016 * this.config.animationSpeed; // ~60fps
        
        // Update shader uniforms
        if (this.particleMaterial) {
            this.particleMaterial.uniforms.time.value = this.time;
        }
        
        if (this.glowMaterial) {
            this.glowMaterial.uniforms.time.value = this.time;
        }
        
        // Auto rotation
        if (this.config.autoRotate && this.particles) {
            this.particles.rotation.y += 0.005 * this.config.animationSpeed;
            this.sentimentSphere.rotation.y += 0.01 * this.config.animationSpeed;
        } else {
            // Manual rotation based on mouse
            this.particles.rotation.x += (this.targetRotationX - this.particles.rotation.x) * 0.05;
            this.particles.rotation.y += (this.targetRotationY - this.particles.rotation.y) * 0.05;
        }
        
        // Particle animation
        if (this.config.particleAnimation && this.particleGeometry) {
            const positions = this.particleGeometry.attributes.position.array;
            
            for (let i = 0; i < this.config.particleCount; i++) {
                const i3 = i * 3;
                
                // Gentle floating motion
                positions[i3 + 1] += Math.sin(this.time + i * 0.01) * 0.002;
            }
            
            this.particleGeometry.attributes.position.needsUpdate = true;
        }
        
        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }
    
    dispose() {
        // Clean up resources
        if (this.particleGeometry) this.particleGeometry.dispose();
        if (this.particleMaterial) this.particleMaterial.dispose();
        if (this.sentimentSphere) {
            this.sentimentSphere.geometry.dispose();
            this.sentimentSphere.material.dispose();
        }
        this.renderer.dispose();
    }
}

// Global instance
let visualization = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for CSS to load
    setTimeout(() => {
        visualization = new SentimentVisualization('threeCanvas');
        
        // Expose to global scope for other scripts
        window.sentimentVisualization = visualization;
        
        console.log('3D Sentiment Visualization initialized');
    }, 100);
});
