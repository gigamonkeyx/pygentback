// Elite Visual Effects - High-quality WebGL and Canvas 2D effects system
class EliteVisualEffects {
    constructor(canvas, ctx) {
        this.canvas = canvas;
        this.ctx = ctx;
        
        // Effect storage
        this.effects = [];
        this.particles = [];
        this.screenShakes = [];
        
        // WebGL context for advanced effects
        this.initWebGL();
        
        // Shader programs
        this.shaders = {};
        this.initShaders();
        
        // Particle systems
        this.particleSystems = new Map();
        
        // Screen effects
        this.screenEffects = {
            shake: { x: 0, y: 0, intensity: 0, duration: 0 },
            flash: { intensity: 0, color: '#ffffff', duration: 0 },
            distortion: { intensity: 0, duration: 0 }
        };
        
        // Performance settings
        this.maxParticles = 1000;
        this.maxEffects = 50;
        
        console.log('Elite Visual Effects system initialized');
    }
    
    initWebGL() {
        try {
            // Create a separate WebGL canvas for effects
            this.webglCanvas = document.createElement('canvas');
            this.webglCanvas.width = this.canvas.width;
            this.webglCanvas.height = this.canvas.height;
            this.webglCanvas.style.position = 'absolute';
            this.webglCanvas.style.top = '0';
            this.webglCanvas.style.left = '0';
            this.webglCanvas.style.pointerEvents = 'none';
            this.webglCanvas.style.zIndex = '10';
            
            this.gl = this.webglCanvas.getContext('webgl2') || this.webglCanvas.getContext('webgl');
            
            if (this.gl) {
                this.canvas.parentNode.appendChild(this.webglCanvas);
                this.setupWebGL();
                console.log('WebGL effects enabled');
            } else {
                console.log('WebGL not available, using Canvas 2D effects only');
            }
        } catch (error) {
            console.log('WebGL initialization failed:', error);
            this.gl = null;
        }
    }
    
    setupWebGL() {
        if (!this.gl) return;
        
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        this.gl.viewport(0, 0, this.webglCanvas.width, this.webglCanvas.height);
        
        // Create basic quad geometry for particle rendering
        this.quadBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        
        const quadVertices = new Float32Array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1
        ]);
        
        this.gl.bufferData(this.gl.ARRAY_BUFFER, quadVertices, this.gl.STATIC_DRAW);
    }
    
    initShaders() {
        if (!this.gl) return;
        
        // Particle shader
        this.shaders.particle = this.createShaderProgram(
            // Vertex shader
            `#version 300 es
            in vec2 a_position;
            in vec2 a_texCoord;
            uniform mat4 u_matrix;
            uniform vec2 u_particlePos;
            uniform float u_size;
            uniform float u_rotation;
            out vec2 v_texCoord;
            
            void main() {
                vec2 rotated = vec2(
                    cos(u_rotation) * a_position.x - sin(u_rotation) * a_position.y,
                    sin(u_rotation) * a_position.x + cos(u_rotation) * a_position.y
                );
                vec2 scaled = rotated * u_size;
                vec2 position = u_particlePos + scaled;
                gl_Position = u_matrix * vec4(position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }`,
            
            // Fragment shader
            `#version 300 es
            precision highp float;
            in vec2 v_texCoord;
            uniform vec4 u_color;
            uniform float u_time;
            out vec4 outColor;
            
            void main() {
                vec2 center = v_texCoord - 0.5;
                float dist = length(center);
                
                // Create particle shape with soft edges
                float alpha = smoothstep(0.5, 0.3, dist);
                
                // Add some sparkle effect
                float sparkle = sin(u_time * 10.0 + dist * 20.0) * 0.1 + 0.9;
                
                outColor = vec4(u_color.rgb * sparkle, u_color.a * alpha);
            }`
        );
        
        // Energy field shader
        this.shaders.energyField = this.createShaderProgram(
            `#version 300 es
            in vec2 a_position;
            in vec2 a_texCoord;
            uniform mat4 u_matrix;
            out vec2 v_texCoord;
            
            void main() {
                gl_Position = u_matrix * vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }`,
            
            `#version 300 es
            precision highp float;
            in vec2 v_texCoord;
            uniform float u_time;
            uniform vec4 u_color;
            uniform vec2 u_center;
            uniform float u_intensity;
            out vec4 outColor;
            
            float noise(vec2 p) {
                return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
            }
            
            void main() {
                vec2 uv = v_texCoord;
                vec2 center = u_center;
                
                // Create energy field pattern
                float dist = distance(uv, center);
                float wave1 = sin(dist * 20.0 - u_time * 3.0) * 0.5 + 0.5;
                float wave2 = sin(dist * 15.0 + u_time * 2.0) * 0.5 + 0.5;
                
                // Add noise for organic feel
                float n = noise(uv * 10.0 + u_time * 0.1);
                
                float energy = (wave1 * wave2 + n * 0.3) * u_intensity;
                energy *= smoothstep(1.0, 0.0, dist * 2.0);
                
                outColor = vec4(u_color.rgb, u_color.a * energy);
            }`
        );
    }
    
    createShaderProgram(vertexSource, fragmentSource) {
        if (!this.gl) return null;
        
        const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentSource);
        
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Shader program linking failed:', this.gl.getProgramInfoLog(program));
            return null;
        }
        
        return program;
    }
    
    compileShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compilation failed:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    // Effect creation methods
    addFoodConsumptionEffect(x, y, color) {
        // Burst of particles
        for (let i = 0; i < 15; i++) {
            const angle = (Math.PI * 2 * i) / 15;
            const speed = 50 + Math.random() * 100;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                size: 3 + Math.random() * 4,
                color: color,
                alpha: 1,
                life: 1000 + Math.random() * 500,
                maxLife: 1000 + Math.random() * 500,
                type: 'burst'
            });
        }
        
        // Central glow effect
        this.effects.push({
            type: 'glow',
            x: x,
            y: y,
            size: 0,
            maxSize: 30,
            color: color,
            alpha: 0.8,
            life: 800,
            maxLife: 800,
            growth: true
        });
    }
    
    addPowerupCollectionEffect(x, y, powerupType) {
        const colors = {
            speed: '#00ffff',
            strength: '#ff4444',
            health: '#44ff44',
            energy: '#ffff44',
            skill: '#ff44ff'
        };
        
        const color = colors[powerupType] || '#ffffff';
        
        // Spiral particles
        for (let i = 0; i < 20; i++) {
            const angle = (Math.PI * 2 * i) / 20;
            const radius = 5 + Math.random() * 10;
            
            this.particles.push({
                x: x + Math.cos(angle) * radius,
                y: y + Math.sin(angle) * radius,
                vx: Math.cos(angle) * 30,
                vy: Math.sin(angle) * 30 - 50, // Upward drift
                size: 2 + Math.random() * 3,
                color: color,
                alpha: 1,
                life: 1500,
                maxLife: 1500,
                type: 'spiral',
                angularVelocity: 0.1
            });
        }
        
        // Screen flash
        this.addScreenFlash(color, 0.3, 200);
    }
    
    addDamageEffect(x, y, damage) {
        // Damage number
        this.effects.push({
            type: 'damageText',
            x: x,
            y: y,
            text: `-${damage}`,
            color: '#ff4444',
            size: Math.min(24, 12 + damage * 0.5),
            alpha: 1,
            vy: -50,
            life: 2000,
            maxLife: 2000
        });
        
        // Impact particles
        for (let i = 0; i < Math.min(damage, 10); i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 20 + Math.random() * 80;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                size: 1 + Math.random() * 2,
                color: '#ff6666',
                alpha: 1,
                life: 800,
                maxLife: 800,
                type: 'impact'
            });
        }
    }
    
    addHeadCollisionEffect(x, y) {
        // Intense explosion effect
        for (let i = 0; i < 30; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 100 + Math.random() * 200;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                size: 2 + Math.random() * 5,
                color: ['#ffff00', '#ff8800', '#ff4400'][Math.floor(Math.random() * 3)],
                alpha: 1,
                life: 1200,
                maxLife: 1200,
                type: 'explosion'
            });
        }
        
        // Shockwave
        this.effects.push({
            type: 'shockwave',
            x: x,
            y: y,
            size: 0,
            maxSize: 100,
            color: '#ffffff',
            alpha: 0.8,
            life: 600,
            maxLife: 600,
            lineWidth: 3
        });
        
        // Screen shake
        this.addScreenShake(15);
    }
    
    addWallCollisionEffect(x, y) {
        // Spark particles
        for (let i = 0; i < 12; i++) {
            const angle = Math.random() * Math.PI;
            const speed = 50 + Math.random() * 100;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - 30,
                size: 1 + Math.random() * 2,
                color: '#ffaa00',
                alpha: 1,
                life: 600,
                maxLife: 600,
                type: 'spark'
            });
        }
    }
    
    addVictoryEffect(x, y, color) {
        // Victory celebration
        for (let i = 0; i < 50; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 50 + Math.random() * 150;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - 100,
                size: 3 + Math.random() * 6,
                color: color,
                alpha: 1,
                life: 3000,
                maxLife: 3000,
                type: 'celebration',
                gravity: 20
            });
        }
        
        // Continuous fireworks effect
        let fireworkCount = 0;
        const fireworkInterval = setInterval(() => {
            if (fireworkCount >= 10) {
                clearInterval(fireworkInterval);
                return;
            }
            
            const fx = x + (Math.random() - 0.5) * 200;
            const fy = y + (Math.random() - 0.5) * 200;
            
            for (let i = 0; i < 20; i++) {
                const angle = (Math.PI * 2 * i) / 20;
                const speed = 30 + Math.random() * 70;
                
                this.particles.push({
                    x: fx,
                    y: fy,
                    vx: Math.cos(angle) * speed,
                    vy: Math.sin(angle) * speed,
                    size: 2 + Math.random() * 4,
                    color: ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'][Math.floor(Math.random() * 5)],
                    alpha: 1,
                    life: 2000,
                    maxLife: 2000,
                    type: 'firework'
                });
            }
            
            fireworkCount++;
        }, 300);
    }
    
    addScreenShake(intensity) {
        this.screenEffects.shake = {
            intensity: intensity,
            duration: 300,
            x: 0,
            y: 0
        };
    }
    
    addScreenFlash(color, intensity, duration) {
        this.screenEffects.flash = {
            color: color,
            intensity: intensity,
            duration: duration,
            maxDuration: duration
        };
    }
    
    addIronScalesEffect(snake) {
        // Metallic armor effect
        const head = snake.segments[0];
        const x = head.x * 8 + 4; // Assuming gridSize of 8
        const y = head.y * 8 + 4;
        
        // Create metallic particles around the snake
        for (let i = 0; i < 5; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = 10 + Math.random() * 15;
            
            this.particles.push({
                x: x + Math.cos(angle) * radius,
                y: y + Math.sin(angle) * radius,
                vx: Math.cos(angle) * 10,
                vy: Math.sin(angle) * 10,
                size: 1 + Math.random() * 2,
                color: '#c0c0c0',
                alpha: 0.8,
                life: 1000,
                maxLife: 1000,
                type: 'metallic'
            });
        }
        
        if (this.gl && this.shaders.energyField) {
            // Add WebGL energy field effect
            this.addEnergyFieldEffect(x, y, '#c0c0c0', 0.3, 1000);
        }
    }
    
    addEnergyFieldEffect(x, y, color, intensity, duration) {
        if (!this.gl || !this.shaders.energyField) return;
        
        this.effects.push({
            type: 'webgl_energy',
            x: x,
            y: y,
            color: color,
            intensity: intensity,
            life: duration,
            maxLife: duration,
            shader: this.shaders.energyField
        });
    }
    
    addLightningEffect(startX, startY, endX, endY, color = '#00ffff') {
        // Create lightning bolt effect
        const segments = [];
        const numSegments = 8;
        
        for (let i = 0; i <= numSegments; i++) {
            const t = i / numSegments;
            const x = startX + (endX - startX) * t;
            const y = startY + (endY - startY) * t;
            
            // Add random jitter
            const jitterX = (Math.random() - 0.5) * 20;
            const jitterY = (Math.random() - 0.5) * 20;
            
            segments.push({
                x: x + jitterX,
                y: y + jitterY
            });
        }
        
        this.effects.push({
            type: 'lightning',
            segments: segments,
            color: color,
            alpha: 1,
            life: 400,
            maxLife: 400,
            lineWidth: 2 + Math.random() * 3
        });
        
        // Add glow particles along the bolt
        segments.forEach(segment => {
            this.particles.push({
                x: segment.x,
                y: segment.y,
                vx: (Math.random() - 0.5) * 20,
                vy: (Math.random() - 0.5) * 20,
                size: 1 + Math.random() * 2,
                color: color,
                alpha: 0.8,
                life: 600,
                maxLife: 600,
                type: 'electric'
            });
        });
    }
    
    // Update and render methods
    update(deltaTime) {
        this.updateEffects(deltaTime);
        this.updateParticles(deltaTime);
        this.updateScreenEffects(deltaTime);
        
        // Clean up old effects and particles
        this.cleanupEffects();
    }
    
    updateEffects(deltaTime) {
        this.effects.forEach(effect => {
            effect.life -= deltaTime;
            
            switch (effect.type) {
                case 'glow':
                    if (effect.growth && effect.size < effect.maxSize) {
                        effect.size += deltaTime * 0.1;
                    }
                    effect.alpha = effect.life / effect.maxLife;
                    break;
                    
                case 'shockwave':
                    effect.size = effect.maxSize * (1 - effect.life / effect.maxLife);
                    effect.alpha = effect.life / effect.maxLife;
                    break;
                    
                case 'damageText':
                    effect.y += effect.vy * deltaTime * 0.001;
                    effect.alpha = effect.life / effect.maxLife;
                    break;
                    
                case 'lightning':
                    effect.alpha = Math.max(0, effect.life / effect.maxLife);
                    break;
                    
                case 'webgl_energy':
                    effect.intensity = (effect.life / effect.maxLife) * 0.3;
                    break;
            }
        });
    }
    
    updateParticles(deltaTime) {
        this.particles.forEach(particle => {
            particle.life -= deltaTime;
            particle.alpha = Math.max(0, particle.life / particle.maxLife);
            
            // Update position
            particle.x += particle.vx * deltaTime * 0.001;
            particle.y += particle.vy * deltaTime * 0.001;
            
            // Apply gravity for certain particle types
            if (particle.gravity) {
                particle.vy += particle.gravity * deltaTime * 0.001;
            }
            
            // Type-specific updates
            switch (particle.type) {
                case 'spiral':
                    if (particle.angularVelocity) {
                        const angle = Date.now() * particle.angularVelocity * 0.001;
                        particle.x += Math.cos(angle) * 0.5;
                        particle.y += Math.sin(angle) * 0.5;
                    }
                    break;
                    
                case 'electric':
                    // Electric particles flicker
                    particle.alpha *= 0.5 + Math.random() * 0.5;
                    break;
                    
                case 'metallic':
                    // Metallic particles reflect light
                    particle.alpha = 0.8 + Math.sin(Date.now() * 0.01) * 0.2;
                    break;
            }
            
            // Size changes over time
            if (particle.type === 'explosion') {
                particle.size *= 0.995;
            }
        });
    }
    
    updateScreenEffects(deltaTime) {
        // Screen shake
        if (this.screenEffects.shake.duration > 0) {
            this.screenEffects.shake.duration -= deltaTime;
            const intensity = this.screenEffects.shake.intensity * (this.screenEffects.shake.duration / 300);
            
            this.screenEffects.shake.x = (Math.random() - 0.5) * intensity;
            this.screenEffects.shake.y = (Math.random() - 0.5) * intensity;
        } else {
            this.screenEffects.shake.x = 0;
            this.screenEffects.shake.y = 0;
        }
        
        // Screen flash
        if (this.screenEffects.flash.duration > 0) {
            this.screenEffects.flash.duration -= deltaTime;
            this.screenEffects.flash.intensity = 
                (this.screenEffects.flash.duration / this.screenEffects.flash.maxDuration) * 0.3;
        }
    }
    
    cleanupEffects() {
        this.effects = this.effects.filter(effect => effect.life > 0);
        this.particles = this.particles.filter(particle => particle.life > 0);
        
        // Limit particle count for performance
        if (this.particles.length > this.maxParticles) {
            this.particles.splice(0, this.particles.length - this.maxParticles);
        }
        
        if (this.effects.length > this.maxEffects) {
            this.effects.splice(0, this.effects.length - this.maxEffects);
        }
    }
    
    render(ctx) {
        // Apply screen shake
        ctx.save();
        ctx.translate(this.screenEffects.shake.x, this.screenEffects.shake.y);
        
        // Render Canvas 2D effects
        this.renderCanvas2DEffects(ctx);
        this.renderCanvas2DParticles(ctx);
        
        // Render WebGL effects
        if (this.gl) {
            this.renderWebGLEffects();
        }
        
        ctx.restore();
        
        // Render screen flash (not affected by shake)
        this.renderScreenFlash(ctx);
    }
    
    renderCanvas2DEffects(ctx) {
        this.effects.forEach(effect => {
            if (effect.type === 'webgl_energy') return; // Skip WebGL effects
            
            ctx.save();
            ctx.globalAlpha = effect.alpha;
            
            switch (effect.type) {
                case 'glow':
                    ctx.fillStyle = effect.color;
                    ctx.shadowColor = effect.color;
                    ctx.shadowBlur = effect.size;
                    ctx.beginPath();
                    ctx.arc(effect.x, effect.y, effect.size * 0.3, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                    break;
                    
                case 'shockwave':
                    ctx.strokeStyle = effect.color;
                    ctx.lineWidth = effect.lineWidth;
                    ctx.shadowColor = effect.color;
                    ctx.shadowBlur = 10;
                    ctx.beginPath();
                    ctx.arc(effect.x, effect.y, effect.size, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                    break;
                    
                case 'damageText':
                    ctx.fillStyle = effect.color;
                    ctx.font = `bold ${effect.size}px Arial`;
                    ctx.textAlign = 'center';
                    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                    ctx.shadowBlur = 2;
                    ctx.fillText(effect.text, effect.x, effect.y);
                    ctx.shadowBlur = 0;
                    break;
                    
                case 'lightning':
                    ctx.strokeStyle = effect.color;
                    ctx.lineWidth = effect.lineWidth;
                    ctx.shadowColor = effect.color;
                    ctx.shadowBlur = 15;
                    ctx.lineCap = 'round';
                    
                    ctx.beginPath();
                    effect.segments.forEach((segment, index) => {
                        if (index === 0) {
                            ctx.moveTo(segment.x, segment.y);
                        } else {
                            ctx.lineTo(segment.x, segment.y);
                        }
                    });
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                    break;
            }
            
            ctx.restore();
        });
    }
    
    renderCanvas2DParticles(ctx) {
        this.particles.forEach(particle => {
            ctx.save();
            ctx.globalAlpha = particle.alpha;
            ctx.fillStyle = particle.color;
            
            // Add glow for certain particle types
            if (['electric', 'metallic', 'celebration'].includes(particle.type)) {
                ctx.shadowColor = particle.color;
                ctx.shadowBlur = particle.size * 2;
            }
            
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.shadowBlur = 0;
            ctx.restore();
        });
    }
    
    renderWebGLEffects() {
        if (!this.gl) return;
        
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        
        // Render energy field effects
        const energyEffects = this.effects.filter(effect => effect.type === 'webgl_energy');
        
        energyEffects.forEach(effect => {
            if (!effect.shader) return;
            
            this.gl.useProgram(effect.shader);
            
            // Set uniforms
            const timeLocation = this.gl.getUniformLocation(effect.shader, 'u_time');
            const colorLocation = this.gl.getUniformLocation(effect.shader, 'u_color');
            const centerLocation = this.gl.getUniformLocation(effect.shader, 'u_center');
            const intensityLocation = this.gl.getUniformLocation(effect.shader, 'u_intensity');
            
            if (timeLocation) this.gl.uniform1f(timeLocation, Date.now() * 0.001);
            if (colorLocation) {
                const color = this.hexToRgb(effect.color);
                this.gl.uniform4f(colorLocation, color.r, color.g, color.b, effect.intensity);
            }
            if (centerLocation) {
                this.gl.uniform2f(centerLocation, effect.x / this.canvas.width, effect.y / this.canvas.height);
            }
            if (intensityLocation) this.gl.uniform1f(intensityLocation, effect.intensity);
            
            // Render quad
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
            
            const positionLocation = this.gl.getAttribLocation(effect.shader, 'a_position');
            const texCoordLocation = this.gl.getAttribLocation(effect.shader, 'a_texCoord');
            
            if (positionLocation >= 0) {
                this.gl.enableVertexAttribArray(positionLocation);
                this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 16, 0);
            }
            
            if (texCoordLocation >= 0) {
                this.gl.enableVertexAttribArray(texCoordLocation);
                this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 16, 8);
            }
            
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        });
    }
    
    renderScreenFlash(ctx) {
        if (this.screenEffects.flash.duration <= 0) return;
        
        ctx.save();
        ctx.fillStyle = this.screenEffects.flash.color;
        ctx.globalAlpha = this.screenEffects.flash.intensity;
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.restore();
    }
    
    // Utility methods
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16) / 255,
            g: parseInt(result[2], 16) / 255,
            b: parseInt(result[3], 16) / 255
        } : { r: 1, g: 1, b: 1 };
    }
    
    // Cleanup method
    dispose() {
        if (this.gl) {
            // Clean up WebGL resources
            Object.values(this.shaders).forEach(shader => {
                if (shader) this.gl.deleteProgram(shader);
            });
            
            if (this.quadBuffer) {
                this.gl.deleteBuffer(this.quadBuffer);
            }
            
            if (this.webglCanvas && this.webglCanvas.parentNode) {
                this.webglCanvas.parentNode.removeChild(this.webglCanvas);
            }
        }
        
        this.effects = [];
        this.particles = [];
        this.particleSystems.clear();
    }
}
