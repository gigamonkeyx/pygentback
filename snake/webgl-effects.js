// WebGL Effects System - GPU-accelerated visual effects for Snake Battle Arena
class WebGLEffects {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.programs = {};
        this.buffers = {};
        this.textures = {};
        this.particles = [];
        this.maxParticles = 2000;
        this.framebuffer = null;
        this.backgroundTexture = null;
        
        // Initialize WebGL with error handling
        if (!this.initWebGL()) {
            console.warn('WebGL initialization failed - falling back to 2D mode');
            return;
        }
        
        if (this.gl) {
            try {
                this.initShaders();
                this.initBuffers();
                this.initFramebuffer();
                console.log('WebGL effects fully initialized');
            } catch (error) {
                console.error('WebGL setup error:', error);
                this.gl = null;
            }
        }
    }    initWebGL() {
        try {
            this.gl = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl');
            if (!this.gl) {
                console.warn('WebGL not supported, falling back to 2D canvas');
                return false;
            }
            
            // Enable blending for transparency
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
            
            console.log('WebGL context created successfully');
            return true;
        } catch (e) {
            console.warn('WebGL initialization failed:', e);
            this.gl = null;
            return false;
        }
    }

    initShaders() {
        // Iron Scales armor glow shader
        const armorVertexShader = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            uniform vec2 u_resolution;
            uniform float u_time;
            varying vec2 v_texCoord;
            varying float v_glow;
            
            void main() {
                vec2 position = ((a_position / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
                gl_Position = vec4(position, 0, 1);
                v_texCoord = a_texCoord;
                v_glow = sin(u_time * 3.0) * 0.3 + 0.7;
            }
        `;

        const armorFragmentShader = `
            precision mediump float;
            varying vec2 v_texCoord;
            varying float v_glow;
            uniform float u_time;
            uniform vec3 u_color;
            uniform float u_intensity;
            
            void main() {
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(v_texCoord, center);
                
                // Create metallic armor pattern
                float pattern = sin(v_texCoord.x * 20.0 + u_time) * sin(v_texCoord.y * 20.0 + u_time * 0.7);
                float armor = smoothstep(0.3, 0.7, pattern + 0.5);
                
                // Add glow effect
                float glow = 1.0 - smoothstep(0.0, 0.8, dist);
                glow *= v_glow * u_intensity;
                
                vec3 finalColor = u_color * armor + vec3(0.8, 0.9, 1.0) * glow;
                float alpha = armor * 0.6 + glow * 0.8;
                
                gl_FragColor = vec4(finalColor, alpha);
            }
        `;

        // Particle system shader
        const particleVertexShader = `
            attribute vec2 a_position;
            attribute vec2 a_velocity;
            attribute float a_life;
            attribute float a_size;
            attribute vec3 a_color;
            
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform float u_deltaTime;
            
            varying vec3 v_color;
            varying float v_alpha;
            
            void main() {
                vec2 pos = a_position + a_velocity * u_deltaTime;
                vec2 position = ((pos / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
                
                gl_Position = vec4(position, 0, 1);
                gl_PointSize = a_size * (a_life / 100.0);
                
                v_color = a_color;
                v_alpha = a_life / 100.0;
            }
        `;

        const particleFragmentShader = `
            precision mediump float;
            varying vec3 v_color;
            varying float v_alpha;
            
            void main() {
                vec2 center = gl_PointCoord - vec2(0.5);
                float dist = length(center);
                
                if (dist > 0.5) discard;
                
                float intensity = 1.0 - smoothstep(0.0, 0.5, dist);
                gl_FragColor = vec4(v_color, v_alpha * intensity);
            }
        `;

        // Lightning effect shader
        const lightningVertexShader = `
            attribute vec2 a_position;
            uniform vec2 u_resolution;
            varying vec2 v_position;
            
            void main() {
                vec2 position = ((a_position / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
                gl_Position = vec4(position, 0, 1);
                v_position = a_position;
            }
        `;

        const lightningFragmentShader = `
            precision mediump float;
            varying vec2 v_position;
            uniform float u_time;
            uniform vec2 u_start;
            uniform vec2 u_end;
            uniform float u_intensity;
            
            float noise(vec2 st) {
                return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
            }
            
            void main() {
                vec2 dir = normalize(u_end - u_start);
                vec2 perp = vec2(-dir.y, dir.x);
                
                float t = dot(v_position - u_start, dir) / length(u_end - u_start);
                float d = abs(dot(v_position - u_start, perp));
                
                if (t < 0.0 || t > 1.0) discard;
                
                float branch = noise(vec2(t * 10.0, u_time * 5.0)) - 0.5;
                float thickness = 2.0 + sin(t * 20.0 + u_time * 10.0) * 1.0;
                
                if (d > thickness + abs(branch) * 10.0) discard;
                
                float intensity = (thickness - d) / thickness;
                intensity *= u_intensity;
                
                vec3 color = vec3(0.8, 0.9, 1.0) + vec3(0.2, 0.1, 0.0) * sin(u_time * 15.0);
                gl_FragColor = vec4(color, intensity);
            }
        `;

        // Compile and link shaders
        this.programs.armor = this.createProgram(armorVertexShader, armorFragmentShader);
        this.programs.particles = this.createProgram(particleVertexShader, particleFragmentShader);
        this.programs.lightning = this.createProgram(lightningVertexShader, lightningFragmentShader);
    }

    createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentSource);
        
        if (!vertexShader || !fragmentShader) return null;
        
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(program));
            return null;
        }
        
        return program;
    }

    compileShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }

    initBuffers() {
        // Quad buffer for full-screen effects
        const quadVertices = new Float32Array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1
        ]);
        
        this.buffers.quad = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.quad);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, quadVertices, this.gl.STATIC_DRAW);
        
        // Particle buffers
        this.buffers.particlePosition = this.gl.createBuffer();
        this.buffers.particleVelocity = this.gl.createBuffer();
        this.buffers.particleLife = this.gl.createBuffer();
        this.buffers.particleSize = this.gl.createBuffer();
        this.buffers.particleColor = this.gl.createBuffer();
    }

    initFramebuffer() {
        this.framebuffer = this.gl.createFramebuffer();
        this.backgroundTexture = this.gl.createTexture();
        
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.backgroundTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.backgroundTexture, 0);
    }

    // Create Iron Scales armor effect
    createIronScalesEffect(snake) {
        if (!this.gl || !snake.skills.includes('Iron Scales')) return;
        
        // Add metallic particles around the snake
        for (let i = 0; i < 20; i++) {
            const angle = (i / 20) * Math.PI * 2;
            const radius = 30 + Math.random() * 20;
            const head = snake.body[0];
            
            this.particles.push({
                x: head.x + Math.cos(angle) * radius,
                y: head.y + Math.sin(angle) * radius,
                vx: Math.cos(angle) * 2,
                vy: Math.sin(angle) * 2,
                life: 60,
                maxLife: 60,
                size: 4 + Math.random() * 4,
                color: [0.7, 0.8, 0.9], // Metallic blue-silver
                type: 'armor'
            });
        }
    }

    // Create damage absorption effect
    createDamageAbsorptionEffect(snake) {
        if (!this.gl) return;
        
        const head = snake.body[0];
        
        // Create shockwave effect
        for (let i = 0; i < 30; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 3 + Math.random() * 5;
            
            this.particles.push({
                x: head.x,
                y: head.y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 40,
                maxLife: 40,
                size: 6 + Math.random() * 6,
                color: [1.0, 0.8, 0.2], // Golden shockwave
                type: 'shockwave'
            });
        }
    }

    // Create lightning strike effect for special attacks
    createLightningStrike(from, to) {
        if (!this.gl || !this.programs.lightning) return;
        
        const lightning = {
            start: from,
            end: to,
            intensity: 1.0,
            duration: 30,
            timeLeft: 30
        };
        
        this.particles.push(lightning);
    }

    // Update all particles and effects
    update(deltaTime) {
        if (!this.gl) return;
        
        this.particles = this.particles.filter(particle => {
            if (particle.type === 'lightning') {
                particle.timeLeft--;
                particle.intensity *= 0.95;
                return particle.timeLeft > 0;
            }
            
            particle.x += particle.vx * deltaTime;
            particle.y += particle.vy * deltaTime;
            particle.life--;
            
            // Apply gravity to certain particle types
            if (particle.type === 'shockwave') {
                particle.vy += 0.2;
            }
            
            // Fade based on life
            particle.alpha = particle.life / particle.maxLife;
            
            return particle.life > 0;
        });
    }

    // Render Iron Scales armor glow on snake
    renderArmorGlow(snake, time) {
        if (!this.gl || !this.programs.armor || !snake.skills.includes('Iron Scales')) return;
        
        const program = this.programs.armor;
        this.gl.useProgram(program);
        
        // Set uniforms
        const resolutionLocation = this.gl.getUniformLocation(program, 'u_resolution');
        const timeLocation = this.gl.getUniformLocation(program, 'u_time');
        const colorLocation = this.gl.getUniformLocation(program, 'u_color');
        const intensityLocation = this.gl.getUniformLocation(program, 'u_intensity');
        
        this.gl.uniform2f(resolutionLocation, this.canvas.width, this.canvas.height);
        this.gl.uniform1f(timeLocation, time / 1000.0);
        this.gl.uniform3f(colorLocation, 0.6, 0.7, 0.9); // Steel blue
        this.gl.uniform1f(intensityLocation, snake.lives / snake.maxLives);
        
        // Render armor effect around each snake segment
        snake.body.forEach((segment, index) => {
            this.renderArmorSegment(segment, snake.currentWidth || 20, program);
        });
    }

    renderArmorSegment(segment, width, program) {
        const size = width + 10; // Armor extends beyond snake body
        
        // Create quad around segment
        const vertices = new Float32Array([
            segment.x - size/2, segment.y - size/2, 0, 0,
            segment.x + size/2, segment.y - size/2, 1, 0,
            segment.x - size/2, segment.y + size/2, 0, 1,
            segment.x + size/2, segment.y + size/2, 1, 1
        ]);
        
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.DYNAMIC_DRAW);
        
        const positionLocation = this.gl.getAttribLocation(program, 'a_position');
        const texCoordLocation = this.gl.getAttribLocation(program, 'a_texCoord');
        
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.enableVertexAttribArray(texCoordLocation);
        
        this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 16, 0);
        this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 16, 8);
        
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 4, 0);
        
        this.gl.deleteBuffer(buffer);
    }

    // Render all particles
    renderParticles(time) {
        if (!this.gl || !this.programs.particles || this.particles.length === 0) return;
        
        const program = this.programs.particles;
        this.gl.useProgram(program);
        
        // Prepare particle data
        const positions = [];
        const velocities = [];
        const lives = [];
        const sizes = [];
        const colors = [];
        
        this.particles.forEach(particle => {
            if (particle.type !== 'lightning') {
                positions.push(particle.x, particle.y);
                velocities.push(particle.vx, particle.vy);
                lives.push(particle.life);
                sizes.push(particle.size);
                colors.push(...particle.color);
            }
        });
        
        if (positions.length === 0) return;
        
        // Upload data to GPU
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particlePosition);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.DYNAMIC_DRAW);
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleVelocity);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(velocities), this.gl.DYNAMIC_DRAW);
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleLife);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(lives), this.gl.DYNAMIC_DRAW);
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleSize);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(sizes), this.gl.DYNAMIC_DRAW);
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleColor);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(colors), this.gl.DYNAMIC_DRAW);
        
        // Set uniforms
        const resolutionLocation = this.gl.getUniformLocation(program, 'u_resolution');
        const timeLocation = this.gl.getUniformLocation(program, 'u_time');
        
        this.gl.uniform2f(resolutionLocation, this.canvas.width, this.canvas.height);
        this.gl.uniform1f(timeLocation, time / 1000.0);
        
        // Set vertex attributes
        const positionLocation = this.gl.getAttribLocation(program, 'a_position');
        const velocityLocation = this.gl.getAttribLocation(program, 'a_velocity');
        const lifeLocation = this.gl.getAttribLocation(program, 'a_life');
        const sizeLocation = this.gl.getAttribLocation(program, 'a_size');
        const colorLocation = this.gl.getAttribLocation(program, 'a_color');
        
        // Position
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particlePosition);
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
        
        // Velocity
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleVelocity);
        this.gl.enableVertexAttribArray(velocityLocation);
        this.gl.vertexAttribPointer(velocityLocation, 2, this.gl.FLOAT, false, 0, 0);
        
        // Life
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleLife);
        this.gl.enableVertexAttribArray(lifeLocation);
        this.gl.vertexAttribPointer(lifeLocation, 1, this.gl.FLOAT, false, 0, 0);
        
        // Size
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleSize);
        this.gl.enableVertexAttribArray(sizeLocation);
        this.gl.vertexAttribPointer(sizeLocation, 1, this.gl.FLOAT, false, 0, 0);
        
        // Color
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.particleColor);
        this.gl.enableVertexAttribArray(colorLocation);
        this.gl.vertexAttribPointer(colorLocation, 3, this.gl.FLOAT, false, 0, 0);
        
        // Render particles
        this.gl.drawArrays(this.gl.POINTS, 0, positions.length / 2);
    }

    // Main render function
    render(snakes, time, deltaTime) {
        if (!this.gl) return;
        
        this.update(deltaTime);
        
        // Clear and setup viewport
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0, 0, 0, 0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        
        // Render armor glows for snakes with Iron Scales
        snakes.forEach(snake => {
            if (snake && snake.lives > 0) {
                this.renderArmorGlow(snake, time);
            }
        });
        
        // Render particles
        this.renderParticles(time);
    }

    // Clean up resources
    dispose() {
        if (!this.gl) return;
        
        Object.values(this.programs).forEach(program => {
            if (program) this.gl.deleteProgram(program);
        });
        
        Object.values(this.buffers).forEach(buffer => {
            if (buffer) this.gl.deleteBuffer(buffer);
        });
        
        if (this.framebuffer) this.gl.deleteFramebuffer(this.framebuffer);
        if (this.backgroundTexture) this.gl.deleteTexture(this.backgroundTexture);
    }
}
