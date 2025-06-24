// Elite Game Master - Orchestrates the entire Snake Evolution Arena experience
class EliteGameMaster {
    constructor() {
        // Core systems
        this.canvas = null;
        this.ctx = null;
        this.visualEffects = null;
        this.battleEngine = null;
        this.evolutionSystem = null;
        this.tournamentSystem = null;
        this.snakeGenerator = null;
        
        // Game state
        this.gameMode = 'menu'; // menu, tournament, battle, spectate
        this.isInitialized = false;
        this.isPaused = false;
        
        // Performance monitoring
        this.performance = {
            fps: 0,
            frameTime: 0,
            lastFrameTime: 0,
            frameCount: 0
        };
        
        // Configuration
        this.config = {
            targetFPS: 60,
            enableVSync: true,
            enableWebGL: true,
            enableHDR: true,
            enableParticles: true,
            maxParticles: 1000,
            enableSound: false, // For future implementation
            autoStartTournament: true
        };
        
        // UI state
        this.ui = {
            elements: new Map(),
            panels: new Map(),
            activePanel: null
        };
        
        console.log('Elite Game Master initializing...');
    }
    
    async initialize() {
        try {
            // Initialize canvas and context
            await this.initializeCanvas();
            
            // Initialize core systems
            await this.initializeSystems();
            
            // Initialize UI
            await this.initializeUI();
            
            // Bind events
            this.bindEvents();
            
            // Set initial game mode
            this.setGameMode('menu');
            
            this.isInitialized = true;
            
            console.log('Elite Game Master initialized successfully');
            
            // Auto-start tournament if configured
            if (this.config.autoStartTournament) {
                setTimeout(() => {
                    this.startTournamentMode();
                }, 1000);
            }
            
            return true;
            
        } catch (error) {
            console.error('Elite Game Master initialization failed:', error);
            this.showErrorMessage('Initialization failed: ' + error.message);
            return false;
        }
    }
    
    async initializeCanvas() {
        // Find or create canvas
        this.canvas = document.getElementById('game-canvas');
        if (!this.canvas) {
            this.canvas = document.createElement('canvas');
            this.canvas.id = 'game-canvas';
            
            const arenaContainer = document.querySelector('.arena-container');
            if (arenaContainer) {
                arenaContainer.appendChild(this.canvas);
            } else {
                document.body.appendChild(this.canvas);
            }
        }
        
        // Get 2D context
        this.ctx = this.canvas.getContext('2d');
        if (!this.ctx) {
            throw new Error('Failed to get 2D rendering context');
        }
        
        // Configure canvas
        this.setupCanvas();
        
        console.log('Canvas initialized:', this.canvas.width, 'x', this.canvas.height);
    }
    
    setupCanvas() {
        // Set canvas size
        const containerRect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = Math.max(1200, containerRect.width);
        this.canvas.height = Math.max(800, containerRect.height);
        
        // High DPI support
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        this.ctx.scale(dpr, dpr);
        
        // High quality rendering
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
        this.ctx.textBaseline = 'middle';
    }
    
    async initializeSystems() {
        console.log('Initializing core systems...');
        
        // Initialize Visual Effects System
        this.visualEffects = new EliteVisualEffects(this.canvas, this.ctx);
        
        // Initialize Snake Generator
        this.snakeGenerator = new EliteSnakeGenerator();
        
        // Initialize Battle Engine
        this.battleEngine = new EliteBattleEngine(this.canvas, this.ctx, this.visualEffects);
        
        // Initialize Evolution System
        this.evolutionSystem = new EliteEvolutionSystem(this.visualEffects);
        
        // Initialize Tournament System
        this.tournamentSystem = new EliteTournamentSystem(
            this.battleEngine,
            this.evolutionSystem,
            this.visualEffects
        );
        
        console.log('Core systems initialized');
    }
    
    async initializeUI() {
        console.log('Initializing UI systems...');
        
        // Create main UI panels
        this.createControlPanel();
        this.createInfoPanel();
        this.createSettingsPanel();
        this.createStatsPanel();
        
        // Initialize UI state
        this.updateUI();
        
        console.log('UI systems initialized');
    }
    
    createControlPanel() {
        const panel = document.createElement('div');
        panel.id = 'control-panel';
        panel.className = 'ui-panel control-panel';
        
        panel.innerHTML = `
            <h3>🎮 Elite Arena Controls</h3>
            <div class="control-section">
                <h4>Game Modes</h4>
                <button id="btn-tournament" class="control-btn primary">🏆 Start Tournament</button>
                <button id="btn-single-battle" class="control-btn">⚔️ Single Battle</button>
                <button id="btn-spectate" class="control-btn">👁️ Spectate Mode</button>
            </div>
            <div class="control-section">
                <h4>Tournament Controls</h4>
                <button id="btn-pause" class="control-btn">⏸️ Pause</button>
                <button id="btn-resume" class="control-btn">▶️ Resume</button>
                <button id="btn-reset" class="control-btn danger">🔄 Reset</button>
            </div>
            <div class="control-section">
                <h4>Settings</h4>
                <button id="btn-settings" class="control-btn">⚙️ Settings</button>
                <button id="btn-fullscreen" class="control-btn">🖥️ Fullscreen</button>
            </div>
        `;
        
        const leftPanel = document.querySelector('.left-panel');
        if (leftPanel) {
            leftPanel.insertBefore(panel, leftPanel.firstChild);
        }
        
        this.ui.panels.set('control', panel);
        
        // Bind control events
        this.bindControlEvents(panel);
    }
    
    bindControlEvents(panel) {
        // Tournament controls        const tournamentBtn = panel.querySelector('#btn-tournament');
        if (tournamentBtn) {
            tournamentBtn.addEventListener('click', () => {
                this.startTournamentMode();
            });
        }
        
        panel.querySelector('#btn-single-battle')?.addEventListener('click', () => {
            this.startSingleBattleMode();
        });
        
        panel.querySelector('#btn-spectate')?.addEventListener('click', () => {
            this.startSpectateMode();
        });
        
        // Pause/Resume
        panel.querySelector('#btn-pause')?.addEventListener('click', () => {
            this.pauseGame();
        });
        
        panel.querySelector('#btn-resume')?.addEventListener('click', () => {
            this.resumeGame();
        });
        
        // Reset
        panel.querySelector('#btn-reset')?.addEventListener('click', () => {
            this.resetGame();
        });
        
        // Settings
        panel.querySelector('#btn-settings')?.addEventListener('click', () => {
            this.showSettingsPanel();
        });
        
        // Fullscreen
        panel.querySelector('#btn-fullscreen')?.addEventListener('click', () => {
            this.toggleFullscreen();
        });
    }
    
    createInfoPanel() {
        const panel = document.createElement('div');
        panel.id = 'info-panel';
        panel.className = 'ui-panel info-panel';
        
        panel.innerHTML = `
            <h3>📊 Game Information</h3>
            <div class="info-section">
                <div class="info-item">
                    <span class="info-label">Game Mode:</span>
                    <span id="info-mode" class="info-value">Menu</span>
                </div>
                <div class="info-item">
                    <span class="info-label">FPS:</span>
                    <span id="info-fps" class="info-value">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Frame Time:</span>
                    <span id="info-frametime" class="info-value">0ms</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Active Effects:</span>
                    <span id="info-effects" class="info-value">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Particles:</span>
                    <span id="info-particles" class="info-value">0</span>
                </div>
            </div>
            <div id="current-battle-info" class="info-section">
                <!-- Battle info populated dynamically -->
            </div>
        `;
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            rightPanel.appendChild(panel);
        }
        
        this.ui.panels.set('info', panel);
    }
    
    createSettingsPanel() {
        const panel = document.createElement('div');
        panel.id = 'settings-panel';
        panel.className = 'ui-panel settings-panel hidden';
        
        panel.innerHTML = `
            <h3>⚙️ Settings</h3>
            <div class="settings-section">
                <h4>Graphics</h4>
                <label class="setting-item">
                    <input type="checkbox" id="setting-webgl" ${this.config.enableWebGL ? 'checked' : ''}>
                    <span>Enable WebGL Effects</span>
                </label>
                <label class="setting-item">
                    <input type="checkbox" id="setting-hdr" ${this.config.enableHDR ? 'checked' : ''}>
                    <span>Enable HDR Rendering</span>
                </label>
                <label class="setting-item">
                    <input type="checkbox" id="setting-particles" ${this.config.enableParticles ? 'checked' : ''}>
                    <span>Enable Particle Effects</span>
                </label>
            </div>
            <div class="settings-section">
                <h4>Performance</h4>
                <label class="setting-item">
                    <span>Max Particles:</span>
                    <input type="range" id="setting-max-particles" min="100" max="2000" value="${this.config.maxParticles}">
                    <span id="particles-value">${this.config.maxParticles}</span>
                </label>
                <label class="setting-item">
                    <input type="checkbox" id="setting-vsync" ${this.config.enableVSync ? 'checked' : ''}>
                    <span>Enable VSync</span>
                </label>
            </div>
            <div class="settings-section">
                <h4>Tournament</h4>
                <label class="setting-item">
                    <input type="checkbox" id="setting-auto-start" ${this.config.autoStartTournament ? 'checked' : ''}>
                    <span>Auto-start Tournament</span>
                </label>
            </div>
            <div class="settings-actions">
                <button id="settings-apply" class="control-btn primary">Apply Settings</button>
                <button id="settings-close" class="control-btn">Close</button>
            </div>
        `;
        
        document.body.appendChild(panel);
        this.ui.panels.set('settings', panel);
        
        // Bind settings events
        this.bindSettingsEvents(panel);
    }
    
    bindSettingsEvents(panel) {
        // Update particle count display
        const particleRange = panel.querySelector('#setting-max-particles');
        const particleValue = panel.querySelector('#particles-value');
        
        particleRange?.addEventListener('input', (e) => {
            particleValue.textContent = e.target.value;
        });
        
        // Apply settings
        panel.querySelector('#settings-apply')?.addEventListener('click', () => {
            this.applySettings();
        });
        
        // Close settings
        panel.querySelector('#settings-close')?.addEventListener('click', () => {
            this.hideSettingsPanel();
        });
    }
    
    createStatsPanel() {
        const panel = document.createElement('div');
        panel.id = 'stats-panel';
        panel.className = 'ui-panel stats-panel';
        
        panel.innerHTML = `
            <h3>📈 Performance Stats</h3>
            <div id="stats-content">
                <canvas id="fps-chart" width="280" height="120"></canvas>
                <div class="stats-details">
                    <div class="stat-item">
                        <span>Avg FPS:</span>
                        <span id="avg-fps">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Min FPS:</span>
                        <span id="min-fps">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Max FPS:</span>
                        <span id="max-fps">0</span>
                    </div>
                </div>
            </div>
        `;
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            rightPanel.appendChild(panel);
        }
        
        this.ui.panels.set('stats', panel);
        
        // Initialize FPS chart
        this.initializeFPSChart();
    }
    
    initializeFPSChart() {
        const canvas = document.getElementById('fps-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        this.fpsChart = {
            canvas: canvas,
            ctx: ctx,
            data: [],
            maxDataPoints: 60
        };
    }
    
    bindEvents() {
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            this.handleKeyDown(e);
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Visibility change (pause when tab not visible)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseGame();
            } else {
                this.resumeGame();
            }
        });
        
        // Battle events
        document.addEventListener('battleEnd', (e) => {
            this.handleBattleEnd(e.detail);
        });
        
        document.addEventListener('tournamentComplete', (e) => {
            this.handleTournamentComplete(e.detail);
        });
    }
    
    handleKeyDown(event) {
        if (event.code === 'Space') {
            event.preventDefault();
            this.togglePause();
        }
        
        if (event.code === 'KeyT') {
            event.preventDefault();
            this.startTournamentMode();
        }
        
        if (event.code === 'KeyF') {
            event.preventDefault();
            this.toggleFullscreen();
        }
        
        if (event.code === 'Escape') {
            event.preventDefault();
            if (this.ui.activePanel) {
                this.hideActivePanel();
            } else {
                this.showMainMenu();
            }
        }
    }
    
    handleResize() {
        this.setupCanvas();
        
        if (this.battleEngine) {
            this.battleEngine.gameWidth = this.canvas.width;
            this.battleEngine.gameHeight = this.canvas.height;
        }
    }
    
    handleBattleEnd(battleDetail) {
        console.log('Battle ended:', battleDetail.winner.name, 'wins');
        this.updateBattleInfo();
    }
    
    handleTournamentComplete(tournamentDetail) {
        console.log('Tournament completed! Champion:', tournamentDetail.champion.name);
        this.showVictoryScreen(tournamentDetail.champion);
    }
    
    // Game Mode Management    setGameMode(mode) {
        console.log(`Setting game mode: ${mode}`);
        
        this.gameMode = mode;
        
        switch (mode) {
            case 'menu':
                // Don't call showMainMenu here to avoid circular calls
                this.stopGameLoop();
                break;
            case 'tournament':
                this.startGameLoop();
                break;
            case 'battle':
                this.startGameLoop();
                break;
            case 'spectate':
                this.startGameLoop();
                break;
        }
        
        // Update UI after mode change
        this.updateUI();
    }
    
    startTournamentMode() {
        console.log('Starting tournament mode');
        
        this.setGameMode('tournament');
        
        if (this.tournamentSystem) {
            this.tournamentSystem.startNewTournament();
        }
    }
    
    startSingleBattleMode() {        console.log('Starting single battle mode');
        
        this.setGameMode('battle');
        
        // Generate two random snakes for battle
        const snake1 = this.snakeGenerator.createSnake();
        const snake2 = this.snakeGenerator.createSnake();
        
        if (this.battleEngine) {
            this.battleEngine.startBattle(snake1, snake2);
        }
    }
    
    startSpectateMode() {
        console.log('Starting spectate mode');
        
        this.setGameMode('spectate');
        
        // Start a continuous series of random battles
        this.startSpectateLoop();
    }
      startSpectateLoop() {
        const spectateNextBattle = () => {
            if (this.gameMode !== 'spectate') return;
            
            const snake1 = this.snakeGenerator.createSnake();
            const snake2 = this.snakeGenerator.createSnake();
            
            this.battleEngine.startBattle(snake1, snake2);
            
            // Schedule next battle
            setTimeout(spectateNextBattle, 60000); // New battle every minute
        };
        
        spectateNextBattle();
    }
    
    // Game Control
    pauseGame() {
        this.isPaused = true;
        
        if (this.battleEngine) {
            this.battleEngine.togglePause();
        }
        
        if (this.tournamentSystem) {
            this.tournamentSystem.togglePause();
        }
        
        console.log('Game paused');
        this.updateUI();
    }
    
    resumeGame() {
        this.isPaused = false;
        
        if (this.battleEngine && this.battleEngine.isPaused) {
            this.battleEngine.togglePause();
        }
        
        if (this.tournamentSystem && this.tournamentSystem.isPaused) {
            this.tournamentSystem.togglePause();
        }
        
        console.log('Game resumed');
        this.updateUI();
    }
    
    togglePause() {
        if (this.isPaused) {
            this.resumeGame();
        } else {
            this.pauseGame();
        }
    }
    
    resetGame() {
        console.log('Resetting game');
        
        // Stop current systems
        if (this.battleEngine) {
            this.battleEngine.isRunning = false;
        }
        
        if (this.tournamentSystem) {
            this.tournamentSystem.isRunning = false;
        }
        
        // Clear effects
        if (this.visualEffects) {
            this.visualEffects.effects = [];
            this.visualEffects.particles = [];
        }
        
        // Return to menu
        this.setGameMode('menu');
    }
    
    // Game Loop
    startGameLoop() {
        if (this.gameLoopRunning) return;
        
        this.gameLoopRunning = true;
        this.lastFrameTime = performance.now();
        
        const gameLoop = (currentTime) => {
            if (!this.gameLoopRunning) return;
            
            // Calculate frame timing
            const deltaTime = currentTime - this.lastFrameTime;
            this.lastFrameTime = currentTime;
            
            // Update performance stats
            this.updatePerformanceStats(deltaTime);
            
            // Update game systems
            this.update(deltaTime);
            
            // Render everything
            this.render();
            
            // Continue loop
            if (this.config.enableVSync) {
                requestAnimationFrame(gameLoop);
            } else {
                setTimeout(() => requestAnimationFrame(gameLoop), 1000 / this.config.targetFPS);
            }
        };
        
        requestAnimationFrame(gameLoop);
        console.log('Game loop started');
    }
    
    stopGameLoop() {
        this.gameLoopRunning = false;
        console.log('Game loop stopped');
    }
    
    update(deltaTime) {
        // Update visual effects
        if (this.visualEffects && this.config.enableParticles) {
            this.visualEffects.update(deltaTime);
        }
        
        // Battle engine updates are handled internally
        // Tournament system updates are handled internally
        
        // Update UI periodically
        this.updateUI();
    }
    
    render() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Render based on game mode
        switch (this.gameMode) {
            case 'menu':
                this.renderMainMenu();
                break;
            case 'tournament':
            case 'battle':
            case 'spectate':
                this.renderGameplay();
                break;
        }
        
        // Render UI overlays
        this.renderUIOverlays();
    }
    
    renderMainMenu() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Title
        this.ctx.fillStyle = '#00ffff';
        this.ctx.font = 'bold 48px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🐍 ELITE SNAKE EVOLUTION ARENA', centerX, centerY - 100);
        
        // Subtitle
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '24px Arial';
        this.ctx.fillText('High-Resolution Battle Royale with Advanced Evolution', centerX, centerY - 50);
        
        // Instructions
        this.ctx.fillStyle = '#cccccc';
        this.ctx.font = '18px Arial';
        this.ctx.fillText('Press T to start Tournament Mode', centerX, centerY + 20);
        this.ctx.fillText('Press F for Fullscreen', centerX, centerY + 50);
        this.ctx.fillText('Press ESC for Settings', centerX, centerY + 80);
        
        // Version info
        this.ctx.fillStyle = '#666666';
        this.ctx.font = '12px Arial';
        this.ctx.fillText('Elite Edition v2.0 - Next Generation Snake AI', centerX, centerY + 150);
    }
    
    renderGameplay() {
        // Battle engine handles its own rendering
        // Visual effects handle their own rendering
        
        // Render additional game info
        this.renderGameInfo();
    }
    
    renderGameInfo() {
        // Game mode indicator
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 10, 200, 30);
        
        this.ctx.fillStyle = '#00ffff';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Mode: ${this.gameMode.toUpperCase()}`, 20, 30);
        
        // FPS counter
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(this.canvas.width - 100, 10, 80, 30);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`FPS: ${Math.round(this.performance.fps)}`, this.canvas.width - 20, 30);
    }
    
    renderUIOverlays() {
        // Pause overlay
        if (this.isPaused) {
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = 'bold 36px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('PAUSED', this.canvas.width / 2, this.canvas.height / 2);
            
            this.ctx.font = '18px Arial';
            this.ctx.fillText('Press SPACE to resume', this.canvas.width / 2, this.canvas.height / 2 + 40);
        }
    }
    
    // UI Management
    updateUI() {
        // Update game mode display
        const modeElement = document.getElementById('info-mode');
        if (modeElement) {
            modeElement.textContent = this.gameMode.charAt(0).toUpperCase() + this.gameMode.slice(1);
        }
        
        // Update performance display
        this.updatePerformanceDisplay();
        
        // Update battle info
        this.updateBattleInfo();
        
        // Update control button states
        this.updateControlButtons();
    }
    
    updatePerformanceDisplay() {
        const fpsElement = document.getElementById('info-fps');
        const frameTimeElement = document.getElementById('info-frametime');
        const effectsElement = document.getElementById('info-effects');
        const particlesElement = document.getElementById('info-particles');
        
        if (fpsElement) {
            fpsElement.textContent = Math.round(this.performance.fps);
        }
        
        if (frameTimeElement) {
            frameTimeElement.textContent = this.performance.frameTime.toFixed(1) + 'ms';
        }
        
        if (effectsElement && this.visualEffects) {
            effectsElement.textContent = this.visualEffects.effects.length;
        }
        
        if (particlesElement && this.visualEffects) {
            particlesElement.textContent = this.visualEffects.particles.length;
        }
        
        // Update FPS chart
        this.updateFPSChart();
    }
    
    updateBattleInfo() {
        const battleInfoElement = document.getElementById('current-battle-info');
        if (!battleInfoElement) return;
        
        if (this.battleEngine && this.battleEngine.currentBattle) {
            const battle = this.battleEngine.currentBattle;
            const elapsed = Date.now() - battle.startTime;
            const remaining = Math.max(0, battle.config.timeLimit - elapsed);
            
            battleInfoElement.innerHTML = `
                <h4>Current Battle</h4>
                <div class="battle-fighters">
                    <div style="color: ${battle.snake1.appearance.primaryColor}">
                        ${battle.snake1.name} (${battle.snake1.health}/${battle.snake1.maxHealth} HP)
                    </div>
                    <div>VS</div>
                    <div style="color: ${battle.snake2.appearance.primaryColor}">
                        ${battle.snake2.name} (${battle.snake2.health}/${battle.snake2.maxHealth} HP)
                    </div>
                </div>
                <div class="battle-timer">
                    Time Remaining: ${(remaining / 1000).toFixed(1)}s
                </div>
            `;
        } else {
            battleInfoElement.innerHTML = '<p>No active battle</p>';
        }
    }
    
    updateControlButtons() {
        const pauseBtn = document.getElementById('btn-pause');
        const resumeBtn = document.getElementById('btn-resume');
        
        if (pauseBtn) {
            pauseBtn.disabled = this.isPaused || this.gameMode === 'menu';
        }
        
        if (resumeBtn) {
            resumeBtn.disabled = !this.isPaused || this.gameMode === 'menu';
        }
    }
    
    updateFPSChart() {
        if (!this.fpsChart) return;
        
        // Add current FPS to data
        this.fpsChart.data.push(this.performance.fps);
        
        // Keep only recent data points
        if (this.fpsChart.data.length > this.fpsChart.maxDataPoints) {
            this.fpsChart.data.shift();
        }
        
        // Draw chart
        const ctx = this.fpsChart.ctx;
        const canvas = this.fpsChart.canvas;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.fpsChart.data.length < 2) return;
        
        // Draw FPS line
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const maxFPS = Math.max(60, Math.max(...this.fpsChart.data));
        const stepX = canvas.width / (this.fpsChart.maxDataPoints - 1);
        
        this.fpsChart.data.forEach((fps, index) => {
            const x = index * stepX;
            const y = canvas.height - (fps / maxFPS) * canvas.height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw target FPS line
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        const targetY = canvas.height - (this.config.targetFPS / maxFPS) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(0, targetY);
        ctx.lineTo(canvas.width, targetY);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    updatePerformanceStats(deltaTime) {
        this.performance.frameCount++;
        this.performance.frameTime = deltaTime;
        this.performance.fps = 1000 / deltaTime;
        
        // Update averages periodically
        if (this.performance.frameCount % 60 === 0) {
            const avgFpsElement = document.getElementById('avg-fps');
            const minFpsElement = document.getElementById('min-fps');
            const maxFpsElement = document.getElementById('max-fps');
            
            if (this.fpsChart && this.fpsChart.data.length > 0) {
                const avgFps = this.fpsChart.data.reduce((a, b) => a + b, 0) / this.fpsChart.data.length;
                const minFps = Math.min(...this.fpsChart.data);
                const maxFps = Math.max(...this.fpsChart.data);
                
                if (avgFpsElement) avgFpsElement.textContent = Math.round(avgFps);
                if (minFpsElement) minFpsElement.textContent = Math.round(minFps);
                if (maxFpsElement) maxFpsElement.textContent = Math.round(maxFps);
            }
        }
    }
    
    // Settings Management
    showSettingsPanel() {
        const panel = this.ui.panels.get('settings');
        if (panel) {
            panel.classList.remove('hidden');
            this.ui.activePanel = panel;
        }
    }
    
    hideSettingsPanel() {
        const panel = this.ui.panels.get('settings');
        if (panel) {
            panel.classList.add('hidden');
            this.ui.activePanel = null;
        }
    }
    
    hideActivePanel() {
        if (this.ui.activePanel) {
            this.ui.activePanel.classList.add('hidden');
            this.ui.activePanel = null;
        }
    }
    
    applySettings() {
        const panel = this.ui.panels.get('settings');
        if (!panel) return;
        
        // Get setting values
        this.config.enableWebGL = panel.querySelector('#setting-webgl')?.checked || false;
        this.config.enableHDR = panel.querySelector('#setting-hdr')?.checked || false;
        this.config.enableParticles = panel.querySelector('#setting-particles')?.checked || false;
        this.config.maxParticles = parseInt(panel.querySelector('#setting-max-particles')?.value) || 1000;
        this.config.enableVSync = panel.querySelector('#setting-vsync')?.checked || false;
        this.config.autoStartTournament = panel.querySelector('#setting-auto-start')?.checked || false;
        
        // Apply settings to systems
        if (this.visualEffects) {
            this.visualEffects.maxParticles = this.config.maxParticles;
        }
        
        console.log('Settings applied:', this.config);
        this.hideSettingsPanel();
    }
    
    // Utility Methods    showMainMenu() {
        // Set game mode directly without circular call
        this.gameMode = 'menu';
        this.stopGameLoop();
        
        // Show the menu UI
        this.hideActivePanel();
        this.showUI();
    }
    
    showVictoryScreen(champion) {
        // Create victory overlay
        const overlay = document.createElement('div');
        overlay.className = 'victory-overlay';
        overlay.innerHTML = `
            <div class="victory-content">
                <h1>🏆 TOURNAMENT CHAMPION! 🏆</h1>
                <h2 style="color: ${champion.appearance.primaryColor}">${champion.name}</h2>
                <div class="champion-stats">
                    <p>Generation: ${champion.generation}</p>
                    <p>Tournament Wins: ${champion.tournamentStats.wins}</p>
                    <p>Total Damage: ${champion.tournamentStats.totalDamageDealt}</p>
                    <p>Evolutions: ${champion.tournamentStats.evolutionsGained}</p>
                </div>
                <button id="new-tournament-btn" class="victory-btn">🎮 New Tournament</button>
                <button id="close-victory-btn" class="victory-btn">✕ Close</button>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Bind events
        overlay.querySelector('#new-tournament-btn')?.addEventListener('click', () => {
            document.body.removeChild(overlay);
            this.startTournamentMode();
        });
        
        overlay.querySelector('#close-victory-btn')?.addEventListener('click', () => {
            document.body.removeChild(overlay);
        });
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (document.body.contains(overlay)) {
                document.body.removeChild(overlay);
            }
        }, 10000);
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <h3>⚠️ Error</h3>
            <p>${message}</p>
            <button onclick="this.parentElement.remove()">Close</button>
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            if (document.body.contains(errorDiv)) {
                document.body.removeChild(errorDiv);
            }
        }, 5000);
    }
    
    // Cleanup
    dispose() {
        console.log('Disposing Elite Game Master...');
        
        this.stopGameLoop();
        
        if (this.visualEffects) {
            this.visualEffects.dispose();
        }
        
        if (this.tournamentSystem) {
            this.tournamentSystem.dispose();
        }
        
        // Remove event listeners
        document.removeEventListener('keydown', this.handleKeyDown);
        window.removeEventListener('resize', this.handleResize);
        document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    }
}

// Initialize the Elite Game Master when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🐍 Starting Elite Snake Evolution Arena...');
    
    window.eliteGameMaster = new EliteGameMaster();
    const success = await window.eliteGameMaster.initialize();
    
    if (success) {
        console.log('🎮 Elite Snake Evolution Arena ready!');
    } else {
        console.error('❌ Failed to initialize Elite Snake Evolution Arena');
    }
});
