// Elite Game Master - Orchestrates all systems for the Snake Evolution Arena
console.log('📜 Elite Game Master script loaded');

class EliteGameMaster {
    constructor() {
        console.log('🏗️ EliteGameMaster constructor called');
        this.canvas = null;
        this.ctx = null;
        this.gameMode = 'menu';
        this.isInitialized = false;
        this.gameLoopRunning = false;
        this.animationId = null;
        
        // Core systems
        this.snakeGenerator = null;
        this.battleEngine = null;
        this.visualEffects = null;
        this.evolutionSystem = null;
        this.tournamentSystem = null;
        
        // UI panels
        this.ui = {
            activePanel: null,
            panels: {}
        };
        
        // Configuration
        this.config = {
            enableWebGL: true,
            enableHDR: true,
            enableParticles: true,
            maxParticles: 1000,
            enableVSync: true,
            autoStartTournament: false
        };
        
        // Performance monitoring
        this.performance = {
            fps: 60,
            frameTime: 16.67,
            lastFrameTime: 0,
            frameCount: 0,
            fpsHistory: []
        };
        
        // Bind methods
        this.gameLoop = this.gameLoop.bind(this);
        this.handleResize = this.handleResize.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleVisibilityChange = this.handleVisibilityChange.bind(this);
    }
    
    async initialize() {
        try {
            console.log('🎮 Initializing Elite Game Master...');
            
            // Get canvas
            this.canvas = document.getElementById('game-canvas');
            if (!this.canvas) {
                throw new Error('Game canvas not found');
            }
            
            this.ctx = this.canvas.getContext('2d');
            if (!this.ctx) {
                throw new Error('Could not get canvas context');
            }
            
            // Set high resolution
            this.setupHighResolution();
            
            // Initialize systems
            await this.initializeSystems();
            
            // Create UI
            this.createUI();
            
            // Bind events
            this.bindEvents();
            
            // Show main menu
            this.showMainMenu();
            
            this.isInitialized = true;
            console.log('✅ Elite Game Master initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize Elite Game Master:', error);
            this.showErrorMessage('Failed to initialize game: ' + error.message);
            return false;
        }
    }
    
    setupHighResolution() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        // Set actual size
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        
        // Scale back down using CSS
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        // Scale the drawing context
        this.ctx.scale(dpr, dpr);
        
        console.log('📺 High resolution setup complete', {
            width: this.canvas.width,
            height: this.canvas.height,
            dpr: dpr
        });
    }
      async initializeSystems() {
        try {
            // Initialize snake generator
            this.snakeGenerator = new EliteSnakeGenerator();
            console.log('🐍 Snake generator initialized');
            
            // Initialize visual effects first (needed by battle engine)
            this.visualEffects = new EliteVisualEffects(this.canvas, this.ctx);
            console.log('✨ Visual effects initialized');
            
            // Initialize battle engine with visual effects
            this.battleEngine = new EliteBattleEngine(this.canvas, this.ctx, this.visualEffects);
            console.log('⚔️ Battle engine initialized');
            
            // Initialize evolution system
            this.evolutionSystem = new EliteEvolutionSystem();
            console.log('🧬 Evolution system initialized');
            
            // Initialize tournament system
            this.tournamentSystem = new EliteTournamentSystem();
            console.log('🏆 Tournament system initialized');
            
            // Bind system events
            this.bindSystemEvents();
            
        } catch (error) {
            console.error('❌ System initialization failed:', error);
            throw error;
        }
    }
    
    bindSystemEvents() {
        // Battle engine events
        this.canvas.addEventListener('battleEnd', (event) => {
            this.handleBattleEnd(event.detail);
        });
        
        // Tournament events
        this.canvas.addEventListener('tournamentComplete', (event) => {
            this.handleTournamentComplete(event.detail);
        });
    }
      createUI() {
        // Use existing HTML panels instead of creating new ones
        this.ui.panels.control = document.querySelector('.left-panel');
        this.ui.panels.info = document.querySelector('.right-panel');
        this.ui.panels.settings = document.querySelector('.settings-overlay');
        this.ui.panels.stats = document.querySelector('.stats-overlay');
        
        // Bind events to existing buttons
        this.bindExistingUIEvents();
    }
      bindExistingUIEvents() {
        // Look for control buttons in the existing HTML
        const tournamentBtn = document.getElementById('start-tournament');
        if (tournamentBtn) {
            tournamentBtn.addEventListener('click', () => {
                this.startTournamentMode();
            });
        }
        
        const singleBattleBtn = document.getElementById('single-battle');
        if (singleBattleBtn) {
            singleBattleBtn.addEventListener('click', () => {
                this.startSingleBattle();
            });
        }
        
        const spectateBtn = document.getElementById('spectate-mode');
        if (spectateBtn) {
            spectateBtn.addEventListener('click', () => {
                this.startSpectateMode();
            });
        }
          // Add other UI event bindings as needed
        console.log('✅ UI events bound to existing elements');
    }
    
    bindEvents() {
        // Keyboard events
        document.addEventListener('keydown', this.handleKeyDown);
        
        // Window events
        window.addEventListener('resize', this.handleResize);
        document.addEventListener('visibilitychange', this.handleVisibilityChange);
        
        // Mouse events for canvas
        this.canvas.addEventListener('click', (event) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            this.handleCanvasClick(x, y);
        });
    }
    
    handleKeyDown(event) {
        // F11 for fullscreen
        if (event.key === 'F11') {
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
        
        // Space for pause/resume
        if (event.code === 'Space') {
            event.preventDefault();
            if (this.gameLoopRunning) {
                this.pauseGame();
            } else {
                this.resumeGame();
            }
        }
    }
    
    handleResize() {
        if (this.canvas) {
            this.setupHighResolution();
        }
    }
    
    handleCanvasClick(x, y) {
        // Handle canvas interactions
        console.log('Canvas clicked at:', x, y);
    }
    
    handleBattleEnd(battleDetail) {
        console.log('Battle ended:', battleDetail);
    }
    
    handleTournamentComplete(tournamentDetail) {
        console.log('Tournament completed:', tournamentDetail);
        this.showVictoryScreen(tournamentDetail.champion);
    }
    
    setGameMode(mode) {
        console.log('Setting game mode:', mode);
        this.gameMode = mode;
        
        switch (mode) {
            case 'menu':
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
        
        this.updateUI();
    }
    
    startTournamentMode() {
        console.log('🏆 Starting tournament mode...');
        
        if (!this.tournamentSystem) {
            this.showErrorMessage('Tournament system not initialized');
            return;
        }
        
        try {
            // Generate tournament snakes
            const snakes = [];
            for (let i = 0; i < 16; i++) {
                snakes.push(this.snakeGenerator.generateRandomSnake());
            }
            
            // Initialize tournament
            this.tournamentSystem.initialize(snakes);
            
            // Set mode and start
            this.setGameMode('tournament');
            
            console.log('✅ Tournament started with', snakes.length, 'snakes');
            
        } catch (error) {
            console.error('❌ Failed to start tournament:', error);
            this.showErrorMessage('Failed to start tournament: ' + error.message);
        }
    }
    
    startSingleBattle() {
        console.log('⚔️ Starting single battle...');
        
        try {
            // Generate two random snakes
            const snake1 = this.snakeGenerator.generateRandomSnake();
            const snake2 = this.snakeGenerator.generateRandomSnake();
            
            // Initialize battle
            this.battleEngine.initializeBattle(snake1, snake2);
            
            // Set mode and start
            this.setGameMode('battle');
            
            console.log('✅ Single battle started:', snake1.name, 'vs', snake2.name);
            
        } catch (error) {
            console.error('❌ Failed to start battle:', error);
            this.showErrorMessage('Failed to start battle: ' + error.message);
        }
    }
    
    startSpectateMode() {
        console.log('👁️ Starting spectate mode...');
        this.setGameMode('spectate');
    }
    
    pauseGame() {
        if (this.gameLoopRunning) {
            this.stopGameLoop();
            console.log('⏸️ Game paused');
        }
    }
    
    resumeGame() {
        if (!this.gameLoopRunning && this.gameMode !== 'menu') {
            this.startGameLoop();
            console.log('▶️ Game resumed');
        }
    }
    
    resetGame() {
        console.log('🔄 Resetting game...');
        this.stopGameLoop();
        
        // Reset systems
        if (this.battleEngine) {
            this.battleEngine.reset();
        }
        if (this.tournamentSystem) {
            this.tournamentSystem.reset();
        }
        
        this.setGameMode('menu');
    }
    
    startGameLoop() {
        if (!this.gameLoopRunning) {
            this.gameLoopRunning = true;
            this.performance.lastFrameTime = performance.now();
            this.gameLoop();
            console.log('🎮 Game loop started');
        }
    }
    
    stopGameLoop() {
        if (this.gameLoopRunning) {
            this.gameLoopRunning = false;
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
            console.log('⏹️ Game loop stopped');
        }
    }
    
    gameLoop() {
        if (!this.gameLoopRunning) return;
        
        const currentTime = performance.now();
        const deltaTime = currentTime - this.performance.lastFrameTime;
        
        // Update performance stats
        this.updatePerformanceStats(currentTime, deltaTime);
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update and render based on mode
        switch (this.gameMode) {
            case 'tournament':
                this.updateTournament(deltaTime);
                this.renderTournament();
                break;
            case 'battle':
                this.updateBattle(deltaTime);
                this.renderBattle();
                break;
            case 'spectate':
                this.updateSpectate(deltaTime);
                this.renderSpectate();
                break;
        }
        
        // Render UI overlay
        this.renderUIOverlay();
        
        // Continue loop
        this.performance.lastFrameTime = currentTime;
        this.animationId = requestAnimationFrame(() => this.gameLoop());
    }
    
    updatePerformanceStats(currentTime, deltaTime) {
        this.performance.frameTime = deltaTime;
        this.performance.fps = 1000 / deltaTime;
        this.performance.frameCount++;
        
        // Update FPS history
        if (this.performance.frameCount % 10 === 0) {
            this.performance.fpsHistory.push(this.performance.fps);
            if (this.performance.fpsHistory.length > 100) {
                this.performance.fpsHistory.shift();
            }
            this.updateFPSDisplay();
        }
    }
    
    updateTournament(deltaTime) {
        if (this.tournamentSystem) {
            this.tournamentSystem.update(deltaTime);
        }
    }
    
    updateBattle(deltaTime) {
        if (this.battleEngine) {
            this.battleEngine.update(deltaTime);
        }
    }
    
    updateSpectate(deltaTime) {
        // Spectate mode updates
    }
    
    renderTournament() {
        if (this.tournamentSystem) {
            this.tournamentSystem.render(this.ctx);
        }
    }
    
    renderBattle() {
        if (this.battleEngine) {
            this.battleEngine.render();
        }
    }
    
    renderSpectate() {
        // Spectate mode rendering
    }
    
    renderUIOverlay() {
        // Render debug info
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 10, 200, 60);
        
        this.ctx.fillStyle = 'white';
        this.ctx.font = '14px monospace';
        this.ctx.fillText('Mode: ' + this.gameMode.toUpperCase(), 20, 30);
        this.ctx.fillText('FPS: ' + Math.round(this.performance.fps), 20, 50);
        this.ctx.fillText('Frame: ' + this.performance.frameTime.toFixed(1) + 'ms', 20, 70);
        
        // Render additional info in top-right
        this.ctx.textAlign = 'right';
        this.ctx.fillText('FPS: ' + Math.round(this.performance.fps), this.canvas.width - 20, 30);
        this.ctx.textAlign = 'left';
    }
    
    updateUI() {
        this.updateStatsPanel();
        this.updateInfoPanel();
    }
    
    updateStatsPanel() {
        const fpsDisplay = document.getElementById('fps-display');
        const frameTimeDisplay = document.getElementById('frame-time-display');
        const modeDisplay = document.getElementById('mode-display');
        
        if (fpsDisplay) fpsDisplay.textContent = Math.round(this.performance.fps);
        if (frameTimeDisplay) frameTimeDisplay.textContent = this.performance.frameTime.toFixed(1) + 'ms';
        if (modeDisplay) modeDisplay.textContent = this.gameMode.charAt(0).toUpperCase() + this.gameMode.slice(1);
    }
    
    updateInfoPanel() {
        const battleInfo = document.getElementById('battle-info');
        const tournamentInfo = document.getElementById('tournament-info');
        
        if (battleInfo && this.battleEngine) {
            const battle = this.battleEngine.getCurrentBattle();
            if (battle) {
                battleInfo.innerHTML = `
                    <h4>Current Battle</h4>
                    <div class="battle-fighters">
                        <div style="color: ${battle.snake1.appearance.primaryColor}">
                            ${battle.snake1.name} (${battle.snake1.health} HP)
                        </div>
                        <div style="color: ${battle.snake2.appearance.primaryColor}">
                            ${battle.snake2.name} (${battle.snake2.health} HP)
                        </div>
                    </div>
                `;
            } else {
                battleInfo.innerHTML = '<p>No active battle</p>';
            }
        }
        
        if (tournamentInfo && this.tournamentSystem) {
            const tournament = this.tournamentSystem.getCurrentTournament();
            if (tournament) {
                tournamentInfo.innerHTML = `
                    <h4>Tournament Progress</h4>
                    <p>Round: ${tournament.currentRound}</p>
                    <p>Remaining: ${tournament.remainingSnakes.length}</p>
                `;
            } else {
                tournamentInfo.innerHTML = '<p>No active tournament</p>';
            }
        }
    }
    
    updateFPSDisplay() {
        if (this.fpsChart && this.fpsChart.ctx) {
            const ctx = this.fpsChart.ctx;
            const canvas = this.fpsChart.canvas;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw FPS chart
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 1;
            ctx.beginPath();
            
            const history = this.performance.fpsHistory;
            const maxFPS = 120;
            
            for (let i = 0; i < history.length; i++) {
                const x = (i / history.length) * canvas.width;
                const y = canvas.height - (history[i] / maxFPS) * canvas.height;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }
    }
    
    showUI() {
        const panels = Object.values(this.ui.panels);
        panels.forEach(panel => {
            if (panel.id !== 'settings-panel') {
                panel.style.display = 'block';
            }
        });
    }
    
    hideUI() {
        const panels = Object.values(this.ui.panels);
        panels.forEach(panel => {
            panel.style.display = 'none';
        });
    }
    
    showSettingsPanel() {
        this.hideActivePanel();
        this.ui.panels.settings.style.display = 'block';
        this.ui.activePanel = this.ui.panels.settings;
    }
    
    hideSettingsPanel() {
        this.ui.panels.settings.style.display = 'none';
        this.ui.activePanel = null;
    }
    
    hideActivePanel() {
        if (this.ui.activePanel) {
            this.ui.activePanel.style.display = 'none';
            this.ui.activePanel = null;
        }
    }
    
    applySettings() {
        const panel = this.ui.panels.settings;
        
        const webglCheckbox = panel.querySelector('#setting-webgl');
        const hdrCheckbox = panel.querySelector('#setting-hdr');
        const particleCheckbox = panel.querySelector('#setting-particles');
        const maxParticlesSlider = panel.querySelector('#setting-max-particles');
        const vsyncCheckbox = panel.querySelector('#setting-vsync');
        const autoStartCheckbox = panel.querySelector('#setting-auto-start');
        
        this.config.enableWebGL = webglCheckbox ? webglCheckbox.checked : false;
        this.config.enableHDR = hdrCheckbox ? hdrCheckbox.checked : false;
        this.config.enableParticles = particleCheckbox ? particleCheckbox.checked : false;
        this.config.maxParticles = maxParticlesSlider ? parseInt(maxParticlesSlider.value) : 1000;
        this.config.enableVSync = vsyncCheckbox ? vsyncCheckbox.checked : false;
        this.config.autoStartTournament = autoStartCheckbox ? autoStartCheckbox.checked : false;
        
        console.log('⚙️ Settings applied:', this.config);
        
        // Apply changes to systems
        if (this.visualEffects) {
            this.visualEffects.updateConfig(this.config);
        }
        
        this.hideSettingsPanel();
    }
    
    handleVisibilityChange() {
        if (document.hidden && this.gameLoopRunning) {
            this.pauseGame();
        }
    }
    
    showMainMenu() {
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
        const newTournamentBtn = overlay.querySelector('#new-tournament-btn');
        if (newTournamentBtn) {
            newTournamentBtn.addEventListener('click', () => {
                document.body.removeChild(overlay);
                this.startTournamentMode();
            });
        }
        
        const closeBtn = overlay.querySelector('#close-victory-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                document.body.removeChild(overlay);
                this.showMainMenu();
            });
        }
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
    
    dispose() {
        console.log('🧹 Disposing Elite Game Master...');
        
        // Stop game loop
        this.stopGameLoop();
        
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
