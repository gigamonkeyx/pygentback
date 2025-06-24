// Elite Game Master - Simplified version for debugging
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
        
        // Core systems - will be initialized in initializeSystems()
        this.snakeGenerator = null;
        this.visualEffects = null;
        this.battleEngine = null;
        this.evolutionSystem = null;
        this.tournamentSystem = null;
        
        // UI state
        this.ui = {
            panels: {},
            overlays: {},
            buttons: {}
        };
        
        // Game state
        this.currentBattle = null;
        this.currentTournament = null;
        
        // Configuration
        this.config = {
            enableWebGL: true,
            enableHDR: true,
            enableParticles: true,
            maxParticles: 1000,
            enableVSync: true,
            autoStartTournament: false
        };
        
        // Performance tracking
        this.performance = {
            lastFrameTime: 0,
            fpsHistory: [],
            currentFPS: 60
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
            this.canvas = document.getElementById('game-canvas') || document.getElementById('battle-canvas');
            if (!this.canvas) {
                throw new Error('Game canvas not found (looking for game-canvas or battle-canvas)');
            }
            
            this.ctx = this.canvas.getContext('2d');
            if (!this.ctx) {
                throw new Error('Could not get canvas context');
            }
            
            // Set high resolution
            this.setupHighResolution();
            
            // Initialize systems
            await this.initializeSystems();
            
            // Create UI (simplified)
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
    }    setupHighResolution() {
        // Set a fixed size for testing
        this.canvas.width = 1000;
        this.canvas.height = 700;
        
        // Ensure CSS size matches
        this.canvas.style.width = '1000px';
        this.canvas.style.height = '700px';
        
        console.log(`✅ High resolution setup: ${this.canvas.width}x${this.canvas.height}`);
        
        // Draw a test pattern to show the canvas is working
        this.drawTestPattern();
    }
      drawTestPattern() {
        // Clear with gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
        gradient.addColorStop(0, '#001122');
        gradient.addColorStop(1, '#112233');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid pattern
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        
        const gridSize = 50;
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // Draw center indicator
        this.ctx.fillStyle = '#00ffff';
        this.ctx.font = '32px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🐍 Elite Snake Evolution Arena', this.canvas.width / 2, this.canvas.height / 2 - 20);
        
        this.ctx.font = '18px Arial';
        this.ctx.fillText('Click the buttons to start battles!', this.canvas.width / 2, this.canvas.height / 2 + 20);
        
        console.log('✅ Test pattern drawn');
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
        // For the clean test page, bind to existing buttons
        this.bindExistingButtons();
        
        console.log('✅ UI connected');
    }
    
    bindExistingButtons() {
        // Wait a bit for DOM to be ready, then bind buttons
        setTimeout(() => {
            const demoBattleBtn = document.getElementById('btn-demo-battle');
            const demoTournamentBtn = document.getElementById('btn-demo-tournament');
            const pauseBtn = document.getElementById('btn-pause');
            const resetBtn = document.getElementById('btn-reset');
            
            if (demoBattleBtn) {
                demoBattleBtn.addEventListener('click', () => {
                    console.log('🎮 Demo battle requested');
                    this.startDemoBattle();
                });
                console.log('✅ Demo battle button bound');
            }
            
            if (demoTournamentBtn) {
                demoTournamentBtn.addEventListener('click', () => {
                    console.log('🏆 Demo tournament requested');
                    this.startDemoTournament();
                });
                console.log('✅ Demo tournament button bound');
            }
            
            if (pauseBtn) {
                pauseBtn.addEventListener('click', () => {
                    console.log('⏸️ Pause/Resume requested');
                    this.togglePause();
                });
                console.log('✅ Pause button bound');
            }
            
            if (resetBtn) {
                resetBtn.addEventListener('click', () => {
                    console.log('🔄 Reset requested');
                    this.resetGame();
                });
                console.log('✅ Reset button bound');
            }
        }, 100);
    }
    
    bindEvents() {
        // Keyboard events
        document.addEventListener('keydown', this.handleKeyDown);
        
        // Window events
        window.addEventListener('resize', this.handleResize);
        document.addEventListener('visibilitychange', this.handleVisibilityChange);
        
        console.log('✅ Events bound');
    }
    
    handleKeyDown(event) {
        switch (event.code) {
            case 'Space':
                event.preventDefault();
                if (this.gameMode === 'battle') {
                    this.togglePause();
                }
                break;
            case 'Escape':
                event.preventDefault();
                this.showMainMenu();
                break;
            case 'KeyR':
                if (event.ctrlKey) {
                    event.preventDefault();
                    this.resetGame();
                }
                break;
        }
    }
    
    handleResize() {
        if (this.canvas && this.ctx) {
            this.setupHighResolution();
        }
    }
    
    handleVisibilityChange() {
        if (document.hidden && this.gameLoopRunning) {
            this.pauseGame();
        }
    }
    
    handleBattleEnd(battleDetail) {
        console.log('⚔️ Battle ended:', battleDetail);
        // Handle battle end logic
    }
    
    handleTournamentComplete(tournamentDetail) {
        console.log('🏆 Tournament completed:', tournamentDetail);
        // Handle tournament completion logic
    }
      showMainMenu() {
        this.gameMode = 'menu';
        this.stopGameLoop();
        
        // Draw the test pattern instead of plain menu
        this.drawTestPattern();
        
        console.log('📋 Main menu displayed');
    }
      startDemoBattle() {
        try {
            console.log('🎮 Starting demo battle...');
            
            // Check if systems are ready
            if (!this.snakeGenerator) {
                throw new Error('Snake generator not initialized');
            }
            if (!this.battleEngine) {
                throw new Error('Battle engine not initialized');
            }
              // Generate two demo snakes
            console.log('🐍 Generating first snake...');
            const snake1 = this.snakeGenerator.createSnake();
            console.log('🐍 Generating second snake...');
            const snake2 = this.snakeGenerator.createSnake();
            
            console.log('Generated snakes:', snake1.name, 'vs', snake2.name);
            
            // Draw the snakes on canvas for visual feedback
            this.drawTestPattern();
            
            // Draw snake info
            this.ctx.fillStyle = '#ffff00';
            this.ctx.font = '20px Arial';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(`🐍 ${snake1.name}`, 50, 100);
            this.ctx.fillText(`Species: ${snake1.species}`, 70, 130);
            this.ctx.fillText(`Skills: ${snake1.skills.length}`, 70, 160);
            
            this.ctx.textAlign = 'right';
            this.ctx.fillText(`${snake2.name} 🐍`, this.canvas.width - 50, 100);
            this.ctx.fillText(`Species: ${snake2.species}`, this.canvas.width - 70, 130);
            this.ctx.fillText(`Skills: ${snake2.skills.length}`, this.canvas.width - 70, 160);
            
            // VS indicator
            this.ctx.fillStyle = '#ff0000';
            this.ctx.font = 'bold 48px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('VS', this.canvas.width / 2, this.canvas.height / 2);
            
            // Start battle
            console.log('⚔️ Initializing battle...');
            this.gameMode = 'battle';
            this.currentBattle = this.battleEngine.startBattle(snake1, snake2);
            
            // Start game loop
            this.startGameLoop();
            
            console.log('✅ Demo battle started successfully');
            
        } catch (error) {
            console.error('❌ Failed to start demo battle:', error);
            this.showErrorMessage('Failed to start demo battle: ' + error.message);
        }
    }
    
    startDemoTournament() {
        try {
            console.log('🏆 Starting demo tournament...');
            
            // Generate tournament snakes
            const snakes = [];
            for (let i = 0; i < 8; i++) {
                snakes.push(this.snakeGenerator.createSnake());
            }
            
            console.log('Generated tournament snakes:', snakes.map(s => s.name));
            
            // Start tournament
            this.gameMode = 'tournament';
            this.currentTournament = this.tournamentSystem.createTournament(snakes);
            
            // Start first battle
            const firstMatch = this.tournamentSystem.getNextMatch(this.currentTournament);
            if (firstMatch) {
                this.currentBattle = this.battleEngine.startBattle(firstMatch.snake1, firstMatch.snake2);
                this.startGameLoop();
            }
            
            console.log('✅ Demo tournament started');
            
        } catch (error) {
            console.error('❌ Failed to start demo tournament:', error);
        }
    }
    
    startGameLoop() {
        if (!this.gameLoopRunning) {
            this.gameLoopRunning = true;
            this.performance.lastFrameTime = performance.now();
            this.gameLoop();
            console.log('🔄 Game loop started');
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
        this.performance.lastFrameTime = currentTime;
        
        // Update FPS
        this.updateFPS(deltaTime);
        
        try {
            // Clear canvas
            this.ctx.fillStyle = '#001122';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Update and render based on game mode
            switch (this.gameMode) {
                case 'battle':
                    if (this.currentBattle && this.battleEngine) {
                        this.battleEngine.update(deltaTime);
                        this.battleEngine.render();
                    }
                    break;
                case 'tournament':
                    if (this.currentBattle && this.battleEngine) {
                        this.battleEngine.update(deltaTime);
                        this.battleEngine.render();
                    }
                    break;
                default:
                    this.showMainMenu();
                    return;
            }
            
            // Render effects
            if (this.visualEffects) {
                this.visualEffects.update(deltaTime);
                this.visualEffects.render();
            }
            
        } catch (error) {
            console.error('❌ Game loop error:', error);
            this.stopGameLoop();
        }
        
        // Schedule next frame
        this.animationId = requestAnimationFrame(this.gameLoop);
    }
    
    updateFPS(deltaTime) {
        const fps = Math.round(1000 / deltaTime);
        this.performance.currentFPS = fps;
        
        // Keep FPS history for monitoring
        this.performance.fpsHistory.push(fps);
        if (this.performance.fpsHistory.length > 60) {
            this.performance.fpsHistory.shift();
        }
    }
    
    togglePause() {
        if (this.gameLoopRunning) {
            this.pauseGame();
        } else {
            this.resumeGame();
        }
    }
    
    pauseGame() {
        this.stopGameLoop();
        console.log('⏸️ Game paused');
    }
    
    resumeGame() {
        this.startGameLoop();
        console.log('▶️ Game resumed');
    }
    
    resetGame() {
        this.stopGameLoop();
        this.currentBattle = null;
        this.currentTournament = null;
        this.showMainMenu();
        console.log('🔄 Game reset');
    }
    
    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 0, 0, 0.9);
            color: white;
            padding: 20px;
            border-radius: 10px;
            z-index: 10000;
            font-family: Arial, sans-serif;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
        `;
        errorDiv.innerHTML = `
            <h3>❌ Error</h3>
            <p>${message}</p>
            <button onclick="this.parentElement.remove()" style="background: #cc0000; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Close</button>
        `;
        document.body.appendChild(errorDiv);
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
