// Main Game Engine - Controls the Snake Battle Arena
class SnakeBattleGame {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Initialize WebGL effects system with error handling
        try {
            this.webglEffects = new WebGLEffects(this.canvas);
            this.effectsEnabled = true;
            console.log('WebGL effects initialized successfully');
        } catch (error) {
            console.warn('WebGL effects failed to initialize:', error);
            this.webglEffects = null;
            this.effectsEnabled = false;
        }
        
        // Game state
        this.gameRunning = false;
        this.gamePaused = false;
        this.gameSpeed = 150; // milliseconds between updates
        this.speedMultiplier = 1;
        
        // Game objects
        this.snake1 = null;
        this.snake2 = null;
        this.food = null;
        this.powerups = [];        // Visual effects
        this.particles = [];
        this.screenShake = 0;
        this.armorEffectTime = 0;
        
        // Systems
        this.snakeGenerator = new SnakeGenerator();
        this.learningSystem = new LearningSystem();
        this.ai = new SnakeAI(this.learningSystem);
        this.tournamentSystem = new TournamentSystem(this.snakeGenerator, this.learningSystem);
        
        // Game mode
        this.gameMode = 'normal'; // 'normal' or 'tournament'
        this.currentTournamentMatch = null;
        
        // Game loop
        this.lastUpdate = 0;
        this.gameLoop = null;
        
        // UI elements
        this.initializeUI();
          // Initialize first generation
        this.initializeGame();
    }

    initializeUI() {
        // Button event listeners with error handling
        try {
            document.getElementById('startBtn').addEventListener('click', () => this.startGame());
            document.getElementById('tournamentBtn').addEventListener('click', () => this.startTournament());
            document.getElementById('pauseBtn').addEventListener('click', () => this.pauseGame());
            document.getElementById('resetBtn').addEventListener('click', () => this.resetGame());
            document.getElementById('speedBtn').addEventListener('click', () => this.toggleSpeed());
            document.getElementById('effectsBtn').addEventListener('click', () => this.toggleEffects());
            
            // Update effects button based on WebGL availability
            const effectsBtn = document.getElementById('effectsBtn');
            if (!this.webglEffects) {
                effectsBtn.textContent = '⚡ GPU Effects: N/A';
                effectsBtn.style.background = 'linear-gradient(45deg, #7f8c8d, #95a5a6)';
                effectsBtn.disabled = true;
                this.effectsEnabled = false;
            }
            
            console.log('UI initialized successfully');
        } catch (error) {
            console.error('UI initialization error:', error);
        }
    }

    initializeGame() {
        // Create initial snakes
        this.snake1 = this.createSnakeObject(this.snakeGenerator.createSnake(), 1, '#ff6b6b');
        this.snake2 = this.createSnakeObject(this.snakeGenerator.createSnake(), 2, '#4ecdc4');
        
        // Position snakes
        this.resetSnakePositions();
        
        // Create food
        this.spawnFood();
        
        // Update UI
        this.updateSnakeUI();
        
        // Start with a log entry
        this.learningSystem.addLogEntry('New generation of AI snakes entered the arena!');
    }

    createSnakeObject(snakeData, id, color) {
        return {
            ...snakeData,
            id: id,
            body: [],
            direction: 'right',
            color: color,
            speed: this.gameSpeed,
            powerups: [],
            invincible: false,
            invincibleTime: 0,
            speedBoost: false,
            speedBoostTime: 0,
            // Initialize width evolution if not present
            baseWidth: snakeData.baseWidth || 10,
            currentWidth: snakeData.currentWidth || 10,
            maxWidth: snakeData.maxWidth || 25,
            widthGrowthRate: snakeData.widthGrowthRate || 1,
            lengthCapReached: snakeData.lengthCapReached || false
        };
    }

    resetSnakePositions() {
        // Snake 1 starts on the left
        this.snake1.body = [
            { x: 100, y: this.canvas.height / 2 },
            { x: 80, y: this.canvas.height / 2 },
            { x: 60, y: this.canvas.height / 2 }
        ];
        this.snake1.direction = 'right';

        // Snake 2 starts on the right
        this.snake2.body = [
            { x: this.canvas.width - 100, y: this.canvas.height / 2 },
            { x: this.canvas.width - 80, y: this.canvas.height / 2 },
            { x: this.canvas.width - 60, y: this.canvas.height / 2 }
        ];
        this.snake2.direction = 'left';
    }

    startGame() {
        if (!this.gameRunning) {
            this.gameRunning = true;
            this.gamePaused = false;
            this.lastUpdate = performance.now();
            this.gameLoop = requestAnimationFrame(() => this.update());
            
            document.getElementById('startBtn').textContent = 'Stop';
            this.learningSystem.addLogEntry('Battle commenced!');
        } else {
            this.stopGame();
        }
    }

    stopGame() {
        this.gameRunning = false;
        this.gamePaused = false;
        if (this.gameLoop) {
            cancelAnimationFrame(this.gameLoop);
            this.gameLoop = null;
        }
        document.getElementById('startBtn').textContent = 'Start Battle';
    }

    pauseGame() {
        if (this.gameRunning) {
            this.gamePaused = !this.gamePaused;            document.getElementById('pauseBtn').textContent = this.gamePaused ? 'Resume' : 'Pause';
            
            if (!this.gamePaused) {
                this.lastUpdate = performance.now();
                this.gameLoop = requestAnimationFrame(() => this.update());
            }
        }
    }

    resetGame() {
        this.stopGame();
        
        // Reset tournament mode
        this.gameMode = 'normal';
        this.currentTournamentMatch = null;
        
        // Hide tournament UI
        document.getElementById('tournament-panel').style.display = 'none';
        document.getElementById('tournament-ladder').style.display = 'none';
        
        // Reset tournament system
        this.tournamentSystem.reset();
        
        // Get the last winner for elite breeding
        const lastWinner = this.getLastWinner();
        
        // Evolve snakes based on performance - ONLY WINNERS BREED
        const deadSnakes = this.learningSystem.getEvolutionCandidates();
        const [newSnake1, newSnake2] = this.snakeGenerator.createEvolutionPair(deadSnakes, lastWinner);
        
        // Transfer learning
        if (deadSnakes.length > 0 || lastWinner) {
            this.learningSystem.transferLearning('snake1', newSnake1);
            this.learningSystem.transferLearning('snake2', newSnake2);
        }
        
        // Create new snake objects
        this.snake1 = this.createSnakeObject(newSnake1, 1, '#ff6b6b');
        this.snake2 = this.createSnakeObject(newSnake2, 2, '#4ecdc4');
        
        // Reset positions
        this.resetSnakePositions();
        
        // Clear powerups
        this.powerups = [];
        
        // Spawn new food
        this.spawnFood();
        
        // Reset UI
        this.updateSnakeUI();
        
        // Log evolution with winners circle info
        this.logEvolutionInfo(newSnake1, newSnake2, lastWinner);
    }
    
    getLastWinner() {
        // Return the snake that has lives > 0
        if (this.snake1 && this.snake1.lives > 0) return this.snake1;
        if (this.snake2 && this.snake2.lives > 0) return this.snake2;
        return null;
    }
      logEvolutionInfo(newSnake1, newSnake2, winner) {
        this.learningSystem.addLogEntry(`🧬 EVOLUTION: Gen ${this.snakeGenerator.generation}`);
        this.learningSystem.addLogEntry(`New competitors: ${newSnake1.name} vs ${newSnake2.name}`);
        
        if (winner) {
            this.learningSystem.addLogEntry(`👑 ${winner.name} joins the Winners Circle!`);
        }
        
        // Show breeding info
        if (newSnake1.championOffspring) {
            this.learningSystem.addLogEntry(`🏆 ${newSnake1.name} is champion offspring!`);
        }
        if (newSnake2.championOffspring) {
            this.learningSystem.addLogEntry(`🏆 ${newSnake2.name} is champion offspring!`);
        }
        
        // Show winners circle status
        const winnersStatus = this.snakeGenerator.getWinnersCircleStatus();
        if (winnersStatus.length > 0) {
            this.learningSystem.addLogEntry(`🏆 Winners Circle (${winnersStatus.length}/6):`);
            winnersStatus.forEach(champ => {
                this.learningSystem.addLogEntry(`  ${champ.name} - Gen ${champ.generation}, Used ${champ.timesUsed}x, ${champ.generationsLeft} gens left`);
            });
        } else {
            this.learningSystem.addLogEntry(`Winners Circle empty - using standard breeding`);
        }
        
        if (newSnake1.isEvolved) {
            this.learningSystem.addLogEntry(`${newSnake1.name} evolved with enhanced traits`);
        }
        if (newSnake2.isEvolved) {
            this.learningSystem.addLogEntry(`${newSnake2.name} evolved with enhanced traits`);
        }
    }

    toggleSpeed() {
        const speeds = [1, 2, 4, 8];
        const currentIndex = speeds.indexOf(this.speedMultiplier);
        const nextIndex = (currentIndex + 1) % speeds.length;
        this.speedMultiplier = speeds[nextIndex];
        
        document.getElementById('speedBtn').textContent = `Speed: ${this.speedMultiplier}x`;
    }

    toggleEffects() {
        if (!this.webglEffects) {
            this.showPowerupMessage('❌ WebGL Effects not available on this device');
            return;
        }
        
        this.effectsEnabled = !this.effectsEnabled;
        const effectsBtn = document.getElementById('effectsBtn');
        
        if (this.effectsEnabled) {
            effectsBtn.textContent = '⚡ GPU Effects: ON';
            effectsBtn.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
            this.showPowerupMessage('🔥 GPU Effects Enabled - Iron Scales will look EPIC!');
        } else {
            effectsBtn.textContent = '⚡ GPU Effects: OFF';
            effectsBtn.style.background = 'linear-gradient(45deg, #7f8c8d, #95a5a6)';
            this.showPowerupMessage('💤 GPU Effects Disabled - Basic rendering mode');
        }
    }    update() {
        if (!this.gameRunning || this.gamePaused) {
            // Continue animation loop even when paused
            this.gameLoop = requestAnimationFrame(() => this.update());
            return;
        }
        
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastUpdate;
        
        if (deltaTime >= this.gameSpeed / this.speedMultiplier) {
            this.gameLogic();
            this.render();
            this.lastUpdate = currentTime;
        }
        
        this.gameLoop = requestAnimationFrame(() => this.update());
    }

    gameLogic() {
        // Update powerup timers
        this.updatePowerupTimers();
        
        // AI decisions
        const gameState = {
            canvas: this.canvas,
            snake1: this.snake1,
            snake2: this.snake2,
            food: this.food,
            powerups: this.powerups
        };
        
        this.snake1.direction = this.ai.getNextMove(this.snake1, gameState);
        this.snake2.direction = this.ai.getNextMove(this.snake2, gameState);
        
        // Move snakes
        this.moveSnake(this.snake1);
        this.moveSnake(this.snake2);
        
        // Check collisions
        this.checkCollisions();
          // Remove expired powerups
        this.powerups = this.powerups.filter(powerup => {
            const age = Date.now() - powerup.spawnTime;
            if (age > powerup.duration) {
                this.learningSystem.addLogEntry(`${powerup.type.charAt(0).toUpperCase() + powerup.type.slice(1)} powerup expired`);
                return false;
            }
            return true;
        });
        
        // Spawn powerups more frequently to encourage movement
        if (Math.random() < 0.012) { // Increased from 0.008
            this.spawnPowerup();
        }
        
        // Add random food respawn to keep snakes active
        if (Math.random() < 0.002) { // Occasionally respawn food to new location
            this.spawnFood();
            this.learningSystem.addLogEntry('Food relocated to encourage exploration!');
        }
          // Update survival time
        this.snake1.survivalTime += this.gameSpeed / this.speedMultiplier;
        this.snake2.survivalTime += this.gameSpeed / this.speedMultiplier;
          // Update UI
        this.updateSnakeUI();
        
        // Update tournament ladder if in tournament mode
        if (this.gameMode === 'tournament') {
            this.updateCurrentMatchDisplay();
        }
        
        // Update winners circle display
        if (this.learningSystem.updateWinnersCircleDisplay) {
            this.learningSystem.updateWinnersCircleDisplay();
        }
    }

    moveSnake(snake) {
        if (snake.lives <= 0) return;
        
        const head = { ...snake.body[0] };
        let moveDistance = 20;
        
        // Apply skill-based speed modifications
        if (snake.skills.includes('Speed Boost')) {
            moveDistance *= 1.2; // 20% faster
        }
        if (snake.skills.includes('Lightning Strike') && snake.body.length === 3) {
            moveDistance *= 1.5; // 50% faster on first move
        }
        
        // Apply speed boost powerup
        const actualDistance = snake.speedBoost ? moveDistance * 1.5 : moveDistance;
        
        switch (snake.direction) {
            case 'up': head.y -= actualDistance; break;
            case 'down': head.y += actualDistance; break;
            case 'left': head.x -= actualDistance; break;
            case 'right': head.x += actualDistance; break;
        }
          snake.body.unshift(head);
        
        // Handle growth - remove tail unless growing
        if (snake.growing && snake.growing > 0) {
            if (typeof snake.growing === 'number' && snake.growing > 1) {
                // Multi-segment growth
                snake.growing--;
            } else {
                // Single segment growth
                snake.growing = false;
            }
            // Don't remove tail when growing - snake gets longer
        } else {
            // Normal movement - remove tail
            snake.body.pop();
        }
    }

    checkCollisions() {
        [this.snake1, this.snake2].forEach(snake => {
            if (snake.lives <= 0) return;
            
            const head = snake.body[0];
            
            // Wall collision
            if (head.x < 0 || head.x >= this.canvas.width || 
                head.y < 0 || head.y >= this.canvas.height) {
                this.handleSnakeDeath(snake, 'wall collision');
                return;
            }
            
            // Self collision
            for (let i = 1; i < snake.body.length; i++) {
                if (this.isColliding(head, snake.body[i])) {
                    this.handleSnakeDeath(snake, 'self collision');
                    return;
                }
            }
              // Enemy collision (if not invincible)
            if (!snake.invincible) {
                const enemy = snake === this.snake1 ? this.snake2 : this.snake1;
                if (enemy && enemy.lives > 0) {
                    for (let segment of enemy.body) {
                        if (this.isColliding(head, segment)) {
                            this.handleSnakeDeath(snake, 'enemy collision');
                            
                            // Record learning
                            this.learningSystem.recordAction(
                                snake.id === 1 ? 'snake1' : 'snake2',
                                'enemyAvoidance',
                                'failure',
                                { snake: snake, position: head }
                            );
                            return;
                        }
                    }
                }
            }
            
            // Food collision
            if (this.food && this.isColliding(head, this.food)) {
                this.eatFood(snake);
            }
            
            // Powerup collision
            this.powerups.forEach((powerup, index) => {
                if (this.isColliding(head, powerup)) {
                    this.collectPowerup(snake, powerup, index);
                }
            });
        });
    }

    isColliding(pos1, pos2, tolerance = 15) {
        return Math.abs(pos1.x - pos2.x) < tolerance && Math.abs(pos1.y - pos2.y) < tolerance;
    }

    eatFood(snake) {
        // Check if snake has reached its width cap
        if (snake.lengthCapReached) {
            // Snake is at max size - only give points, no growth
            snake.score += 15; // More points since no growth
            snake.foodEaten++;
            
            this.showPowerupMessage(`${snake.name} is at maximum size! +15 points!`);
              // Apply efficient metabolism skill for max-size snakes
            if (snake.skills.includes('Efficient Metabolism')) {
                snake.score += 15; // Extra bonus points
            }
            
            // Regeneration skill effect
            if (snake.skills.includes('Regeneration') && snake.body.length < 10) {
                snake.growing = true; // Can still grow even at max width
                this.showPowerupMessage(`${snake.name} regenerated a segment!`);
            }
            
            // Magnetic Field effect
            if (snake.skills.includes('Magnetic Field')) {
                // Spawn food closer to this snake
                this.spawnFoodNear(snake.body[0]);
            }
        } else {
            // Normal growth behavior
            snake.growing = true;
            snake.score += 10;
            snake.foodEaten++;
            
            // Update width based on length
            this.updateSnakeWidth(snake);
            
            // Visual growth feedback
            this.showPowerupMessage(`${snake.name} ate food and grew! Width: ${Math.round(snake.currentWidth)}`);
            
            // Apply efficient metabolism skill
            if (snake.skills.includes('Efficient Metabolism')) {
                snake.score += 5; // Bonus points
                // Chance for double growth
                if (Math.random() < 0.3) {
                    snake.growing = 2; // Grow by 2 segments
                    this.showPowerupMessage(`${snake.name} had efficient metabolism - double growth!`);
                }
            }
        }
        
        this.spawnFood();
        
        // Record learning
        this.learningSystem.recordAction(
            snake.id === 1 ? 'snake1' : 'snake2',
            'foodSeeking',
            'success',
            { snake: snake, position: snake.body[0] }
        );
        
        this.learningSystem.addLogEntry(`${snake.name} found food! Score: ${snake.score}, Length: ${snake.body.length}, Width: ${Math.round(snake.currentWidth)}`);
    }

    updateSnakeWidth(snake) {
        const length = snake.body.length;
        
        // Calculate width based on length and growth rate
        const widthProgression = Math.floor(length / 3) * snake.widthGrowthRate;
        snake.currentWidth = Math.min(snake.baseWidth + widthProgression, snake.maxWidth);
        
        // Check if we've reached the width cap
        if (snake.currentWidth >= snake.maxWidth && !snake.lengthCapReached) {
            snake.lengthCapReached = true;
            this.learningSystem.addLogEntry(`${snake.name} reached maximum width (${snake.maxWidth}px) - growth stopped!`);
            
            // Bonus for reaching max size
            snake.score += 50;
            snake.experience += 100;
            
            // Evolution event
            this.learningSystem.recordAction(
                snake.id === 1 ? 'snake1' : 'snake2',
                'evolution',
                'success',
                { 
                    snake: snake, 
                    position: snake.body[0],
                    adaptation: `reached maximum evolution size of ${snake.maxWidth}px width`
                }
            );
        }
    }

    collectPowerup(snake, powerup, index) {
        snake.powerupsCollected++;
        snake.score += 25;
        
        switch (powerup.type) {
            case 'speed':
                snake.speedBoost = true;
                snake.speedBoostTime = 5000; // 5 seconds
                this.showPowerupMessage(`${snake.name} gained Speed Boost!`);
                break;
                
            case 'invincibility':
                snake.invincible = true;
                snake.invincibleTime = 3000; // 3 seconds
                this.showPowerupMessage(`${snake.name} is now Invincible!`);
                break;
                
            case 'score':
                snake.score += 50;
                this.showPowerupMessage(`${snake.name} gained bonus points!`);
                break;
                
            case 'shrink':
                if (snake.body.length > 3) {
                    snake.body.pop();
                    snake.body.pop();
                    this.showPowerupMessage(`${snake.name} shrank but gained agility!`);
                }
                break;
        }
        
        this.powerups.splice(index, 1);
          // Record learning
        this.learningSystem.recordAction(
            snake.id === 1 ? 'snake1' : 'snake2',
            'powerupCollection',
            'success',
            { snake: snake, position: snake.body[0] }
        );
    }

    handleSnakeDeath(snake, cause) {
        // Check for defensive skills before applying damage
        let shouldTakeDamage = true;
        
        // Iron Scales - 25% chance to completely deflect damage + visual effect
        if (snake.skills && snake.skills.includes('Iron Scales')) {
            if (Math.random() < 0.25) {
                shouldTakeDamage = false;
                this.showPowerupMessage(`⚔️ ${snake.name}'s Iron Scales deflected the ${cause}!`);
                this.screenShake = 10; // Add screen shake for dramatic effect
                
                // Create metallic deflection effect
                if (this.webglEffects && this.effectsEnabled) {
                    this.webglEffects.createDamageAbsorptionEffect(snake);
                    this.webglEffects.createIronScalesEffect(snake);
                }
                return;
            }
        }
        
        // Thick Skin - 25% chance to absorb damage
        if (snake.skills && snake.skills.includes('Thick Skin')) {
            if (Math.random() < 0.25) {
                shouldTakeDamage = false;
                this.showPowerupMessage(`🛡️ ${snake.name}'s Thick Skin absorbed the ${cause}!`);
                
                // Create protective barrier effect
                if (this.webglEffects && this.effectsEnabled) {
                    this.webglEffects.createDamageAbsorptionEffect(snake);
                }
            }
        }
        
        // Titan Blood - immunity to certain damage types
        if (snake.skills && snake.skills.includes('Titan Blood')) {
            if (cause === 'poison' || cause === 'venom' || cause.includes('status')) {
                shouldTakeDamage = false;
                this.showPowerupMessage(`⚡ ${snake.name}'s Titan Blood provides immunity to ${cause}!`);
                
                // Create lightning immunity effect
                if (this.webglEffects && this.effectsEnabled) {
                    const head = snake.body[0];
                    this.webglEffects.createLightningStrike(
                        {x: head.x, y: head.y - 50},
                        {x: head.x, y: head.y + 50}
                    );
                }
            }
        }
        
        if (!shouldTakeDamage) {
            return; // No damage taken
        }
        
        snake.lives--;
        
        // Show Iron Scales effect when damage is taken
        if (snake.skills && snake.skills.includes('Iron Scales')) {
            if (snake.lives > 0) {
                this.showPowerupMessage(`⚔️ ${snake.name}'s Iron Scales reduced the damage! ${Math.ceil(snake.lives/2)} hearts remaining`);
                
                // Create armor cracking effect
                if (this.webglEffects && this.effectsEnabled) {
                    this.webglEffects.createIronScalesEffect(snake);
                }
            } else {
                this.showPowerupMessage(`💀 ${snake.name}'s Iron Scales finally shattered!`);
                this.screenShake = 15; // Bigger shake when armor breaks
                
                // Create armor shattering effect
                if (this.webglEffects && this.effectsEnabled) {
                    const head = snake.body[0];
                    // Create explosion of metallic particles
                    for (let i = 0; i < 50; i++) {
                        const angle = Math.random() * Math.PI * 2;
                        const speed = 5 + Math.random() * 10;
                        this.webglEffects.particles.push({
                            x: head.x,
                            y: head.y,
                            vx: Math.cos(angle) * speed,
                            vy: Math.sin(angle) * speed,
                            life: 60,
                            maxLife: 60,
                            size: 3 + Math.random() * 5,
                            color: [0.8, 0.8, 0.9], // Shattered metal
                            type: 'shatter'
                        });
                    }
                }            }
        }
        
        if (snake.lives <= 0) {
            const enemy = snake === this.snake1 ? this.snake2 : this.snake1;
            if (enemy && enemy.lives > 0) {
                enemy.victories++;
                enemy.score += 100;
            }
            
            snake.defeats++;
            
            // Record battle result
            this.learningSystem.recordBattleResult(
                snake.lives <= 0 ? enemy : null,
                snake.lives <= 0 ? snake : null,
                { duration: snake.survivalTime, cause: cause }
            );
            
            this.learningSystem.addLogEntry(`${snake.name} died from ${cause}! Lives remaining: ${snake.lives}`);
            
            if (snake.lives <= 0) {
                // Game over - declare winner
                this.endGame(enemy, snake, cause);
            }
        } else {
            // Respawn
            this.respawnSnake(snake);
            this.learningSystem.addLogEntry(`${snake.name} respawned! Lives remaining: ${snake.lives}`);    }

    endGame(winner, loser, cause) {
        this.stopGame();
        
        // Record match result
        if (this.gameMode === 'tournament') {
            this.tournamentSystem.recordMatchResult(winner, loser);
            
            // Show brief victory message
            this.showPowerupMessage(`🏆 ${winner.name} wins! Advancing to next round...`);
            
            // Auto-advance tournament after short delay
            setTimeout(() => {
                this.advanceTournament();
            }, 2000);
        } else {
            // Normal game mode - show victory screen
            this.showVictoryScreen(winner, loser, cause);
        }
    }

    advanceTournament() {
        const nextMatch = this.tournamentSystem.getNextMatch();
        
        console.log('advanceTournament called, nextMatch:', nextMatch);
        this.learningSystem.addLogEntry(`🔍 DEBUG: Advancing tournament, next match: ${nextMatch ? 'Found' : 'None'}`);
        
        if (nextMatch) {
            if (nextMatch.type === 'tournament_complete') {
                // Tournament finished!
                this.learningSystem.addLogEntry('🏆 Tournament complete! Crowning champion...');
                this.completeTournament(nextMatch.winner);
            } else {
                // Next match
                this.learningSystem.addLogEntry(`🔄 Setting up next match: ${nextMatch.snake1?.name} vs ${nextMatch.snake2?.name}`);
                this.currentTournamentMatch = nextMatch;
                
                this.snake1 = this.createSnakeObject(nextMatch.snake1, 1, '#ff6b6b');
                this.snake2 = this.createSnakeObject(nextMatch.snake2, 2, '#4ecdc4');
                
                this.resetSnakePositions();
                this.spawnFood();
                this.powerups = [];
                
                this.updateTournamentUI();
                this.updateSnakeUI();
                
                // Auto-start next match
                this.learningSystem.addLogEntry('⏱️ Starting next match in 1 second...');
                setTimeout(() => {
                    this.startGame();
                }, 1000);
            }
        } else {
            this.learningSystem.addLogEntry('❌ ERROR: No next match found!');
            console.error('No next match found in tournament advancement');
        }
    }
    
    completeTournament(champion) {
        this.gameMode = 'normal';
        this.currentTournamentMatch = null;
        
        // Show epic championship screen
        this.showChampionshipVictory(champion);
        
        // Update UI
        this.updateTournamentUI();
        
        this.learningSystem.addLogEntry('🎊 TOURNAMENT COMPLETED! 🎊');
        this.learningSystem.addLogEntry(`The ultimate champion: ${champion.name}`);
        
        if (champion.uberSkills && champion.uberSkills.length > 0) {
            champion.uberSkills.forEach(skill => {
                this.learningSystem.addLogEntry(`👑 ${skill.name}: ${skill.description}`);
            });
        }
    }
    
    showChampionshipVictory(champion) {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(155, 89, 182, 0.9), rgba(142, 68, 173, 0.9));
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            color: white;
            font-family: Arial, sans-serif;
            animation: championPulse 2s infinite;
        `;
        
        const uberSkillsHTML = champion.uberSkills && champion.uberSkills.length > 0 
            ? `<div style="margin: 20px 0;">
                <h3 style="color: #FFD700;">🌟 Uber Skills Acquired 🌟</h3>
                ${champion.uberSkills.map(skill => 
                    `<div style="margin: 5px 0; padding: 10px; background: rgba(255, 215, 0, 0.2); border-radius: 5px;">
                        <strong>${skill.name}</strong> (${skill.rarity})<br>
                        <small>${skill.description}</small>
                    </div>`
                ).join('')}
            </div>`
            : '';
        
        overlay.innerHTML = `
            <div style="text-align: center; padding: 40px; background: rgba(0, 0, 0, 0.3); border-radius: 20px; backdrop-filter: blur(10px); border: 4px solid #FFD700;">
                <h1 style="font-size: 4em; margin: 0; color: #FFD700; text-shadow: 3px 3px 6px rgba(0,0,0,0.8);">👑 ULTIMATE CHAMPION! 👑</h1>
                <h2 style="font-size: 3em; margin: 20px 0; color: #ffffff; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);">${champion.name}</h2>
                <div style="margin: 30px 0;">
                    <p style="font-size: 1.5em;">Survived 40-snake tournament</p>
                    <p style="font-size: 1.2em;">Rounds Won: ${champion.roundsWon || 0}</p>
                    <p style="font-size: 1.2em;">Genes Absorbed: ${(champion.genesAbsorbed || []).length}</p>
                    <p style="font-size: 1.2em;">Final Fitness: ${Math.round(this.tournamentSystem.calculateTournamentFitness(champion))}</p>
                </div>
                ${uberSkillsHTML}
                <p style="font-size: 1em; margin: 30px 0; opacity: 0.8;">Click anywhere to continue...</p>
            </div>
        `;
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes championPulse {
                0%, 100% { filter: brightness(1); }
                50% { filter: brightness(1.2); }
            }
        `;
        document.head.appendChild(style);
        
        overlay.addEventListener('click', () => {
            document.body.removeChild(overlay);
            document.head.removeChild(style);
        });
        
        document.body.appendChild(overlay);
    }

    showVictoryScreen(winner, loser, cause) {
        // Create victory overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            color: white;
            font-family: Arial, sans-serif;
        `;
        
        overlay.innerHTML = `
            <div style="text-align: center; padding: 40px; background: rgba(255, 255, 255, 0.1); border-radius: 20px; backdrop-filter: blur(10px);">
                <h1 style="font-size: 3em; margin: 0; color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);">🏆 VICTORY! 🏆</h1>
                <h2 style="font-size: 2em; margin: 20px 0; color: ${winner.color};">${winner.name}</h2>
                <p style="font-size: 1.5em; margin: 10px 0;">defeats</p>
                <h3 style="font-size: 1.8em; margin: 20px 0; color: ${loser.color}; opacity: 0.7;">${loser.name}</h3>
                <div style="margin: 30px 0;">
                    <p style="font-size: 1.2em; margin: 10px 0;">Final Score: ${winner.score} - ${loser.score}</p>
                    <p style="font-size: 1em; margin: 10px 0; opacity: 0.8;">Eliminated by: ${cause}</p>
                    <p style="font-size: 1em; margin: 10px 0; opacity: 0.8;">Battle Duration: ${Math.round(winner.survivalTime / 1000)}s</p>
                </div>
                <p style="font-size: 1em; margin: 20px 0; opacity: 0.6;">Evolution begins in 5 seconds...</p>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Remove overlay after delay
        setTimeout(() => {
            if (document.body.contains(overlay)) {
                document.body.removeChild(overlay);
            }
        }, 5000);
    }

    respawnSnake(snake) {
        // Reset to starting position with temporary invincibility
        if (snake.id === 1) {
            snake.body = [
                { x: 100, y: this.canvas.height / 2 },
                { x: 80, y: this.canvas.height / 2 },
                { x: 60, y: this.canvas.height / 2 }
            ];
            snake.direction = 'right';
        } else {
            snake.body = [
                { x: this.canvas.width - 100, y: this.canvas.height / 2 },
                { x: this.canvas.width - 80, y: this.canvas.height / 2 },
                { x: this.canvas.width - 60, y: this.canvas.height / 2 }
            ];
            snake.direction = 'left';
        }
        
        snake.invincible = true;
        snake.invincibleTime = 2000; // 2 seconds of invincibility
    }

    updatePowerupTimers() {
        [this.snake1, this.snake2].forEach(snake => {
            if (snake.invincibleTime > 0) {
                snake.invincibleTime -= this.gameSpeed / this.speedMultiplier;
                if (snake.invincibleTime <= 0) {
                    snake.invincible = false;
                }
            }
            
            if (snake.speedBoostTime > 0) {
                snake.speedBoostTime -= this.gameSpeed / this.speedMultiplier;
                if (snake.speedBoostTime <= 0) {
                    snake.speedBoost = false;
                }
            }
        });
    }

    spawnFood() {
        const margin = 40;
        let attempts = 0;
        const maxAttempts = 50;
        
        do {
            this.food = {
                x: margin + Math.random() * (this.canvas.width - margin * 2),
                y: margin + Math.random() * (this.canvas.height - margin * 2)
            };
            
            // Make sure food doesn't spawn on snakes
            const foodCollides = [this.snake1, this.snake2].some(snake => {
                return snake && snake.body && snake.body.some(segment => this.isColliding(this.food, segment, 30));
            });
            
            attempts++;
            if (!foodCollides) break;
            
        } while (attempts < maxAttempts);
        
        // Add visual spawn effect
        if (this.food) {
            this.learningSystem.addLogEntry(`Fresh food spawned at (${Math.round(this.food.x)}, ${Math.round(this.food.y)})`);
        }
    }

    spawnPowerup() {
        if (this.powerups.length >= 4) return; // Increased max powerups from 3 to 4
        
        const types = ['speed', 'invincibility', 'score', 'shrink'];
        const type = types[Math.floor(Math.random() * types.length)];
        
        const margin = 50;
        const powerup = {
            x: margin + Math.random() * (this.canvas.width - margin * 2),
            y: margin + Math.random() * (this.canvas.height - margin * 2),
            type: type,
            spawnTime: Date.now(),
            duration: 20000 // Increased from 15 to 20 seconds
        };
        
        // Make sure powerup doesn't spawn on snakes
        const powerupCollides = [this.snake1, this.snake2].some(snake => {
            return snake && snake.body && snake.body.some(segment => this.isColliding(powerup, segment, 40));
        });
        
        if (!powerupCollides) {
            this.powerups.push(powerup);
            this.learningSystem.addLogEntry(`${type.charAt(0).toUpperCase() + type.slice(1)} powerup spawned at (${Math.round(powerup.x)}, ${Math.round(powerup.y)})!`);
        }
    }

    showPowerupMessage(message) {
        const display = document.getElementById('powerupDisplay');
        display.textContent = message;
        display.style.display = 'block';
        
        setTimeout(() => {
            display.style.display = 'none';
        }, 2000);
    }

    render() {
        // Clear canvas
        this.ctx.fillStyle = '#2c3e50';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
          // Apply screen shake effect
        if (this.screenShake > 0) {
            const shakeX = (Math.random() - 0.5) * this.screenShake;
            const shakeY = (Math.random() - 0.5) * this.screenShake;
            this.ctx.translate(shakeX, shakeY);
            this.screenShake *= 0.9; // Decay shake
            if (this.screenShake < 0.5) this.screenShake = 0;
        }
        
        // Draw grid
        this.drawGrid();
          // Draw food with pulsing effect
        if (this.food) {
            const pulse = Math.sin(Date.now() * 0.008) * 0.3 + 0.7;
            const size = 12 * pulse;
            
            // Outer glow
            this.ctx.fillStyle = 'rgba(231, 76, 60, 0.5)';
            this.ctx.beginPath();
            this.ctx.arc(this.food.x, this.food.y, size + 4, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Main food
            this.ctx.fillStyle = '#e74c3c';
            this.ctx.beginPath();
            this.ctx.arc(this.food.x, this.food.y, size, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Highlight
            this.ctx.fillStyle = '#ff6b6b';
            this.ctx.beginPath();
            this.ctx.arc(this.food.x - 3, this.food.y - 3, size * 0.4, 0, 2 * Math.PI);
            this.ctx.fill();
        }
        
        // Draw powerups
        this.powerups.forEach(powerup => {
            this.drawPowerup(powerup);
        });
          // Draw snakes (only alive ones)
        if (this.snake1.lives > 0) {
            this.drawSnake(this.snake1);
        }
        if (this.snake2.lives > 0) {
            this.drawSnake(this.snake2);
        }
        
        // Draw "ELIMINATED" text for dead snakes
        if (this.snake1.lives <= 0) {
            this.drawEliminatedText(this.snake1, 'left');
        }
        if (this.snake2.lives <= 0) {
            this.drawEliminatedText(this.snake2, 'right');
        }
        
        // Update and draw particles
        this.updateParticles();
        
        // Render WebGL effects on top
        if (this.webglEffects && this.effectsEnabled) {
            const currentTime = Date.now();
            const deltaTime = currentTime - this.lastUpdate;
            this.webglEffects.render([this.snake1, this.snake2], currentTime, deltaTime);
        }
        
        // Reset canvas transform after screen shake
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    updateParticles() {
        // Update particles if they exist
        if (this.particles && this.particles.length > 0) {
            this.particles = this.particles.filter(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;
                particle.life--;
                
                // Apply gravity to some particle types
                if (particle.type === 'shockwave' || particle.type === 'shatter') {
                    particle.vy += 0.2;
                }
                
                // Draw particle
                this.ctx.save();
                this.ctx.globalAlpha = particle.life / particle.maxLife;
                this.ctx.fillStyle = `rgb(${Math.floor(particle.color[0] * 255)}, ${Math.floor(particle.color[1] * 255)}, ${Math.floor(particle.color[2] * 255)})`;
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, particle.size, 0, 2 * Math.PI);
                this.ctx.fill();
                this.ctx.restore();
                
                return particle.life > 0;
            });
        }
    }

    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        
        for (let x = 0; x < this.canvas.width; x += 20) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += 20) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    drawSnake(snake) {
        if (snake.lives <= 0) return;
        
        // Special visual effects for passive and uber skills
        const hasUberSkills = snake.uberSkills && snake.uberSkills.length > 0;
        let glowEffect = false;
        let specialColor = null;
        let metallic = false;
        
        // Iron Scales effect - metallic blue glow
        if (snake.skills && snake.skills.includes('Iron Scales')) {
            specialColor = '#B0C4DE'; // Steel blue metallic
            glowEffect = true;
            metallic = true;
        }
        
        // Other defensive skills
        if (snake.skills && snake.skills.includes('Titan Blood')) {
            specialColor = '#FF4500'; // Orange-red
            glowEffect = true;
        }
        
        if (snake.skills && snake.skills.includes('Thick Skin') && !metallic) {
            specialColor = '#8FBC8F'; // Dark sea green
        }
        
        if (hasUberSkills) {
            // Check for specific uber skills that change appearance
            snake.uberSkills.forEach(skill => {
                switch (skill.name) {
                    case 'Genesis Blood':
                        specialColor = '#FF6B9D'; // Pink glow for regeneration
                        glowEffect = true;
                        break;
                    case 'Void Walker':
                        specialColor = '#9B59B6'; // Purple for phase abilities
                        glowEffect = true;
                        break;
                    case 'Time Predator':
                        specialColor = '#F39C12'; // Orange for time manipulation
                        glowEffect = true;
                        break;
                    case 'Alpha Genome':
                        specialColor = '#FFD700'; // Gold for perfection
                        glowEffect = true;
                        break;
                    case 'God Serpent':
                        specialColor = '#FF0000'; // Red for ultimate power
                        glowEffect = true;
                        break;
                }
            });
        }
        
        snake.body.forEach((segment, index) => {
            // Body color with uber skill effects
            let color = snake.color;
              // Apply special color effects
            if (specialColor && glowEffect) {
                const intensity = Math.sin(Date.now() * 0.005) * 0.3 + 0.4;
                color = this.blendColors(snake.color, specialColor, intensity);
            }
            
            // Special effects
            if (snake.invincible) {
                const flash = Math.sin(Date.now() * 0.01) > 0;
                color = flash ? '#ffffff' : color;
            }
            
            if (snake.speedBoost) {
                const intensity = Math.sin(Date.now() * 0.02) * 0.3 + 0.7;
                color = this.blendColors(color, '#ffff00', intensity);
            }
            
            // Iron Scales metallic effect
            if (metallic) {
                // Draw metallic base
                this.ctx.fillStyle = '#2C3E50'; // Dark base
                const size = index === 0 ? 12 : 10;
                this.ctx.fillRect(
                    segment.x - size / 2,
                    segment.y - size / 2,
                    size,
                    size
                );
                
                // Add metallic highlights
                const highlight = Math.sin(Date.now() * 0.008 + index * 0.5) * 0.3 + 0.7;
                this.ctx.fillStyle = this.blendColors('#B0C4DE', '#E6E6FA', highlight);
                this.ctx.fillRect(
                    segment.x - size / 2 + 1,
                    segment.y - size / 2 + 1,
                    size - 2,
                    size - 3
                );
                
                // Add armor ridges
                this.ctx.fillStyle = '#708090';
                if (index % 2 === 0) {
                    this.ctx.fillRect(
                        segment.x - size / 4,
                        segment.y - size / 2,
                        size / 2,
                        1
                    );
                }
            } else {
                // Draw glow effect for other special skills
                if (glowEffect && specialColor) {
                    this.ctx.shadowColor = specialColor;
                    this.ctx.shadowBlur = 15;
                }
                
                this.ctx.fillStyle = color;
                
                // Head is slightly larger
                const size = index === 0 ? 12 : 10;
                
                this.ctx.fillRect(
                    segment.x - size / 2,
                    segment.y - size / 2,
                    size,
                    size
                );
            }
            
            // Reset shadow for other elements
            this.ctx.shadowBlur = 0;
            
            // Draw eyes on head
            if (index === 0) {
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fillRect(segment.x - 3, segment.y - 3, 2, 2);
                this.ctx.fillRect(segment.x + 1, segment.y - 3, 2, 2);
                
                // Special eye effects for certain uber skills
                if (hasUberSkills) {
                    snake.uberSkills.forEach(skill => {
                        if (skill.name === 'Quantum Mind') {
                            // Glowing blue eyes for prediction
                            this.ctx.fillStyle = '#00BFFF';
                            this.ctx.fillRect(segment.x - 3, segment.y - 3, 2, 2);
                            this.ctx.fillRect(segment.x + 1, segment.y - 3, 2, 2);
                        }
                    });
                }
            }
            
            // Draw crown for champions with multiple uber skills
            if (index === 0 && hasUberSkills && snake.uberSkills.length >= 3) {
                this.ctx.fillStyle = '#FFD700';
                this.ctx.font = '16px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText('👑', segment.x, segment.y - 15);
            }
        });
    }

    drawPowerup(powerup) {
        const colors = {
            speed: '#f39c12',
            invincibility: '#9b59b6',
            score: '#f1c40f',
            shrink: '#1abc9c'
        };
        
        const symbols = {
            speed: '⚡',
            invincibility: '🛡️',
            score: '💎',
            shrink: '🔄'
        };
        
        // Pulsing effect
        const pulse = Math.sin(Date.now() * 0.005) * 0.2 + 0.8;
        const size = 15 * pulse;
        
        this.ctx.fillStyle = colors[powerup.type];
        this.ctx.fillRect(
            powerup.x - size / 2,
            powerup.y - size / 2,
            size,
            size
        );
        
        // Symbol
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(symbols[powerup.type], powerup.x, powerup.y + 4);
    }

    updateSnakeUI() {
        [this.snake1, this.snake2].forEach((snake, index) => {
            const prefix = index === 0 ? 'snake1' : 'snake2';
            
            // Name with special indicator for uber snakes
            const nameElement = document.getElementById(`${prefix}-name`);
            const hasUberSkills = snake.uberSkills && snake.uberSkills.length > 0;
            nameElement.textContent = snake.name;
            if (hasUberSkills) {
                nameElement.innerHTML = `${snake.name} <span style="color: #FFD700;">✨</span>`;
            }
              // Lives and score with Iron Scales support
            const livesElement = document.getElementById(`${prefix}-lives`);
            if (snake.skills && snake.skills.includes('Iron Scales')) {
                // Show hearts with half-heart system
                const fullHearts = Math.floor(snake.lives / 2);
                const hasHalfHeart = snake.lives % 2 === 1;
                const emptyHearts = Math.floor((snake.maxLives - snake.lives) / 2);
                
                let heartsDisplay = '❤️'.repeat(fullHearts);
                if (hasHalfHeart) heartsDisplay += '💔';
                heartsDisplay += '🖤'.repeat(emptyHearts);
                
                livesElement.innerHTML = `${heartsDisplay} <small>(${snake.lives}/${snake.maxLives})</small>`;
                livesElement.style.color = '#ff6b6b';
                livesElement.title = 'Iron Scales: Double health with half-heart system';
            } else {
                livesElement.textContent = snake.lives;
                livesElement.style.color = '';
                livesElement.title = '';
            }
            document.getElementById(`${prefix}-score`).textContent = snake.score;
            
            // Stats
            const statsContainer = document.getElementById(`${prefix}-stats`);
            statsContainer.innerHTML = Object.entries(snake.stats)
                .map(([stat, value]) => 
                    `<div class="stat-item">${stat.charAt(0).toUpperCase() + stat.slice(1)}: ${value}</div>`
                ).join('');
              // Skills with tooltips
            const skillsContainer = document.getElementById(`${prefix}-skills`);
            const skillDescriptions = this.getSkillDescriptions();
            skillsContainer.innerHTML = snake.skills
                .map(skill => {
                    const description = skillDescriptions[skill] || 'Special ability';
                    return `<span class="skill-item" title="${description}" style="cursor: help;">${skill}</span>`;
                })
                .join('');
            
            // Add Uber Skills section if present
            if (hasUberSkills) {
                const uberSkillsHtml = snake.uberSkills
                    .map(skill => {
                        const rarityColors = {
                            'legendary': '#FFD700',
                            'mythic': '#FF6B6B',
                            'cosmic': '#9B59B6'
                        };
                        const color = rarityColors[skill.rarity] || '#FFD700';
                        return `<div class="uber-skill-item" style="background: ${color}20; border: 1px solid ${color}; color: ${color}; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 0.8em;">
                            <strong>${skill.name}</strong><br>
                            <small style="opacity: 0.8;">${skill.description}</small>
                        </div>`;
                    }).join('');
                
                skillsContainer.innerHTML += `
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,215,0,0.3);">
                        <div style="color: #FFD700; font-weight: bold; margin-bottom: 5px;">⚡ Uber Skills:</div>
                        ${uberSkillsHtml}
                    </div>
                `;
            }
            
            // Pedigree
            const pedigreeContainer = document.getElementById(`${prefix}-pedigree`);
            pedigreeContainer.innerHTML = snake.pedigree
                .map(line => `<div class="generation">${line}</div>`)
                .join('');
                
            // Add genetic absorption history in tournament mode
            if (this.gameMode === 'tournament' && snake.genesAbsorbed && snake.genesAbsorbed.length > 0) {
                const absorptionHtml = snake.genesAbsorbed
                    .slice(-3) // Show last 3 absorptions
                    .map(absorption => `
                        <div class="absorption" style="background: rgba(144, 238, 144, 0.1); padding: 5px; margin: 2px 0; border-radius: 3px; font-size: 0.8em;">
                            <strong>From ${absorption.from}:</strong> ${absorption.changes.join(', ')}
                        </div>
                    `).join('');
                
                pedigreeContainer.innerHTML += `
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(144,238,144,0.3);">
                        <div style="color: #90EE90; font-weight: bold; margin-bottom: 5px;">🧬 Recent Absorptions:</div>
                        ${absorptionHtml}
                    </div>
                `;
            }
        });
    }

    blendColors(color1, color2, ratio) {
        // Simple color blending - in a real implementation you'd convert to RGB
        return color1; // Simplified for now
    }

    drawEliminatedText(snake, side) {
        this.ctx.save();
        
        // Semi-transparent background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        const x = side === 'left' ? 50 : this.canvas.width - 200;
        this.ctx.fillRect(x, this.canvas.height / 2 - 40, 150, 80);
        
        // Eliminated text
        this.ctx.fillStyle = '#ff4444';
        this.ctx.font = 'bold 20px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('ELIMINATED', x + 75, this.canvas.height / 2 - 10);
        
        this.ctx.fillStyle = snake.color;
        this.ctx.font = '16px Arial';
        this.ctx.fillText(snake.name, x + 75, this.canvas.height / 2 + 15);
        
        this.ctx.restore();
    }
    
    startTournament() {
        this.gameMode = 'tournament';
        this.currentTournamentMatch = this.tournamentSystem.initializeTournament();
        
        if (this.currentTournamentMatch) {
            // Set up the current match
            this.snake1 = this.createSnakeObject(this.currentTournamentMatch.snake1, 1, '#ff6b6b');
            this.snake2 = this.createSnakeObject(this.currentTournamentMatch.snake2, 2, '#4ecdc4');
            
            this.resetSnakePositions();
            this.spawnFood();
            this.powerups = [];
            
            // Show tournament UI
            document.getElementById('tournament-panel').style.display = 'block';
            this.updateTournamentUI();
            
            // Start the match
            this.startGame();
            
            this.learningSystem.addLogEntry(`🏟️ TOURNAMENT MATCH: ${this.currentTournamentMatch.snake1.name} vs ${this.currentTournamentMatch.snake2.name}`);
        }
    }
      updateTournamentUI() {
        const status = this.tournamentSystem.getTournamentStatus();
        const bracket = this.tournamentSystem.getBracketDisplay();
        
        const infoPanel = document.getElementById('tournament-info');
        if (status.active) {
            infoPanel.innerHTML = `
                <div><strong>Round ${status.round}/${status.maxRounds}</strong></div>
                <div>Match ${status.matchNumber}/${status.totalMatches}</div>
                <div>${status.currentBracket} survivors remaining</div>
            `;
        } else if (status.winner) {
            infoPanel.innerHTML = `
                <div style="color: #ffd700; font-weight: bold;">🏆 TOURNAMENT COMPLETE!</div>
                <div>Champion: ${status.winner.name}</div>
                <div>Uber Skills: ${(status.winner.uberSkills || []).length}</div>
            `;
        }
        
        const bracketPanel = document.getElementById('tournament-bracket');
        if (bracket && bracket.survivors) {
            bracketPanel.innerHTML = bracket.survivors
                .sort((a, b) => b.fitness - a.fitness)
                .map((snake, index) => `
                    <div style="margin: 2px 0; padding: 3px; background: rgba(255, 215, 0, ${index < 3 ? 0.2 : 0.1}); border-radius: 3px;">
                        <strong>${snake.name}</strong> (Seed #${snake.seed})<br>
                        <small>Rounds Won: ${snake.roundsWon} • Fitness: ${snake.fitness} • Uber Skills: ${snake.uberSkills}</small>
                    </div>
                `).join('');
        }
        
        if (bracket && bracket.champion) {
            bracketPanel.innerHTML = `
                <div style="text-align: center; padding: 15px; background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; border-radius: 8px; font-weight: bold;">
                    🏆 TOURNAMENT CHAMPION 🏆<br>
                    ${bracket.champion.name}<br>
                    <small>Rounds: ${bracket.champion.totalRounds} • Fitness: ${bracket.champion.finalFitness}</small><br>
                    <small>Uber Skills: ${bracket.champion.uberSkills.map(s => s.name).join(', ')}</small>
                </div>
            `;
        }
        
        // Update the main tournament ladder display
        this.updateTournamentLadder();
    }
    
    updateTournamentLadder() {
        const ladderElement = document.getElementById('tournament-ladder');
        const status = this.tournamentSystem.getTournamentStatus();
        const bracket = this.tournamentSystem.getBracketDisplay();
        
        if (this.gameMode === 'tournament' && status.active) {
            ladderElement.style.display = 'block';
            
            // Update ladder status
            const ladderStatus = document.getElementById('ladder-status');
            ladderStatus.innerHTML = `
                <div>Round ${status.round}/${status.maxRounds}</div>
                <div>Match ${status.matchNumber}/${status.totalMatches}</div>
                <div>${status.currentBracket} fighters remaining</div>
            `;
            
            // Update current match display
            this.updateCurrentMatchDisplay();
            
            // Update bracket visualization
            this.updateBracketVisualization();
            
            // Update survivors list
            this.updateSurvivorsList();
            
        } else if (this.gameMode === 'tournament' && status.winner) {
            // Tournament completed
            ladderElement.style.display = 'block';
            const ladderStatus = document.getElementById('ladder-status');
            ladderStatus.innerHTML = `
                <div style="color: #ffd700; font-weight: bold;">🏆 TOURNAMENT COMPLETE!</div>
                <div>Champion: ${status.winner.name}</div>
            `;
            
            this.updateBracketVisualization();
            this.updateSurvivorsList();
            
        } else {
            ladderElement.style.display = 'none';
        }
    }
    
    updateCurrentMatchDisplay() {
        const currentMatchDiv = document.getElementById('current-match-display');
        
        if (this.currentTournamentMatch && this.gameRunning) {
            const match = this.currentTournamentMatch;
            currentMatchDiv.innerHTML = `
                <div class="match-fighter">
                    <span style="color: #ff6b6b;">${match.snake1.name}</span>
                    <span>Score: ${this.snake1?.score || 0}</span>
                </div>
                <div class="vs-divider">⚔️ VS ⚔️</div>
                <div class="match-fighter">
                    <span style="color: #4ecdc4;">${match.snake2.name}</span>
                    <span>Score: ${this.snake2?.score || 0}</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                    Round ${match.round} • Match ${match.matchNumber}
                </div>
            `;
        } else if (this.currentTournamentMatch) {
            currentMatchDiv.innerHTML = `
                <div style="text-align: center; opacity: 0.7;">
                    Next Match: ${this.currentTournamentMatch.snake1.name} vs ${this.currentTournamentMatch.snake2.name}
                </div>
            `;
        } else {
            currentMatchDiv.innerHTML = `
                <div style="text-align: center; opacity: 0.7;">
                    No active match
                </div>
            `;
        }
    }
    
    updateBracketVisualization() {
        const bracketDiv = document.getElementById('bracket-visualization');
        const status = this.tournamentSystem.getTournamentStatus();
        
        if (!status.active && !status.winner) {
            bracketDiv.innerHTML = '<div style="text-align: center; opacity: 0.7;">Tournament not started</div>';
            return;
        }
        
        // Create bracket rounds visualization
        const rounds = [
            { name: 'Round 1', count: 40 },
            { name: 'Round 2', count: 20 },
            { name: 'Round 3', count: 10 },
            { name: 'Round 4', count: 5 },
            { name: 'Round 5', count: 3 },
            { name: 'Semifinals', count: 2 },
            { name: 'FINAL', count: 1 }
        ];
        
        let bracketHtml = '';
        
        rounds.forEach((round, index) => {
            const roundNumber = index + 1;
            const isCurrentRound = roundNumber === status.round;
            const isPastRound = roundNumber < status.round;
            const isFutureRound = roundNumber > status.round;
            
            let roundClass = '';
            if (isPastRound) roundClass = 'eliminated';
            else if (isCurrentRound) roundClass = 'survivor';
            
            bracketHtml += `
                <div class="bracket-round" data-round="${round.name}">
                    <div class="bracket-box ${roundClass}">
                        ${round.count}
                        ${isCurrentRound ? '<br><small>ACTIVE</small>' : ''}
                        ${isPastRound ? '<br><small>COMPLETE</small>' : ''}
                    </div>
                    ${index < rounds.length - 1 ? '<div class="bracket-arrow"></div>' : ''}
                </div>
            `;
        });
        
        if (status.winner) {
            bracketHtml += `
                <div style="margin-top: 20px; text-align: center; padding: 15px; background: linear-gradient(45deg, rgba(255, 215, 0, 0.3), rgba(255, 165, 0, 0.3)); border-radius: 10px; border: 2px solid #ffd700;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #ffd700;">👑 CHAMPION 👑</div>
                    <div style="font-size: 1.1em; margin-top: 5px;">${status.winner.name}</div>
                </div>
            `;
        }
        
        bracketDiv.innerHTML = bracketHtml;
    }
    
    updateSurvivorsList() {
        const survivorsDiv = document.getElementById('survivors-list');
        const bracket = this.tournamentSystem.getBracketDisplay();
        const status = this.tournamentSystem.getTournamentStatus();
        
        if (status.winner) {
            // Tournament complete - show champion
            survivorsDiv.innerHTML = `
                <div class="survivor-entry champion">
                    <div>
                        <div class="survivor-name">👑 ${status.winner.name}</div>
                        <div class="survivor-stats">TOURNAMENT CHAMPION</div>
                    </div>
                    <div>
                        ${(status.winner.uberSkills || []).length > 0 ? 
                            `<span class="uber-skill-indicator">${status.winner.uberSkills.length} Uber Skills</span>` 
                            : ''}
                    </div>
                </div>
            `;
            return;
        }
        
        if (!bracket || !bracket.survivors) {
            survivorsDiv.innerHTML = '<div style="text-align: center; opacity: 0.7;">No survivors data</div>';
            return;
        }
        
        const survivorsHtml = bracket.survivors
            .sort((a, b) => b.fitness - a.fitness)
            .map((survivor, index) => {
                const isTopThree = index < 3;
                const medals = ['🥇', '🥈', '🥉'];
                
                return `
                    <div class="survivor-entry ${isTopThree ? 'champion' : ''}">
                        <div>
                            <div class="survivor-name">
                                ${isTopThree ? medals[index] + ' ' : ''}${survivor.name}
                            </div>
                            <div class="survivor-stats">
                                Seed #${survivor.seed} • Rounds: ${survivor.roundsWon} • Fitness: ${survivor.fitness}
                            </div>
                        </div>
                        <div>
                            ${survivor.uberSkills > 0 ? 
                                `<span class="uber-skill-indicator">${survivor.uberSkills} Uber</span>` 
                                : ''}
                        </div>
                    </div>
                `;
            }).join('');
        
        survivorsDiv.innerHTML = survivorsHtml || '<div style="text-align: center; opacity: 0.7;">No survivors</div>';
    }
    
    enterFeedingMode(winner, loser, cause) {
        // Create feeding mode overlay
        const feedingOverlay = document.createElement('div');
        feedingOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(139, 0, 0, 0.9), rgba(75, 0, 130, 0.9));
            z-index: 1000;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            color: white;
        `;
        
        // Create feeding canvas
        const feedingCanvas = document.createElement('canvas');
        feedingCanvas.width = 800;
        feedingCanvas.height = 400;
        feedingCanvas.style.cssText = `
            border: 3px solid #ff4444;
            border-radius: 15px;
            box-shadow: 0 0 30px rgba(255, 68, 68, 0.8);
            background: #1a1a2e;
        `;
        
        // Information panel
        const infoPanel = document.createElement('div');
        infoPanel.style.cssText = `
            margin-top: 30px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 15px;
            max-width: 800px;
            text-align: center;
            border: 2px solid #ffd700;
        `;
        
        feedingOverlay.appendChild(feedingCanvas);
        feedingOverlay.appendChild(infoPanel);
        document.body.appendChild(feedingOverlay);
        
        // Start feeding animation
        this.animateFeeding(feedingCanvas, winner, loser, infoPanel, cause);
    }
    
    animateFeeding(canvas, winner, loser, infoPanel, cause) {
        const ctx = canvas.getContext('2d');
        let animationStep = 0;
        const totalSteps = 180; // 6 seconds at 30fps
        let currentInfo = 0;
        
        // Create loser body segments to be consumed
        const loserSegments = [...loser.body].map((segment, index) => ({
            x: segment.x,
            y: segment.y,
            consumed: false,
            consumeFrame: Math.floor(Math.random() * totalSteps * 0.6) + 30
        }));
        
        // Position winner at the loser's head
        const winnerHead = { ...loser.body[0] };
        
        // Information to display during feeding
        const feedingInfo = [
            {
                title: "🥊 BATTLE RESULT",
                content: `${winner.name} defeats ${loser.name}!\nCause: ${cause}`,
                delay: 30
            },
            {
                title: "🧬 GENETIC ABSORPTION",
                content: winner.lastAbsorption || "No beneficial genes found",
                delay: 60
            },
            {
                title: "⚡ MUTATION CHECK",
                content: winner.lastMutation || "No mutations occurred",
                delay: 90
            },
            {
                title: "📈 POWER EVOLUTION",
                content: this.generatePowerSummary(winner),
                delay: 120
            },
            {
                title: "🏆 VICTORY STATS",
                content: `Rounds Won: ${winner.roundsWon || 0}\nTotal Score: ${winner.score}\nUber Skills: ${(winner.uberSkills || []).length}`,
                delay: 150
            }
        ];
        
        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw dark battlefield
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 20) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 20) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            // Draw remaining loser segments
            loserSegments.forEach((segment, index) => {
                if (!segment.consumed && animationStep >= segment.consumeFrame) {
                    segment.consumed = true;
                    // Add consumption particle effect
                    this.createConsumptionParticles(ctx, segment.x, segment.y);
                }
                
                if (!segment.consumed) {
                    // Fading loser segments
                    const alpha = Math.max(0, 1 - (animationStep / totalSteps));
                    ctx.fillStyle = `rgba(255, 100, 100, ${alpha})`;
                    ctx.fillRect(segment.x - 10, segment.y - 10, 20, 20);
                    
                    // Death glow effect
                    const glowSize = 5 + Math.sin(animationStep * 0.3) * 3;
                    ctx.shadowColor = '#ff4444';
                    ctx.shadowBlur = glowSize;
                    ctx.fillRect(segment.x - 10, segment.y - 10, 20, 20);
                    ctx.shadowBlur = 0;
                }
            });
            
            // Draw growing winner
            const growthFactor = 1 + (animationStep / totalSteps) * 0.5;
            const segmentSize = 20 * growthFactor;
            
            // Winner head (eating)
            ctx.fillStyle = winner.color;
            ctx.shadowColor = winner.color;
                       ctx.shadowBlur = 10 + Math.sin(animationStep * 0.2) * 5;
            ctx.fillRect(winnerHead.x - segmentSize/2, winnerHead.y - segmentSize/2, segmentSize, segmentSize);
            ctx.shadowBlur = 0;
            
            // Feeding mouth animation
            if (animationStep % 20 < 10) {
                ctx.fillStyle = '#ffff00';
                ctx.fillRect(winnerHead.x - 3, winnerHead.y - 3, 6, 6);
            }
            
            // Energy absorption effect
            if (animationStep > 30) {
                this.drawEnergyAbsorption(ctx, winnerHead, animationStep);
            }
            
            // Update info panel
            feedingInfo.forEach((info, index) => {
                if (animationStep >= info.delay && currentInfo === index) {
                    infoPanel.innerHTML = `
                        <h2 style="margin: 0 0 15px 0; color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                            ${info.title}
                        </h2>
                        <div style="font-size: 1.1em; line-height: 1.4; white-space: pre-line;">
                            ${info.content}
                        </div>
                        ${winner.uberSkills && winner.uberSkills.length > 0 ? `
                            <div style="margin-top: 15px; padding: 10px; background: rgba(255, 215, 0, 0.2); border-radius: 8px;">
                                <strong style="color: #ffd700;">✨ UBER SKILLS ACQUIRED ✨</strong><br>
                                ${winner.uberSkills.map(skill => `
                                    <div style="color: ${this.getRarityColor(skill.rarity)};">• ${skill.name} (${skill.rarity})</div>
                                `).join('')}
                            </div>
                        ` : ''}
                    `;
                    currentInfo++;
                }
            });
            
            animationStep++;
              if (animationStep < totalSteps) {
                requestAnimationFrame(animate);
            } else {
                // Feeding complete - show evolution results and next match button
                this.showEvolutionResults(feedingOverlay, winner, loser, cause);
            }
        };
        
        animate();
    }
    
    createConsumptionParticles(ctx, x, y) {
        // Create particle burst effect when segment is consumed
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const distance = 15 + Math.random() * 10;
            const particleX = x + Math.cos(angle) * distance;
            const particleY = y + Math.sin(angle) * distance;
            
            ctx.fillStyle = `rgba(255, 215, 0, ${0.8 - Math.random() * 0.3})`;
            ctx.fillRect(particleX - 2, particleY - 2, 4, 4);
        }
    }
    
    drawEnergyAbsorption(ctx, center, frame) {
        // Draw energy spirals flowing into winner
        const spirals = 5;
        for (let i = 0; i < spirals; i++) {
            const baseAngle = (i / spirals) * Math.PI * 2;
            const spiralRadius = 60 - (frame % 60);
            const x = center.x + Math.cos(baseAngle + frame * 0.1) * spiralRadius;
            const y = center.y + Math.sin(baseAngle + frame * 0.1) * spiralRadius;
            
            ctx.fillStyle = `rgba(255, 215, 0, ${spiralRadius / 60})`;
            ctx.shadowColor = '#ffd700';
            ctx.shadowBlur = 5;
            ctx.fillRect(x - 3, y - 3, 6, 6);
            ctx.shadowBlur = 0;
        }
    }
    
    generatePowerSummary(winner) {
        const stats = Object.entries(winner.stats)
            .map(([key, value]) => `${key}: ${value}`)
            .join('\n');
        
        const skills = winner.skills && winner.skills.length > 0
            ? `\nSkills: ${winner.skills.join(', ')}`
            : '';
        
        return stats + skills;
    }
    
    getRarityColor(rarity) {
        switch (rarity) {
            case 'legendary': return '#ff6b35';
            case 'mythic': return '#9b59b6';
            case 'cosmic': return '#e74c3c';
            default: return '#ffd700';
        }
    }    showEvolutionResults(feedingOverlay, winner, loser, cause) {
        console.log('showEvolutionResults called, gameMode:', this.gameMode);
        console.log('currentTournamentMatch:', this.currentTournamentMatch);
        
        // Set flag to pause game logic and keep evolution results visible
        this.evolutionResultsVisible = true;
        this.evolutionOverlay = feedingOverlay;
        
        // Clear all existing content in the overlay
        feedingOverlay.innerHTML = '';
        
        // Reset overlay styling to ensure it stays visible
        feedingOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(139, 0, 0, 0.9), rgba(75, 0, 130, 0.9));
            z-index: 1000;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            color: white;
        `;
        
        // Create evolution results display
        const resultsPanel = document.createElement('div');
        resultsPanel.style.cssText = `
            padding: 30px;
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.95), rgba(30, 30, 60, 0.95));
            border-radius: 20px;
            border: 3px solid #ffd700;
            box-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
            color: white;
            font-family: Arial, sans-serif;
            max-width: 900px;
            max-height: 80vh;
            overflow-y: auto;
        `;
        
        // Generate detailed evolution report
        const evolutionReport = this.generateEvolutionReport(winner, loser, cause);
        
        // Generate button HTML based on game mode
        let buttonsHtml = '';
        if (this.gameMode === 'tournament') {
            buttonsHtml = `
                <button id="nextMatchBtn" style="
                    padding: 15px 40px;
                    font-size: 1.2em;
                    font-weight: bold;
                    background: linear-gradient(45deg, #9b59b6, #8e44ad);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(155, 89, 182, 0.4);
                    transition: all 0.3s;
                    margin-right: 15px;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(155, 89, 182, 0.6)'"
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(155, 89, 182, 0.4)'">
                    🏟️ Next Match
                </button>
                <button id="closeEvolutionBtn" style="
                    padding: 15px 40px;
                    font-size: 1.2em;
                    font-weight: bold;
                    background: linear-gradient(45deg, #64b5f6, #1976d2);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(100, 181, 246, 0.4);
                    transition: all 0.3s;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(100, 181, 246, 0.6)'"
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(100, 181, 246, 0.4)'">
                    📊 Study Evolution
                </button>
            `;
        } else {
            buttonsHtml = `
                <button id="closeEvolutionBtn" style="
                    padding: 15px 40px;
                    font-size: 1.2em;
                    font-weight: bold;
                    background: linear-gradient(45deg, #64b5f6, #1976d2);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(100, 181, 246, 0.4);
                    transition: all 0.3s;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(100, 181, 246, 0.6)'"
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(100, 181, 246, 0.4)'">
                    🔄 Continue
                </button>
            `;
        }
        
        resultsPanel.innerHTML = `
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="margin: 0; color: #ffd700; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);">
                    🧬 EVOLUTION COMPLETE 🧬
                </h1>
                <div style="font-size: 1.3em; margin-top: 10px; color: ${winner.color};">
                    ${winner.name} has consumed ${loser.name}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 30px;">
                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                    <h3 style="margin: 0 0 15px 0; color: #90EE90; text-align: center;">🧬 Genetic Absorption</h3>
                    <div style="background: rgba(144, 238, 144, 0.2); padding: 15px; border-radius: 10px; border-left: 4px solid #90EE90;">
                        ${winner.lastAbsorption ? 
                            `<div style="font-size: 1.1em; margin-bottom: 10px;"><strong>Genes Absorbed:</strong></div>
                             <div style="line-height: 1.4;">${winner.lastAbsorption}</div>` 
                            : '<div style="opacity: 0.7;">No beneficial genes found to absorb</div>'}
                    </div>
                </div>
                
                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                    <h3 style="margin: 0 0  15px 0; color: #FFD700; text-align: center;">⚡ Mutation Analysis</h3>
                    <div style="background: rgba(255, 215, 0, 0.2); padding: 15px; border-radius: 10px; border-left: 4px solid #FFD700;">
                        ${winner.lastMutation ? 
                            `<div style="font-size: 1.1em; margin-bottom: 10px;"><strong>New Mutation:</strong></div>
                             <div style="line-height: 1.4; color: #FFD700; font-weight: bold;">${winner.lastMutation}</div>` 
                            : '<div style="opacity: 0.7;">No mutations occurred during this evolution</div>'}
                    </div>
                </div>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 25px;">
                <h3 style="margin: 0 0 15px 0; color: #87CEEB; text-align: center;">📊 Evolution Summary</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4 style="margin: 0 0 10px 0; color: #ffffff;">Current Stats:</h4>
                        <div style="background: rgba(0, 0, 0, 0.3); padding: 10px; border-radius: 8px; font-family: monospace;">
                            ${Object.entries(winner.stats).map(([key, value]) => 
                                `<div>${key}: ${value}</div>`
                            ).join('')}
                        </div>
                    </div>
                    <div>
                        <h4 style="margin: 0 0 10px 0; color: #ffffff;">Skills & Abilities:</h4>
                        <div style="background: rgba(0, 0, 0, 0.3); padding: 10px; border-radius: 8px;">
                            <div><strong>Skills:</strong> ${(winner.skills || []).length > 0 ? winner.skills.join(', ') : 'None'}</div>
                            <div style="margin-top: 8px;"><strong>Uber Skills:</strong> ${(winner.uberSkills || []).length}</div>
                            ${(winner.uberSkills || []).length > 0 ? 
                                `<div style="margin-top: 5px; font-size: 0.9em;">
                                    ${winner.uberSkills.map(skill => 
                                        `<div style="color: ${this.getRarityColor(skill.rarity)};">• ${skill.name} (${skill.rarity})</div>`
                                    ).join('')}
                                </div>` 
                                : ''}
                        </div>
                    </div>
                </div>
            </div>
            
            ${this.gameMode === 'tournament' ? `
                <div style="background: rgba(155, 89, 182, 0.2); padding: 20px; border-radius: 15px; margin-bottom: 25px;">
                    <h3 style="margin: 0 0 15px 0; color: #9b59b6; text-align: center;">🏟️ Tournament Status</h3>
                    <div style="text-align: center; font-size: 1.1em;">
                        <div>Round ${this.currentTournamentMatch?.round || 'N/A'} Complete</div>
                        <div style="margin: 10px 0;">
                            Match ${this.currentTournamentMatch?.matchNumber || 'N/A'}/${this.currentTournamentMatch?.totalMatches || 'N/A'}
                        </div>
                        <div style="color: #ffd700; font-weight: bold;">
                            ${winner.name} advances to the next round!
                        </div>
                    </div>
                </div>            ` : ''}
            
            <div style="text-align: center; margin-top: 30px;">
                ${buttonsHtml}
            </div>
        `;        
        // Add the results panel to the overlay
        feedingOverlay.appendChild(resultsPanel);
        
        // Ensure overlay stays visible with additional checks
        setTimeout(() => {
            if (!document.body.contains(feedingOverlay)) {
                console.log('Overlay was removed, re-adding...');
                document.body.appendChild(feedingOverlay);
            }
            
            // Force visibility
            feedingOverlay.style.display = 'flex';
            feedingOverlay.style.visibility = 'visible';
              console.log('Evolution results should now be visible');
        }, 100);        // Add event listeners with a slight delay to ensure DOM elements exist
        setTimeout(() => {
            const nextMatchBtn = document.getElementById('nextMatchBtn');
            if (nextMatchBtn) {
                console.log('Next Match button found and adding event listener');
                nextMatchBtn.addEventListener('click', () => {
                    console.log('Next Match button clicked');
                    this.evolutionResultsVisible = false;
                    document.body.removeChild(feedingOverlay);
                    this.proceedToNextMatch(winner, loser, cause);
                });
            } else {
                console.log('Next Match button NOT found. GameMode:', this.gameMode);
                console.log('Available buttons:', feedingOverlay.querySelectorAll('button'));
            }
              const closeEvolutionBtn = document.getElementById('closeEvolutionBtn');
            if (closeEvolutionBtn) {
                console.log('Close Evolution button found and adding event listener');
                closeEvolutionBtn.addEventListener('click', () => {
                    console.log('Close Evolution button clicked');
                    this.evolutionResultsVisible = false;
                    document.body.removeChild(feedingOverlay);
                    if (this.gameMode === 'tournament') {
                        this.proceedToNextMatch(winner, loser, cause);
                    } else {
                        this.completeFeedingMode(winner, loser, cause);
                    }
                });
            } else {
                console.log('Close Evolution button NOT found');
            }
        }, 200);
        
        // Also add a keyboard shortcut as backup for tournament progression
        if (this.gameMode === 'tournament') {
            const keyHandler = (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    console.log('Keyboard shortcut used to advance tournament');
                    document.body.removeChild(feedingOverlay);
                    this.proceedToNextMatch(winner, loser, cause);
                    document.removeEventListener('keydown', keyHandler);
                }
            };
            document.addEventListener('keydown', keyHandler);
            
            // Remove event listener after 30 seconds to prevent memory leaks
            setTimeout(() => {
                document.removeEventListener('keydown', keyHandler);
            }, 30000);
        }
    }
    
    generateEvolutionReport(winner, loser, cause) {
        const report = {
            battleSummary: `${winner.name} defeated ${loser.name} by ${cause}`,
            geneticChanges: winner.lastAbsorption || 'No genetic material absorbed',
            mutations: winner.lastMutation || 'No mutations occurred',
            currentStats: { ...winner.stats },
            skills: [...(winner.skills || [])],
            uberSkills: [...(winner.uberSkills || [])],
            tournamentProgress: this.gameMode === 'tournament' ? {
                round: this.currentTournamentMatch?.round,
                match: this.currentTournamentMatch?.matchNumber,
                totalMatches: this.currentTournamentMatch?.totalMatches
            } : null
        };
        
        return report;
    }
      proceedToNextMatch(winner, loser, cause) {
        this.learningSystem.addLogEntry(`🍽️ ${winner.name} has consumed ${loser.name} and absorbed their power!`);
        this.learningSystem.addLogEntry(`Evolution complete. Proceeding to next tournament match...`);
        
        // Clear evolution results display
        this.evolutionResultsVisible = false;
        if (this.evolutionOverlay) {
            this.evolutionOverlay.remove();
            this.evolutionOverlay = null;
        }
        
        // Proceed to next match
        this.advanceTournament();
    }
    
    completeFeedingMode(winner, loser, cause) {
        if (this.gameMode === 'tournament') {
            // This should only be called if manually advancing tournament
            this.learningSystem.addLogEntry(`🍽️ ${winner.name} has consumed ${loser.name} and absorbed their power!`);
            this.advanceTournament();
        } else {
            // Normal mode - evolve and continue
            this.learningSystem.addLogEntry(`🏆 BATTLE ENDED! ${winner.name} defeats ${loser.name}!`);
            this.learningSystem.addLogEntry(`Victory Cause: ${loser.name} eliminated by ${cause}`);
            this.learningSystem.addLogEntry(`🍽️ ${winner.name} consumed their opponent and evolved!`);
            
            // Auto-evolve after feeding
            setTimeout(() => {
                this.resetGame();
            }, 1000);
        }
    }
      // Skill descriptions for tooltips with real effects
    getSkillDescriptions() {
        return {
            // Basic Skills
            'Regeneration': 'Slowly heals damage over time',
            'Speed Boost': 'Moves 20% faster than normal snakes', 
            'Thick Skin': 'Takes 25% less damage from collisions',
            'Sharp Turn': 'Can make tighter turns without slowing down',
            'Food Sense': 'Detects food from 2x the normal distance',
            'Danger Sense': 'Avoids walls and enemies more effectively',
            'Wall Hugger': 'Prefers moving along walls for safety',
            'Center Seeker': 'Gravitates toward the center of the arena',
            'Power Hunter': 'Actively seeks out power-ups',
            'Survival Instinct': 'Becomes more cautious when health is low',
            'Quick Reflex': 'Reacts 30% faster to threats',
            'Efficient Metabolism': 'Gains more energy from food',
            'Territory Control': 'Defends claimed areas aggressively',
            'Adaptive Learning': 'Gets smarter throughout the match',
            'Risk Assessment': 'Carefully evaluates dangerous moves',
            'Pattern Recognition': 'Learns enemy movement patterns',
            'Strategic Retreat': 'Knows when to back down from fights',
            'Opportunist': 'Exploits enemy mistakes effectively',
            'Better Eyes': 'Extended vision range for spotting threats',
            'Eagle Vision': 'Can see through walls and obstacles',
            'Sixth Sense': 'Predicts enemy movements intuitively',
            'Fortune Hunter': 'Increases power-up spawn rate nearby',
            
            // Advanced Skills
            'Titan Blood': 'Immunity to poison and status effects',
            'Venom Glands': 'Bite attacks poison enemies over time',
            'Camouflage': 'Briefly becomes invisible when threatened',
            'Lightning Strike': 'First attack each battle deals double damage',
            'Iron Scales': 'Double health with half-heart system (6 lives instead of 3)',
            'Hypnotic Gaze': 'Can briefly stun enemies with eye contact',
            'Coil Master': 'Can wrap around enemies to restrict movement',
            'Phase Walker': 'Can pass through walls for short periods',
            'Berserker Rage': 'Damage increases as health decreases',
            'Ancient Wisdom': 'Starts each battle with bonus intelligence',
            'Split Tail': 'Can sacrifice tail segments to escape death',
            'Magnetic Field': 'Attracts nearby food and power-ups',
            'Time Dilation': 'Slows down time during critical moments',
            'Shadow Clone': 'Creates temporary decoy when health is low',
            'Alpha Predator': 'All other snakes fear this one and keep distance'
        };
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.game = new SnakeBattleGame();
});
