// Elite Battle Engine - High-resolution snake combat system with detailed physics and effects
class EliteBattleEngine {
    constructor(canvas, ctx, visualEffects) {
        this.canvas = canvas;
        this.ctx = ctx;
        this.visualEffects = visualEffects;
        
        // High-resolution settings
        this.gameWidth = 1400;
        this.gameHeight = 800;
        this.gridSize = 8; // Smaller grid for higher resolution
        
        // Battle state
        this.isRunning = false;
        this.isPaused = false;
        this.battleId = 0;
        this.currentBattle = null;
        
        // Physics and timing
        this.lastTime = 0;
        this.deltaTime = 0;
        this.gameSpeed = 120; // Base game speed (lower = faster)
        this.frameCounter = 0;
        
        // Food system
        this.foods = [];
        this.maxFood = 15;
        this.foodSpawnRate = 0.3;
        
        // Powerups system
        this.powerups = [];
        this.maxPowerups = 5;
        this.powerupSpawnRate = 0.1;
        
        // Combat tracking
        this.combatLog = [];
        this.battleStats = {
            hits: 0,
            dodges: 0,
            skillActivations: 0,
            foodEaten: 0,
            powerupsCollected: 0,
            distance: 0
        };
        
        // Performance monitoring
        this.fps = 0;
        this.fpsCounter = 0;
        this.fpsLastTime = 0;
        
        this.init();
    }
    
    init() {
        this.setupCanvas();
        this.bindEvents();
        console.log('Elite Battle Engine initialized with high-resolution settings');
    }
    
    setupCanvas() {
        // High DPI support
        const dpr = window.devicePixelRatio || 1;
        
        this.canvas.width = this.gameWidth * dpr;
        this.canvas.height = this.gameHeight * dpr;
        
        this.canvas.style.width = this.gameWidth + 'px';
        this.canvas.style.height = this.gameHeight + 'px';
        
        this.ctx.scale(dpr, dpr);
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
    }
    
    bindEvents() {
        // Battle control events
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                this.togglePause();
            }
            if (e.code === 'KeyR') {
                e.preventDefault();
                this.restartBattle();
            }
        });
    }
    
    startBattle(snake1, snake2, battleConfig = {}) {
        this.battleId++;
        this.currentBattle = {
            id: this.battleId,
            snake1: this.initializeSnake(snake1, 'left'),
            snake2: this.initializeSnake(snake2, 'right'),
            config: {
                timeLimit: battleConfig.timeLimit || 120000, // 2 minutes
                winCondition: battleConfig.winCondition || 'elimination',
                environment: battleConfig.environment || 'arena',
                ...battleConfig
            },
            startTime: Date.now(),
            winner: null,
            reason: null
        };
        
        this.resetBattleState();
        this.isRunning = true;
        this.gameLoop();
        
        console.log(`Battle ${this.battleId} started: ${snake1.name} vs ${snake2.name}`);
        this.logCombatEvent(`Battle commenced: ${snake1.name} vs ${snake2.name}`);
        
        return this.battleId;
    }
    
    initializeSnake(snakeData, side) {
        const snake = {
            ...snakeData,
            // Position setup
            segments: [],
            direction: { x: 0, y: 0 },
            nextDirection: { x: 0, y: 0 },
            
            // Combat stats
            health: snakeData.maxHealth || 100,
            maxHealth: snakeData.maxHealth || 100,
            energy: 100,
            maxEnergy: 100,
            
            // Movement
            speed: this.calculateSpeed(snakeData),
            moveTimer: 0,
            lastMoveTime: 0,
            
            // AI state
            aiState: {
                target: null,
                strategy: 'aggressive',
                lastDecision: Date.now(),
                pathfinding: [],
                dangerLevel: 0
            },
            
            // Visual effects
            effects: [],
            particles: [],
            
            // Combat tracking
            kills: 0,
            damage: 0,
            skillCooldowns: {},
            activeSkills: [],
            
            // Evolution tracking
            experienceGained: 0,
            mutationsEarned: [],
            
            // Side positioning
            side: side
        };
        
        this.initializeSnakePosition(snake, side);
        this.initializeSnakeSkills(snake);
        
        return snake;
    }
    
    initializeSnakePosition(snake, side) {
        const centerY = Math.floor(this.gameHeight / this.gridSize / 2);
        const length = 5; // Starting length
        
        if (side === 'left') {
            const startX = 10;
            snake.direction = { x: 1, y: 0 };
            for (let i = 0; i < length; i++) {
                snake.segments.push({
                    x: startX - i,
                    y: centerY,
                    age: i
                });
            }
        } else {
            const startX = Math.floor(this.gameWidth / this.gridSize) - 10;
            snake.direction = { x: -1, y: 0 };
            for (let i = 0; i < length; i++) {
                snake.segments.push({
                    x: startX + i,
                    y: centerY,
                    age: i
                });
            }
        }
    }
    
    initializeSnakeSkills(snake) {
        snake.skillCooldowns = {};
        snake.activeSkills = [];
        
        snake.skills.forEach(skill => {
            snake.skillCooldowns[skill] = 0;
        });
    }
    
    calculateSpeed(snake) {
        let baseSpeed = this.gameSpeed;
        
        // Speed modifications based on stats and skills
        const speedStat = snake.stats.speed || 50;
        const speedMultiplier = 0.5 + (speedStat / 100);
        
        if (snake.skills.includes('Speed Boost')) {
            baseSpeed *= 0.7; // Faster (lower is faster)
        }
        if (snake.skills.includes('Lightning Reflexes')) {
            baseSpeed *= 0.8;
        }
        if (snake.skills.includes('Swift Strike')) {
            baseSpeed *= 0.75;
        }
        
        return Math.max(20, baseSpeed / speedMultiplier);
    }
    
    gameLoop(currentTime = 0) {
        if (!this.isRunning) return;
        
        this.deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;
        
        this.updateFPS(currentTime);
        
        if (!this.isPaused) {
            this.update(this.deltaTime);
        }
        
        this.render();
        
        requestAnimationFrame((time) => this.gameLoop(time));
    }
    
    update(deltaTime) {
        if (!this.currentBattle) return;
        
        this.frameCounter++;
        
        // Update snakes
        this.updateSnake(this.currentBattle.snake1, deltaTime);
        this.updateSnake(this.currentBattle.snake2, deltaTime);
        
        // Update food and powerups
        this.updateFood(deltaTime);
        this.updatePowerups(deltaTime);
        
        // Check collisions
        this.checkCollisions();
        
        // Update effects and particles
        this.updateEffects(deltaTime);
        
        // Check win conditions
        this.checkWinConditions();
        
        // Update battle timer
        this.updateBattleTimer();
    }
    
    updateSnake(snake, deltaTime) {
        // Update AI decision making
        this.updateSnakeAI(snake, deltaTime);
        
        // Update movement timer
        snake.moveTimer += deltaTime;
        
        // Move snake if enough time has passed
        if (snake.moveTimer >= snake.speed) {
            this.moveSnake(snake);
            snake.moveTimer = 0;
            snake.lastMoveTime = Date.now();
        }
        
        // Update energy and health regeneration
        this.updateSnakeVitals(snake, deltaTime);
        
        // Update skill cooldowns
        this.updateSkillCooldowns(snake, deltaTime);
        
        // Update active effects
        this.updateSnakeEffects(snake, deltaTime);
        
        // Update particles
        this.updateSnakeParticles(snake, deltaTime);
    }
    
    updateSnakeAI(snake, deltaTime) {
        const now = Date.now();
        if (now - snake.aiState.lastDecision < 50) return; // AI decision rate limiting
        
        const opponent = snake === this.currentBattle.snake1 ? this.currentBattle.snake2 : this.currentBattle.snake1;
        
        // Analyze battlefield
        const analysis = this.analyzeField(snake, opponent);
        
        // Choose strategy based on health and position
        if (snake.health < snake.maxHealth * 0.3) {
            snake.aiState.strategy = 'defensive';
        } else if (analysis.advantageScore > 20) {
            snake.aiState.strategy = 'aggressive';
        } else {
            snake.aiState.strategy = 'opportunistic';
        }
        
        // Make movement decision
        const decision = this.makeAIDecision(snake, opponent, analysis);
        this.executeAIDecision(snake, decision);
        
        snake.aiState.lastDecision = now;
    }
    
    analyzeField(snake, opponent) {
        const head = snake.segments[0];
        const opponentHead = opponent.segments[0];
        
        return {
            distanceToOpponent: Math.abs(head.x - opponentHead.x) + Math.abs(head.y - opponentHead.y),
            nearbyFood: this.findNearbyFood(head, 10),
            nearbyPowerups: this.findNearbyPowerups(head, 15),
            dangerLevel: this.calculateDangerLevel(snake, opponent),
            advantageScore: this.calculateAdvantageScore(snake, opponent),
            escapeRoutes: this.findEscapeRoutes(head),
            walls: this.getWallDistances(head)
        };
    }
    
    makeAIDecision(snake, opponent, analysis) {
        const decisions = [];
        
        // Strategic decisions based on current strategy
        switch (snake.aiState.strategy) {
            case 'aggressive':
                decisions.push(...this.getAggressiveDecisions(snake, opponent, analysis));
                break;
            case 'defensive':
                decisions.push(...this.getDefensiveDecisions(snake, opponent, analysis));
                break;
            case 'opportunistic':
                decisions.push(...this.getOpportunisticDecisions(snake, opponent, analysis));
                break;
        }
        
        // Add general decisions
        decisions.push(...this.getGeneralDecisions(snake, analysis));
        
        // Weight and select best decision
        return this.selectBestDecision(decisions, snake, analysis);
    }
    
    getAggressiveDecisions(snake, opponent, analysis) {
        const decisions = [];
        const head = snake.segments[0];
        const opponentHead = opponent.segments[0];
        
        // Chase opponent
        if (analysis.distanceToOpponent > 5) {
            const chaseDirection = this.getDirectionTowards(head, opponentHead);
            decisions.push({
                type: 'move',
                direction: chaseDirection,
                weight: 70,
                reason: 'chase_opponent'
            });
        }
        
        // Use combat skills
        if (this.canUseSkill(snake, 'Venom Strike') && analysis.distanceToOpponent < 3) {
            decisions.push({
                type: 'skill',
                skill: 'Venom Strike',
                weight: 90,
                reason: 'venom_strike_opportunity'
            });
        }
        
        // Intercept opponent's path
        const interceptPoint = this.calculateInterceptPoint(snake, opponent);
        if (interceptPoint) {
            const interceptDirection = this.getDirectionTowards(head, interceptPoint);
            decisions.push({
                type: 'move',
                direction: interceptDirection,
                weight: 60,
                reason: 'intercept_path'
            });
        }
        
        return decisions;
    }
    
    getDefensiveDecisions(snake, opponent, analysis) {
        const decisions = [];
        const head = snake.segments[0];
        
        // Avoid opponent
        if (analysis.distanceToOpponent < 8) {
            const avoidDirection = this.getDirectionAway(head, opponent.segments[0]);
            decisions.push({
                type: 'move',
                direction: avoidDirection,
                weight: 80,
                reason: 'avoid_opponent'
            });
        }
        
        // Use defensive skills
        if (this.canUseSkill(snake, 'Iron Scales') && snake.health < snake.maxHealth * 0.5) {
            decisions.push({
                type: 'skill',
                skill: 'Iron Scales',
                weight: 85,
                reason: 'defensive_boost'
            });
        }
        
        // Seek healing/food
        if (analysis.nearbyFood.length > 0) {
            const nearestFood = analysis.nearbyFood[0];
            const foodDirection = this.getDirectionTowards(head, nearestFood);
            decisions.push({
                type: 'move',
                direction: foodDirection,
                weight: 75,
                reason: 'seek_healing'
            });
        }
        
        return decisions;
    }
    
    getOpportunisticDecisions(snake, opponent, analysis) {
        const decisions = [];
        const head = snake.segments[0];
        
        // Collect powerups
        if (analysis.nearbyPowerups.length > 0) {
            const nearestPowerup = analysis.nearbyPowerups[0];
            const powerupDirection = this.getDirectionTowards(head, nearestPowerup);
            decisions.push({
                type: 'move',
                direction: powerupDirection,
                weight: 65,
                reason: 'collect_powerup'
            });
        }
        
        // Position for advantage
        const strategicPosition = this.findStrategicPosition(snake, opponent);
        if (strategicPosition) {
            const strategicDirection = this.getDirectionTowards(head, strategicPosition);
            decisions.push({
                type: 'move',
                direction: strategicDirection,
                weight: 50,
                reason: 'strategic_positioning'
            });
        }
        
        return decisions;
    }
    
    getGeneralDecisions(snake, analysis) {
        const decisions = [];
        const head = snake.segments[0];
        
        // Avoid walls
        Object.entries(analysis.walls).forEach(([direction, distance]) => {
            if (distance < 5) {
                const oppositeDirection = this.getOppositeDirection(direction);
                decisions.push({
                    type: 'move',
                    direction: this.stringToDirection(oppositeDirection),
                    weight: 40 + (5 - distance) * 10,
                    reason: 'avoid_wall'
                });
            }
        });
        
        // Avoid self-collision
        const dangerousDirections = this.getDangerousDirections(snake);
        dangerousDirections.forEach(dir => {
            decisions.push({
                type: 'avoid',
                direction: dir,
                weight: 100,
                reason: 'avoid_self_collision'
            });
        });
        
        return decisions;
    }
    
    selectBestDecision(decisions, snake, analysis) {
        if (decisions.length === 0) {
            // Default: move forward
            return {
                type: 'move',
                direction: snake.direction,
                weight: 10,
                reason: 'default_forward'
            };
        }
        
        // Filter out impossible moves
        const validDecisions = decisions.filter(decision => {
            if (decision.type === 'move') {
                return this.isValidMove(snake, decision.direction);
            }
            return true;
        });
        
        if (validDecisions.length === 0) {
            // Emergency: find any safe direction
            const safeDirections = this.getSafeDirections(snake);
            if (safeDirections.length > 0) {
                return {
                    type: 'move',
                    direction: safeDirections[0],
                    weight: 5,
                    reason: 'emergency_safe_move'
                };
            }
        }
        
        // Select highest weighted decision
        validDecisions.sort((a, b) => b.weight - a.weight);
        return validDecisions[0];
    }
    
    executeAIDecision(snake, decision) {
        if (!decision) return;
        
        switch (decision.type) {
            case 'move':
                snake.nextDirection = decision.direction;
                break;
            case 'skill':
                this.activateSkill(snake, decision.skill);
                break;
            case 'avoid':
                // Already handled in decision making
                break;
        }
        
        // Log significant decisions
        if (decision.weight > 70) {
            this.logCombatEvent(`${snake.name}: ${decision.reason} (weight: ${decision.weight})`);
        }
    }
    
    moveSnake(snake) {
        // Update direction if there's a queued direction change
        if (snake.nextDirection.x !== 0 || snake.nextDirection.y !== 0) {
            // Prevent 180-degree turns
            if (!(snake.direction.x === -snake.nextDirection.x && snake.direction.y === -snake.nextDirection.y)) {
                snake.direction = { ...snake.nextDirection };
            }
        }
        
        // Calculate new head position
        const head = snake.segments[0];
        const newHead = {
            x: head.x + snake.direction.x,
            y: head.y + snake.direction.y,
            age: 0
        };
        
        // Add new head
        snake.segments.unshift(newHead);
        
        // Age all segments
        snake.segments.forEach(segment => segment.age++);
        
        // Check if snake should grow (from eating food)
        if (!snake.shouldGrow) {
            snake.segments.pop();
        } else {
            snake.shouldGrow = false;
        }
        
        // Track distance traveled
        this.battleStats.distance++;
    }
    
    updateSnakeVitals(snake, deltaTime) {
        // Energy regeneration
        if (snake.energy < snake.maxEnergy) {
            snake.energy = Math.min(snake.maxEnergy, snake.energy + deltaTime * 0.01);
        }
        
        // Health regeneration (if has regeneration skill)
        if (snake.skills.includes('Regeneration') && snake.health < snake.maxHealth) {
            snake.health = Math.min(snake.maxHealth, snake.health + deltaTime * 0.005);
        }
        
        // Skill-based vitals updates
        if (snake.skills.includes('Metabolic Boost')) {
            snake.energy = Math.min(snake.maxEnergy, snake.energy + deltaTime * 0.005);
        }
    }
    
    updateSkillCooldowns(snake, deltaTime) {
        Object.keys(snake.skillCooldowns).forEach(skill => {
            if (snake.skillCooldowns[skill] > 0) {
                snake.skillCooldowns[skill] = Math.max(0, snake.skillCooldowns[skill] - deltaTime);
            }
        });
    }
    
    updateFood(deltaTime) {
        // Spawn new food
        if (this.foods.length < this.maxFood && Math.random() < this.foodSpawnRate * deltaTime * 0.001) {
            this.spawnFood();
        }
        
        // Update existing food (could add decay, movement, etc.)
        this.foods.forEach(food => {
            food.age = (food.age || 0) + deltaTime;
        });
        
        // Remove old food
        this.foods = this.foods.filter(food => (food.age || 0) < 30000); // 30 seconds max age
    }
    
    spawnFood() {
        const gridWidth = Math.floor(this.gameWidth / this.gridSize);
        const gridHeight = Math.floor(this.gameHeight / this.gridSize);
        
        let attempts = 0;
        let position;
        
        do {
            position = {
                x: Math.floor(Math.random() * (gridWidth - 4)) + 2,
                y: Math.floor(Math.random() * (gridHeight - 4)) + 2
            };
            attempts++;
        } while (this.isPositionOccupied(position) && attempts < 50);
        
        if (attempts < 50) {
            this.foods.push({
                ...position,
                type: 'normal',
                value: 10,
                age: 0
            });
        }
    }
    
    updatePowerups(deltaTime) {
        // Spawn new powerups
        if (this.powerups.length < this.maxPowerups && Math.random() < this.powerupSpawnRate * deltaTime * 0.001) {
            this.spawnPowerup();
        }
        
        // Update existing powerups
        this.powerups.forEach(powerup => {
            powerup.age = (powerup.age || 0) + deltaTime;
            powerup.pulsePhase = (powerup.pulsePhase || 0) + deltaTime * 0.005;
        });
        
        // Remove old powerups
        this.powerups = this.powerups.filter(powerup => (powerup.age || 0) < 20000); // 20 seconds max age
    }
    
    spawnPowerup() {
        const gridWidth = Math.floor(this.gameWidth / this.gridSize);
        const gridHeight = Math.floor(this.gameHeight / this.gridSize);
        
        let attempts = 0;
        let position;
        
        do {
            position = {
                x: Math.floor(Math.random() * (gridWidth - 4)) + 2,
                y: Math.floor(Math.random() * (gridHeight - 4)) + 2
            };
            attempts++;
        } while (this.isPositionOccupied(position) && attempts < 50);
        
        if (attempts < 50) {
            const powerupTypes = ['speed', 'strength', 'health', 'energy', 'skill'];
            const type = powerupTypes[Math.floor(Math.random() * powerupTypes.length)];
            
            this.powerups.push({
                ...position,
                type: type,
                age: 0,
                pulsePhase: 0
            });
        }
    }
    
    checkCollisions() {
        if (!this.currentBattle) return;
        
        const snake1 = this.currentBattle.snake1;
        const snake2 = this.currentBattle.snake2;
        
        // Snake vs food collisions
        this.checkFoodCollisions(snake1);
        this.checkFoodCollisions(snake2);
        
        // Snake vs powerup collisions
        this.checkPowerupCollisions(snake1);
        this.checkPowerupCollisions(snake2);
        
        // Snake vs wall collisions
        this.checkWallCollisions(snake1);
        this.checkWallCollisions(snake2);
        
        // Snake vs snake collisions
        this.checkSnakeCollisions(snake1, snake2);
        
        // Snake vs self collisions
        this.checkSelfCollisions(snake1);
        this.checkSelfCollisions(snake2);
    }
    
    checkFoodCollisions(snake) {
        const head = snake.segments[0];
        
        for (let i = this.foods.length - 1; i >= 0; i--) {
            const food = this.foods[i];
            if (head.x === food.x && head.y === food.y) {
                // Food consumed
                this.consumeFood(snake, food);
                this.foods.splice(i, 1);
                this.battleStats.foodEaten++;
                
                this.logCombatEvent(`${snake.name} consumed food (+${food.value} health)`);
                break;
            }
        }
    }
    
    checkPowerupCollisions(snake) {
        const head = snake.segments[0];
        
        for (let i = this.powerups.length - 1; i >= 0; i--) {
            const powerup = this.powerups[i];
            if (head.x === powerup.x && head.y === powerup.y) {
                // Powerup collected
                this.collectPowerup(snake, powerup);
                this.powerups.splice(i, 1);
                this.battleStats.powerupsCollected++;
                
                this.logCombatEvent(`${snake.name} collected ${powerup.type} powerup`);
                break;
            }
        }
    }
    
    consumeFood(snake, food) {
        // Heal snake
        snake.health = Math.min(snake.maxHealth, snake.health + food.value);
        
        // Grow snake
        snake.shouldGrow = true;
        
        // Add visual effect
        this.visualEffects.addFoodConsumptionEffect(
            food.x * this.gridSize + this.gridSize / 2,
            food.y * this.gridSize + this.gridSize / 2,
            snake.appearance.primaryColor
        );
        
        // Experience gain
        snake.experienceGained += 5;
    }
    
    collectPowerup(snake, powerup) {
        switch (powerup.type) {
            case 'speed':
                snake.speed = Math.max(20, snake.speed * 0.8); // Temporary speed boost
                snake.effects.push({
                    type: 'speed_boost',
                    duration: 5000,
                    startTime: Date.now()
                });
                break;
            case 'strength':
                snake.effects.push({
                    type: 'strength_boost',
                    duration: 8000,
                    startTime: Date.now(),
                    multiplier: 1.5
                });
                break;
            case 'health':
                snake.health = Math.min(snake.maxHealth, snake.health + 30);
                break;
            case 'energy':
                snake.energy = snake.maxEnergy;
                break;
            case 'skill':
                // Temporarily reduce all skill cooldowns
                Object.keys(snake.skillCooldowns).forEach(skill => {
                    snake.skillCooldowns[skill] = Math.max(0, snake.skillCooldowns[skill] - 3000);
                });
                break;
        }
        
        // Add visual effect
        this.visualEffects.addPowerupCollectionEffect(
            powerup.x * this.gridSize + this.gridSize / 2,
            powerup.y * this.gridSize + this.gridSize / 2,
            powerup.type
        );
        
        // Experience gain
        snake.experienceGained += 10;
    }
    
    checkWallCollisions(snake) {
        const head = snake.segments[0];
        const gridWidth = Math.floor(this.gameWidth / this.gridSize);
        const gridHeight = Math.floor(this.gameHeight / this.gridSize);
        
        if (head.x < 0 || head.x >= gridWidth || head.y < 0 || head.y >= gridHeight) {
            this.damageSnake(snake, 25, 'wall_collision');
            this.visualEffects.addWallCollisionEffect(
                head.x * this.gridSize + this.gridSize / 2,
                head.y * this.gridSize + this.gridSize / 2
            );
        }
    }
    
    checkSnakeCollisions(snake1, snake2) {
        const head1 = snake1.segments[0];
        const head2 = snake2.segments[0];
        
        // Head-to-head collision
        if (head1.x === head2.x && head1.y === head2.y) {
            this.handleHeadToHeadCollision(snake1, snake2);
            return;
        }
        
        // Snake 1 head vs Snake 2 body
        for (let i = 1; i < snake2.segments.length; i++) {
            const segment = snake2.segments[i];
            if (head1.x === segment.x && head1.y === segment.y) {
                this.damageSnake(snake1, 30, 'body_collision');
                this.logCombatEvent(`${snake1.name} collided with ${snake2.name}'s body`);
                break;
            }
        }
        
        // Snake 2 head vs Snake 1 body
        for (let i = 1; i < snake1.segments.length; i++) {
            const segment = snake1.segments[i];
            if (head2.x === segment.x && head2.y === segment.y) {
                this.damageSnake(snake2, 30, 'body_collision');
                this.logCombatEvent(`${snake2.name} collided with ${snake1.name}'s body`);
                break;
            }
        }
    }
    
    handleHeadToHeadCollision(snake1, snake2) {
        const damage1 = this.calculateCollisionDamage(snake1, snake2);
        const damage2 = this.calculateCollisionDamage(snake2, snake1);
        
        this.damageSnake(snake1, damage2, 'head_collision');
        this.damageSnake(snake2, damage1, 'head_collision');
        
        this.visualEffects.addHeadCollisionEffect(
            snake1.segments[0].x * this.gridSize + this.gridSize / 2,
            snake1.segments[0].y * this.gridSize + this.gridSize / 2
        );
        
        this.logCombatEvent(`Head-to-head collision: ${snake1.name} vs ${snake2.name}`);
        this.battleStats.hits++;
    }
    
    calculateCollisionDamage(attacker, defender) {
        let baseDamage = 20;
        
        // Factor in stats
        const attackerStrength = attacker.stats.strength || 50;
        const defenderEndurance = defender.stats.endurance || 50;
        
        baseDamage = baseDamage * (attackerStrength / 100) * (100 / (defenderEndurance + 50));
        
        // Factor in skills
        if (attacker.skills.includes('Venom Strike')) {
            baseDamage *= 1.3;
        }
        if (attacker.skills.includes('Iron Scales')) {
            baseDamage *= 1.2;
        }
        if (defender.skills.includes('Thick Skin')) {
            baseDamage *= 0.7;
        }
        if (defender.skills.includes('Damage Reduction')) {
            baseDamage *= 0.8;
        }
        
        // Factor in active effects
        attacker.effects.forEach(effect => {
            if (effect.type === 'strength_boost') {
                baseDamage *= effect.multiplier || 1.5;
            }
        });
        
        return Math.floor(baseDamage);
    }
    
    checkSelfCollisions(snake) {
        const head = snake.segments[0];
        
        for (let i = 1; i < snake.segments.length; i++) {
            const segment = snake.segments[i];
            if (head.x === segment.x && head.y === segment.y) {
                this.damageSnake(snake, 40, 'self_collision');
                this.logCombatEvent(`${snake.name} collided with itself`);
                break;
            }
        }
    }
    
    damageSnake(snake, damage, source) {
        // Apply damage reduction from skills/effects
        let actualDamage = damage;
        
        if (snake.skills.includes('Iron Scales')) {
            actualDamage = Math.floor(actualDamage * 0.5); // 50% damage reduction
        }
        if (snake.skills.includes('Thick Skin')) {
            actualDamage = Math.floor(actualDamage * 0.8); // 20% damage reduction
        }
        
        snake.health = Math.max(0, snake.health - actualDamage);
        
        // Add damage visual effect
        this.visualEffects.addDamageEffect(
            snake.segments[0].x * this.gridSize + this.gridSize / 2,
            snake.segments[0].y * this.gridSize + this.gridSize / 2,
            actualDamage
        );
        
        // Screen shake for significant damage
        if (actualDamage > 20) {
            this.visualEffects.addScreenShake(actualDamage * 0.5);
        }
        
        this.logCombatEvent(`${snake.name} took ${actualDamage} damage from ${source}`);
    }
    
    updateEffects(deltaTime) {
        if (!this.currentBattle) return;
        
        [this.currentBattle.snake1, this.currentBattle.snake2].forEach(snake => {
            // Update snake effects
            snake.effects = snake.effects.filter(effect => {
                const elapsed = Date.now() - effect.startTime;
                return elapsed < effect.duration;
            });
            
            // Update particles
            snake.particles = snake.particles.filter(particle => {
                particle.life -= deltaTime;
                particle.x += particle.vx * deltaTime * 0.001;
                particle.y += particle.vy * deltaTime * 0.001;
                particle.alpha = Math.max(0, particle.life / particle.maxLife);
                return particle.life > 0;
            });
        });
        
        // Update visual effects
        this.visualEffects.update(deltaTime);
    }
    
    checkWinConditions() {
        if (!this.currentBattle || this.currentBattle.winner) return;
        
        const snake1 = this.currentBattle.snake1;
        const snake2 = this.currentBattle.snake2;
        
        // Health-based elimination
        if (snake1.health <= 0 && snake2.health <= 0) {
            // Tie - longer snake wins, or random if same length
            if (snake1.segments.length > snake2.segments.length) {
                this.endBattle(snake1, 'elimination_tie_length');
            } else if (snake2.segments.length > snake1.segments.length) {
                this.endBattle(snake2, 'elimination_tie_length');
            } else {
                const winner = Math.random() < 0.5 ? snake1 : snake2;
                this.endBattle(winner, 'elimination_tie_random');
            }
        } else if (snake1.health <= 0) {
            this.endBattle(snake2, 'elimination');
        } else if (snake2.health <= 0) {
            this.endBattle(snake1, 'elimination');
        }
        
        // Time limit
        const elapsed = Date.now() - this.currentBattle.startTime;
        if (elapsed >= this.currentBattle.config.timeLimit) {
            // Judge by health percentage, then length, then random
            const health1Pct = snake1.health / snake1.maxHealth;
            const health2Pct = snake2.health / snake2.maxHealth;
            
            if (health1Pct > health2Pct) {
                this.endBattle(snake1, 'time_limit_health');
            } else if (health2Pct > health1Pct) {
                this.endBattle(snake2, 'time_limit_health');
            } else if (snake1.segments.length > snake2.segments.length) {
                this.endBattle(snake1, 'time_limit_length');
            } else if (snake2.segments.length > snake1.segments.length) {
                this.endBattle(snake2, 'time_limit_length');
            } else {
                const winner = Math.random() < 0.5 ? snake1 : snake2;
                this.endBattle(winner, 'time_limit_random');
            }
        }
    }
    
    endBattle(winner, reason) {
        this.currentBattle.winner = winner;
        this.currentBattle.reason = reason;
        this.currentBattle.endTime = Date.now();
        this.currentBattle.duration = this.currentBattle.endTime - this.currentBattle.startTime;
        
        this.isRunning = false;
        
        this.logCombatEvent(`Battle ended: ${winner.name} wins by ${reason}`);
        
        // Trigger battle end effects
        this.visualEffects.addVictoryEffect(
            winner.segments[0].x * this.gridSize + this.gridSize / 2,
            winner.segments[0].y * this.gridSize + this.gridSize / 2,
            winner.appearance.primaryColor
        );
        
        // Fire battle end event
        document.dispatchEvent(new CustomEvent('battleEnd', {
            detail: {
                battle: this.currentBattle,
                winner: winner,
                reason: reason,
                stats: this.battleStats
            }
        }));
    }
    
    // Helper methods
    findNearbyFood(position, radius) {
        return this.foods.filter(food => {
            const distance = Math.abs(food.x - position.x) + Math.abs(food.y - position.y);
            return distance <= radius;
        }).sort((a, b) => {
            const distA = Math.abs(a.x - position.x) + Math.abs(a.y - position.y);
            const distB = Math.abs(b.x - position.x) + Math.abs(b.y - position.y);
            return distA - distB;
        });
    }
    
    findNearbyPowerups(position, radius) {
        return this.powerups.filter(powerup => {
            const distance = Math.abs(powerup.x - position.x) + Math.abs(powerup.y - position.y);
            return distance <= radius;
        }).sort((a, b) => {
            const distA = Math.abs(a.x - position.x) + Math.abs(a.y - position.y);
            const distB = Math.abs(b.x - position.x) + Math.abs(b.y - position.y);
            return distA - distB;
        });
    }
    
    getDirectionTowards(from, to) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        
        if (Math.abs(dx) > Math.abs(dy)) {
            return { x: dx > 0 ? 1 : -1, y: 0 };
        } else {
            return { x: 0, y: dy > 0 ? 1 : -1 };
        }
    }
    
    getDirectionAway(from, away) {
        const dx = from.x - away.x;
        const dy = from.y - away.y;
        
        if (Math.abs(dx) > Math.abs(dy)) {
            return { x: dx > 0 ? 1 : -1, y: 0 };
        } else {
            return { x: 0, y: dy > 0 ? 1 : -1 };
        }
    }
    
    isValidMove(snake, direction) {
        const head = snake.segments[0];
        const newPos = {
            x: head.x + direction.x,
            y: head.y + direction.y
        };
        
        // Check bounds
        const gridWidth = Math.floor(this.gameWidth / this.gridSize);
        const gridHeight = Math.floor(this.gameHeight / this.gridSize);
        
        if (newPos.x < 0 || newPos.x >= gridWidth || newPos.y < 0 || newPos.y >= gridHeight) {
            return false;
        }
        
        // Check self-collision
        for (let i = 1; i < snake.segments.length; i++) {
            const segment = snake.segments[i];
            if (newPos.x === segment.x && newPos.y === segment.y) {
                return false;
            }
        }
        
        return true;
    }
    
    getSafeDirections(snake) {
        const directions = [
            { x: 0, y: -1 }, // up
            { x: 1, y: 0 },  // right
            { x: 0, y: 1 },  // down
            { x: -1, y: 0 }  // left
        ];
        
        return directions.filter(dir => this.isValidMove(snake, dir));
    }
    
    isPositionOccupied(position) {
        // Check if position is occupied by snakes
        if (this.currentBattle) {
            const allSegments = [
                ...this.currentBattle.snake1.segments,
                ...this.currentBattle.snake2.segments
            ];
            
            return allSegments.some(segment => 
                segment.x === position.x && segment.y === position.y
            );
        }
        
        return false;
    }
    
    canUseSkill(snake, skillName) {
        return snake.skills.includes(skillName) && 
               snake.skillCooldowns[skillName] <= 0 && 
               snake.energy >= 20; // Basic energy cost
    }
    
    activateSkill(snake, skillName) {
        if (!this.canUseSkill(snake, skillName)) return false;
        
        snake.energy -= 20; // Basic energy cost
        snake.skillCooldowns[skillName] = this.getSkillCooldown(skillName);
        
        this.battleStats.skillActivations++;
        
        // Skill-specific effects would be handled here
        this.logCombatEvent(`${snake.name} activated ${skillName}`);
        
        return true;
    }
    
    getSkillCooldown(skillName) {
        const cooldowns = {
            'Iron Scales': 8000,
            'Venom Strike': 5000,
            'Speed Boost': 6000,
            'Lightning Reflexes': 4000,
            'Regeneration': 10000
        };
        
        return cooldowns[skillName] || 5000;
    }
    
    updateFPS(currentTime) {
        this.fpsCounter++;
        if (currentTime - this.fpsLastTime >= 1000) {
            this.fps = this.fpsCounter;
            this.fpsCounter = 0;
            this.fpsLastTime = currentTime;
        }
    }
    
    updateBattleTimer() {
        // Update UI elements, check time limits, etc.
        const elapsed = Date.now() - this.currentBattle.startTime;
        const remaining = Math.max(0, this.currentBattle.config.timeLimit - elapsed);
        
        // Update timer display
        document.dispatchEvent(new CustomEvent('battleTimer', {
            detail: {
                elapsed: elapsed,
                remaining: remaining,
                total: this.currentBattle.config.timeLimit
            }
        }));
    }
    
    logCombatEvent(message) {
        this.combatLog.push({
            time: Date.now(),
            message: message
        });
        
        // Keep log size manageable
        if (this.combatLog.length > 100) {
            this.combatLog.shift();
        }
        
        console.log(`[Combat] ${message}`);
    }
    
    resetBattleState() {
        this.foods = [];
        this.powerups = [];
        this.combatLog = [];
        this.battleStats = {
            hits: 0,
            dodges: 0,
            skillActivations: 0,
            foodEaten: 0,
            powerupsCollected: 0,
            distance: 0
        };
        this.frameCounter = 0;
    }
    
    togglePause() {
        this.isPaused = !this.isPaused;
        console.log(`Battle ${this.isPaused ? 'paused' : 'resumed'}`);
    }
    
    restartBattle() {
        if (this.currentBattle) {
            this.startBattle(
                this.currentBattle.snake1,
                this.currentBattle.snake2,
                this.currentBattle.config
            );
        }
    }
    
    updateSnakeEffects(snake, deltaTime) {
        if (!snake.activeEffects) {
            snake.activeEffects = [];
        }
        
        // Update active effects
        snake.activeEffects = snake.activeEffects.filter(effect => {
            effect.duration -= deltaTime;
            
            // Apply effect based on type
            switch (effect.type) {
                case 'speed_boost':
                    snake.speed = snake.baseSpeed * effect.multiplier;
                    break;
                case 'damage_boost':
                    snake.damage = snake.baseDamage * effect.multiplier;
                    break;
                case 'regeneration':
                    snake.health = Math.min(snake.maxHealth, snake.health + effect.amount * deltaTime * 0.001);
                    break;
                case 'poison':
                    snake.health = Math.max(0, snake.health - effect.amount * deltaTime * 0.001);
                    break;
                case 'shield':
                    snake.shielded = true;
                    break;
            }
            
            // Remove expired effects
            if (effect.duration <= 0) {
                this.removeEffect(snake, effect);
                return false;
            }
            return true;
        });
    }
    
    updateSnakeParticles(snake, deltaTime) {
        if (!snake.particles) {
            snake.particles = [];
        }
        
        // Update existing particles
        snake.particles = snake.particles.filter(particle => {
            particle.life -= deltaTime;
            particle.x += particle.vx * deltaTime * 0.001;
            particle.y += particle.vy * deltaTime * 0.001;
            particle.alpha = particle.life / particle.maxLife;
            
            return particle.life > 0;
        });
        
        // Add trail particles for moving snakes
        if (snake.velocity && (Math.abs(snake.velocity.x) > 0.1 || Math.abs(snake.velocity.y) > 0.1)) {
            if (Math.random() < 0.3) {
                snake.particles.push({
                    x: snake.x + (Math.random() - 0.5) * 20,
                    y: snake.y + (Math.random() - 0.5) * 20,
                    vx: (Math.random() - 0.5) * 50,
                    vy: (Math.random() - 0.5) * 50,
                    life: 500,
                    maxLife: 500,
                    alpha: 1,
                    color: snake.appearance?.primaryColor || snake.color || '#ffffff',
                    size: Math.random() * 3 + 1
                });
            }
        }
    }
    
    removeEffect(snake, effect) {
        // Reset properties when effect expires
        switch (effect.type) {
            case 'speed_boost':
                snake.speed = snake.baseSpeed || snake.stats?.speed || 1;
                break;
            case 'damage_boost':
                snake.damage = snake.baseDamage || snake.stats?.attack || 10;
                break;
            case 'shield':
                snake.shielded = false;
                break;
        }
    }
    
    render() {
        // Clear canvas with high-quality settings
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.gameWidth, this.gameHeight);
        
        // Render game elements
        this.renderBackground();
        this.renderFood();
        this.renderPowerups();
        
        if (this.currentBattle) {
            this.renderSnake(this.currentBattle.snake1);
            this.renderSnake(this.currentBattle.snake2);
        }
        
        // Render effects and particles
        this.visualEffects.render(this.ctx);
        
        // Render UI elements
        this.renderUI();
    }
    
    renderBackground() {
        // Subtle grid pattern
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 0.5;
        
        const gridWidth = Math.floor(this.gameWidth / this.gridSize);
        const gridHeight = Math.floor(this.gameHeight / this.gridSize);
        
        for (let x = 0; x <= gridWidth; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(x * this.gridSize, 0);
            this.ctx.lineTo(x * this.gridSize, this.gameHeight);
            this.ctx.stroke();
        }
        
        for (let y = 0; y <= gridHeight; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y * this.gridSize);
            this.ctx.lineTo(this.gameWidth, y * this.gridSize);
            this.ctx.stroke();
        }
    }
    
    renderFood() {
        this.foods.forEach(food => {
            const x = food.x * this.gridSize;
            const y = food.y * this.gridSize;
            
            // Pulsing food effect
            const pulse = Math.sin(Date.now() * 0.005) * 0.3 + 0.7;
            const size = this.gridSize * 0.6 * pulse;
            
            this.ctx.fillStyle = '#ff6b6b';
            this.ctx.shadowColor = '#ff6b6b';
            this.ctx.shadowBlur = 10;
            
            this.ctx.beginPath();
            this.ctx.arc(
                x + this.gridSize / 2,
                y + this.gridSize / 2,
                size / 2,
                0,
                Math.PI * 2
            );
            this.ctx.fill();
            
            this.ctx.shadowBlur = 0;
        });
    }
    
    renderPowerups() {
        this.powerups.forEach(powerup => {
            const x = powerup.x * this.gridSize;
            const y = powerup.y * this.gridSize;
            
            // Rotating powerup effect
            const rotation = Date.now() * 0.002;
            const pulse = Math.sin(powerup.pulsePhase) * 0.2 + 0.8;
            
            this.ctx.save();
            this.ctx.translate(x + this.gridSize / 2, y + this.gridSize / 2);
            this.ctx.rotate(rotation);
            this.ctx.scale(pulse, pulse);
            
            // Different colors for different powerup types
            const colors = {
                speed: '#00ffff',
                strength: '#ff4444',
                health: '#44ff44',
                energy: '#ffff44',
                skill: '#ff44ff'
            };
            
            this.ctx.fillStyle = colors[powerup.type] || '#ffffff';
            this.ctx.shadowColor = colors[powerup.type] || '#ffffff';
            this.ctx.shadowBlur = 15;
            
            // Draw diamond shape
            const size = this.gridSize * 0.4;
            this.ctx.beginPath();
            this.ctx.moveTo(0, -size);
            this.ctx.lineTo(size, 0);
            this.ctx.lineTo(0, size);
            this.ctx.lineTo(-size, 0);
            this.ctx.closePath();
            this.ctx.fill();
            
            this.ctx.shadowBlur = 0;
            this.ctx.restore();
        });
    }
    
    renderSnake(snake) {
        if (!snake || !snake.segments || snake.segments.length === 0) return;
        
        snake.segments.forEach((segment, index) => {
            const x = segment.x * this.gridSize;
            const y = segment.y * this.gridSize;
            const isHead = index === 0;
            
            // Progressive size reduction for body segments
            const sizeFactor = isHead ? 1 : Math.max(0.6, 1 - (index * 0.02));
            const segmentSize = this.gridSize * 0.9 * sizeFactor;
            
            // Color variation along body
            let color = snake.appearance.primaryColor;
            if (!isHead) {
                const alpha = Math.max(0.4, 1 - (index * 0.03));
                color = this.hexToRgba(snake.appearance.primaryColor, alpha);
            }
            
            this.ctx.fillStyle = color;
            
            // Add glow effect for heads
            if (isHead) {
                this.ctx.shadowColor = snake.appearance.primaryColor;
                this.ctx.shadowBlur = 12;
            }
            
            // Render segment
            this.ctx.fillRect(
                x + (this.gridSize - segmentSize) / 2,
                y + (this.gridSize - segmentSize) / 2,
                segmentSize,
                segmentSize
            );
            
            // Special effects for head
            if (isHead) {
                this.renderSnakeHead(snake, x, y, segmentSize);
            }
            
            this.ctx.shadowBlur = 0;
        });
        
        // Render snake effects
        this.renderSnakeEffects(snake);
    }
    
    renderSnakeHead(snake, x, y, size) {
        const centerX = x + this.gridSize / 2;
        const centerY = y + this.gridSize / 2;
        
        // Eyes
        this.ctx.fillStyle = '#ffffff';
        const eyeSize = size * 0.15;
        const eyeOffset = size * 0.25;
        
        // Eye positions based on direction
        let eyeX1, eyeY1, eyeX2, eyeY2;
        
        if (snake.direction.x !== 0) {
            // Moving horizontally
            eyeX1 = centerX + snake.direction.x * eyeOffset;
            eyeY1 = centerY - eyeOffset;
            eyeX2 = centerX + snake.direction.x * eyeOffset;
            eyeY2 = centerY + eyeOffset;
        } else {
            // Moving vertically
            eyeX1 = centerX - eyeOffset;
            eyeY1 = centerY + snake.direction.y * eyeOffset;
            eyeX2 = centerX + eyeOffset;
            eyeY2 = centerY + snake.direction.y * eyeOffset;
        }
        
        this.ctx.beginPath();
        this.ctx.arc(eyeX1, eyeY1, eyeSize, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.beginPath();
        this.ctx.arc(eyeX2, eyeY2, eyeSize, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Pupils
        this.ctx.fillStyle = '#000000';
        const pupilSize = eyeSize * 0.6;
        
        this.ctx.beginPath();
        this.ctx.arc(eyeX1, eyeY1, pupilSize, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.beginPath();
        this.ctx.arc(eyeX2, eyeY2, pupilSize, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    renderSnakeEffects(snake) {
        snake.effects.forEach(effect => {
            switch (effect.type) {
                case 'speed_boost':
                    this.renderSpeedBoostEffect(snake);
                    break;
                case 'strength_boost':
                    this.renderStrengthBoostEffect(snake);
                    break;
            }
        });
        
        // Render particles
        snake.particles.forEach(particle => {
            this.ctx.globalAlpha = particle.alpha;
            this.ctx.fillStyle = particle.color;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        this.ctx.globalAlpha = 1;
    }
    
    renderSpeedBoostEffect(snake) {
        // Trail effect for speed boost
        const head = snake.segments[0];
        if (!head) return;
        
        const x = head.x * this.gridSize + this.gridSize / 2;
        const y = head.y * this.gridSize + this.gridSize / 2;
        
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)';
        this.ctx.lineWidth = 2;
        this.ctx.shadowColor = '#00ffff';
        this.ctx.shadowBlur = 8;
        
        // Draw speed lines
        for (let i = 0; i < 3; i++) {
            const offset = (i + 1) * 10;
            this.ctx.beginPath();
            this.ctx.moveTo(x - snake.direction.x * offset, y - snake.direction.y * offset);
            this.ctx.lineTo(x - snake.direction.x * (offset + 15), y - snake.direction.y * (offset + 15));
            this.ctx.stroke();
        }
        
        this.ctx.shadowBlur = 0;
    }
    
    renderStrengthBoostEffect(snake) {
        // Pulsing red aura for strength boost
        const head = snake.segments[0];
        if (!head) return;
        
        const x = head.x * this.gridSize + this.gridSize / 2;
        const y = head.y * this.gridSize + this.gridSize / 2;
        
        const pulse = Math.sin(Date.now() * 0.01) * 0.5 + 0.5;
        
        this.ctx.strokeStyle = `rgba(255, 0, 0, ${0.3 + pulse * 0.4})`;
        this.ctx.lineWidth = 3;
        this.ctx.shadowColor = '#ff0000';
        this.ctx.shadowBlur = 15;
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.gridSize * 0.8, 0, Math.PI * 2);
        this.ctx.stroke();
        
        this.ctx.shadowBlur = 0;
    }
    
    renderUI() {
        // Performance counter
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.font = '12px monospace';
        this.ctx.fillText(`FPS: ${this.fps}`, 10, 20);
        
        if (this.currentBattle) {
            // Battle timer
            const elapsed = Date.now() - this.currentBattle.startTime;
            const remaining = Math.max(0, this.currentBattle.config.timeLimit - elapsed);
            const minutes = Math.floor(remaining / 60000);
            const seconds = Math.floor((remaining % 60000) / 1000);
            
            this.ctx.fillText(`Time: ${minutes}:${seconds.toString().padStart(2, '0')}`, 10, 40);
        }
    }
    
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
}
