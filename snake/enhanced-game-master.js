// Enhanced Working Game Master with Elite Features
console.log('📜 Loading Enhanced Game Master...');

class SimpleGameMaster {
    constructor() {
        console.log('🏗️ SimpleGameMaster constructor called');
        this.canvas = null;
        this.ctx = null;
        this.gameMode = 'menu';
        this.isInitialized = false;
        this.animationId = null;
        this.restartCooldown = false;
        
        // Battle state
        this.snakes = [];
        this.battleActive = false;
        
        // Elite systems
        this.snakeGenerator = null;
        this.battleEngine = null;
        this.evolutionSystem = null;
        this.visualEffects = null;
        this.tournamentSystem = null;
        
        // Performance
        this.performance = {
            currentFPS: 60,
            lastFrameTime: 0
        };
    }
    
    async initialize() {
        try {
            console.log('🎮 Initializing Enhanced Game Master...');
            
            // Get canvas
            this.canvas = document.getElementById('game-canvas') || 
                         document.getElementById('battle-canvas') ||
                         document.getElementById('debug-canvas');
                         
            if (!this.canvas) {
                throw new Error('No canvas found');
            }
            
            this.ctx = this.canvas.getContext('2d');
            if (!this.ctx) {
                throw new Error('Could not get canvas context');
            }
            
            // Canvas setup
            this.canvas.width = 1000;
            this.canvas.height = 700;
            
            console.log(`✅ Canvas: ${this.canvas.width}x${this.canvas.height}`);
            
            // Initialize elite systems
            this.initializeSystems();
            
            // Mark as initialized
            this.isInitialized = true;
            console.log('✅ Enhanced Game Master ready!');
            
            // Draw initial menu
            this.drawMenu();
            
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize Enhanced Game Master:', error);
            return false;
        }
    }
    
    initializeSystems() {
        try {
            if (typeof EliteSnakeGenerator !== 'undefined') {
                this.snakeGenerator = new EliteSnakeGenerator();
                console.log('✅ Elite Snake Generator ready');
            }
            
            if (typeof EliteBattleEngine !== 'undefined') {
                this.battleEngine = new EliteBattleEngine(this.canvas, this.ctx);
                console.log('✅ Elite Battle Engine ready');
            }
            
            if (typeof EliteEvolutionSystem !== 'undefined') {
                this.evolutionSystem = new EliteEvolutionSystem();
                console.log('✅ Elite Evolution System ready');
            }
            
            if (typeof EliteVisualEffects !== 'undefined') {
                this.visualEffects = new EliteVisualEffects(this.canvas, this.ctx);
                console.log('✅ Elite Visual Effects ready');
            }
            
            if (typeof EliteTournamentSystem !== 'undefined') {
                this.tournamentSystem = new EliteTournamentSystem();
                console.log('✅ Elite Tournament System ready');
            }
            
        } catch (error) {
            console.error('❌ System initialization error:', error);
        }
    }
    
    startDemoBattle() {
        try {
            console.log('🎮 Starting enhanced demo battle...');
            
            if (!this.snakeGenerator) {
                throw new Error('Snake generator not available');
            }
              // Generate snakes with full abilities
            const snake1 = this.snakeGenerator.createSnake();
            const snake2 = this.snakeGenerator.createSnake();
            
            // Ensure snake compatibility with battle engine
            this.prepareSnakeForBattle(snake1);
            this.prepareSnakeForBattle(snake2);
            
            console.log(`Generated: ${snake1.name} vs ${snake2.name}`);
            console.log(`Snake1:`, snake1.stats, snake1.skills);
            console.log(`Snake2:`, snake2.stats, snake2.skills);
            
            // Use elite battle engine if available
            if (this.battleEngine && this.battleEngine.startBattle) {
                console.log('🚀 Using Elite Battle Engine');
                this.battleEngine.startBattle(snake1, snake2, {
                    onBattleEnd: (winner, loser) => {
                        this.handleEliteBattleEnd(winner, loser);
                    }
                });
            } else {
                console.log('⚡ Using Enhanced Battle System');
                this.battleActive = true;
                this.gameMode = 'battle';
                this.startEnhancedBattle(snake1, snake2);
            }
            
            console.log('✅ Enhanced demo battle started');
            
        } catch (error) {
            console.error('❌ Failed to start demo battle:', error);
            this.showError('Battle failed: ' + error.message);
        }
    }
    
    startEnhancedBattle(snake1, snake2) {
        let animationStep = 0;
        const maxSteps = 450; // 7.5 seconds for complex battles
        console.log('🎬 Starting enhanced battle animation...');
        
        // Initialize snake battle states
        let snake1State = {
            x: 200, y: this.canvas.height / 2,
            health: snake1.stats?.health || 100,
            energy: snake1.stats?.energy || 100,
            action: 'idle',
            actionCooldown: 0,
            speedBoost: 0
        };
        
        let snake2State = {
            x: this.canvas.width - 200, y: this.canvas.height / 2,
            health: snake2.stats?.health || 100,
            energy: snake2.stats?.energy || 100,
            action: 'idle',
            actionCooldown: 0,
            speedBoost: 0
        };
        
        const animate = () => {
            if (!this.battleActive) {
                console.log('❌ Battle stopped');
                return;
            }
            
            if (animationStep >= maxSteps || snake1State.health <= 0 || snake2State.health <= 0) {
                this.endEnhancedBattle(snake1, snake2, snake1State, snake2State);
                return;
            }
            
            // Clear canvas
            this.ctx.fillStyle = '#001122';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw enhanced arena
            this.drawEnhancedArena();
            
            // Update snake AI and actions
            this.updateSnakeAI(snake1, snake1State, snake2State, animationStep);
            this.updateSnakeAI(snake2, snake2State, snake1State, animationStep);
            
            // Draw snakes with abilities
            this.drawEnhancedSnake(snake1, snake1State);
            this.drawEnhancedSnake(snake2, snake2State);
            
            // Draw battle UI
            this.drawBattleUI(snake1, snake1State, snake2, snake2State, animationStep / maxSteps);
            
            // Apply visual effects
            if (this.visualEffects) {
                this.visualEffects.renderBattleEffects(animationStep);
            }
            
            animationStep++;
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    updateSnakeAI(snake, myState, enemyState, frame) {
        // Reduce cooldowns
        if (myState.actionCooldown > 0) myState.actionCooldown--;
        if (myState.speedBoost > 0) myState.speedBoost--;
        
        // Calculate distance and direction
        const dx = enemyState.x - myState.x;
        const dy = enemyState.y - myState.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // AI decision making
        if (myState.actionCooldown === 0 && myState.energy > 20) {
            if (snake.skills && snake.skills.length > 0) {
                const skill = snake.skills[Math.floor(Math.random() * snake.skills.length)];
                this.useSkill(snake, skill, myState, enemyState, distance);
                myState.actionCooldown = 30;
            }
        }
        
        // Movement AI based on health and strategy
        const healthRatio = myState.health / 100;
        const moveSpeed = myState.speedBoost > 0 ? 3 : 2;
        
        if (healthRatio > 0.5 && distance > 80) {
            // Aggressive - seek enemy
            myState.x += Math.sign(dx) * moveSpeed;
            myState.y += Math.sign(dy) * (moveSpeed * 0.7);
            myState.action = 'seeking';
        } else if (healthRatio < 0.3 && distance < 120) {
            // Defensive - flee
            myState.x -= Math.sign(dx) * (moveSpeed * 0.8);
            myState.y -= Math.sign(dy) * (moveSpeed * 0.6);
            myState.action = 'fleeing';
        } else {
            // Tactical positioning
            myState.x += Math.sin(frame * 0.05) * moveSpeed;
            myState.y += Math.cos(frame * 0.03) * (moveSpeed * 0.8);
            myState.action = 'positioning';
        }
        
        // Boundaries
        myState.x = Math.max(40, Math.min(this.canvas.width - 40, myState.x));
        myState.y = Math.max(40, Math.min(this.canvas.height - 40, myState.y));
        
        // Energy regeneration
        if (myState.energy < 100) myState.energy += 0.3;
    }
    
    useSkill(snake, skill, myState, enemyState, distance) {
        const skillName = skill.name || skill;
        console.log(`💥 ${snake.name} uses ${skillName}!`);
        
        if (skillName.includes('Strike') || skillName.includes('Attack')) {
            if (distance < 80) {
                const damage = (snake.stats?.attack || 20) + Math.random() * 15;
                enemyState.health -= damage;
                myState.energy -= 15;
                console.log(`⚡ ${skillName} hit for ${Math.round(damage)} damage!`);
            }
        } else if (skillName.includes('Speed') || skillName.includes('Boost')) {
            myState.speedBoost = 60;
            myState.energy -= 10;
            console.log(`🏃 Speed boost activated!`);
        } else if (skillName.includes('Heal') || skillName.includes('Regenerate')) {
            myState.health = Math.min(100, myState.health + 20);
            myState.energy -= 20;
            console.log(`💚 Healing activated!`);
        } else {
            // Generic skill effect
            if (distance < 100) {
                const damage = Math.random() * 12 + 8;
                enemyState.health -= damage;
                myState.energy -= 12;
            }
        }
    }
    
    drawEnhancedArena() {
        // Border with glow
        this.ctx.strokeStyle = '#00ffff';
        this.ctx.lineWidth = 4;
        this.ctx.strokeRect(15, 15, this.canvas.width - 30, this.canvas.height - 30);
        
        // Grid pattern
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.2)';
        this.ctx.lineWidth = 1;
        for (let x = 60; x < this.canvas.width; x += 60) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 25);
            this.ctx.lineTo(x, this.canvas.height - 25);
            this.ctx.stroke();
        }
        for (let y = 60; y < this.canvas.height; y += 60) {
            this.ctx.beginPath();
            this.ctx.moveTo(25, y);
            this.ctx.lineTo(this.canvas.width - 25, y);
            this.ctx.stroke();
        }
    }
    
    drawEnhancedSnake(snake, state) {
        const radius = 28;
        
        // Dynamic coloring based on state
        let bodyColor = snake.appearance?.primaryColor || snake.color || '#ff0000';
        if (state.speedBoost > 0) {
            bodyColor = this.brightenColor(bodyColor, 40);
        }
        if (state.action === 'seeking') {
            bodyColor = this.brightenColor(bodyColor, 20);
        }
        
        // Main body
        this.ctx.fillStyle = bodyColor;
        this.ctx.beginPath();
        this.ctx.arc(state.x, state.y, radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Health-based outline
        const healthRatio = state.health / 100;
        this.ctx.strokeStyle = healthRatio > 0.6 ? '#00ff00' : 
                             healthRatio > 0.3 ? '#ffff00' : '#ff0000';
        this.ctx.lineWidth = 4;
        this.ctx.stroke();
        
        // Eyes with direction awareness
        this.ctx.fillStyle = snake.appearance?.eyeColor || '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(state.x - 10, state.y - 10, 5, 0, Math.PI * 2);
        this.ctx.arc(state.x + 10, state.y - 10, 5, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Pupils
        this.ctx.fillStyle = '#000000';
        this.ctx.beginPath();
        this.ctx.arc(state.x - 10, state.y - 10, 2, 0, Math.PI * 2);
        this.ctx.arc(state.x + 10, state.y - 10, 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Name and info
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(snake.name, state.x, state.y - 40);
        
        // Action status
        this.ctx.font = '11px Arial';
        const actionColor = state.action === 'seeking' ? '#ff6600' : 
                          state.action === 'fleeing' ? '#6666ff' : '#ffff00';
        this.ctx.fillStyle = actionColor;
        this.ctx.fillText(state.action, state.x, state.y + 50);
        
        // Health bar
        const barWidth = 50;
        const barHeight = 6;
        this.ctx.fillStyle = '#333333';
        this.ctx.fillRect(state.x - barWidth/2, state.y - 55, barWidth, barHeight);
        this.ctx.fillStyle = healthRatio > 0.6 ? '#00ff00' : 
                           healthRatio > 0.3 ? '#ffff00' : '#ff0000';
        this.ctx.fillRect(state.x - barWidth/2, state.y - 55, barWidth * healthRatio, barHeight);
        
        // Energy bar
        const energyRatio = state.energy / 100;
        this.ctx.fillStyle = '#111111';
        this.ctx.fillRect(state.x - barWidth/2, state.y - 47, barWidth, 4);
        this.ctx.fillStyle = '#00aaff';
        this.ctx.fillRect(state.x - barWidth/2, state.y - 47, barWidth * energyRatio, 4);
    }
    
    drawBattleUI(snake1, state1, snake2, state2, progress) {
        // Progress bar
        this.ctx.fillStyle = '#333333';
        this.ctx.fillRect(this.canvas.width / 2 - 200, this.canvas.height - 50, 400, 30);
        this.ctx.fillStyle = '#00ff00';
        this.ctx.fillRect(this.canvas.width / 2 - 200, this.canvas.height - 50, 400 * progress, 30);
        
        // Progress text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 18px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`Battle Progress: ${Math.round(progress * 100)}%`, this.canvas.width / 2, this.canvas.height - 60);
        
        // Snake info panels
        this.ctx.textAlign = 'left';
        this.ctx.font = '16px Arial';
        this.ctx.fillStyle = '#ffffff';
        
        // Snake 1 panel
        this.ctx.fillText(`${snake1.name}`, 30, 30);
        this.ctx.font = '14px Arial';
        this.ctx.fillText(`Health: ${Math.round(state1.health)}/100`, 30, 55);
        this.ctx.fillText(`Energy: ${Math.round(state1.energy)}/100`, 30, 75);
        this.ctx.fillText(`Action: ${state1.action}`, 30, 95);
        this.ctx.fillText(`Skills: ${snake1.skills?.length || 0}`, 30, 115);
        
        // Snake 2 panel
        this.ctx.textAlign = 'right';
        this.ctx.font = '16px Arial';
        this.ctx.fillText(`${snake2.name}`, this.canvas.width - 30, 30);
        this.ctx.font = '14px Arial';
        this.ctx.fillText(`Health: ${Math.round(state2.health)}/100`, this.canvas.width - 30, 55);
        this.ctx.fillText(`Energy: ${Math.round(state2.energy)}/100`, this.canvas.width - 30, 75);
        this.ctx.fillText(`Action: ${state2.action}`, this.canvas.width - 30, 95);
        this.ctx.fillText(`Skills: ${snake2.skills?.length || 0}`, this.canvas.width - 30, 115);
    }
    
    brightenColor(color, amount) {
        const hex = color.replace('#', '');
        const r = Math.min(255, parseInt(hex.substr(0,2), 16) + amount);
        const g = Math.min(255, parseInt(hex.substr(2,2), 16) + amount);
        const b = Math.min(255, parseInt(hex.substr(4,2), 16) + amount);
        return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
    }
    
    endEnhancedBattle(snake1, snake2, state1, state2) {
        this.battleActive = false;
        
        // Determine winner
        const winner = state1.health > state2.health ? snake1 : snake2;
        const loser = winner === snake1 ? snake2 : snake1;
        const winnerState = winner === snake1 ? state1 : state2;
        const loserState = winner === snake1 ? state2 : state1;
        
        console.log(`🏆 Enhanced battle ended! Winner: ${winner.name} (${Math.round(winnerState.health)} HP)`);
        
        // Draw final scene
        this.drawEnhancedArena();
        this.drawEnhancedSnake(snake1, state1);
        this.drawEnhancedSnake(snake2, state2);
        
        // Victory overlay
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = 'bold 40px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🏆 VICTORY! 🏆', this.canvas.width / 2, this.canvas.height / 2 - 80);
        
        this.ctx.fillStyle = winner.appearance?.primaryColor || winner.color || '#00ff00';
        this.ctx.font = 'bold 32px Arial';
        this.ctx.fillText(winner.name, this.canvas.width / 2, this.canvas.height / 2 - 20);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '20px Arial';
        this.ctx.fillText(`Defeated ${loser.name} in epic combat!`, this.canvas.width / 2, this.canvas.height / 2 + 20);
        this.ctx.fillText(`Final: ${Math.round(winnerState.health)}HP vs ${Math.round(loserState.health)}HP`, this.canvas.width / 2, this.canvas.height / 2 + 50);
        
        // Show skills used
        if (winner.skills && winner.skills.length > 0) {
            this.ctx.font = '16px Arial';
            this.ctx.fillText(`Winning Skills: ${winner.skills.slice(0,3).map(s => s.name || s).join(', ')}`, this.canvas.width / 2, this.canvas.height / 2 + 80);
        }
        
        // Evolution check
        if (this.evolutionSystem) {
            const evolvedWinner = this.evolutionSystem.evolveSnake(winner, 'victory');
            if (evolvedWinner && evolvedWinner !== winner) {
                console.log(`🧬 ${winner.name} evolved after victory!`);
                this.ctx.fillStyle = '#ff00ff';
                this.ctx.fillText('🧬 EVOLUTION UNLOCKED! 🧬', this.canvas.width / 2, this.canvas.height / 2 + 110);
            }
        }
        
        // Auto-restart with cooldown
        this.handleBattleRestart();
    }
    
    handleEliteBattleEnd(winner, loser) {
        console.log(`🏆 Elite battle ended! Winner: ${winner.name}`);
        
        if (this.evolutionSystem) {
            const evolved = this.evolutionSystem.evolveSnake(winner, 'victory');
            if (evolved && evolved !== winner) {
                console.log(`🧬 ${winner.name} evolved after elite victory!`);
            }
        }
        
        this.handleBattleRestart();
    }
    
    handleBattleRestart() {
        if (this.restartCooldown) {
            console.log('⏸️ Battle restart on cooldown');
            return;
        }
        
        if (window.location.pathname.includes('auto-battle.html')) {
            this.restartCooldown = true;
            setTimeout(() => {
                this.restartCooldown = false;
                console.log('🔄 Auto-restarting enhanced battle...');
                this.startDemoBattle();
            }, 4000);
        } else {
            setTimeout(() => {
                this.gameMode = 'menu';
                this.drawMenu();
            }, 4000);
        }
    }
    
    drawMenu() {
        this.ctx.fillStyle = '#001122';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = 'bold 48px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🐍 ELITE SNAKE ARENA 🐍', this.canvas.width / 2, 200);
        
        this.ctx.fillStyle = '#00ffff';
        this.ctx.font = '24px Arial';
        this.ctx.fillText('Enhanced with AI, Skills & Evolution', this.canvas.width / 2, 250);
    }
    
    showError(message) {
        console.error('❌ Error:', message);
    }
    
    prepareSnakeForBattle(snake) {
        // Ensure required properties for battle engine
        if (!snake.maxHealth) snake.maxHealth = snake.health || 100;
        if (!snake.baseSpeed) snake.baseSpeed = snake.stats?.speed || 1;
        if (!snake.baseDamage) snake.baseDamage = snake.stats?.attack || 10;
        if (!snake.velocity) snake.velocity = { x: 0, y: 0 };
        if (!snake.skillCooldowns) snake.skillCooldowns = {};
        if (!snake.activeEffects) snake.activeEffects = [];
        if (!snake.particles) snake.particles = [];
        
        // Initialize skill cooldowns
        if (snake.skills) {
            snake.skills.forEach(skill => {
                const skillName = skill.name || skill;
                snake.skillCooldowns[skillName] = 0;
            });
        }
        
        // Ensure color property exists
        if (!snake.color && snake.appearance?.primaryColor) {
            snake.color = snake.appearance.primaryColor;
        } else if (!snake.color) {
            snake.color = '#ff0000';
        }
        
        console.log(`✅ Snake ${snake.name} prepared for battle`);
    }
}

// Global instance
window.eliteGameMaster = null;

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing Enhanced Game Master...');
    window.eliteGameMaster = new SimpleGameMaster();
    await window.eliteGameMaster.initialize();
});

console.log('✅ Enhanced Game Master module loaded');
