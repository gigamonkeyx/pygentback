// Simplified Working Game Master
console.log('📜 Loading Simple Game Master...');

class SimpleGameMaster {    constructor() {
        console.log('🏗️ SimpleGameMaster constructor called');
        this.canvas = null;
        this.ctx = null;
        this.gameMode = 'menu';
        this.isInitialized = false;
        this.animationId = null;
        this.restartCooldown = false; // Prevent infinite restart loops
        
        // Simple state
        this.snakes = [];
        this.battleActive = false;
        
        // Performance
        this.performance = {
            currentFPS: 60,
            lastFrameTime: 0
        };
    }
    
    async initialize() {
        try {
            console.log('🎮 Initializing Simple Game Master...');
            
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
            
            // Simple canvas setup
            this.canvas.width = 1000;
            this.canvas.height = 700;
            
            console.log(`✅ Canvas: ${this.canvas.width}x${this.canvas.height}`);
            
            // Initialize simple systems
            this.initializeSystems();
            
            // Draw initial state
            this.drawMenu();
            
            this.isInitialized = true;
            console.log('✅ Simple Game Master initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize Simple Game Master:', error);
            return false;
        }
    }
      initializeSystems() {
        // Initialize all elite systems for full functionality
        try {
            if (typeof EliteSnakeGenerator !== 'undefined') {
                this.snakeGenerator = new EliteSnakeGenerator();
                console.log('✅ Elite Snake Generator ready');
            } else {
                console.warn('⚠️ EliteSnakeGenerator not available');
            }
            
            // Initialize Battle Engine for advanced combat
            if (typeof EliteBattleEngine !== 'undefined') {
                this.battleEngine = new EliteBattleEngine(this.canvas, this.ctx);
                console.log('✅ Elite Battle Engine ready');
            } else {
                console.warn('⚠️ EliteBattleEngine not available - using simple battles');
            }
            
            // Initialize Evolution System for mutations
            if (typeof EliteEvolutionSystem !== 'undefined') {
                this.evolutionSystem = new EliteEvolutionSystem();
                console.log('✅ Elite Evolution System ready');
            } else {
                console.warn('⚠️ EliteEvolutionSystem not available');
            }
            
            // Initialize Visual Effects for enhanced graphics
            if (typeof EliteVisualEffects !== 'undefined') {
                this.visualEffects = new EliteVisualEffects(this.canvas, this.ctx);
                console.log('✅ Elite Visual Effects ready');
            } else {
                console.warn('⚠️ EliteVisualEffects not available');
            }
            
            // Initialize Tournament System
            if (typeof EliteTournamentSystem !== 'undefined') {
                this.tournamentSystem = new EliteTournamentSystem();
                console.log('✅ Elite Tournament System ready');
            } else {
                console.warn('⚠️ EliteTournamentSystem not available');
            }
            
        } catch (error) {
            console.error('❌ System initialization error:', error);
        }
    }
    
    drawMenu() {
        // Clear canvas
        this.ctx.fillStyle = '#001122';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        
        for (let x = 0; x < this.canvas.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // Draw title
        this.ctx.fillStyle = '#00ffff';
        this.ctx.font = 'bold 32px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🐍 Elite Snake Evolution Arena', this.canvas.width / 2, this.canvas.height / 2 - 50);
        
        this.ctx.font = '18px Arial';
        this.ctx.fillText('Ready for Epic Battles!', this.canvas.width / 2, this.canvas.height / 2 + 20);
        
        console.log('✅ Menu drawn');
    }
      startDemoBattle() {
        try {
            console.log('🎮 Starting demo battle...');
            
            if (!this.snakeGenerator) {
                throw new Error('Snake generator not available');
            }
            
            // Generate snakes with full abilities
            const snake1 = this.snakeGenerator.createSnake();
            const snake2 = this.snakeGenerator.createSnake();
            
            console.log(`Generated: ${snake1.name} vs ${snake2.name}`);
            console.log(`Snake1 stats:`, snake1.stats);
            console.log(`Snake1 skills:`, snake1.skills);
            console.log(`Snake2 stats:`, snake2.stats);
            console.log(`Snake2 skills:`, snake2.skills);
            
            // Use elite battle engine if available
            if (this.battleEngine && this.battleEngine.startBattle) {
                console.log('🚀 Using Elite Battle Engine');
                this.battleEngine.startBattle(snake1, snake2, {
                    onBattleEnd: (winner, loser) => {
                        this.handleBattleEnd(winner, loser);
                    }
                });
            } else {
                console.log('⚡ Using Enhanced Simple Battle System');
                // Enhanced battle with AI behaviors
                this.drawBattle(snake1, snake2);
                this.battleActive = true;
                this.gameMode = 'battle';
                this.startAdvancedBattleAnimation(snake1, snake2);
            }
            
            console.log('✅ Demo battle started');
            
        } catch (error) {
            console.error('❌ Failed to start demo battle:', error);
            this.showError('Battle failed: ' + error.message);
        }
    }
    
    drawBattle(snake1, snake2) {
        // Clear canvas
        this.ctx.fillStyle = '#001122';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw arena border
        this.ctx.strokeStyle = '#00ffff';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(10, 10, this.canvas.width - 20, this.canvas.height - 20);
        
        // Draw snake 1 (left side)
        this.ctx.fillStyle = snake1.color || '#ff0000';
        this.ctx.beginPath();
        this.ctx.arc(200, this.canvas.height / 2, 30, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(snake1.name, 200, this.canvas.height / 2 - 50);
        this.ctx.fillText(snake1.species || 'Elite Snake', 200, this.canvas.height / 2 + 60);
        
        // Draw snake 2 (right side)
        this.ctx.fillStyle = snake2.color || '#0000ff';
        this.ctx.beginPath();
        this.ctx.arc(this.canvas.width - 200, this.canvas.height / 2, 30, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(snake2.name, this.canvas.width - 200, this.canvas.height / 2 - 50);
        this.ctx.fillText(snake2.species || 'Elite Snake', this.canvas.width - 200, this.canvas.height / 2 + 60);
        
        // VS text
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = 'bold 48px Arial';
        this.ctx.fillText('VS', this.canvas.width / 2, this.canvas.height / 2);
        
        // Skills display
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Skills: ${snake1.skills ? snake1.skills.length : 0}`, 50, 100);
        
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`Skills: ${snake2.skills ? snake2.skills.length : 0}`, this.canvas.width - 50, 100);
    }
      startAdvancedBattleAnimation(snake1, snake2) {
        let animationStep = 0;
        const maxSteps = 450; // 7.5 seconds for more complex battles
        console.log('🎬 Starting advanced battle animation...');
        
        // Initialize snake positions and states
        let snake1State = {
            x: 200, y: this.canvas.height / 2,
            health: snake1.stats?.health || 100,
            energy: snake1.stats?.energy || 100,
            action: 'idle',
            actionCooldown: 0
        };
        
        let snake2State = {
            x: this.canvas.width - 200, y: this.canvas.height / 2,
            health: snake2.stats?.health || 100,
            energy: snake2.stats?.energy || 100,
            action: 'idle',
            actionCooldown: 0
        };
        
        const animate = () => {
            if (!this.battleActive) {
                console.log('❌ Battle stopped - battleActive is false');
                return;
            }
            
            if (animationStep >= maxSteps || snake1State.health <= 0 || snake2State.health <= 0) {
                console.log('⏰ Battle complete');
                this.endAdvancedBattle(snake1, snake2, snake1State, snake2State);
                return;
            }
            
            // Clear canvas
            this.ctx.fillStyle = '#001122';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw arena with effects
            this.drawAdvancedArena();
            
            // Update snake AI and actions
            this.updateSnakeAI(snake1, snake1State, snake2State, animationStep);
            this.updateSnakeAI(snake2, snake2State, snake1State, animationStep);
            
            // Draw snakes with abilities
            this.drawAdvancedSnake(snake1, snake1State);
            this.drawAdvancedSnake(snake2, snake2State);
            
            // Draw battle UI
            this.drawBattleUI(snake1, snake1State, snake2, snake2State, animationStep / maxSteps);
            
            // Apply visual effects if available
            if (this.visualEffects) {
                this.visualEffects.renderBattleEffects(animationStep);
            }
            
            animationStep++;
            this.animationId = requestAnimationFrame(animate);
        };
        
        // Initialize battle state
        this.battleActive = true;
        animate();
    }
        let animationStep = 0;
        const maxSteps = 300; // Increase to 5 seconds at 60fps for more visible battle
        console.log('🎬 Starting battle animation...');
        
        const animate = () => {
            if (!this.battleActive) {
                console.log('❌ Battle stopped - battleActive is false');
                return;
            }
            
            if (animationStep >= maxSteps) {
                console.log('⏰ Animation complete, ending battle');
                this.endBattle(snake1, snake2);
                return;
            }
            
            // Clear canvas first
            this.ctx.fillStyle = '#001122';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Add animation effects
            const progress = animationStep / maxSteps;
            
            // More dramatic snake movement - snakes move toward each other
            const baseDistance = 150;
            const moveProgress = Math.min(progress * 2, 1); // Move faster in first half
            
            const snake1X = 200 + moveProgress * baseDistance + Math.sin(animationStep * 0.2) * 30;
            const snake1Y = this.canvas.height / 2 + Math.cos(animationStep * 0.15) * 20;
            const snake2X = this.canvas.width - 200 - moveProgress * baseDistance + Math.sin(animationStep * 0.2 + Math.PI) * 30;
            const snake2Y = this.canvas.height / 2 + Math.cos(animationStep * 0.15 + Math.PI) * 20;
            
            // Draw arena border
            this.ctx.strokeStyle = '#00ffff';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(10, 10, this.canvas.width - 20, this.canvas.height - 20);
            
            // Draw animated snakes
            this.ctx.fillStyle = snake1.appearance?.primaryColor || snake1.color || '#ff0000';
            this.ctx.beginPath();
            this.ctx.arc(snake1X, snake1Y, 25, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            this.ctx.fillStyle = snake2.appearance?.primaryColor || snake2.color || '#0000ff';
            this.ctx.beginPath();
            this.ctx.arc(snake2X, snake2Y, 25, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
            
            // Snake names
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = 'bold 14px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(snake1.name, snake1X, snake1Y - 40);
            this.ctx.fillText(snake2.name, snake2X, snake2Y - 40);
            
            // Battle effects
            if (animationStep % 30 === 0) {
                console.log(`💥 Battle frame ${animationStep}/${maxSteps} - ${snake1.name} vs ${snake2.name}`);
            }
            
            // Progress bar
            this.ctx.fillStyle = '#333333';
            this.ctx.fillRect(this.canvas.width / 2 - 150, this.canvas.height - 40, 300, 25);
            this.ctx.fillStyle = '#00ff00';
            this.ctx.fillRect(this.canvas.width / 2 - 150, this.canvas.height - 40, 300 * progress, 25);
            
            // Progress text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '16px Arial';
            this.ctx.fillText(`Battle: ${Math.round(progress * 100)}%`, this.canvas.width / 2, this.canvas.height - 50);
            
            animationStep++;
            this.animationId = requestAnimationFrame(animate);
        };
        
        // Start the animation
        this.battleActive = true;
        animate();
    }
      endBattle(snake1, snake2) {
        this.battleActive = false;
        
        // Determine winner based on stats (more realistic than random)
        const snake1Power = (snake1.stats?.attack || 50) + (snake1.stats?.defense || 50) + (snake1.skills?.length || 0) * 10;
        const snake2Power = (snake2.stats?.attack || 50) + (snake2.stats?.defense || 50) + (snake2.skills?.length || 0) * 10;
        
        // Add some randomness but favor stronger snake
        const winner = (snake1Power + Math.random() * 20) > (snake2Power + Math.random() * 20) ? snake1 : snake2;
        const loser = winner === snake1 ? snake2 : snake1;
        
        // Draw final result
        this.drawBattle(snake1, snake2);
        
        // Winner announcement
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = 'bold 36px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🏆 WINNER! 🏆', this.canvas.width / 2, this.canvas.height / 2 - 60);
        
        this.ctx.fillStyle = winner.color || '#00ff00';
        this.ctx.font = 'bold 28px Arial';
        this.ctx.fillText(winner.name, this.canvas.width / 2, this.canvas.height / 2);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '18px Arial';
        this.ctx.fillText(`Defeated ${loser.name} in epic combat!`, this.canvas.width / 2, this.canvas.height / 2 + 40);
        
        const reason = snake1Power > snake2Power ? 'Superior Combat Skills!' : 'Lucky Critical Hit!';
        this.ctx.fillText(reason, this.canvas.width / 2, this.canvas.height / 2 + 65);        console.log(`🏆 Battle ended! Winner: ${winner.name}`);
        
        // Prevent infinite loops - add a restart cooldown
        if (this.restartCooldown) {
            console.log('⏸️ Battle restart on cooldown, skipping...');
            return;
        }
        
        // Check if we're in auto-battle mode (called from auto-battle.html)
        // If so, start a new battle after delay
        if (window.location.pathname.includes('auto-battle.html')) {
            this.restartCooldown = true;
            setTimeout(() => {
                this.restartCooldown = false;
                console.log('🔄 Auto-restarting battle...');
                this.startDemoBattle();
            }, 3000); // Increased delay to 3 seconds
        } else {
            // Auto-return to menu after 3 seconds for other modes
            setTimeout(() => {
                this.gameMode = 'menu';
                this.drawMenu();
            }, 3000);
        }
    }
    
    startDemoTournament() {
        try {
            console.log('🏆 Starting demo tournament...');
            
            if (!this.snakeGenerator) {
                throw new Error('Snake generator not available');
            }
            
            // Generate 4 snakes for a quick tournament
            const snakes = [];
            for (let i = 0; i < 4; i++) {
                snakes.push(this.snakeGenerator.createSnake());
            }
            
            console.log('Tournament snakes:', snakes.map(s => s.name));
            
            // Draw tournament bracket
            this.drawTournament(snakes);
            
            console.log('✅ Demo tournament started');
            
        } catch (error) {
            console.error('❌ Failed to start demo tournament:', error);
            this.showError('Tournament failed: ' + error.message);
        }
    }
    
    drawTournament(snakes) {
        // Clear canvas
        this.ctx.fillStyle = '#001122';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Title
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = 'bold 32px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🏆 TOURNAMENT BRACKET', this.canvas.width / 2, 60);
        
        // Draw bracket
        const positions = [
            { x: 200, y: 200 },
            { x: 200, y: 300 },
            { x: 200, y: 450 },
            { x: 200, y: 550 }
        ];
        
        snakes.forEach((snake, index) => {
            const pos = positions[index];
            
            // Snake circle
            this.ctx.fillStyle = snake.color || `hsl(${index * 90}, 70%, 50%)`;
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, 25, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Snake name
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '16px Arial';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(snake.name, pos.x + 40, pos.y + 5);
            this.ctx.fillText(snake.species || 'Elite Snake', pos.x + 40, pos.y + 20);
        });
        
        // Draw bracket lines
        this.ctx.strokeStyle = '#00ffff';
        this.ctx.lineWidth = 2;
        
        // Semi-final lines
        this.ctx.beginPath();
        this.ctx.moveTo(250, 200);
        this.ctx.lineTo(400, 250);
        this.ctx.moveTo(250, 300);
        this.ctx.lineTo(400, 250);
        this.ctx.moveTo(250, 450);
        this.ctx.lineTo(400, 500);
        this.ctx.moveTo(250, 550);
        this.ctx.lineTo(400, 500);
        this.ctx.stroke();
        
        // Final line
        this.ctx.beginPath();
        this.ctx.moveTo(400, 250);
        this.ctx.lineTo(500, 375);
        this.ctx.moveTo(400, 500);
        this.ctx.lineTo(500, 375);
        this.ctx.stroke();
        
        // Trophy
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = '48px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🏆', 650, 375);
    }
    
    togglePause() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
            console.log('⏸️ Game paused');
        } else if (this.battleActive) {
            // Resume would need to restart animation
            console.log('▶️ Game resumed');
        }
    }
    
    resetGame() {
        this.battleActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.gameMode = 'menu';
        this.drawMenu();
        console.log('🔄 Game reset');
    }
    
    showError(message) {
        // Draw error on canvas
        this.ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('❌ ERROR', this.canvas.width / 2, this.canvas.height / 2 - 30);
        
        this.ctx.font = '18px Arial';
        this.ctx.fillText(message, this.canvas.width / 2, this.canvas.height / 2 + 10);
        
        this.ctx.font = '14px Arial';
        this.ctx.fillText('Check console for details', this.canvas.width / 2, this.canvas.height / 2 + 40);
    }
    
    updateSnakeAI(snake, myState, enemyState, frame) {
        // Reduce action cooldowns
        if (myState.actionCooldown > 0) myState.actionCooldown--;
        
        // Calculate distance to enemy
        const dx = enemyState.x - myState.x;
        const dy = enemyState.y - myState.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // AI decision making based on snake skills and stats
        if (myState.actionCooldown === 0 && myState.energy > 20) {
            // Use snake abilities
            if (snake.skills && snake.skills.length > 0) {
                const skill = snake.skills[Math.floor(Math.random() * snake.skills.length)];
                this.useSkill(snake, skill, myState, enemyState);
                myState.actionCooldown = 30; // 0.5 second cooldown
            }
        }
        
        // Movement AI - seek or flee based on health
        const healthRatio = myState.health / 100;
        if (healthRatio > 0.5 && distance > 100) {
            // Aggressive - move toward enemy
            myState.x += Math.sign(dx) * 2;
            myState.y += Math.sign(dy) * 1;
            myState.action = 'seeking';
        } else if (healthRatio < 0.3 && distance < 150) {
            // Defensive - flee
            myState.x -= Math.sign(dx) * 1.5;
            myState.y -= Math.sign(dy) * 1;
            myState.action = 'fleeing';
        } else {
            // Circling/positioning
            myState.x += Math.sin(frame * 0.05) * 1.5;
            myState.y += Math.cos(frame * 0.03) * 1;
            myState.action = 'positioning';
        }
        
        // Regenerate energy slowly
        if (myState.energy < 100) myState.energy += 0.2;
    }
    
    useSkill(snake, skill, myState, enemyState) {
        console.log(`💥 ${snake.name} uses ${skill.name || skill}!`);
        
        // Calculate distance for range checks
        const dx = enemyState.x - myState.x;
        const dy = enemyState.y - myState.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Skill effects based on skill name/type
        if (skill.name && skill.name.includes('Strike') || skill.includes && skill.includes('Strike')) {
            if (distance < 80) {
                const damage = (snake.stats?.attack || 20) + Math.random() * 10;
                enemyState.health -= damage;
                myState.energy -= 15;
                console.log(`⚡ Strike hit for ${Math.round(damage)} damage!`);
            }
        } else if (skill.name && skill.name.includes('Speed') || skill.includes && skill.includes('Speed')) {
            // Speed boost
            myState.speedBoost = 60; // 1 second boost
            myState.energy -= 10;
        } else if (skill.name && skill.name.includes('Heal') || skill.includes && skill.includes('Heal')) {
            // Self heal
            myState.health = Math.min(100, myState.health + 15);
            myState.energy -= 20;
        } else {
            // Generic attack
            if (distance < 100) {
                const damage = Math.random() * 15 + 5;
                enemyState.health -= damage;
                myState.energy -= 10;
            }
        }
    }
    
    drawAdvancedArena() {
        // Enhanced arena with effects
        this.ctx.strokeStyle = '#00ffff';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(10, 10, this.canvas.width - 20, this.canvas.height - 20);
        
        // Grid with glow effect
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.3)';
        this.ctx.lineWidth = 1;
        for (let x = 50; x < this.canvas.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 20);
            this.ctx.lineTo(x, this.canvas.height - 20);
            this.ctx.stroke();
        }
        for (let y = 50; y < this.canvas.height; y += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(20, y);
            this.ctx.lineTo(this.canvas.width - 20, y);
            this.ctx.stroke();
        }
    }
    
    drawAdvancedSnake(snake, state) {
        // Snake body with advanced rendering
        const radius = 25;
        
        // Action-based colors and effects
        let bodyColor = snake.appearance?.primaryColor || snake.color || '#ff0000';
        if (state.action === 'seeking') {
            bodyColor = this.brightenColor(bodyColor, 20);
        } else if (state.action === 'fleeing') {
            bodyColor = this.darkenColor(bodyColor, 20);
        }
        
        // Draw main body
        this.ctx.fillStyle = bodyColor;
        this.ctx.beginPath();
        this.ctx.arc(state.x, state.y, radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Health-based outline
        const healthRatio = state.health / 100;
        this.ctx.strokeStyle = healthRatio > 0.5 ? '#00ff00' : healthRatio > 0.25 ? '#ffff00' : '#ff0000';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        
        // Eyes with expression
        this.ctx.fillStyle = snake.appearance?.eyeColor || '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(state.x - 8, state.y - 8, 4, 0, Math.PI * 2);
        this.ctx.arc(state.x + 8, state.y - 8, 4, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Pupils (direction-based)
        this.ctx.fillStyle = '#000000';
        this.ctx.beginPath();
        this.ctx.arc(state.x - 8, state.y - 8, 2, 0, Math.PI * 2);
        this.ctx.arc(state.x + 8, state.y - 8, 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Name and status
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(snake.name, state.x, state.y - 35);
        
        // Action indicator
        this.ctx.font = '10px Arial';
        this.ctx.fillStyle = state.action === 'seeking' ? '#ff6600' : 
                           state.action === 'fleeing' ? '#6666ff' : '#ffff00';
        this.ctx.fillText(state.action, state.x, state.y + 45);
        
        // Health bar
        const barWidth = 40;
        const barHeight = 4;
        this.ctx.fillStyle = '#333333';
        this.ctx.fillRect(state.x - barWidth/2, state.y - 50, barWidth, barHeight);
        this.ctx.fillStyle = healthRatio > 0.5 ? '#00ff00' : healthRatio > 0.25 ? '#ffff00' : '#ff0000';
        this.ctx.fillRect(state.x - barWidth/2, state.y - 50, barWidth * healthRatio, barHeight);
    }
    
    drawBattleUI(snake1, state1, snake2, state2, progress) {
        // Progress bar
        this.ctx.fillStyle = '#333333';
        this.ctx.fillRect(this.canvas.width / 2 - 150, this.canvas.height - 40, 300, 25);
        this.ctx.fillStyle = '#00ff00';
        this.ctx.fillRect(this.canvas.width / 2 - 150, this.canvas.height - 40, 300 * progress, 25);
        
        // Progress text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`Battle: ${Math.round(progress * 100)}%`, this.canvas.width / 2, this.canvas.height - 50);
        
        // Snake stats display
        this.ctx.textAlign = 'left';
        this.ctx.font = '14px Arial';
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(`${snake1.name}`, 30, 30);
        this.ctx.fillText(`Health: ${Math.round(state1.health)}`, 30, 50);
        this.ctx.fillText(`Energy: ${Math.round(state1.energy)}`, 30, 70);
        this.ctx.fillText(`Action: ${state1.action}`, 30, 90);
        
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`${snake2.name}`, this.canvas.width - 30, 30);
        this.ctx.fillText(`Health: ${Math.round(state2.health)}`, this.canvas.width - 30, 50);
        this.ctx.fillText(`Energy: ${Math.round(state2.energy)}`, this.canvas.width - 30, 70);
        this.ctx.fillText(`Action: ${state2.action}`, this.canvas.width - 30, 90);
    }
    
    brightenColor(color, amount) {
        // Simple color brightening
        const hex = color.replace('#', '');
        const r = Math.min(255, parseInt(hex.substr(0,2), 16) + amount);
        const g = Math.min(255, parseInt(hex.substr(2,2), 16) + amount);
        const b = Math.min(255, parseInt(hex.substr(4,2), 16) + amount);
        return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
    }
    
    darkenColor(color, amount) {
        // Simple color darkening
        const hex = color.replace('#', '');
        const r = Math.max(0, parseInt(hex.substr(0,2), 16) - amount);
        const g = Math.max(0, parseInt(hex.substr(2,2), 16) - amount);
        const b = Math.max(0, parseInt(hex.substr(4,2), 16) - amount);
        return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
    }
    
    endAdvancedBattle(snake1, snake2, state1, state2) {
        this.battleActive = false;
        
        // Determine winner based on health
        const winner = state1.health > state2.health ? snake1 : snake2;
        const loser = winner === snake1 ? snake2 : snake1;
        const winnerState = winner === snake1 ? state1 : state2;
        const loserState = winner === snake1 ? state2 : state1;
        
        // Draw final result with enhanced display
        this.drawAdvancedArena();
        this.drawAdvancedSnake(snake1, state1);
        this.drawAdvancedSnake(snake2, state2);
        
        // Victory overlay
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = 'bold 36px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('🏆 WINNER! 🏆', this.canvas.width / 2, this.canvas.height / 2 - 60);
        
        this.ctx.fillStyle = winner.appearance?.primaryColor || winner.color || '#00ff00';
        this.ctx.font = 'bold 28px Arial';
        this.ctx.fillText(winner.name, this.canvas.width / 2, this.canvas.height / 2);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '18px Arial';
        this.ctx.fillText(`Defeated ${loser.name} in epic combat!`, this.canvas.width / 2, this.canvas.height / 2 + 40);
        this.ctx.fillText(`Final Health: ${Math.round(winnerState.health)} vs ${Math.round(loserState.health)}`, this.canvas.width / 2, this.canvas.height / 2 + 65);
        
        console.log(`🏆 Advanced Battle ended! Winner: ${winner.name} (${Math.round(winnerState.health)} HP)`);
        
        // Handle evolution/mutation if evolution system is available
        if (this.evolutionSystem) {
            const evolvedWinner = this.evolutionSystem.evolveSnake(winner, 'victory');
            if (evolvedWinner && evolvedWinner !== winner) {
                console.log(`🧬 ${winner.name} evolved after victory!`);
            }
        }
        
        // Auto-restart logic with cooldown
        if (this.restartCooldown) {
            console.log('⏸️ Battle restart on cooldown, skipping...');
            return;
        }
        
        if (window.location.pathname.includes('auto-battle.html')) {
            this.restartCooldown = true;
            setTimeout(() => {
                this.restartCooldown = false;
                console.log('🔄 Auto-restarting advanced battle...');
                this.startDemoBattle();
            }, 4000); // Longer delay to appreciate the results
        } else {
            setTimeout(() => {
                this.gameMode = 'menu';
                this.drawMenu();
            }, 4000);
        }
    }
    
    handleBattleEnd(winner, loser) {
        // Handle elite battle engine results
        console.log(`🏆 Elite Battle ended! Winner: ${winner.name}`);
        
        // Apply evolution if available
        if (this.evolutionSystem) {
            const evolvedWinner = this.evolutionSystem.evolveSnake(winner, 'victory');
            if (evolvedWinner && evolvedWinner !== winner) {
                console.log(`🧬 ${winner.name} evolved after elite victory!`);
            }
        }
        
        // Auto-restart for continuous battles
        if (window.location.pathname.includes('auto-battle.html') && !this.restartCooldown) {
            this.restartCooldown = true;
            setTimeout(() => {
                this.restartCooldown = false;
                this.startDemoBattle();
            }, 3000);
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🐍 Starting Simple Snake Arena...');
    
    window.eliteGameMaster = new SimpleGameMaster();
    const success = await window.eliteGameMaster.initialize();
    
    if (success) {
        console.log('🎮 Simple Snake Arena ready!');
    } else {
        console.error('❌ Failed to initialize Simple Snake Arena');
    }
});
