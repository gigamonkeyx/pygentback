// Learning System - Handles AI adaptation and evolution
class LearningSystem {
    constructor() {
        this.learningData = {
            snake1: this.initializeLearningData(),
            snake2: this.initializeLearningData()
        };
        this.battleHistory = [];
        this.deadSnakes = [];
        this.logEntries = [];
        this.maxLogEntries = 50;
    }

    initializeLearningData() {
        return {
            patterns: {
                wallAvoidance: { success: 0, attempts: 0 },
                foodSeeking: { success: 0, attempts: 0 },
                powerupCollection: { success: 0, attempts: 0 },
                enemyAvoidance: { success: 0, attempts: 0 },
                centerPlay: { success: 0, attempts: 0 },
                edgePlay: { success: 0, attempts: 0 }
            },
            preferences: {
                aggressiveness: 0.5,
                exploration: 0.5,
                riskTaking: 0.5,
                cooperation: 0.1
            },
            memory: {
                dangerousPositions: [],
                safePositions: [],
                foodHotspots: [],
                powerupLocations: []
            },
            adaptations: []
        };
    }

    recordAction(snakeId, action, outcome, context = {}) {
        const data = this.learningData[snakeId];
        if (!data) return;

        // Record pattern learning
        if (data.patterns[action]) {
            data.patterns[action].attempts++;
            if (outcome === 'success') {
                data.patterns[action].attempts++;
            }
        }

        // Update memory based on context
        this.updateMemory(snakeId, action, outcome, context);

        // Adapt preferences based on outcomes
        this.adaptPreferences(snakeId, action, outcome);

        // Log significant learning events
        this.logLearningEvent(snakeId, action, outcome, context);
    }

    updateMemory(snakeId, action, outcome, context) {
        const memory = this.learningData[snakeId].memory;
        const position = context.position;

        if (!position) return;

        if (outcome === 'danger' || outcome === 'death') {
            memory.dangerousPositions.push({
                x: position.x,
                y: position.y,
                timestamp: Date.now(),
                action: action
            });

            // Keep only recent dangerous positions
            if (memory.dangerousPositions.length > 20) {
                memory.dangerousPositions = memory.dangerousPositions.slice(-20);
            }
        } else if (outcome === 'success') {
            if (action === 'foodSeeking') {
                memory.foodHotspots.push({
                    x: position.x,
                    y: position.y,
                    timestamp: Date.now()
                });

                if (memory.foodHotspots.length > 15) {
                    memory.foodHotspots = memory.foodHotspots.slice(-15);
                }
            } else {
                memory.safePositions.push({
                    x: position.x,
                    y: position.y,
                    timestamp: Date.now()
                });

                if (memory.safePositions.length > 25) {
                    memory.safePositions = memory.safePositions.slice(-25);
                }
            }
        }
    }

    adaptPreferences(snakeId, action, outcome) {
        const prefs = this.learningData[snakeId].preferences;
        const adaptationRate = 0.05;

        switch (action) {
            case 'enemyAvoidance':
                if (outcome === 'success') {
                    prefs.aggressiveness = Math.max(0, prefs.aggressiveness - adaptationRate);
                } else {
                    prefs.aggressiveness = Math.min(1, prefs.aggressiveness + adaptationRate);
                }
                break;

            case 'exploration':
                if (outcome === 'success') {
                    prefs.exploration = Math.min(1, prefs.exploration + adaptationRate);
                } else {
                    prefs.exploration = Math.max(0, prefs.exploration - adaptationRate);
                }
                break;

            case 'riskTaking':
                if (outcome === 'danger') {
                    prefs.riskTaking = Math.max(0, prefs.riskTaking - adaptationRate * 2);
                } else if (outcome === 'success') {
                    prefs.riskTaking = Math.min(1, prefs.riskTaking + adaptationRate);
                }
                break;
        }
    }

    logLearningEvent(snakeId, action, outcome, context) {
        const snake = context.snake;
        if (!snake) return;

        let message = '';
        
        switch (action) {
            case 'wallAvoidance':
                if (outcome === 'success') {
                    message = `${snake.name} learned to avoid wall collision`;
                } else {
                    message = `${snake.name} hit a wall and learned from it`;
                }
                break;

            case 'foodSeeking':
                if (outcome === 'success') {
                    message = `${snake.name} successfully found food using learned patterns`;
                } else {
                    message = `${snake.name} missed food opportunity, adjusting strategy`;
                }
                break;

            case 'powerupCollection':
                if (outcome === 'success') {
                    message = `${snake.name} collected powerup using improved tactics`;
                }
                break;

            case 'enemyAvoidance':
                if (outcome === 'success') {
                    message = `${snake.name} successfully avoided enemy collision`;
                } else {
                    message = `${snake.name} collided with enemy, learning defensive patterns`;
                }
                break;

            case 'adaptation':
                message = `${snake.name} ${context.adaptation}`;
                break;

            case 'evolution':
                message = context.message || `${snake.name} has evolved`;
                break;
        }

        if (message) {
            this.addLogEntry(message);
        }
    }

    addLogEntry(message) {
        const entry = {
            message: message,
            timestamp: new Date().toLocaleTimeString()
        };

        this.logEntries.unshift(entry);

        if (this.logEntries.length > this.maxLogEntries) {
            this.logEntries = this.logEntries.slice(0, this.maxLogEntries);
        }

        // Update UI
        this.updateLogDisplay();
    }

    updateLogDisplay() {
        const logElement = document.getElementById('learning-log');
        if (!logElement) return;

        logElement.innerHTML = this.logEntries
            .slice(0, 10) // Show only latest 10 entries
            .map(entry => `<div class="log-entry">[${entry.timestamp}] ${entry.message}</div>`)
            .join('');
    }

    updateWinnersCircleDisplay() {
        const winnersElement = document.getElementById('winners-circle');
        if (!winnersElement) return;
        
        const winnersStatus = window.game.snakeGenerator.getWinnersCircleStatus();
        
        if (winnersStatus.length === 0) {
            winnersElement.innerHTML = '<div style="opacity: 0.6;">No champions yet...</div>';
            return;
        }
        
        winnersElement.innerHTML = winnersStatus
            .map(champ => `
                <div style="margin: 3px 0; padding: 3px; background: rgba(255, 215, 0, 0.1); border-radius: 3px;">
                    <strong>${champ.name}</strong><br>
                    <small>Gen ${champ.generation} • Fitness: ${Math.round(champ.fitness)} • Used: ${champ.timesUsed}x • ${champ.generationsLeft} gens left</small>
                </div>
            `).join('');
    }

    getActionAdvice(snakeId, gameState) {
        const data = this.learningData[snakeId];
        if (!data) return null;

        const advice = {
            avoidPositions: [],
            seekPositions: [],
            preferredActions: [],
            riskAssessment: 'medium'
        };

        // Add dangerous positions to avoid
        data.memory.dangerousPositions.forEach(pos => {
            if (Date.now() - pos.timestamp < 30000) { // 30 seconds
                advice.avoidPositions.push(pos);
            }
        });

        // Add food hotspots to seek
        data.memory.foodHotspots.forEach(pos => {
            if (Date.now() - pos.timestamp < 60000) { // 1 minute
                advice.seekPositions.push(pos);
            }
        });

        // Determine risk assessment
        if (data.preferences.riskTaking < 0.3) {
            advice.riskAssessment = 'low';
        } else if (data.preferences.riskTaking > 0.7) {
            advice.riskAssessment = 'high';
        }

        // Suggest preferred actions based on learning
        const patterns = data.patterns;
        Object.keys(patterns).forEach(pattern => {
            const successRate = patterns[pattern].attempts > 0 
                ? patterns[pattern].success / patterns[pattern].attempts 
                : 0;
            
            if (successRate > 0.6) {
                advice.preferredActions.push(pattern);
            }
        });

        return advice;
    }

    recordBattleResult(winner, loser, battleData) {
        const battle = {
            winner: winner ? winner.name : 'Draw',
            loser: loser ? loser.name : 'Draw',
            timestamp: Date.now(),
            battleLength: battleData.duration || 0,
            winnerScore: winner ? winner.score : 0,
            loserScore: loser ? loser.score : 0
        };

        this.battleHistory.push(battle);

        if (loser) {
            this.deadSnakes.push({...loser});
            this.logLearningEvent(loser.id === 1 ? 'snake1' : 'snake2', 'evolution', 'death', {
                snake: loser,
                message: `${loser.name} died and will be replaced by evolved offspring`
            });
        }

        if (winner) {
            this.recordAction(
                winner.id === 1 ? 'snake1' : 'snake2',
                'victory',
                'success',
                { snake: winner, battleData: battleData }
            );
        }
    }

    getEvolutionCandidates() {
        return this.deadSnakes.slice(-10); // Return last 10 dead snakes for evolution
    }

    analyzePerformance(snake) {
        const analysis = {
            strengths: [],
            weaknesses: [],
            suggestions: []
        };

        const data = this.learningData[snake.id === 1 ? 'snake1' : 'snake2'];
        if (!data) return analysis;

        // Analyze patterns
        Object.keys(data.patterns).forEach(pattern => {
            const p = data.patterns[pattern];
            if (p.attempts > 5) {
                const successRate = p.success / p.attempts;
                if (successRate > 0.7) {
                    analysis.strengths.push(`Excellent ${pattern.replace(/([A-Z])/g, ' $1').toLowerCase()}`);
                } else if (successRate < 0.3) {
                    analysis.weaknesses.push(`Poor ${pattern.replace(/([A-Z])/g, ' $1').toLowerCase()}`);
                    analysis.suggestions.push(`Practice ${pattern.replace(/([A-Z])/g, ' $1').toLowerCase()}`);
                }
            }
        });

        // Analyze preferences
        const prefs = data.preferences;
        if (prefs.aggressiveness > 0.8) {
            analysis.strengths.push('Highly aggressive playstyle');
        } else if (prefs.aggressiveness < 0.2) {
            analysis.weaknesses.push('Too passive');
            analysis.suggestions.push('Be more aggressive when safe');
        }

        if (prefs.exploration > 0.7) {
            analysis.strengths.push('Good exploration instincts');
        } else if (prefs.exploration < 0.3) {
            analysis.suggestions.push('Explore more areas of the arena');
        }

        return analysis;
    }

    reset() {
        this.learningData.snake1 = this.initializeLearningData();
        this.learningData.snake2 = this.initializeLearningData();
        this.logEntries = [];
        this.updateLogDisplay();
        this.addLogEntry('Learning system reset - fresh start for new generation');
    }

    transferLearning(oldSnakeId, newSnake) {
        const oldData = this.learningData[oldSnakeId];
        const newData = this.initializeLearningData();
        
        // Transfer some learned preferences (50% inheritance)
        Object.keys(oldData.preferences).forEach(pref => {
            newData.preferences[pref] = (oldData.preferences[pref] + newData.preferences[pref]) / 2;
        });

        // Transfer some memory (25% inheritance)
        newData.memory.dangerousPositions = oldData.memory.dangerousPositions.slice(-5);
        newData.memory.safePositions = oldData.memory.safePositions.slice(-8);
        newData.memory.foodHotspots = oldData.memory.foodHotspots.slice(-5);

        this.learningData[oldSnakeId] = newData;

        this.addLogEntry(`${newSnake.name} inherited learned behaviors from previous generation`);
    }
}

// Export for use in other files
window.LearningSystem = LearningSystem;
