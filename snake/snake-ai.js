// Snake AI - Intelligent decision making for AI snakes
class SnakeAI {
    constructor(learning) {
        this.learning = learning;
        this.decisionHistory = [];
        this.maxHistoryLength = 20;
    }

    getNextMove(snake, gameState) {
        const currentDirection = snake.direction;
        const head = snake.body[0];
        const advice = this.learning.getActionAdvice(
            snake.id === 1 ? 'snake1' : 'snake2', 
            gameState
        );

        // Analyze the game state
        const analysis = this.analyzeGameState(snake, gameState);
          // Calculate move priorities
        const movePriorities = this.calculateMovePriorities(snake, analysis, advice, gameState);
          // Apply passive skills
        this.applyPersonality(snake, movePriorities, analysis);
        
        // Apply Uber Skills - special abilities
        this.processUberSkills(snake, analysis, movePriorities, gameState);
        
        // Choose the best move
        const chosenMove = this.chooseBestMove(movePriorities, currentDirection);
        
        // Record the decision for learning
        this.recordDecision(snake, chosenMove, analysis);
        
        return chosenMove;
    }

    analyzeGameState(snake, gameState) {
        const head = snake.body[0];
        const analysis = {
            threats: [],
            opportunities: [],
            distances: {},
            quadrant: this.getQuadrant(head, gameState.canvas),
            spaceAvailable: {},
            enemyThreat: null
        };

        // Analyze walls
        analysis.distances.walls = {
            up: head.y,
            down: gameState.canvas.height - head.y,
            left: head.x,
            right: gameState.canvas.width - head.x
        };

        // Find closest food
        if (gameState.food) {
            const foodDistance = this.calculateDistance(head, gameState.food);
            analysis.distances.food = foodDistance;
            analysis.opportunities.push({
                type: 'food',
                position: gameState.food,
                distance: foodDistance,
                priority: this.calculateFoodPriority(snake, gameState.food, foodDistance)
            });
        }        // Analyze powerups with enhanced vision
        gameState.powerups.forEach(powerup => {
            const distance = this.calculateDistance(head, powerup);
            let visionRange = 200; // Base vision range
            
            // Enhanced vision skills
            if (snake.skills.includes('Better Eyes')) {
                visionRange = 300;
            }
            if (snake.skills.includes('Eagle Vision')) {
                visionRange = 400;
            }
            if (snake.skills.includes('Sixth Sense')) {
                visionRange = 350;
            }
            
            // Only consider powerups within vision range
            if (distance <= visionRange) {
                analysis.opportunities.push({
                    type: 'powerup',
                    subtype: powerup.type,
                    position: powerup,
                    distance: distance,
                    priority: this.calculatePowerupPriority(snake, powerup, distance)
                });
            }
        });

        // Analyze enemy snake
        const enemy = snake.id === 1 ? gameState.snake2 : gameState.snake1;
        if (enemy && enemy.body.length > 0) {
            const enemyHead = enemy.body[0];
            const enemyDistance = this.calculateDistance(head, enemyHead);
              analysis.enemyThreat = {
                head: enemyHead,
                position: enemyHead, // Add position property for consistency
                body: enemy.body,
                distance: enemyDistance,
                direction: enemy.direction,
                dangerLevel: this.assessEnemyDanger(snake, enemy, enemyDistance)
            };

            // Check for collision threats
            enemy.body.forEach((segment, index) => {
                const segmentDistance = this.calculateDistance(head, segment);
                if (segmentDistance < 60) { // Nearby threat
                    analysis.threats.push({
                        type: 'enemy_body',
                        position: segment,
                        distance: segmentDistance,
                        severity: index === 0 ? 'high' : 'medium'
                    });
                }
            });
        }

        // Check self-collision threats
        snake.body.slice(1).forEach((segment, index) => {
            const segmentDistance = this.calculateDistance(head, segment);
            if (segmentDistance < 60) {
                analysis.threats.push({
                    type: 'self_body',
                    position: segment,
                    distance: segmentDistance,
                    severity: 'high'
                });
            }
        });

        // Analyze available space in each direction
        ['up', 'down', 'left', 'right'].forEach(direction => {
            analysis.spaceAvailable[direction] = this.calculateAvailableSpace(head, direction, gameState);
        });

        return analysis;
    }    calculateMovePriorities(snake, analysis, advice, gameState) {
        const priorities = {
            up: 0,
            down: 0,
            left: 0,
            right: 0
        };

        const head = snake.body[0];

        // Safety first - avoid immediate threats
        analysis.threats.forEach(threat => {
            const direction = this.getDirectionTowards(head, threat.position);
            const avoidDirection = this.getOppositeDirection(direction);
            
            const penalty = threat.severity === 'high' ? -1000 : -500;
            priorities[direction] += penalty;
            priorities[avoidDirection] += Math.abs(penalty) * 0.3;
        });        // Avoid walls - but not too aggressively to prevent wall hugging
        Object.keys(analysis.distances.walls).forEach(direction => {
            const distance = analysis.distances.walls[direction];
            if (distance < 30) {
                priorities[direction] -= (30 - distance) * 50; // Stronger penalty for very close walls
            } else if (distance < 60) {
                priorities[direction] -= (60 - distance) * 10; // Lighter penalty for moderate distance
            }
        });        // Seek opportunities with powerup competition
        analysis.opportunities.forEach(opportunity => {
            const direction = this.getDirectionTowards(head, opportunity.position);
            let bonus = opportunity.priority;
            
            // EXTRA bonus for food - make it very attractive
            if (opportunity.type === 'food') {
                bonus *= 2; // Double the food seeking behavior
                
                // If snake is small, make food even more attractive
                if (snake.body.length < 6) {
                    bonus *= 1.5;
                }
            }
            
            // POWERUP seeking behavior
            if (opportunity.type === 'powerup') {
                // Competitive bonus if enemy is also near the powerup
                const enemy = snake.id === 1 ? gameState.snake2 : gameState.snake1;
                if (enemy && enemy.body.length > 0) {
                    const enemyDistance = this.calculateDistance(enemy.body[0], opportunity.position);
                    if (enemyDistance < opportunity.distance + 50) {
                        bonus *= 1.6; // Competition bonus
                    }
                }
                
                // Extra bonus for valuable powerups
                if (opportunity.subtype === 'invincibility' || opportunity.subtype === 'speed') {
                    bonus *= 1.4;
                }
            }
            
            priorities[direction] += bonus;
            
            // Also slightly boost adjacent directions for better pathfinding
            if (opportunity.type === 'food' || opportunity.type === 'powerup') {
                const adjacentDirs = this.getAdjacentDirections(direction);
                adjacentDirs.forEach(adjDir => {
                    priorities[adjDir] += bonus * 0.25;
                });
            }
        });// Consider available space - but prioritize food over space
        Object.keys(analysis.spaceAvailable).forEach(direction => {
            const space = analysis.spaceAvailable[direction];
            priorities[direction] += space * 1; // Reduced from 2 to 1
        });        // FOOD OVERRIDE - if there's food, make it the dominant factor
        if (analysis.opportunities.some(op => op.type === 'food')) {
            const foodOp = analysis.opportunities.find(op => op.type === 'food');
            const foodDirection = this.getDirectionTowards(head, foodOp.position);
            
            // Massive food bonus to override other behaviors
            priorities[foodDirection] += 600; // Reduced from 800 to allow powerup competition
            
            // Reduce wall hugging when food is available
            Object.keys(priorities).forEach(dir => {
                if (dir !== foodDirection && analysis.distances.walls[dir] < 100) {
                    priorities[dir] -= 100; // Penalty for staying near walls when food is available
                }
            });
        }
        
        // POWERUP OVERRIDE - if there's a valuable powerup nearby, compete for it
        const valuablePowerups = analysis.opportunities.filter(op => 
            op.type === 'powerup' && 
            (op.subtype === 'invincibility' || op.subtype === 'speed' || op.distance < 100)
        );
        
        if (valuablePowerups.length > 0) {
            const bestPowerup = valuablePowerups.reduce((best, current) => 
                current.priority > best.priority ? current : best
            );
            
            const powerupDirection = this.getDirectionTowards(head, bestPowerup.position);
            priorities[powerupDirection] += 500; // Strong powerup seeking
            
            // If enemy is closer, increase urgency
            const enemy = snake.id === 1 ? gameState.snake2 : gameState.snake1;
            if (enemy && enemy.body.length > 0) {
                const enemyDistance = this.calculateDistance(enemy.body[0], bestPowerup.position);
                if (enemyDistance < bestPowerup.distance) {
                    priorities[powerupDirection] += 300; // URGENT competition
                }
            }
        }

        // Apply learning advice
        if (advice) {
            advice.avoidPositions.forEach(pos => {
                const direction = this.getDirectionTowards(head, pos);
                priorities[direction] -= 200;
            });

            advice.seekPositions.forEach(pos => {
                const direction = this.getDirectionTowards(head, pos);
                priorities[direction] += 150;
            });
        }

        // Process Uber Skills - special abilities that affect decision making
        this.processUberSkills(snake, analysis, priorities, gameState);

        return priorities;
    }

    applyPersonality(snake, priorities, analysis) {
        // Apply passive skills
        snake.skills.forEach(skill => {
            switch (skill) {
                case 'Speed Boost':
                    // Prefer direct paths
                    if (analysis.opportunities.length > 0) {
                        const closest = analysis.opportunities[0];
                        const direction = this.getDirectionTowards(snake.body[0], closest.position);
                        priorities[direction] += 100;
                    }
                    break;

                case 'Sharp Turn':
                    // Less penalty for direction changes
                    Object.keys(priorities).forEach(direction => {
                        if (direction !== snake.direction && direction !== this.getOppositeDirection(snake.direction)) {
                            priorities[direction] += 50;
                        }
                    });
                    break;                case 'Wall Hugger':
                    // Only slight preference for edges, not dominant behavior
                    if (analysis.distances.walls.up < 80) priorities.up += 20;
                    if (analysis.distances.walls.down < 80) priorities.down += 20;
                    if (analysis.distances.walls.left < 80) priorities.left += 20;
                    if (analysis.distances.walls.right < 80) priorities.right += 20;
                    break;

                case 'Center Seeker':
                    // Strong preference for center
                    const centerX = 400; // Canvas width / 2
                    const centerY = 300; // Canvas height / 2
                    const head = snake.body[0];
                    const distanceToCenter = this.calculateDistance(head, {x: centerX, y: centerY});
                    
                    if (distanceToCenter > 100) {
                        const centerDirection = this.getDirectionTowards(head, {x: centerX, y: centerY});
                        priorities[centerDirection] += 100;
                    }
                    break;

                case 'Danger Sense':
                    // Extra penalty for threats
                    analysis.threats.forEach(threat => {
                        const direction = this.getDirectionTowards(snake.body[0], threat.position);
                        priorities[direction] -= 100;
                    });
                    break;

                case 'Food Sense':
                    // Extra bonus for food
                    if (analysis.opportunities.some(op => op.type === 'food')) {
                        const foodOp = analysis.opportunities.find(op => op.type === 'food');
                        const direction = this.getDirectionTowards(snake.body[0], foodOp.position);
                        priorities[direction] += 150;
                    }
                    break;                case 'Power Hunter':
                    // Actively seek power-ups with high priority
                    analysis.opportunities.forEach(opp => {
                        if (opp.type === 'powerup') {
                            const direction = this.getDirectionTowards(snake.body[0], opp.position);
                            priorities[direction] += 150; // Very high priority
                        }
                    });
                    break;

                case 'Quick Reflex':
                    // Enhanced threat avoidance
                    analysis.threats.forEach(threat => {
                        if (threat.distance < 60) { // Close threats
                            const direction = this.getDirectionTowards(snake.body[0], threat.position);
                            priorities[direction] -= 200; // Double penalty
                        }
                    });
                    break;

                case 'Survival Instinct':
                    // More defensive when health/size is low
                    if (snake.body.length < 5) {
                        analysis.threats.forEach(threat => {
                            const direction = this.getDirectionTowards(snake.body[0], threat.position);
                            priorities[direction] -= 150;
                        });
                        // Prefer safer moves
                        priorities.up += snake.body.length < 3 ? 80 : 40;
                        priorities.down += snake.body.length < 3 ? 80 : 40;
                        priorities.left += snake.body.length < 3 ? 80 : 40;
                        priorities.right += snake.body.length < 3 ? 80 : 40;
                    }
                    break;

                case 'Territory Control':
                    // Defend current area, avoid wandering too far
                    const homeQuadrant = analysis.quadrant;
                    // Penalize moves that take us away from our territory
                    Object.keys(priorities).forEach(direction => {
                        const newQuadrant = this.predictQuadrant(snake.body[0], direction);
                        if (newQuadrant !== homeQuadrant) {
                            priorities[direction] -= 60;
                        }
                    });
                    break;

                case 'Opportunist':
                    // Bonus for moves that exploit enemy vulnerabilities
                    if (analysis.enemyThreat && analysis.enemyThreat.distance > 40) {
                        // Enemy is far, be more aggressive toward food
                        analysis.opportunities.forEach(opp => {
                            if (opp.type === 'food') {
                                const direction = this.getDirectionTowards(snake.body[0], opp.position);
                                priorities[direction] += 120;
                            }
                        });
                    }
                    break;                case 'Risk Assessment':
                    // More careful evaluation of risky moves
                    Object.keys(priorities).forEach(direction => {
                        try {
                            const riskLevel = this.evaluateRisk(snake, direction, analysis);
                            if (riskLevel > 0.7) { // High risk
                                priorities[direction] -= 100;
                            } else if (riskLevel < 0.3) { // Low risk
                                priorities[direction] += 50;
                            }
                        } catch (error) {
                            console.warn('Risk evaluation error for direction:', direction, error);
                        }
                    });
                    break;

                case 'Better Eyes':
                    // Enhanced food detection (already handled in analyzeGameState with extended range)
                    // Double the bonus for distant food
                    analysis.opportunities.forEach(opp => {
                        if (opp.type === 'food' && opp.distance > 100) {
                            const direction = this.getDirectionTowards(snake.body[0], opp.position);
                            priorities[direction] += 80;
                        }
                    });
                    break;

                case 'Efficient Metabolism':
                    // Prioritize food even more
                    analysis.opportunities.forEach(opp => {
                        if (opp.type === 'food') {
                            const direction = this.getDirectionTowards(snake.body[0], opp.position);
                            priorities[direction] += 100;
                        }
                    });
                    break;

                case 'Alpha Predator':
                    // Aggressive toward enemies, bonus for confrontation
                    if (analysis.enemyThreat && analysis.enemyThreat.distance < 100) {
                        const direction = this.getDirectionTowards(snake.body[0], analysis.enemyThreat.position);
                        priorities[direction] += 80; // Approach enemy
                    }
                    break;

                case 'Strategic Retreat':
                    // Smart retreating when overwhelmed
                    const threatCount = analysis.threats.length;
                    if (threatCount > 2 || (analysis.enemyThreat && analysis.enemyThreat.distance < 40)) {
                        // Find escape routes
                        Object.keys(priorities).forEach(direction => {
                            const isEscape = analysis.threats.every(threat => {
                                const directionToThreat = this.getDirectionTowards(snake.body[0], threat.position);
                                return direction !== directionToThreat;
                            });
                            if (isEscape) {
                                priorities[direction] += 120;
                            }
                        });
                    }
                    break;
            }
        });

        // Apply stat-based personality
        const stats = snake.stats;
        
        // Aggression affects enemy interaction
        if (analysis.enemyThreat) {
            const aggressionFactor = (stats.aggression - 50) * 2;
            if (aggressionFactor > 0) {
                // More aggressive - approach enemy
                const direction = this.getDirectionTowards(snake.body[0], analysis.enemyThreat.head);
                priorities[direction] += aggressionFactor;
            } else {
                // Less aggressive - avoid enemy
                const direction = this.getDirectionAwayFrom(snake.body[0], analysis.enemyThreat.head);
                priorities[direction] += Math.abs(aggressionFactor);
            }
        }        // Intelligence affects decision quality (reduces randomness)
        const intelligenceFactor = stats.intelligence / 100;
        const randomNoise = (1 - intelligenceFactor) * 30; // Reduced noise
        
        Object.keys(priorities).forEach(direction => {
            priorities[direction] += (Math.random() - 0.5) * randomNoise;
        });

        // Speed affects preference for direct paths to food
        if (stats.speed > 70 && analysis.opportunities.some(op => op.type === 'food')) {
            const foodOp = analysis.opportunities.find(op => op.type === 'food');
            const foodDirection = this.getDirectionTowards(snake.body[0], foodOp.position);
            priorities[foodDirection] += 150; // Speed demons go straight for food
        }

        // Add exploration behavior to prevent wall patrolling
        if (!analysis.opportunities.some(op => op.type === 'food')) {
            // No food visible - encourage exploration toward center
            const centerX = 400;
            const centerY = 300;
            const centerDirection = this.getDirectionTowards(snake.body[0], {x: centerX, y: centerY});
            priorities[centerDirection] += 200; // Explore toward center
        }
    }

    chooseBestMove(priorities, currentDirection) {
        // Remove reverse direction (can't go backwards)
        const oppositeDirection = this.getOppositeDirection(currentDirection);
        delete priorities[oppositeDirection];

        // Find the best move
        let bestDirection = null;
        let bestPriority = -Infinity;

        Object.entries(priorities).forEach(([direction, priority]) => {
            if (priority > bestPriority) {
                bestPriority = priority;
                bestDirection = direction;
            }
        });

        return bestDirection || currentDirection;
    }

    recordDecision(snake, chosenMove, analysis) {
        const decision = {
            snake: snake.name,
            move: chosenMove,
            threats: analysis.threats.length,
            opportunities: analysis.opportunities.length,
            timestamp: Date.now()
        };

        this.decisionHistory.push(decision);
        
        if (this.decisionHistory.length > this.maxHistoryLength) {
            this.decisionHistory.shift();
        }

        // Record learning data
        if (analysis.opportunities.length > 0) {
            this.learning.recordAction(
                snake.id === 1 ? 'snake1' : 'snake2',
                'exploration',
                'attempt',
                { snake: snake, position: snake.body[0] }
            );
        }

        if (analysis.threats.length > 0) {
            this.learning.recordAction(
                snake.id === 1 ? 'snake1' : 'snake2',
                'wallAvoidance',
                'attempt',
                { snake: snake, position: snake.body[0] }
            );
        }
    }

    // Process Uber Skills - special abilities that affect decision making
    processUberSkills(snake, analysis, movePriorities, gameState) {
        if (!snake.uberSkills || snake.uberSkills.length === 0) return;

        snake.uberSkills.forEach(uberSkill => {
            switch (uberSkill.name) {
                case 'Quantum Mind':
                    // Enhanced prediction - boost food-seeking priority
                    this.enhanceQuantumMindPrediction(snake, analysis, movePriorities, gameState);
                    break;
                    
                case 'Time Predator':
                    // More aggressive behavior - prioritize enemy confrontation
                    this.enhanceTimePredatorAggression(snake, analysis, movePriorities);
                    break;
                    
                case 'Void Walker':
                    // Less wall avoidance due to phase ability
                    this.enhanceVoidWalkerMovement(snake, analysis, movePriorities);
                    break;
                    
                case 'Genesis Blood':
                    // More risk-taking due to regeneration
                    this.enhanceGenesisBloodRiskTaking(snake, analysis, movePriorities);
                    break;
                    
                case 'Soul Absorber':
                    // Extremely aggressive toward enemies
                    this.enhanceSoulAbsorberAggression(snake, analysis, movePriorities);
                    break;
                    
                case 'Alpha Genome':
                    // Perfect stats lead to more confident play
                    this.enhanceAlphaGenomeConfidence(snake, analysis, movePriorities);
                    break;
                    
                case 'Reality Bender':
                    // Strategic positioning for teleport opportunities
                    this.enhanceRealityBenderStrategy(snake, analysis, movePriorities, gameState);
                    break;
                    
                case 'God Serpent':
                    // Ultimate dominance behavior
                    this.enhanceGodSerpentDominance(snake, analysis, movePriorities, gameState);
                    break;
            }
        });
    }

    enhanceQuantumMindPrediction(snake, analysis, movePriorities, gameState) {
        // Boost food-seeking with perfect prediction
        if (gameState.food) {
            const head = snake.body[0];
            const foodDirection = this.getDirectionTowards(head, gameState.food);
            movePriorities[foodDirection] += 40; // Strong boost for predicted food path
        }
        
        // Predict enemy movements better
        if (analysis.enemyThreat) {
            const predictedEnemyPos = this.predictEnemyPosition(analysis.enemyThreat, 3);
            const avoidDirection = this.getDirectionAway(snake.body[0], predictedEnemyPos);
            movePriorities[avoidDirection] += 25;
        }
    }

    enhanceTimePredatorAggression(snake, analysis, movePriorities) {
        // More aggressive toward enemies
        if (analysis.enemyThreat && analysis.enemyThreat.distance < 100) {
            const attackDirection = this.getDirectionTowards(snake.body[0], analysis.enemyThreat.position);
            movePriorities[attackDirection] += 35; // Aggressive pursuit
        }
    }

    enhanceVoidWalkerMovement(snake, analysis, movePriorities) {
        // Reduce wall avoidance penalties since can phase through
        Object.keys(movePriorities).forEach(direction => {
            const wallDistance = analysis.distances.walls[direction] || 999;
            if (wallDistance < 30) {
                movePriorities[direction] += 20; // Reduce wall fear
            }
        });
    }

    enhanceGenesisBloodRiskTaking(snake, analysis, movePriorities) {
        // Take more risks since can regenerate
        analysis.threats.forEach(threat => {
            if (threat.severity === 'medium') {
                const direction = this.getDirectionTowards(snake.body[0], threat.position);
                movePriorities[direction] += 15; // Less afraid of medium threats
            }
        });
    }

    enhanceSoulAbsorberAggression(snake, analysis, movePriorities) {
        // Extremely aggressive - seek enemy confrontation
        if (analysis.enemyThreat) {
            const attackDirection = this.getDirectionTowards(snake.body[0], analysis.enemyThreat.position);
            movePriorities[attackDirection] += 50; // Maximum aggression
        }
    }

    enhanceAlphaGenomeConfidence(snake, analysis, movePriorities) {
        // More confident movement due to perfect stats
        Object.keys(movePriorities).forEach(direction => {
            movePriorities[direction] += 10; // General confidence boost
        });
    }

    enhanceRealityBenderStrategy(snake, analysis, movePriorities, gameState) {
        // Strategic positioning for teleport usage
        const center = {
            x: gameState.canvas.width / 2,
            y: gameState.canvas.height / 2
        };
        const centerDirection = this.getDirectionTowards(snake.body[0], center);
        movePriorities[centerDirection] += 15; // Prefer center control
    }

    enhanceGodSerpentDominance(snake, analysis, movePriorities, gameState) {
        // Ultimate power - dominate the field
        if (gameState.food) {
            const foodDirection = this.getDirectionTowards(snake.body[0], gameState.food);
            movePriorities[foodDirection] += 60; // Absolute food control
        }
        
        if (analysis.enemyThreat) {
            const attackDirection = this.getDirectionTowards(snake.body[0], analysis.enemyThreat.position);
            movePriorities[attackDirection] += 45; // Divine aggression
        }
    }

    predictEnemyPosition(enemyThreat, steps) {
        // Simple prediction of where enemy will be in 'steps' moves
        const currentPos = enemyThreat.position;
        const direction = enemyThreat.direction;
        
        let futureX = currentPos.x;
        let futureY = currentPos.y;
        
        for (let i = 0; i < steps; i++) {
            switch (direction) {
                case 'up': futureY -= 20; break;
                case 'down': futureY += 20; break;
                case 'left': futureX -= 20; break;
                case 'right': futureX += 20; break;
            }
        }
        
        return { x: futureX, y: futureY };
    }

    evaluateRisk(snake, direction, analysis) {
        const head = snake.body[0];
        const moveDistance = 20;
        let newX = head.x;
        let newY = head.y;
        
        // Calculate new position
        switch (direction) {
            case 'up': newY -= moveDistance; break;
            case 'down': newY += moveDistance; break;
            case 'left': newX -= moveDistance; break;
            case 'right': newX += moveDistance; break;
        }
        
        let riskScore = 0;
        
        // Wall collision risk
        if (newX < 0 || newX >= 800 || newY < 0 || newY >= 600) {
            riskScore += 1.0; // Certain death
        }
        
        // Self collision risk
        const wouldHitSelf = snake.body.some(segment => 
            Math.abs(segment.x - newX) < 20 && Math.abs(segment.y - newY) < 20
        );
        if (wouldHitSelf) {
            riskScore += 1.0;
        }        // Enemy proximity risk
        if (analysis.enemyThreat && analysis.enemyThreat.head) {
            const distanceToEnemy = this.calculateDistance({x: newX, y: newY}, analysis.enemyThreat.head);
            if (distanceToEnemy < 40) {
                riskScore += 0.8;
            } else if (distanceToEnemy < 80) {
                riskScore += 0.4;
            }
        }
        
        // Threat proximity risk
        analysis.threats.forEach(threat => {
            const distanceToThreat = this.calculateDistance({x: newX, y: newY}, threat.position);
            if (distanceToThreat < 30) {
                riskScore += 0.6;
            } else if (distanceToThreat < 60) {
                riskScore += 0.3;
            }
        });
        
        return Math.min(riskScore, 1.0); // Cap at 1.0
    }
    
    predictQuadrant(position, direction) {
        const moveDistance = 20;
        let newX = position.x;
        let newY = position.y;
        
        switch (direction) {
            case 'up': newY -= moveDistance; break;
            case 'down': newY += moveDistance; break;
            case 'left': newX -= moveDistance; break;
            case 'right': newX += moveDistance; break;
        }
        
        return this.getQuadrant({x: newX, y: newY}, {width: 800, height: 600});
    }    // Utility functions
    calculateDistance(pos1, pos2) {
        // Add null checks to prevent crashes
        if (!pos1 || !pos2 || typeof pos1.x === 'undefined' || typeof pos1.y === 'undefined' || 
            typeof pos2.x === 'undefined' || typeof pos2.y === 'undefined') {
            return Infinity; // Return a large distance if positions are invalid
        }
        return Math.sqrt(Math.pow(pos1.x - pos2.x, 2) + Math.pow(pos1.y - pos2.y, 2));
    }

    getDirectionTowards(from, to) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;

        if (Math.abs(dx) > Math.abs(dy)) {
            return dx > 0 ? 'right' : 'left';
        } else {
            return dy > 0 ? 'down' : 'up';
        }
    }

    getDirectionAwayFrom(from, away) {
        const dx = from.x - away.x;
        const dy = from.y - away.y;

        if (Math.abs(dx) > Math.abs(dy)) {
            return dx > 0 ? 'right' : 'left';
        } else {
            return dy > 0 ? 'down' : 'up';
        }
    }

    getOppositeDirection(direction) {
        const opposites = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        };
        return opposites[direction];
    }

    getQuadrant(position, canvas) {
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const margin = 100;

        if (Math.abs(position.x - centerX) < margin && Math.abs(position.y - centerY) < margin) {
            return 'center';
        }

        if (position.x < centerX && position.y < centerY) return 'top-left';
        if (position.x >= centerX && position.y < centerY) return 'top-right';
        if (position.x < centerX && position.y >= centerY) return 'bottom-left';
        return 'bottom-right';
    }

    calculateAvailableSpace(position, direction, gameState) {
        let space = 0;
        let currentPos = { ...position };
        const step = 20;

        while (space < 200) { // Max check distance
            switch (direction) {
                case 'up': currentPos.y -= step; break;
                case 'down': currentPos.y += step; break;
                case 'left': currentPos.x -= step; break;
                case 'right': currentPos.x += step; break;
            }

            // Check bounds
            if (currentPos.x < 0 || currentPos.x >= gameState.canvas.width ||
                currentPos.y < 0 || currentPos.y >= gameState.canvas.height) {
                break;
            }

            // Check collisions
            let collision = false;
            
            [gameState.snake1, gameState.snake2].forEach(snake => {
                if (snake && snake.body) {
                    snake.body.forEach(segment => {
                        if (this.calculateDistance(currentPos, segment) < 15) {
                            collision = true;
                        }
                    });
                }
            });

            if (collision) break;

            space += step;
        }

        return space;
    }    calculateFoodPriority(snake, food, distance) {
        let priority = Math.max(0, 400 - distance * 2); // Much higher base priority
        
        // MUCH higher priority when hungry (shorter snake)
        if (snake.body.length < 5) {
            priority *= 3;
        } else if (snake.body.length < 8) {
            priority *= 2;
        }

        // Apply food sense skill
        if (snake.skills.includes('Food Sense')) {
            priority *= 2; // Double bonus instead of 1.3
        }
        
        // Extra priority if low on lives
        if (snake.lives <= 1) {
            priority *= 2;
        }

        // Competitive bonus - if enemy is closer to food, increase urgency
        const enemy = snake.id === 1 ? window.game.snake2 : window.game.snake1;
        if (enemy && enemy.body.length > 0) {
            const enemyDistance = this.calculateDistance(enemy.body[0], food);
            if (enemyDistance < distance) {
                priority *= 1.8; // Competitive urgency
            }
        }

        return priority;
    }    calculatePowerupPriority(snake, powerup, distance) {
        let priority = Math.max(0, 300 - distance * 1.5); // Increased base priority

        // Different powerups have different values
        switch (powerup.type) {
            case 'speed':
                if (snake.stats.speed < 70) priority *= 2;
                priority += 100; // Base attractiveness
                break;
            case 'invincibility':
                priority *= 2.5; // Very valuable
                priority += 150;
                break;
            case 'score':
                priority *= 1.5;
                priority += 75;
                break;
            case 'shrink':
                if (snake.body.length > 8) priority *= 2;
                priority += 50;
                break;
        }

        // Apply power hunter skill
        if (snake.skills.includes('Power Hunter')) {
            priority *= 2.5; // Increased from 1.5
        }
        
        // Apply better eyes skill - sees powerups from farther away
        if (snake.skills.includes('Better Eyes')) {
            priority *= 1.8;
            if (distance > 150) priority *= 1.5; // Especially good at spotting distant powerups
        }
        
        // Apply eagle vision - ultimate powerup detection
        if (snake.skills.includes('Eagle Vision')) {
            priority *= 2.2;
            priority += 200; // Flat bonus
        }
        
        // Fortune hunter loves all powerups
        if (snake.skills.includes('Fortune Hunter')) {
            priority *= 1.7;
            priority += 100;
        }

        return priority;
    }

    assessEnemyDanger(snake, enemy, distance) {
        if (distance > 100) return 'low';
        if (distance > 60) return 'medium';
        
        // Check if enemy is larger
        if (enemy.body.length > snake.body.length + 2) {
            return 'high';
        }

        // Check if enemy is faster
        if (enemy.stats && enemy.stats.speed > snake.stats.speed + 20) {
            return 'high';
        }

        return distance < 30 ? 'high' : 'medium';
    }

    getAdjacentDirections(direction) {
        const adjacent = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down']
        };
        return adjacent[direction] || [];
    }
}

// Export for use in other files
window.SnakeAI = SnakeAI;
