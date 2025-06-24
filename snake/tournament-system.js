// Tournament System - 40 Snake Battle Royale with Genetic Absorption
class TournamentSystem {
    constructor(snakeGenerator, learningSystem) {
        this.snakeGenerator = snakeGenerator;
        this.learningSystem = learningSystem;
        
        // Tournament structure
        this.tournamentSize = 40;
        this.currentRound = 0;
        this.maxRounds = 6; // 40 -> 20 -> 10 -> 5 -> 3 -> 2 -> 1
        this.tournamentActive = false;
        
        // Current tournament state
        this.currentBracket = [];
        this.roundResults = [];
        this.currentMatchIndex = 0;
        this.tournamentWinner = null;
        
        // Uber Skills - legendary mutations
        this.uberSkills = [
            { name: 'Genesis Blood', description: 'Regenerates lost segments every 10 seconds', rarity: 'legendary' },
            { name: 'Void Walker', description: 'Can phase through walls for 2 seconds every 15 seconds', rarity: 'legendary' },
            { name: 'Time Predator', description: 'Slows time perception, moves at 2x speed for 5 seconds', rarity: 'legendary' },
            { name: 'Soul Absorber', description: 'Gains permanent stat boost from each kill', rarity: 'legendary' },
            { name: 'Reality Bender', description: 'Can teleport to any empty space once per battle', rarity: 'legendary' },
            { name: 'Devourer', description: 'Consumes enemies faster and gains double genetic material', rarity: 'legendary' },
            { name: 'Blood Hunter', description: 'Heals and grows when consuming fallen enemies', rarity: 'legendary' },
            { name: 'Alpha Genome', description: 'All stats +25, immune to negative mutations', rarity: 'mythic' },
            { name: 'Quantum Mind', description: 'Predicts enemy moves 3 seconds ahead', rarity: 'mythic' },
            { name: 'Perfect Evolution', description: 'Always inherits best traits, never weakens', rarity: 'mythic' },
            { name: 'Apex Predator', description: 'Feeding grants temporary invincibility and size boost', rarity: 'mythic' },
            { name: 'Genetic Vampire', description: 'Steals enemy skills permanently when feeding', rarity: 'mythic' },
            { name: 'God Serpent', description: 'Transforms battlefield, controls food spawns', rarity: 'cosmic' },
            { name: 'Universal Code', description: 'Can rewrite own genetic code mid-battle', rarity: 'cosmic' },
            { name: 'Omega Evolution', description: 'Feeding triggers instant evolution and new abilities', rarity: 'cosmic' }
        ];
        
        // Tournament history
        this.tournamentHistory = [];
        this.championLineage = [];
    }

    initializeTournament() {
        this.currentRound = 1;
        this.currentMatchIndex = 0;
        this.tournamentActive = true;
        this.roundResults = [];
        
        // Generate 40 diverse snakes
        this.currentBracket = [];
        for (let i = 0; i < this.tournamentSize; i++) {
            const snake = this.snakeGenerator.createSnake();
            snake.tournamentId = i + 1;
            snake.tournamentSeed = i + 1;
            snake.roundsWon = 0;
            snake.genesAbsorbed = [];
            this.currentBracket.push(snake);
        }
        
        // Shuffle bracket for random matchups
        this.shuffleBracket();
        
        this.learningSystem.addLogEntry('🏟️ TOURNAMENT INITIALIZED: 40 snakes enter, 1 emerges!');
        this.learningSystem.addLogEntry(`Round 1: ${this.currentBracket.length} contestants ready`);
        
        return this.getNextMatch();
    }

    shuffleBracket() {
        for (let i = this.currentBracket.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.currentBracket[i], this.currentBracket[j]] = [this.currentBracket[j], this.currentBracket[i]];
        }
    }    getNextMatch() {
        console.log('getNextMatch called:', {
            tournamentActive: this.tournamentActive,
            currentMatchIndex: this.currentMatchIndex,
            bracketLength: this.currentBracket.length,
            roundResultsLength: this.roundResults.length
        });
        
        if (!this.tournamentActive) {
            console.log('Tournament not active');
            return null;
        }
        
        // Check if round is complete
        if (this.currentMatchIndex >= Math.floor(this.currentBracket.length / 2)) {
            console.log('Round complete, advancing to next round');
            return this.advanceRound();
        }
        
        // Get next pair
        const snake1Index = this.currentMatchIndex * 2;
        const snake2Index = snake1Index + 1;
        
        console.log('Match indices:', { snake1Index, snake2Index, bracketLength: this.currentBracket.length });
        
        if (snake2Index >= this.currentBracket.length) {
            // Odd number - last snake gets bye
            const byeSnake = this.currentBracket[snake1Index];
            byeSnake.roundsWon++;
            this.roundResults.push(byeSnake);
            this.learningSystem.addLogEntry(`${byeSnake.name} advances with a bye`);
            this.currentMatchIndex++;
            return this.getNextMatch();
        }
        
        const match = {
            snake1: this.currentBracket[snake1Index],
            snake2: this.currentBracket[snake2Index],
            round: this.currentRound,
            matchNumber: this.currentMatchIndex + 1,
            totalMatches: Math.floor(this.currentBracket.length / 2)
        };
        
        console.log('Created match:', match);
        this.learningSystem.addLogEntry(`🥊 Round ${this.currentRound} Match ${match.matchNumber}: ${match.snake1.name} vs ${match.snake2.name}`);
        
        return match;
    }

    recordMatchResult(winner, loser) {
        if (!this.tournamentActive) return;
        
        // Genetic absorption - winner takes beneficial genes
        this.performGeneticAbsorption(winner, loser);
        
        // Check for uber mutations
        this.checkForUberMutation(winner);
        
        // Record result
        winner.roundsWon++;
        winner.victories++;
        loser.defeats++;
        
        this.roundResults.push(winner);
        this.currentMatchIndex++;
        
        this.learningSystem.addLogEntry(`🏆 ${winner.name} defeats ${loser.name} and absorbs their genes!`);
        
        // Show genetic changes
        if (winner.lastAbsorption) {
            this.learningSystem.addLogEntry(`  Gene absorption: ${winner.lastAbsorption}`);
        }
        
        if (winner.lastMutation) {
            this.learningSystem.addLogEntry(`  🌟 MUTATION: ${winner.lastMutation}`);
        }
    }

    performGeneticAbsorption(winner, loser) {
        const absorption = [];
        
        // Absorb superior stats (only improvements)
        Object.keys(loser.stats).forEach(stat => {
            if (loser.stats[stat] > winner.stats[stat]) {
                const improvement = Math.floor((loser.stats[stat] - winner.stats[stat]) * 0.3); // 30% of difference
                winner.stats[stat] += improvement;
                absorption.push(`+${improvement} ${stat}`);
            }
        });
        
        // Absorb beneficial skills (no duplicates)
        loser.skills.forEach(skill => {
            if (!winner.skills.includes(skill)) {
                // Random chance to absorb each skill
                if (Math.random() < 0.25) { // 25% chance per skill
                    winner.skills.push(skill);
                    absorption.push(`gained ${skill}`);
                }
            }
        });
        
        // Absorb uber skills
        if (loser.uberSkills) {
            loser.uberSkills.forEach(uberSkill => {
                if (!winner.uberSkills) winner.uberSkills = [];
                if (!winner.uberSkills.some(skill => skill.name === uberSkill.name)) {
                    if (Math.random() < 0.5) { // 50% chance for uber skills
                        winner.uberSkills.push(uberSkill);
                        absorption.push(`inherited ${uberSkill.name} (${uberSkill.rarity})`);
                    }
                }
            });
        }
        
        // Record absorption
        if (!winner.genesAbsorbed) winner.genesAbsorbed = [];
        winner.genesAbsorbed.push({
            from: loser.name,
            round: this.currentRound,
            changes: absorption
        });
        
        winner.lastAbsorption = absorption.length > 0 ? absorption.join(', ') : 'no beneficial genes found';
    }

    checkForUberMutation(snake) {
        // Mutation chance increases with round number
        const baseChance = 0.02; // 2% base
        const roundBonus = this.currentRound * 0.01; // +1% per round
        const mutationChance = baseChance + roundBonus;
        
        if (Math.random() < mutationChance) {
            this.applyUberMutation(snake);
        } else {
            snake.lastMutation = null;
        }
    }

    applyUberMutation(snake) {
        if (!snake.uberSkills) snake.uberSkills = [];
        
        // Filter out already owned skills
        const availableSkills = this.uberSkills.filter(skill => 
            !snake.uberSkills.some(owned => owned.name === skill.name)
        );
        
        if (availableSkills.length === 0) {
            snake.lastMutation = 'mutation attempted but no new uber skills available';
            return;
        }
        
        // Weighted selection based on rarity
        const weights = availableSkills.map(skill => {
            switch (skill.rarity) {
                case 'legendary': return 100;
                case 'mythic': return 20;
                case 'cosmic': return 1;
                default: return 50;
            }
        });
        
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        let random = Math.random() * totalWeight;
        
        for (let i = 0; i < weights.length; i++) {
            random -= weights[i];
            if (random <= 0) {
                const newSkill = availableSkills[i];
                snake.uberSkills.push(newSkill);
                snake.lastMutation = `${newSkill.name} (${newSkill.rarity}) - ${newSkill.description}`;
                
                // Apply immediate stat boosts for some uber skills
                this.applyUberSkillEffects(snake, newSkill);
                break;
            }
        }
    }

    applyUberSkillEffects(snake, uberSkill) {
        switch (uberSkill.name) {
            case 'Alpha Genome':
                Object.keys(snake.stats).forEach(stat => {
                    snake.stats[stat] = Math.min(100, snake.stats[stat] + 25);
                });
                break;
            case 'Soul Absorber':
                // This will be applied during battles
                snake.soulPower = 0;
                break;
            case 'Perfect Evolution':
                snake.perfectEvolution = true;
                break;
        }
    }

    advanceRound() {
        if (this.roundResults.length <= 1) {
            // Tournament complete!
            return this.completeTournament();
        }
        
        // Set up next round
        this.currentBracket = [...this.roundResults];
        this.roundResults = [];
        this.currentMatchIndex = 0;
        this.currentRound++;
        
        this.learningSystem.addLogEntry(`📈 ROUND ${this.currentRound}: ${this.currentBracket.length} survivors advance`);
        
        // Show top performers
        const topPerformers = this.currentBracket
            .sort((a, b) => this.calculateTournamentFitness(b) - this.calculateTournamentFitness(a))
            .slice(0, 3);
        
        topPerformers.forEach((snake, index) => {
            const rank = ['🥇', '🥈', '🥉'][index];
            this.learningSystem.addLogEntry(`${rank} ${snake.name} - Fitness: ${Math.round(this.calculateTournamentFitness(snake))}`);
        });
        
        return this.getNextMatch();
    }

    completeTournament() {
        this.tournamentActive = false;
        this.tournamentWinner = this.roundResults[0];
        
        // Record tournament in history
        const tournament = {
            winner: this.tournamentWinner,
            startTime: this.startTime,
            endTime: Date.now(),
            rounds: this.currentRound - 1,
            genesAbsorbed: this.tournamentWinner.genesAbsorbed || [],
            uberSkills: this.tournamentWinner.uberSkills || []
        };
        
        this.tournamentHistory.push(tournament);
        this.championLineage.push(this.tournamentWinner);
        
        this.learningSystem.addLogEntry('🏆🏆🏆 TOURNAMENT CHAMPION! 🏆🏆🏆');
        this.learningSystem.addLogEntry(`${this.tournamentWinner.name} emerges victorious!`);
        this.learningSystem.addLogEntry(`Final stats: ${Object.entries(this.tournamentWinner.stats).map(([k,v]) => `${k}:${v}`).join(', ')}`);
        
        if (this.tournamentWinner.uberSkills && this.tournamentWinner.uberSkills.length > 0) {
            this.learningSystem.addLogEntry(`Uber Skills: ${this.tournamentWinner.uberSkills.map(s => s.name).join(', ')}`);
        }
        
        return {
            type: 'tournament_complete',
            winner: this.tournamentWinner
        };
    }

    calculateTournamentFitness(snake) {
        let fitness = 0;
        
        // Base stats
        fitness += Object.values(snake.stats).reduce((sum, stat) => sum + stat, 0);
        
        // Tournament performance
        fitness += snake.roundsWon * 200;
        fitness += snake.victories * 100;
        fitness += snake.score * 5;
        
        // Genetic diversity bonus
        fitness += (snake.genesAbsorbed || []).length * 50;
        fitness += (snake.skills || []).length * 25;
        
        // Uber skills massive bonus
        if (snake.uberSkills) {
            snake.uberSkills.forEach(skill => {
                switch (skill.rarity) {
                    case 'legendary': fitness += 500; break;
                    case 'mythic': fitness += 1000; break;
                    case 'cosmic': fitness += 2000; break;
                }
            });
        }
        
        return fitness;
    }

    getTournamentStatus() {
        return {
            active: this.tournamentActive,
            round: this.currentRound,
            maxRounds: this.maxRounds,
            currentBracket: this.currentBracket.length,
            matchNumber: this.currentMatchIndex + 1,
            totalMatches: Math.floor(this.currentBracket.length / 2),
            winner: this.tournamentWinner
        };
    }

    getBracketDisplay() {
        if (!this.tournamentActive && !this.tournamentWinner) return null;
        
        return {
            currentRound: this.currentRound,
            survivors: this.currentBracket.map(snake => ({
                name: snake.name,
                seed: snake.tournamentSeed,
                roundsWon: snake.roundsWon,
                fitness: Math.round(this.calculateTournamentFitness(snake)),
                uberSkills: (snake.uberSkills || []).length
            })),
            champion: this.tournamentWinner ? {
                name: this.tournamentWinner.name,
                totalRounds: this.tournamentWinner.roundsWon,
                finalFitness: Math.round(this.calculateTournamentFitness(this.tournamentWinner)),
                uberSkills: this.tournamentWinner.uberSkills || []
            } : null
        };
    }

    reset() {
        this.currentRound = 0;
        this.currentBracket = [];
        this.roundResults = [];
        this.currentMatchIndex = 0;
        this.tournamentActive = false;
        this.tournamentWinner = null;
    }
}

// Export for use in other files
window.TournamentSystem = TournamentSystem;
