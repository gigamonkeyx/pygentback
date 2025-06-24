// Elite Tournament System - High-stakes evolutionary tournament with progression
class EliteTournamentSystem {
    constructor(battleEngine, evolutionSystem, visualEffects) {
        this.battleEngine = battleEngine;
        this.evolutionSystem = evolutionSystem;
        this.visualEffects = visualEffects;
        
        // Tournament configuration
        this.config = {
            size: 32,           // Total tournament participants
            rounds: 5,          // Number of rounds (32 -> 16 -> 8 -> 4 -> 2 -> 1)
            battleTimeLimit: 90000, // 90 seconds per battle
            restPeriod: 3000,   // 3 seconds between battles
            evolutionEnabled: true,
            feedingMode: true   // Winner absorbs loser's traits
        };
        
        // Tournament state
        this.currentTournament = null;
        this.tournamentHistory = [];
        this.isRunning = false;
        this.isPaused = false;
        
        // UI elements
        this.uiElements = {
            bracket: null,
            ladder: null,
            battleStats: null,
            evolutionPanel: null
        };
        
        // Event handlers
        this.eventHandlers = new Map();
        
        this.initializeUI();
        this.bindEvents();
        
        console.log('Elite Tournament System initialized');
    }
    
    initializeUI() {
        // Find UI elements
        this.uiElements.bracket = document.getElementById('tournament-bracket');
        this.uiElements.ladder = document.getElementById('tournament-ladder');
        this.uiElements.battleStats = document.getElementById('battle-stats');
        this.uiElements.evolutionPanel = document.getElementById('evolution-panel');
        
        // Create UI elements if they don't exist
        this.ensureUIElements();
    }
    
    ensureUIElements() {
        if (!this.uiElements.bracket) {
            this.uiElements.bracket = this.createBracketElement();
        }
        if (!this.uiElements.ladder) {
            this.uiElements.ladder = this.createLadderElement();
        }
        if (!this.uiElements.battleStats) {
            this.uiElements.battleStats = this.createBattleStatsElement();
        }
        if (!this.uiElements.evolutionPanel) {
            this.uiElements.evolutionPanel = this.createEvolutionPanelElement();
        }
    }
    
    createBracketElement() {
        const element = document.createElement('div');
        element.id = 'tournament-bracket';
        element.className = 'tournament-bracket';
        element.innerHTML = '<h3>Tournament Bracket</h3><div class="bracket-content"></div>';
        
        const leftPanel = document.querySelector('.left-panel');
        if (leftPanel) {
            leftPanel.appendChild(element);
        }
        
        return element;
    }
    
    createLadderElement() {
        const element = document.createElement('div');
        element.id = 'tournament-ladder';
        element.className = 'tournament-ladder';
        element.innerHTML = '<h3>Tournament Ladder</h3><div class="ladder-content"></div>';
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            rightPanel.appendChild(element);
        }
        
        return element;
    }
    
    createBattleStatsElement() {
        const element = document.createElement('div');
        element.id = 'battle-stats';
        element.className = 'battle-stats';
        element.innerHTML = '<h3>Battle Statistics</h3><div class="stats-content"></div>';
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            rightPanel.appendChild(element);
        }
        
        return element;
    }
    
    createEvolutionPanelElement() {
        const element = document.createElement('div');
        element.id = 'evolution-panel';
        element.className = 'evolution-panel';
        element.innerHTML = '<h3>Evolution Results</h3><div class="evolution-content"></div>';
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            rightPanel.appendChild(element);
        }
        
        return element;
    }
    
    bindEvents() {
        // Battle end event
        this.eventHandlers.set('battleEnd', (event) => {
            this.handleBattleEnd(event.detail);
        });
        
        // Evolution complete event
        this.eventHandlers.set('evolutionComplete', (event) => {
            this.handleEvolutionComplete(event.detail);
        });
        
        // Register event listeners
        this.eventHandlers.forEach((handler, event) => {
            document.addEventListener(event, handler);
        });
        
        // Control buttons
        document.addEventListener('keydown', (e) => {
            if (e.code === 'KeyT' && !this.isRunning) {
                this.startNewTournament();
            }
            if (e.code === 'KeyP' && this.isRunning) {
                this.togglePause();
            }
        });
    }
    
    async startNewTournament(participants = null) {
        if (this.isRunning) {
            console.log('Tournament already running');
            return;
        }
        
        // Generate participants if not provided
        if (!participants) {
            participants = await this.generateTournamentParticipants();
        }
        
        // Validate participant count
        if (participants.length !== this.config.size) {
            console.error(`Expected ${this.config.size} participants, got ${participants.length}`);
            return;
        }
        
        // Initialize tournament
        this.currentTournament = {
            id: this.generateTournamentId(),
            participants: participants,
            rounds: [],
            currentRound: 0,
            currentMatch: 0,
            champion: null,
            startTime: Date.now(),
            status: 'running'
        };
        
        // Setup first round
        this.setupRound(0, participants);
        
        // Update UI
        this.updateBracketDisplay();
        this.updateLadderDisplay();
        
        this.isRunning = true;
        
        console.log(`Tournament ${this.currentTournament.id} started with ${participants.length} participants`);
        
        // Start first battle
        this.startNextBattle();
        
        return this.currentTournament.id;
    }
    
    async generateTournamentParticipants() {
        const participants = [];
        
        // Use elite snake generator to create diverse participants
        const generator = new EliteSnakeGenerator();
        
        for (let i = 0; i < this.config.size; i++) {
            const snake = generator.createSnake();
            snake.tournamentSeed = i + 1;
            snake.tournamentStats = {
                wins: 0,
                losses: 0,
                totalDamageDealt: 0,
                totalDamageTaken: 0,
                battlesWon: 0,
                evolutionsGained: 0
            };
            participants.push(snake);
        }
        
        // Shuffle participants for random seeding
        this.shuffleArray(participants);
        
        return participants;
    }
    
    setupRound(roundNumber, participants) {
        const round = {
            number: roundNumber,
            matches: [],
            winners: [],
            completed: false
        };
        
        // Create matches by pairing participants
        for (let i = 0; i < participants.length; i += 2) {
            const match = {
                id: `R${roundNumber}M${Math.floor(i / 2)}`,
                snake1: participants[i],
                snake2: participants[i + 1],
                winner: null,
                battleResult: null,
                evolutionResults: null,
                completed: false
            };
            
            round.matches.push(match);
        }
        
        this.currentTournament.rounds[roundNumber] = round;
        
        console.log(`Round ${roundNumber + 1} setup: ${round.matches.length} matches`);
    }
    
    async startNextBattle() {
        if (!this.currentTournament || this.isPaused) return;
        
        const currentRound = this.currentTournament.rounds[this.currentTournament.currentRound];
        if (!currentRound || this.currentTournament.currentMatch >= currentRound.matches.length) {
            // Round complete
            await this.completeCurrentRound();
            return;
        }
        
        const match = currentRound.matches[this.currentTournament.currentMatch];
        
        if (match.completed) {
            this.currentTournament.currentMatch++;
            this.startNextBattle();
            return;
        }
        
        console.log(`Starting battle: ${match.snake1.name} vs ${match.snake2.name}`);
        
        // Update UI to show current battle
        this.updateCurrentBattleDisplay(match);
        
        // Start battle
        const battleId = this.battleEngine.startBattle(match.snake1, match.snake2, {
            timeLimit: this.config.battleTimeLimit,
            winCondition: 'elimination',
            environment: 'tournament_arena'
        });
        
        match.battleId = battleId;
    }
    
    handleBattleEnd(battleDetail) {
        if (!this.currentTournament) return;
        
        const currentRound = this.currentTournament.rounds[this.currentTournament.currentRound];
        const match = currentRound.matches[this.currentTournament.currentMatch];
        
        if (!match || match.completed) return;
        
        // Store battle results
        match.battleResult = battleDetail;
        match.winner = battleDetail.winner;
        match.completed = true;
        
        // Update tournament stats
        this.updateTournamentStats(match);
        
        // Apply evolution to winner
        if (this.config.evolutionEnabled) {
            this.applyBattleEvolution(match);
        }
        
        // Apply feeding mode if enabled
        if (this.config.feedingMode) {
            this.applyFeedingMode(match);
        }
        
        // Update displays
        this.updateBattleStatsDisplay(match);
        this.updateBracketDisplay();
        
        console.log(`Battle completed: ${match.winner.name} wins!`);
        
        // Schedule next battle
        setTimeout(() => {
            this.currentTournament.currentMatch++;
            this.startNextBattle();
        }, this.config.restPeriod);
    }
    
    updateTournamentStats(match) {
        const winner = match.winner;
        const loser = match.winner === match.snake1 ? match.snake2 : match.snake1;
        
        // Winner stats
        winner.tournamentStats.wins++;
        winner.tournamentStats.battlesWon++;
        winner.tournamentStats.totalDamageDealt += match.battleResult.stats.damageDealt || 0;
        winner.tournamentStats.totalDamageTaken += match.battleResult.stats.damageTaken || 0;
        
        // Loser stats
        loser.tournamentStats.losses++;
        loser.tournamentStats.totalDamageDealt += match.battleResult.stats.damageDealt || 0;
        loser.tournamentStats.totalDamageTaken += match.battleResult.stats.damageTaken || 0;
    }
    
    applyBattleEvolution(match) {
        const winner = match.winner;
        const battleResults = {
            victory: true,
            experienceGained: 75 + Math.floor(Math.random() * 50), // 75-125 XP
            dominanceScore: this.calculateDominanceScore(match.battleResult),
            efficiency: this.calculateEfficiency(match.battleResult),
            survivalTime: match.battleResult.duration || 0,
            damageDealt: match.battleResult.stats.damageDealt || 0,
            skillsUsed: match.battleResult.stats.skillActivations || 0
        };
        
        // Apply evolution
        const evolutionResult = this.evolutionSystem.evolveSnake(winner, battleResults);
        match.evolutionResults = evolutionResult;
        
        if (evolutionResult.success) {
            winner.tournamentStats.evolutionsGained++;
            console.log(`${winner.name} evolved! (${evolutionResult.evolutionType})`);
            
            // Show evolution effects
            this.visualEffects.addEvolutionEffect(winner, evolutionResult.evolutionType);
            this.updateEvolutionPanelDisplay(evolutionResult);
        }
    }
    
    applyFeedingMode(match) {
        const winner = match.winner;
        const loser = match.winner === match.snake1 ? match.snake2 : match.snake1;
        
        // Winner absorbs some traits from loser
        const absorption = this.calculateTraitAbsorption(winner, loser);
        
        // Apply absorbed traits
        if (absorption.stats.length > 0) {
            absorption.stats.forEach(stat => {
                winner.stats[stat.name] = Math.min(120, winner.stats[stat.name] + stat.gain);
            });
        }
        
        if (absorption.skills.length > 0) {
            absorption.skills.forEach(skill => {
                if (!winner.skills.includes(skill)) {
                    winner.skills.push(skill);
                }
            });
        }
        
        // Visual feeding effect
        this.visualEffects.addFeedingEffect(winner, loser, absorption);
        
        console.log(`${winner.name} absorbed traits from ${loser.name}`);
    }
    
    calculateTraitAbsorption(winner, loser) {
        const absorption = {
            stats: [],
            skills: [],
            healthGain: 0,
            energyGain: 0
        };
        
        // Absorb some stat points
        Object.keys(loser.stats).forEach(statName => {
            if (loser.stats[statName] > winner.stats[statName] && Math.random() < 0.3) {
                const gain = Math.floor((loser.stats[statName] - winner.stats[statName]) * 0.2);
                if (gain > 0) {
                    absorption.stats.push({ name: statName, gain: gain });
                }
            }
        });
        
        // Chance to absorb a skill
        const availableSkills = loser.skills.filter(skill => !winner.skills.includes(skill));
        if (availableSkills.length > 0 && Math.random() < 0.25) {
            const absorbedSkill = availableSkills[Math.floor(Math.random() * availableSkills.length)];
            absorption.skills.push(absorbedSkill);
        }
        
        // Health and energy absorption
        absorption.healthGain = Math.floor(loser.maxHealth * 0.1);
        absorption.energyGain = Math.floor(loser.maxEnergy * 0.1);
        
        winner.maxHealth = Math.min(300, winner.maxHealth + absorption.healthGain);
        winner.maxEnergy = Math.min(200, winner.maxEnergy + absorption.energyGain);
        winner.health = winner.maxHealth;
        winner.energy = winner.maxEnergy;
        
        return absorption;
    }
    
    calculateDominanceScore(battleResult) {
        // Calculate how dominant the victory was
        let score = 1.0;
        
        if (battleResult.reason === 'elimination') {
            score += 0.5;
        }
        if (battleResult.duration < 30000) { // Quick victory
            score += 0.3;
        }
        if (battleResult.stats.hits > battleResult.stats.dodges) {
            score += 0.2;
        }
        
        return Math.min(2.0, score);
    }
    
    calculateEfficiency(battleResult) {
        // Calculate battle efficiency
        const damageRatio = (battleResult.stats.damageDealt || 1) / Math.max(1, battleResult.stats.damageTaken || 1);
        return Math.min(2.0, damageRatio);
    }
    
    async completeCurrentRound() {
        const currentRound = this.currentTournament.rounds[this.currentTournament.currentRound];
        currentRound.completed = true;
        
        // Collect winners
        currentRound.winners = currentRound.matches.map(match => match.winner);
        
        console.log(`Round ${currentRound.number + 1} completed. ${currentRound.winners.length} winners advance.`);
        
        // Check if tournament is complete
        if (currentRound.winners.length === 1) {
            this.completeTournament(currentRound.winners[0]);
            return;
        }
        
        // Setup next round
        const nextRoundNumber = this.currentTournament.currentRound + 1;
        this.setupRound(nextRoundNumber, currentRound.winners);
        
        // Move to next round
        this.currentTournament.currentRound = nextRoundNumber;
        this.currentTournament.currentMatch = 0;
        
        // Update displays
        this.updateBracketDisplay();
        this.updateLadderDisplay();
        
        // Small pause between rounds
        setTimeout(() => {
            this.startNextBattle();
        }, this.config.restPeriod * 2);
    }
    
    completeTournament(champion) {
        this.currentTournament.champion = champion;
        this.currentTournament.endTime = Date.now();
        this.currentTournament.duration = this.currentTournament.endTime - this.currentTournament.startTime;
        this.currentTournament.status = 'completed';
        
        this.isRunning = false;
        
        console.log(`Tournament completed! Champion: ${champion.name}`);
        
        // Apply champion bonus evolution
        this.applyChampionRewards(champion);
        
        // Victory effects
        this.visualEffects.addVictoryEffect(
            400, 300, // Center of arena
            champion.appearance.primaryColor
        );
        
        // Update displays
        this.updateChampionDisplay(champion);
        this.updateBracketDisplay();
        this.updateLadderDisplay();
        
        // Save tournament to history
        this.tournamentHistory.push({
            ...this.currentTournament,
            timestamp: Date.now()
        });
        
        // Keep history manageable
        if (this.tournamentHistory.length > 10) {
            this.tournamentHistory.shift();
        }
        
        // Fire tournament complete event
        document.dispatchEvent(new CustomEvent('tournamentComplete', {
            detail: {
                tournament: this.currentTournament,
                champion: champion
            }
        }));
    }
    
    applyChampionRewards(champion) {
        // Massive experience bonus for tournament victory
        const championResults = {
            victory: true,
            experienceGained: 200 + Math.floor(Math.random() * 100), // 200-300 XP
            dominanceScore: 2.0,
            efficiency: 2.0,
            survivalTime: 60000,
            damageDealt: 500,
            skillsUsed: 10,
            isTournamentChampion: true
        };
        
        // Guaranteed major evolution
        const evolutionResult = this.evolutionSystem.evolveSnake(champion, championResults);
        
        // Additional champion bonuses
        champion.championTitles = champion.championTitles || [];
        champion.championTitles.push(`Tournament Champion ${Date.now()}`);
        
        // Stat bonuses
        Object.keys(champion.stats).forEach(stat => {
            champion.stats[stat] = Math.min(150, champion.stats[stat] + 10);
        });
        
        // Health and energy bonuses
        champion.maxHealth = Math.min(400, champion.maxHealth + 100);
        champion.maxEnergy = Math.min(250, champion.maxEnergy + 50);
        champion.health = champion.maxHealth;
        champion.energy = champion.maxEnergy;
        
        // Special champion skills
        const championSkills = ['Tournament Victor', 'Elite Warrior', 'Apex Predator'];
        championSkills.forEach(skill => {
            if (!champion.skills.includes(skill)) {
                champion.skills.push(skill);
            }
        });
        
        console.log(`${champion.name} received champion rewards!`);
    }
    
    // UI Update Methods
    updateBracketDisplay() {
        if (!this.uiElements.bracket) return;
        
        const content = this.uiElements.bracket.querySelector('.bracket-content');
        if (!content) return;
        
        if (!this.currentTournament) {
            content.innerHTML = '<p>No active tournament</p>';
            return;
        }
        
        let html = '';
        
        this.currentTournament.rounds.forEach((round, roundIndex) => {
            html += `<div class="round">`;
            html += `<h4>Round ${roundIndex + 1}</h4>`;
            
            round.matches.forEach(match => {
                const status = match.completed ? 'completed' : 
                              (roundIndex === this.currentTournament.currentRound ? 'active' : 'pending');
                
                html += `<div class="match ${status}">`;
                html += `<div class="snake ${match.winner === match.snake1 ? 'winner' : ''}">${match.snake1.name}</div>`;
                html += `<div class="vs">VS</div>`;
                html += `<div class="snake ${match.winner === match.snake2 ? 'winner' : ''}">${match.snake2.name}</div>`;
                if (match.winner) {
                    html += `<div class="winner-indicator">👑 ${match.winner.name}</div>`;
                }
                html += `</div>`;
            });
            
            html += `</div>`;
        });
        
        content.innerHTML = html;
    }
    
    updateLadderDisplay() {
        if (!this.uiElements.ladder) return;
        
        const content = this.uiElements.ladder.querySelector('.ladder-content');
        if (!content) return;
        
        if (!this.currentTournament) {
            content.innerHTML = '<p>No active tournament</p>';
            return;
        }
        
        // Get all participants sorted by performance
        const participants = [...this.currentTournament.participants];
        participants.sort((a, b) => {
            if (a.tournamentStats.wins !== b.tournamentStats.wins) {
                return b.tournamentStats.wins - a.tournamentStats.wins;
            }
            return b.tournamentStats.totalDamageDealt - a.tournamentStats.totalDamageDealt;
        });
        
        let html = '<table class="ladder-table">';
        html += '<tr><th>Rank</th><th>Snake</th><th>W/L</th><th>Dmg</th><th>Evolutions</th></tr>';
        
        participants.forEach((snake, index) => {
            const isChampion = this.currentTournament.champion === snake;
            const isActive = this.isSnakeInCurrentBattle(snake);
            
            html += `<tr class="${isChampion ? 'champion' : ''} ${isActive ? 'active' : ''}">`;
            html += `<td>${index + 1}</td>`;
            html += `<td style="color: ${snake.appearance.primaryColor}">${snake.name}</td>`;
            html += `<td>${snake.tournamentStats.wins}/${snake.tournamentStats.losses}</td>`;
            html += `<td>${snake.tournamentStats.totalDamageDealt}</td>`;
            html += `<td>${snake.tournamentStats.evolutionsGained}</td>`;
            html += `</tr>`;
        });
        
        html += '</table>';
        content.innerHTML = html;
    }
    
    updateCurrentBattleDisplay(match) {
        // Update battle info in UI
        const battleInfo = document.querySelector('.battle-info');
        if (battleInfo) {
            battleInfo.innerHTML = `
                <h3>Current Battle</h3>
                <div class="fighters">
                    <div class="fighter" style="color: ${match.snake1.appearance.primaryColor}">
                        <strong>${match.snake1.name}</strong>
                        <div>HP: ${match.snake1.health}/${match.snake1.maxHealth}</div>
                        <div>Wins: ${match.snake1.tournamentStats.wins}</div>
                    </div>
                    <div class="vs">VS</div>
                    <div class="fighter" style="color: ${match.snake2.appearance.primaryColor}">
                        <strong>${match.snake2.name}</strong>
                        <div>HP: ${match.snake2.health}/${match.snake2.maxHealth}</div>
                        <div>Wins: ${match.snake2.tournamentStats.wins}</div>
                    </div>
                </div>
            `;
        }
    }
    
    updateBattleStatsDisplay(match) {
        if (!this.uiElements.battleStats) return;
        
        const content = this.uiElements.battleStats.querySelector('.stats-content');
        if (!content) return;
        
        const stats = match.battleResult.stats;
        
        const html = `
            <div class="battle-result">
                <h4>Battle Result</h4>
                <p><strong>Winner:</strong> <span style="color: ${match.winner.appearance.primaryColor}">${match.winner.name}</span></p>
                <p><strong>Reason:</strong> ${match.battleResult.reason}</p>
                <p><strong>Duration:</strong> ${(match.battleResult.duration / 1000).toFixed(1)}s</p>
                <div class="stats-grid">
                    <div>Hits: ${stats.hits}</div>
                    <div>Dodges: ${stats.dodges}</div>
                    <div>Skills Used: ${stats.skillActivations}</div>
                    <div>Food Eaten: ${stats.foodEaten}</div>
                </div>
            </div>
        `;
        
        content.innerHTML = html;
    }
    
    updateEvolutionPanelDisplay(evolutionResult) {
        if (!this.uiElements.evolutionPanel) return;
        
        const content = this.uiElements.evolutionPanel.querySelector('.evolution-content');
        if (!content) return;
        
        let html = `
            <div class="evolution-result">
                <h4>${evolutionResult.snakeName} Evolution</h4>
                <p><strong>Type:</strong> ${evolutionResult.evolutionType}</p>
        `;
        
        if (evolutionResult.improvements.length > 0) {
            html += '<h5>Improvements:</h5><ul>';
            evolutionResult.improvements.forEach(improvement => {
                html += `<li>${this.formatImprovement(improvement)}</li>`;
            });
            html += '</ul>';
        }
        
        if (evolutionResult.mutations.length > 0) {
            html += '<h5>Mutations:</h5><ul>';
            evolutionResult.mutations.forEach(mutation => {
                html += `<li><strong>${mutation.name}</strong>: ${mutation.description}</li>`;
            });
            html += '</ul>';
        }
        
        html += '</div>';
        content.innerHTML = html;
    }
    
    updateChampionDisplay(champion) {
        const championPanel = document.querySelector('.champion-panel') || this.createChampionPanel();
        
        championPanel.innerHTML = `
            <h3>🏆 Tournament Champion 🏆</h3>
            <div class="champion-info">
                <h2 style="color: ${champion.appearance.primaryColor}">${champion.name}</h2>
                <p><strong>Generation:</strong> ${champion.generation || 1}</p>
                <p><strong>Tournament Wins:</strong> ${champion.tournamentStats.wins}</p>
                <p><strong>Total Damage:</strong> ${champion.tournamentStats.totalDamageDealt}</p>
                <p><strong>Evolutions:</strong> ${champion.tournamentStats.evolutionsGained}</p>
                <div class="champion-skills">
                    <strong>Skills:</strong>
                    ${champion.skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
        `;
    }
    
    createChampionPanel() {
        const element = document.createElement('div');
        element.className = 'champion-panel';
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) {
            rightPanel.appendChild(element);
        }
        
        return element;
    }
    
    // Utility methods
    formatImprovement(improvement) {
        switch (improvement.type) {
            case 'stat':
                return `${improvement.stat}: +${improvement.improvement} (${improvement.newValue})`;
            case 'new_skill':
                return `New Skill: ${improvement.skill}`;
            case 'skill_upgrade':
                return `${improvement.oldSkill} → ${improvement.newSkill}`;
            case 'vital':
                return `Health: +${improvement.healthIncrease}, Energy: +${improvement.energyIncrease}`;
            default:
                return JSON.stringify(improvement);
        }
    }
    
    isSnakeInCurrentBattle(snake) {
        if (!this.currentTournament || !this.isRunning) return false;
        
        const currentRound = this.currentTournament.rounds[this.currentTournament.currentRound];
        if (!currentRound) return false;
        
        const currentMatch = currentRound.matches[this.currentTournament.currentMatch];
        if (!currentMatch) return false;
        
        return currentMatch.snake1 === snake || currentMatch.snake2 === snake;
    }
    
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
    
    generateTournamentId() {
        return 'tournament_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
    }
    
    togglePause() {
        this.isPaused = !this.isPaused;
        console.log(`Tournament ${this.isPaused ? 'paused' : 'resumed'}`);
    }
    
    // Cleanup
    dispose() {
        // Remove event listeners
        this.eventHandlers.forEach((handler, event) => {
            document.removeEventListener(event, handler);
        });
        
        this.eventHandlers.clear();
        
        // Clear current tournament
        this.currentTournament = null;
        this.isRunning = false;
    }
}
