// Snake Generator - Creates unique AI snakes with attributes and pedigree
class SnakeGenerator {
    constructor() {
        this.nameComponents = {
            prefixes: ['Viper', 'Cobra', 'Python', 'Mamba', 'Boa', 'Adder', 'Rattle', 'King', 'Coral', 'Garter'],
            suffixes: ['Strike', 'Fang', 'Coil', 'Slither', 'Hiss', 'Scale', 'Venom', 'Hunt', 'Swift', 'Wise'],
            modifiers: ['the', 'of', 'from'],
            locations: ['Jungle', 'Desert', 'Mountain', 'River', 'Cave', 'Forest', 'Marsh', 'Valley', 'Peak', 'Grove']
        };        this.passiveSkills = [
            'Regeneration', 'Speed Boost', 'Thick Skin', 'Sharp Turn', 'Food Sense',
            'Danger Sense', 'Wall Hugger', 'Center Seeker', 'Power Hunter', 'Survival Instinct',
            'Quick Reflex', 'Efficient Metabolism', 'Territory Control', 'Adaptive Learning',
            'Risk Assessment', 'Pattern Recognition', 'Strategic Retreat', 'Opportunist',
            'Better Eyes', 'Eagle Vision', 'Sixth Sense', 'Fortune Hunter',
            'Titan Blood', 'Venom Glands', 'Camouflage', 'Lightning Strike', 'Iron Scales',
            'Hypnotic Gaze', 'Coil Master', 'Phase Walker', 'Berserker Rage', 'Ancient Wisdom',
            'Split Tail', 'Magnetic Field', 'Time Dilation', 'Shadow Clone', 'Alpha Predator'
        ];        this.generation = 1;
        this.snakeCounter = 0;
        
        // Winners Circle - elite breeding system
        this.winnersCircle = [];
        this.maxWinnersCircle = 6; // Top 6 champions
        this.championGenerations = 3; // Champions stay for 3 generations
    }

    generateName() {
        const prefix = this.nameComponents.prefixes[Math.floor(Math.random() * this.nameComponents.prefixes.length)];
        const suffix = this.nameComponents.suffixes[Math.floor(Math.random() * this.nameComponents.suffixes.length)];
        
        if (Math.random() < 0.3) {
            const modifier = this.nameComponents.modifiers[Math.floor(Math.random() * this.nameComponents.modifiers.length)];
            const location = this.nameComponents.locations[Math.floor(Math.random() * this.nameComponents.locations.length)];
            return `${prefix}${suffix} ${modifier} ${location}`;
        }
        
        return `${prefix}${suffix}`;
    }

    generateStats() {
        return {
            speed: Math.floor(Math.random() * 100) + 1,
            agility: Math.floor(Math.random() * 100) + 1,
            intelligence: Math.floor(Math.random() * 100) + 1,
            aggression: Math.floor(Math.random() * 100) + 1,
            survival: Math.floor(Math.random() * 100) + 1,
            efficiency: Math.floor(Math.random() * 100) + 1,
            luck: Math.floor(Math.random() * 100) + 1,
            adaptability: Math.floor(Math.random() * 100) + 1
        };
    }

    generateSkills(stats) {
        const skills = [];
        const numSkills = Math.floor(Math.random() * 4) + 2; // 2-5 skills
        
        const availableSkills = [...this.passiveSkills];
        
        for (let i = 0; i < numSkills; i++) {
            if (availableSkills.length === 0) break;
            
            const skillIndex = Math.floor(Math.random() * availableSkills.length);
            skills.push(availableSkills.splice(skillIndex, 1)[0]);
        }
        
        return skills;
    }    generatePedigree(generation = 1, parents = null) {
        const pedigree = [];
          if (parents && parents.length > 0) {
            const parentNames = parents.map(p => p.name).join(' × ');
            pedigree.push(`Gen ${generation}: Child of ${parentNames}`);
            
            // Add champion status if parents are champions
            const championParents = parents.filter(p => p.isChampion || this.winnersCircle.some(c => c.name === p.name));
            if (championParents.length > 0) {
                pedigree.push(`  👑 Elite Champion Bloodline`);
            }
            
            // Add distinct parent lineages with male/female symbols
            parents.forEach((parent, index) => {
                if (parent.pedigree && parent.pedigree.length > 0) {
                    const symbol = index === 0 ? '♂' : '♀';
                    pedigree.push(`  └─ ${symbol} ${parent.pedigree[0]}`);
                }
            });
        } else {
            pedigree.push(`Gen ${generation}: First Generation`);
        }
        
        return pedigree;
    }createSnake(parents = null, isEvolved = false) {
        this.snakeCounter++;
        
        let stats;
        let skills;
        let generation = this.generation;
        
        if (parents && parents.length > 0) {
            // Evolved snake - inherit and mutate from parents
            stats = this.evolveStats(parents);
            skills = this.evolveSkills(parents);
            generation = Math.max(...parents.map(p => p.generation || 1)) + 1;
        } else {
            // New random snake
            stats = this.generateStats();
            skills = this.generateSkills(stats);
        }
        
        const snake = {
            id: this.snakeCounter,
            name: this.generateName(),
            stats: stats,
            skills: skills,
            pedigree: this.generatePedigree(generation, parents),
            generation: generation,
            lives: skills.includes('Iron Scales') ? 6 : 3, // Double health for Iron Scales
            maxLives: skills.includes('Iron Scales') ? 6 : 3, // Track original max health
            score: 0,
            experience: 0,
            victories: 0,
            defeats: 0,
            foodEaten: 0,
            powerupsCollected: 0,
            survivalTime: 0,
            birthTime: Date.now(),
            isEvolved: isEvolved,
            // Width evolution system
            baseWidth: 10,
            currentWidth: 10,
            maxWidth: this.calculateMaxWidth(stats, skills),
            widthGrowthRate: this.calculateWidthGrowthRate(stats, skills),
            lengthCapReached: false
        };
        
        return snake;
    }

    calculateMaxWidth(stats, skills) {
        let maxWidth = 20; // Base maximum width
        
        // Stats influence max width
        maxWidth += Math.floor(stats.strength / 10); // 0-10 bonus
        maxWidth += Math.floor(stats.survival / 15); // 0-6 bonus
        
        // Skills affect max width
        if (skills.includes('Titan Growth')) maxWidth += 15;
        if (skills.includes('Alpha Size')) maxWidth += 12;
        if (skills.includes('Iron Scales')) maxWidth += 8;
        if (skills.includes('Thick Skin')) maxWidth += 6;
        if (skills.includes('Compact Build')) maxWidth = Math.max(15, maxWidth - 10); // Smaller but not tiny
        
        return Math.min(maxWidth, 45); // Cap at 45 pixels
    }

    calculateWidthGrowthRate(stats, skills) {
        let rate = 1; // Base growth rate
        
        // Stats influence growth rate
        if (stats.efficiency > 70) rate += 0.5;
        if (stats.adaptability > 80) rate += 0.3;
        
        // Skills affect growth rate
        if (skills.includes('Efficient Metabolism')) rate += 1;
        if (skills.includes('Titan Growth')) rate += 0.8;
        if (skills.includes('Quick Growth')) rate += 0.6;
        
        return rate;
    }

    evolveStats(parents) {
        const newStats = {};
        const statKeys = Object.keys(parents[0].stats);
        
        statKeys.forEach(key => {
            // Average parent stats
            const avgStat = parents.reduce((sum, parent) => sum + parent.stats[key], 0) / parents.length;
            
            // Add mutation (-20 to +20)
            const mutation = (Math.random() - 0.5) * 40;
            
            // Ensure within bounds
            newStats[key] = Math.max(1, Math.min(100, Math.round(avgStat + mutation)));
        });
        
        return newStats;
    }

    evolveSkills(parents) {
        const parentSkills = [];
        parents.forEach(parent => {
            parentSkills.push(...parent.skills);
        });
        
        // Remove duplicates
        const uniqueParentSkills = [...new Set(parentSkills)];
        
        // Inherit 50-80% of parent skills
        const inheritanceRate = 0.5 + Math.random() * 0.3;
        const inheritedCount = Math.floor(uniqueParentSkills.length * inheritanceRate);
        
        const inheritedSkills = [];
        for (let i = 0; i < inheritedCount && i < uniqueParentSkills.length; i++) {
            inheritedSkills.push(uniqueParentSkills[i]);
        }
        
        // Add 0-2 new random skills
        const newSkillsCount = Math.floor(Math.random() * 3);
        const availableNewSkills = this.passiveSkills.filter(skill => !inheritedSkills.includes(skill));
        
        for (let i = 0; i < newSkillsCount && availableNewSkills.length > 0; i++) {
            const skillIndex = Math.floor(Math.random() * availableNewSkills.length);
            inheritedSkills.push(availableNewSkills.splice(skillIndex, 1)[0]);
        }
        
        return inheritedSkills;
    }    createEvolutionPair(deadSnakes, winner = null) {
        // Add winner to winners circle
        if (winner) {
            this.addToWinnersCircle(winner);
        }
          // ELITE BREEDING: Only winners breed
        if (this.winnersCircle.length >= 2) {
            let parent1, parent2, parent3, parent4;
            
            if (winner && this.winnersCircle.length > 1) {
                // Winner as primary parent for first child
                parent1 = this.winnersCircle.find(champ => champ.name === winner.name) || this.winnersCircle[0];
                
                // Pick best mate from remaining champions for first child
                const availableMates = this.winnersCircle.filter(champ => champ.name !== parent1.name);
                parent2 = availableMates.reduce((best, current) => 
                    this.calculateFitness(current) > this.calculateFitness(best) ? current : best
                );
                
                // For second child, use different parent combinations
                if (this.winnersCircle.length >= 3) {
                    // Get 3rd best champion for diversity
                    const sortedChampions = [...this.winnersCircle].sort((a, b) => 
                        this.calculateFitness(b) - this.calculateFitness(a)
                    );
                    parent3 = sortedChampions[2];
                    parent4 = sortedChampions[1]; // Pair with 2nd best
                } else {
                    // Only 2 champions - reverse the parents for genetic diversity
                    parent3 = parent2;
                    parent4 = parent1;
                }
            } else {
                // Pick different champion combinations
                const sortedChampions = [...this.winnersCircle].sort((a, b) => 
                    this.calculateFitness(b) - this.calculateFitness(a)
                );
                parent1 = sortedChampions[0];
                parent2 = sortedChampions[1];
                
                // Second child gets different combination if available
                if (sortedChampions.length >= 4) {
                    parent3 = sortedChampions[2];
                    parent4 = sortedChampions[3];
                } else if (sortedChampions.length >= 3) {
                    parent3 = sortedChampions[2];
                    parent4 = sortedChampions[0]; // Mix with best
                } else {
                    // Only 2 champions - reverse for diversity
                    parent3 = parent2;
                    parent4 = parent1;
                }
            }
            
            // Track usage
            parent1.championsUsed = (parent1.championsUsed || 0) + 1;
            parent2.championsUsed = (parent2.championsUsed || 0) + 1;
            if (parent3 !== parent1 && parent3 !== parent2) {
                parent3.championsUsed = (parent3.championsUsed || 0) + 1;
            }
            if (parent4 !== parent1 && parent4 !== parent2 && parent4 !== parent3) {
                parent4.championsUsed = (parent4.championsUsed || 0) + 1;
            }
            
            console.log(`ELITE BREEDING 1: ${parent1.name} + ${parent2.name}`);
            console.log(`ELITE BREEDING 2: ${parent3.name} + ${parent4.name}`);
            
            const child1 = this.createSnake([parent1, parent2], true);
            const child2 = this.createSnake([parent3, parent4], true);
            
            // Mark children as champion offspring
            child1.championOffspring = true;
            child2.championOffspring = true;
            child1.breedingInfo = `${parent1.name} × ${parent2.name}`;
            child2.breedingInfo = `${parent3.name} × ${parent4.name}`;
            
            this.generation = Math.max(child1.generation, child2.generation);
            
            console.log(`CHILDREN: ${child1.name} & ${child2.name} from champion parents`);
            
            return [child1, child2];
        }
        
        // Fallback to old system if not enough champions
        if (deadSnakes.length === 0) {
            return [this.createSnake(), this.createSnake()];
        }
        
        // Select parents based on performance
        const sortedSnakes = deadSnakes.sort((a, b) => {
            const scoreA = this.calculateFitness(a);
            const scoreB = this.calculateFitness(b);
            return scoreB - scoreA;
        });
        
        // Top performers have higher chance to be parents
        const parent1 = this.selectParent(sortedSnakes);
        const parent2 = this.selectParent(sortedSnakes, parent1);
        
        const child1 = this.createSnake([parent1, parent2], true);
        const child2 = this.createSnake([parent1, parent2], true);
        
        this.generation = Math.max(child1.generation, child2.generation);
        
        return [child1, child2];
    }

    selectParent(sortedSnakes, exclude = null) {
        // Weighted selection - better snakes more likely to be chosen
        const weights = sortedSnakes.map((snake, index) => {
            if (snake === exclude) return 0;
            return Math.max(1, sortedSnakes.length - index);
        });
        
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        let random = Math.random() * totalWeight;
        
        for (let i = 0; i < weights.length; i++) {
            random -= weights[i];
            if (random <= 0) {
                return sortedSnakes[i];
            }
        }
        
        return sortedSnakes[0];
    }

    calculateFitness(snake) {
        return (
            snake.score * 10 +
            snake.victories * 50 +
            snake.foodEaten * 5 +
            snake.powerupsCollected * 15 +
            snake.survivalTime * 0.1 +
            (snake.lives > 0 ? 25 : 0)
        );
    }
    
    addToWinnersCircle(snake) {
        // Mark snake as champion
        const champion = {
            ...snake,
            championSince: this.generation,
            championsUsed: 0,
            isChampion: true
        };
        
        // Add to winners circle
        this.winnersCircle.unshift(champion);
        
        // Remove old champions
        this.winnersCircle = this.winnersCircle.filter(champ => {
            const generationsInCircle = this.generation - champ.championSince;
            return generationsInCircle < this.championGenerations;
        });
        
        // Keep only top performers
        if (this.winnersCircle.length > this.maxWinnersCircle) {
            this.winnersCircle.sort((a, b) => this.calculateFitness(b) - this.calculateFitness(a));
            this.winnersCircle = this.winnersCircle.slice(0, this.maxWinnersCircle);
        }
        
        return champion;
    }
    
    getWinnersCircleStatus() {
        return this.winnersCircle.map(champ => ({
            name: champ.name,
            generation: champ.championSince,
            victories: champ.victories,
            fitness: this.calculateFitness(champ),
            generationsLeft: this.championGenerations - (this.generation - champ.championSince),
            timesUsed: champ.championsUsed || 0
        }));
    }
}

// Export for use in other files
window.SnakeGenerator = SnakeGenerator;
