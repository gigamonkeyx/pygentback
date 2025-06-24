// Elite Evolution System - Advanced genetics, mutations, and skill evolution
class EliteEvolutionSystem {
    constructor(visualEffects) {
        this.visualEffects = visualEffects;
        
        // Evolution parameters
        this.mutationRate = 0.12; // 12% base mutation rate
        this.crossoverRate = 0.8; // 80% chance of genetic crossover
        this.elitePreservation = 0.1; // Top 10% preserved
        
        // Genetic system
        this.genePool = new Map();
        this.species = new Map();
        this.evolutionHistory = [];
        
        // Mutation weights and probabilities
        this.mutationWeights = {
            stats: 0.4,      // 40% chance for stat mutations
            skills: 0.25,    // 25% chance for skill mutations
            appearance: 0.2, // 20% chance for appearance mutations
            special: 0.1,    // 10% chance for special mutations
            legendary: 0.05  // 5% chance for legendary mutations
        };
        
        // Experience thresholds for evolution triggers
        this.evolutionThresholds = {
            minor: 100,    // Minor evolution after 100 XP
            major: 300,    // Major evolution after 300 XP
            legendary: 800 // Legendary evolution after 800 XP
        };
        
        // Genetic traits and their inheritance patterns
        this.geneticTraits = {
            dominant: ['Iron Scales', 'Venom Strike', 'Speed Boost', 'Thick Skin'],
            recessive: ['Regeneration', 'Phase Step', 'Psychic Awareness'],
            codominant: ['Bioluminescence', 'Pattern Shift', 'Color Change'],
            polygenic: ['Size', 'Intelligence', 'Agility', 'Endurance']
        };
        
        // Mutation catalog
        this.availableMutations = {
            beneficial: [
                { name: 'Enhanced Metabolism', effect: 'energy_regen_boost', rarity: 'common' },
                { name: 'Razor Scales', effect: 'damage_reflection', rarity: 'uncommon' },
                { name: 'Adaptive Camouflage', effect: 'stealth_mode', rarity: 'uncommon' },
                { name: 'Neural Enhancement', effect: 'ai_upgrade', rarity: 'rare' },
                { name: 'Quantum Scales', effect: 'phase_ability', rarity: 'legendary' }
            ],
            neutral: [
                { name: 'Color Shift', effect: 'appearance_change', rarity: 'common' },
                { name: 'Pattern Variation', effect: 'pattern_change', rarity: 'common' },
                { name: 'Size Fluctuation', effect: 'size_change', rarity: 'common' }
            ],
            detrimental: [
                { name: 'Metabolic Inefficiency', effect: 'energy_drain', rarity: 'uncommon' },
                { name: 'Brittle Scales', effect: 'damage_vulnerability', rarity: 'rare' }
            ]
        };
        
        console.log('Elite Evolution System initialized with advanced genetics');
    }
    
    // Main evolution method triggered after battles
    evolveSnake(snake, battleResults) {
        const evolutionReport = {
            snakeId: snake.id,
            snakeName: snake.name,
            generation: snake.generation || 1,
            preEvolution: this.captureSnakeSnapshot(snake),
            mutations: [],
            improvements: [],
            newTraits: [],
            evolutionType: 'none',
            experienceGained: battleResults.experienceGained || 0,
            success: false
        };
        
        // Calculate evolution potential based on battle performance
        const evolutionPotential = this.calculateEvolutionPotential(snake, battleResults);
        
        // Determine evolution type
        if (evolutionPotential.experience >= this.evolutionThresholds.legendary) {
            evolutionReport.evolutionType = 'legendary';
        } else if (evolutionPotential.experience >= this.evolutionThresholds.major) {
            evolutionReport.evolutionType = 'major';
        } else if (evolutionPotential.experience >= this.evolutionThresholds.minor) {
            evolutionReport.evolutionType = 'minor';
        }
        
        // Apply evolution if threshold met
        if (evolutionReport.evolutionType !== 'none') {
            this.applyEvolution(snake, evolutionReport, evolutionPotential);
            evolutionReport.success = true;
            
            // Add visual evolution effect
            this.visualEffects.addEvolutionEffect(snake, evolutionReport.evolutionType);
        }
        
        // Always apply minor improvements and chance for mutations
        this.applyBattleExperience(snake, battleResults, evolutionReport);
        
        // Record evolution in history
        this.recordEvolution(evolutionReport);
        
        return evolutionReport;
    }
    
    calculateEvolutionPotential(snake, battleResults) {
        let baseExperience = battleResults.experienceGained || 0;
        
        // Victory bonus
        if (battleResults.victory) {
            baseExperience += 50;
        }
        
        // Performance multipliers
        const performanceMultipliers = {
            dominance: battleResults.dominanceScore || 1,
            efficiency: battleResults.efficiency || 1,
            survival: battleResults.survivalTime || 1,
            damage: Math.min(2, (battleResults.damageDealt || 0) / 100),
            skills: Math.min(1.5, (battleResults.skillsUsed || 0) / 5)
        };
        
        // Calculate total multiplier
        let totalMultiplier = 1;
        Object.values(performanceMultipliers).forEach(mult => {
            totalMultiplier *= mult;
        });
        
        const finalExperience = Math.floor(baseExperience * totalMultiplier);
        
        return {
            experience: finalExperience,
            multiplier: totalMultiplier,
            baseExperience: baseExperience,
            factors: performanceMultipliers
        };
    }
    
    applyEvolution(snake, evolutionReport, potential) {
        switch (evolutionReport.evolutionType) {
            case 'minor':
                this.applyMinorEvolution(snake, evolutionReport);
                break;
            case 'major':
                this.applyMajorEvolution(snake, evolutionReport);
                break;
            case 'legendary':
                this.applyLegendaryEvolution(snake, evolutionReport);
                break;
        }
        
        // Update snake's generation and evolution metrics
        snake.generation = (snake.generation || 1) + 1;
        snake.totalEvolutions = (snake.totalEvolutions || 0) + 1;
        snake.evolutionHistory = snake.evolutionHistory || [];
        snake.evolutionHistory.push({
            type: evolutionReport.evolutionType,
            timestamp: Date.now(),
            changes: evolutionReport.mutations.concat(evolutionReport.improvements)
        });
    }
    
    applyMinorEvolution(snake, report) {
        // 1-2 stat improvements
        const statsToImprove = this.selectRandomStats(snake.stats, 1, 2);
        
        statsToImprove.forEach(stat => {
            const improvement = 5 + Math.floor(Math.random() * 10); // 5-15 point improvement
            snake.stats[stat] = Math.min(100, snake.stats[stat] + improvement);
            
            report.improvements.push({
                type: 'stat',
                stat: stat,
                improvement: improvement,
                newValue: snake.stats[stat]
            });
        });
        
        // 30% chance for new skill or skill upgrade
        if (Math.random() < 0.3) {
            this.evolveSkills(snake, report, 1);
        }
        
        // 20% chance for appearance mutation
        if (Math.random() < 0.2) {
            this.mutateAppearance(snake, report, 'minor');
        }
    }
    
    applyMajorEvolution(snake, report) {
        // 2-4 stat improvements (larger gains)
        const statsToImprove = this.selectRandomStats(snake.stats, 2, 4);
        
        statsToImprove.forEach(stat => {
            const improvement = 10 + Math.floor(Math.random() * 15); // 10-25 point improvement
            snake.stats[stat] = Math.min(100, snake.stats[stat] + improvement);
            
            report.improvements.push({
                type: 'stat',
                stat: stat,
                improvement: improvement,
                newValue: snake.stats[stat]
            });
        });
        
        // Guaranteed skill evolution
        this.evolveSkills(snake, report, 1, 2);
        
        // 60% chance for beneficial mutation
        if (Math.random() < 0.6) {
            this.applyBeneficialMutation(snake, report);
        }
        
        // 40% chance for appearance evolution
        if (Math.random() < 0.4) {
            this.mutateAppearance(snake, report, 'major');
        }
        
        // Health/energy upgrades
        snake.maxHealth = Math.min(200, snake.maxHealth + 20);
        snake.health = snake.maxHealth;
        snake.maxEnergy = Math.min(150, snake.maxEnergy + 15);
        
        report.improvements.push({
            type: 'vital',
            healthIncrease: 20,
            energyIncrease: 15
        });
    }
    
    applyLegendaryEvolution(snake, report) {
        // Massive stat improvements
        Object.keys(snake.stats).forEach(stat => {
            const improvement = 15 + Math.floor(Math.random() * 20); // 15-35 point improvement
            snake.stats[stat] = Math.min(120, snake.stats[stat] + improvement); // Can exceed normal cap
            
            report.improvements.push({
                type: 'stat',
                stat: stat,
                improvement: improvement,
                newValue: snake.stats[stat]
            });
        });
        
        // Multiple skill evolutions
        this.evolveSkills(snake, report, 2, 4);
        
        // Guaranteed legendary mutation
        this.applyLegendaryMutation(snake, report);
        
        // Dramatic appearance change
        this.mutateAppearance(snake, report, 'legendary');
        
        // Massive health/energy upgrades
        snake.maxHealth = Math.min(300, snake.maxHealth + 50);
        snake.health = snake.maxHealth;
        snake.maxEnergy = Math.min(200, snake.maxEnergy + 30);
        
        // Special legendary trait
        this.applyLegendaryTrait(snake, report);
        
        report.improvements.push({
            type: 'vital',
            healthIncrease: 50,
            energyIncrease: 30
        });
    }
    
    evolveSkills(snake, report, minSkills, maxSkills = null) {
        const numSkills = maxSkills ? 
            minSkills + Math.floor(Math.random() * (maxSkills - minSkills + 1)) : 
            minSkills;
        
        for (let i = 0; i < numSkills; i++) {
            if (Math.random() < 0.6 && snake.skills.length > 0) {
                // Upgrade existing skill
                this.upgradeExistingSkill(snake, report);
            } else {
                // Add new skill
                this.addNewSkill(snake, report);
            }
        }
    }
    
    upgradeExistingSkill(snake, report) {
        const upgradableSkills = snake.skills.filter(skill => !skill.includes('Master') && !skill.includes('Supreme'));
        
        if (upgradableSkills.length === 0) {
            this.addNewSkill(snake, report);
            return;
        }
        
        const skillToUpgrade = upgradableSkills[Math.floor(Math.random() * upgradableSkills.length)];
        const upgradedSkill = this.getSkillUpgrade(skillToUpgrade);
        
        // Replace old skill with upgraded version
        const skillIndex = snake.skills.indexOf(skillToUpgrade);
        snake.skills[skillIndex] = upgradedSkill;
        
        report.improvements.push({
            type: 'skill_upgrade',
            oldSkill: skillToUpgrade,
            newSkill: upgradedSkill
        });
    }
    
    getSkillUpgrade(skill) {
        const upgrades = {
            'Iron Scales': 'Iron Scales Master',
            'Venom Strike': 'Venom Strike Master',
            'Speed Boost': 'Lightning Speed',
            'Thick Skin': 'Armored Hide',
            'Regeneration': 'Advanced Regeneration',
            'Eagle Eyes': 'Hawk Vision',
            'Danger Sense': 'Precognition',
            'Lightning Reflexes': 'Time Dilation',
            'Battle Fury': 'Berserker Rage Master',
            'Energy Vampire': 'Life Drain Master'
        };
        
        return upgrades[skill] || skill + ' Enhanced';
    }
    
    addNewSkill(snake, report) {
        // Get available skills based on snake's stats and current skills
        const availableSkills = this.getAvailableSkills(snake);
        
        if (availableSkills.length === 0) return;
        
        const newSkill = availableSkills[Math.floor(Math.random() * availableSkills.length)];
        snake.skills.push(newSkill);
        
        report.improvements.push({
            type: 'new_skill',
            skill: newSkill
        });
    }
    
    getAvailableSkills(snake) {
        const allSkills = [
            // Combat skills
            'Iron Scales', 'Venom Strike', 'Lightning Reflexes', 'Berserker Rage',
            'Coil Mastery', 'Fang Sharpness', 'Battle Fury', 'Combat Veteran',
            
            // Defensive skills
            'Thick Skin', 'Regeneration', 'Shield Scales', 'Damage Reduction',
            'Evasion Master', 'Fortified Hide', 'Battle Scars', 'Survival Instinct',
            
            // Mobility skills
            'Speed Boost', 'Agile Movement', 'Wall Crawler', 'Phase Step',
            'Swift Strike', 'Momentum', 'Parkour', 'Slipstream',
            
            // Sensory skills
            'Eagle Eyes', 'Danger Sense', 'Food Scanner', 'Threat Detection',
            'Sixth Sense', 'Heat Vision', 'Motion Tracker', 'Radar Sense',
            
            // Metabolic skills
            'Efficient Digestion', 'Power Absorption', 'Energy Vampire',
            'Metabolic Boost', 'Nutrient Optimizer', 'Bio Reactor',
            
            // Special skills
            'Mutation Factor', 'Evolution Catalyst', 'Adaptation',
            'Metamorphosis', 'Genome Shifter', 'Genetic Memory'
        ];
        
        // Filter out skills the snake already has
        return allSkills.filter(skill => !snake.skills.includes(skill));
    }
    
    applyBeneficialMutation(snake, report) {
        const mutations = this.availableMutations.beneficial;
        const availableMutations = mutations.filter(mut => 
            Math.random() < this.getMutationChance(mut.rarity)
        );
        
        if (availableMutations.length === 0) return;
        
        const selectedMutation = availableMutations[Math.floor(Math.random() * availableMutations.length)];
        this.applyMutationEffect(snake, selectedMutation, report);
    }
    
    applyLegendaryMutation(snake, report) {
        const legendaryMutations = [
            { name: 'Transcendent Scales', effect: 'damage_immunity', rarity: 'legendary' },
            { name: 'Temporal Awareness', effect: 'time_manipulation', rarity: 'legendary' },
            { name: 'Quantum Entanglement', effect: 'teleportation', rarity: 'legendary' },
            { name: 'Neural Overdrive', effect: 'super_intelligence', rarity: 'legendary' },
            { name: 'Cellular Regeneration', effect: 'immortality', rarity: 'legendary' }
        ];
        
        const selectedMutation = legendaryMutations[Math.floor(Math.random() * legendaryMutations.length)];
        this.applyMutationEffect(snake, selectedMutation, report);
    }
    
    applyLegendaryTrait(snake, report) {
        const legendaryTraits = [
            'Alpha Predator', 'Apex Evolution', 'Genetic Perfection',
            'Quantum Consciousness', 'Transcendent Being'
        ];
        
        const trait = legendaryTraits[Math.floor(Math.random() * legendaryTraits.length)];
        snake.legendaryTraits = snake.legendaryTraits || [];
        snake.legendaryTraits.push(trait);
        
        report.newTraits.push({
            type: 'legendary',
            trait: trait,
            description: this.getLegendaryTraitDescription(trait)
        });
    }
    
    getLegendaryTraitDescription(trait) {
        const descriptions = {
            'Alpha Predator': 'Dominates all encounters through superior instincts',
            'Apex Evolution': 'Represents the pinnacle of evolutionary achievement',
            'Genetic Perfection': 'Possesses flawless genetic composition',
            'Quantum Consciousness': 'Awareness transcends physical reality',
            'Transcendent Being': 'Has evolved beyond normal biological limitations'
        };
        
        return descriptions[trait] || 'A mysterious legendary trait';
    }
    
    applyMutationEffect(snake, mutation, report) {
        switch (mutation.effect) {
            case 'energy_regen_boost':
                snake.energyRegenRate = (snake.energyRegenRate || 1) * 1.5;
                break;
                
            case 'damage_reflection':
                snake.damageReflection = (snake.damageReflection || 0) + 0.2; // 20% reflection
                break;
                
            case 'stealth_mode':
                snake.skills.push('Adaptive Camouflage');
                break;
                
            case 'ai_upgrade':
                snake.aiLevel = (snake.aiLevel || 1) + 1;
                snake.decisionSpeed = (snake.decisionSpeed || 1) * 1.3;
                break;
                
            case 'phase_ability':
                snake.skills.push('Quantum Phase');
                break;
                
            case 'damage_immunity':
                snake.damageImmunityChance = 0.1; // 10% chance to ignore damage
                break;
                
            case 'time_manipulation':
                snake.skills.push('Temporal Control');
                break;
                
            case 'teleportation':
                snake.skills.push('Quantum Teleport');
                break;
                
            case 'super_intelligence':
                snake.stats.intelligence = Math.min(150, snake.stats.intelligence + 30);
                break;
                
            case 'immortality':
                snake.maxHealth += 100;
                snake.health = snake.maxHealth;
                snake.skills.push('Cellular Regeneration');
                break;
        }
        
        report.mutations.push({
            name: mutation.name,
            effect: mutation.effect,
            rarity: mutation.rarity,
            description: this.getMutationDescription(mutation)
        });
    }
    
    getMutationDescription(mutation) {
        const descriptions = {
            'Enhanced Metabolism': 'Dramatically improved energy regeneration rate',
            'Razor Scales': 'Reflects portion of incoming damage back to attackers',
            'Adaptive Camouflage': 'Can become partially invisible during combat',
            'Neural Enhancement': 'Significantly improved AI decision making',
            'Quantum Scales': 'Ability to phase through attacks occasionally',
            'Transcendent Scales': 'Chance to completely ignore incoming damage',
            'Temporal Awareness': 'Can manipulate time flow during critical moments',
            'Quantum Entanglement': 'Instantaneous teleportation across the battlefield',
            'Neural Overdrive': 'Superhuman intelligence and reaction time',
            'Cellular Regeneration': 'Rapidly heals from any injury'
        };
        
        return descriptions[mutation.name] || 'A mysterious genetic mutation';
    }
    
    mutateAppearance(snake, report, intensity) {
        const appearanceMutations = [];
        
        switch (intensity) {
            case 'minor':
                // Slight color variations
                appearanceMutations.push(this.mutateColor(snake, 0.1));
                break;
                
            case 'major':
                // Significant appearance changes
                appearanceMutations.push(this.mutateColor(snake, 0.3));
                appearanceMutations.push(this.mutatePattern(snake));
                if (Math.random() < 0.3) {
                    appearanceMutations.push(this.mutateSize(snake));
                }
                break;
                
            case 'legendary':
                // Dramatic transformation
                appearanceMutations.push(this.mutateColor(snake, 0.6));
                appearanceMutations.push(this.mutatePattern(snake));
                appearanceMutations.push(this.mutateSize(snake));
                appearanceMutations.push(this.addSpecialEffects(snake));
                break;
        }
        
        appearanceMutations.forEach(mutation => {
            if (mutation) {
                report.mutations.push(mutation);
            }
        });
    }
    
    mutateColor(snake, intensity) {
        const originalColor = snake.appearance.primaryColor;
        
        // Generate new color based on intensity
        let newColor;
        if (intensity < 0.2) {
            // Slight hue shift
            newColor = this.shiftHue(originalColor, 30);
        } else if (intensity < 0.5) {
            // Significant color change
            newColor = this.generateRelatedColor(originalColor);
        } else {
            // Complete color transformation
            newColor = this.generateRandomColor();
        }
        
        snake.appearance.primaryColor = newColor;
        
        return {
            type: 'color_change',
            from: originalColor,
            to: newColor,
            intensity: intensity,
            description: `Color shifted from ${originalColor} to ${newColor}`
        };
    }
    
    mutatePattern(snake) {
        const patterns = ['solid', 'striped', 'spotted', 'gradient', 'metallic', 'iridescent'];
        const oldPattern = snake.appearance.pattern || 'solid';
        let newPattern;
        
        do {
            newPattern = patterns[Math.floor(Math.random() * patterns.length)];
        } while (newPattern === oldPattern);
        
        snake.appearance.pattern = newPattern;
        
        // Add secondary color if pattern requires it
        if (['striped', 'spotted', 'gradient'].includes(newPattern)) {
            snake.appearance.secondaryColor = this.generateComplementaryColor(snake.appearance.primaryColor);
        }
        
        return {
            type: 'pattern_change',
            from: oldPattern,
            to: newPattern,
            description: `Pattern changed from ${oldPattern} to ${newPattern}`
        };
    }
    
    mutateSize(snake) {
        const sizeMultipliers = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5];
        const originalSize = snake.appearance.sizeMultiplier || 1;
        const newSize = sizeMultipliers[Math.floor(Math.random() * sizeMultipliers.length)];
        
        snake.appearance.sizeMultiplier = newSize;
        
        // Adjust stats based on size change
        if (newSize > 1) {
            snake.stats.strength += Math.floor((newSize - 1) * 20);
            snake.stats.speed -= Math.floor((newSize - 1) * 10);
        } else {
            snake.stats.speed += Math.floor((1 - newSize) * 20);
            snake.stats.agility += Math.floor((1 - newSize) * 15);
        }
        
        return {
            type: 'size_change',
            from: originalSize,
            to: newSize,
            description: `Size ${newSize > originalSize ? 'increased' : 'decreased'} by ${Math.abs(newSize - originalSize) * 100}%`
        };
    }
    
    addSpecialEffects(snake) {
        const effects = ['glow', 'sparkles', 'aura', 'trail', 'energy_field'];
        const effect = effects[Math.floor(Math.random() * effects.length)];
        
        snake.appearance.specialEffects = snake.appearance.specialEffects || [];
        snake.appearance.specialEffects.push(effect);
        
        return {
            type: 'special_effect',
            effect: effect,
            description: `Gained ${effect} visual effect`
        };
    }
    
    applyBattleExperience(snake, battleResults, report) {
        // Small stat improvements based on battle performance
        if (battleResults.victory) {
            const statImprovement = 1 + Math.floor(Math.random() * 3);
            const randomStat = this.selectRandomStats(snake.stats, 1)[0];
            snake.stats[randomStat] = Math.min(100, snake.stats[randomStat] + statImprovement);
            
            report.improvements.push({
                type: 'battle_experience',
                stat: randomStat,
                improvement: statImprovement
            });
        }
        
        // Random mutation chance (low probability)
        if (Math.random() < this.mutationRate) {
            const mutationType = this.selectMutationType();
            this.applyRandomMutation(snake, report, mutationType);
        }
    }
    
    selectMutationType() {
        const rand = Math.random();
        let cumulative = 0;
        
        for (const [type, weight] of Object.entries(this.mutationWeights)) {
            cumulative += weight;
            if (rand < cumulative) {
                return type;
            }
        }
        
        return 'stats'; // Fallback
    }
    
    applyRandomMutation(snake, report, type) {
        switch (type) {
            case 'stats':
                this.applyStatMutation(snake, report);
                break;
            case 'skills':
                this.applySkillMutation(snake, report);
                break;
            case 'appearance':
                this.mutateAppearance(snake, report, 'minor');
                break;
            case 'special':
                this.applyBeneficialMutation(snake, report);
                break;
            case 'legendary':
                if (Math.random() < 0.1) { // Only 10% chance even when selected
                    this.applyLegendaryMutation(snake, report);
                }
                break;
        }
    }
    
    applyStatMutation(snake, report) {
        const stat = this.selectRandomStats(snake.stats, 1)[0];
        const change = (Math.random() < 0.8 ? 1 : -1) * (1 + Math.floor(Math.random() * 3));
        const oldValue = snake.stats[stat];
        
        snake.stats[stat] = Math.max(1, Math.min(100, snake.stats[stat] + change));
        
        report.mutations.push({
            type: 'stat_mutation',
            stat: stat,
            change: change,
            oldValue: oldValue,
            newValue: snake.stats[stat]
        });
    }
    
    applySkillMutation(snake, report) {
        if (Math.random() < 0.7) {
            // Add new skill
            this.addNewSkill(snake, report);
        } else {
            // Lose a skill (rare)
            if (snake.skills.length > 2) {
                const lostSkill = snake.skills.splice(Math.floor(Math.random() * snake.skills.length), 1)[0];
                report.mutations.push({
                    type: 'skill_loss',
                    skill: lostSkill
                });
            }
        }
    }
    
    // Breeding and genetic crossover for tournament winners
    breedSnakes(parent1, parent2) {
        const offspring = {
            id: this.generateSnakeId(),
            name: this.generateHybridName(parent1.name, parent2.name),
            generation: Math.max(parent1.generation || 1, parent2.generation || 1) + 1,
            parents: [parent1.id, parent2.id],
            stats: this.crossoverStats(parent1.stats, parent2.stats),
            skills: this.crossoverSkills(parent1.skills, parent2.skills),
            appearance: this.crossoverAppearance(parent1.appearance, parent2.appearance),
            maxHealth: Math.floor((parent1.maxHealth + parent2.maxHealth) / 2) + Math.floor(Math.random() * 20) - 10,
            maxEnergy: Math.floor((parent1.maxEnergy + parent2.maxEnergy) / 2) + Math.floor(Math.random() * 10) - 5,
            evolutionHistory: []
        };
        
        offspring.health = offspring.maxHealth;
        offspring.energy = offspring.maxEnergy;
        
        // Apply breeding mutations
        if (Math.random() < 0.3) {
            this.applyBreedingMutation(offspring);
        }
        
        return offspring;
    }
    
    crossoverStats(stats1, stats2) {
        const offspring = {};
        
        Object.keys(stats1).forEach(stat => {
            if (Math.random() < this.crossoverRate) {
                // Crossover: average with some variance
                const average = (stats1[stat] + stats2[stat]) / 2;
                const variance = Math.floor(Math.random() * 10) - 5;
                offspring[stat] = Math.max(1, Math.min(100, Math.floor(average + variance)));
            } else {
                // Inherit from one parent
                offspring[stat] = Math.random() < 0.5 ? stats1[stat] : stats2[stat];
            }
        });
        
        return offspring;
    }
    
    crossoverSkills(skills1, skills2) {
        const allSkills = [...new Set([...skills1, ...skills2])];
        const offspring = [];
        
        allSkills.forEach(skill => {
            const inheritanceChance = this.getSkillInheritanceChance(skill, skills1, skills2);
            if (Math.random() < inheritanceChance) {
                offspring.push(skill);
            }
        });
        
        // Ensure minimum skill count
        if (offspring.length < 2) {
            const availableSkills = allSkills.filter(skill => !offspring.includes(skill));
            while (offspring.length < 2 && availableSkills.length > 0) {
                const randomSkill = availableSkills.splice(Math.floor(Math.random() * availableSkills.length), 1)[0];
                offspring.push(randomSkill);
            }
        }
        
        return offspring;
    }
    
    getSkillInheritanceChance(skill, skills1, skills2) {
        const inParent1 = skills1.includes(skill);
        const inParent2 = skills2.includes(skill);
        
        if (inParent1 && inParent2) {
            // Both parents have it
            if (this.geneticTraits.dominant.includes(skill)) {
                return 0.9; // 90% chance for dominant traits
            } else {
                return 0.7; // 70% chance for others
            }
        } else if (inParent1 || inParent2) {
            // One parent has it
            if (this.geneticTraits.dominant.includes(skill)) {
                return 0.6; // 60% chance for dominant traits
            } else if (this.geneticTraits.recessive.includes(skill)) {
                return 0.2; // 20% chance for recessive traits
            } else {
                return 0.4; // 40% chance for others
            }
        }
        
        return 0; // Neither parent has it
    }
    
    crossoverAppearance(appearance1, appearance2) {
        const offspring = {};
        
        // Color mixing
        if (Math.random() < 0.5) {
            offspring.primaryColor = appearance1.primaryColor;
        } else {
            offspring.primaryColor = appearance2.primaryColor;
        }
        
        // Sometimes create a blend
        if (Math.random() < 0.2) {
            offspring.primaryColor = this.blendColors(appearance1.primaryColor, appearance2.primaryColor);
        }
        
        // Pattern inheritance
        const patterns = [appearance1.pattern, appearance2.pattern].filter(p => p);
        if (patterns.length > 0) {
            offspring.pattern = patterns[Math.floor(Math.random() * patterns.length)];
        }
        
        // Size inheritance
        const size1 = appearance1.sizeMultiplier || 1;
        const size2 = appearance2.sizeMultiplier || 1;
        offspring.sizeMultiplier = (size1 + size2) / 2 + (Math.random() - 0.5) * 0.2;
        
        return offspring;
    }
    
    // Utility methods
    selectRandomStats(stats, min, max = null) {
        const statNames = Object.keys(stats);
        const count = max ? min + Math.floor(Math.random() * (max - min + 1)) : min;
        
        const selected = [];
        const available = [...statNames];
        
        for (let i = 0; i < Math.min(count, available.length); i++) {
            const index = Math.floor(Math.random() * available.length);
            selected.push(available.splice(index, 1)[0]);
        }
        
        return selected;
    }
    
    getMutationChance(rarity) {
        const chances = {
            common: 0.8,
            uncommon: 0.5,
            rare: 0.2,
            legendary: 0.05
        };
        
        return chances[rarity] || 0.1;
    }
    
    generateSnakeId() {
        return 'snake_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
    }
    
    generateHybridName(name1, name2) {
        const words1 = name1.split(' ');
        const words2 = name2.split(' ');
        
        if (Math.random() < 0.5) {
            return words1[0] + ' ' + (words2[words2.length - 1] || words2[0]);
        } else {
            return words2[0] + ' ' + (words1[words1.length - 1] || words1[0]);
        }
    }
    
    captureSnakeSnapshot(snake) {
        return JSON.parse(JSON.stringify({
            stats: snake.stats,
            skills: snake.skills,
            appearance: snake.appearance,
            health: snake.maxHealth,
            energy: snake.maxEnergy
        }));
    }
    
    recordEvolution(evolutionReport) {
        this.evolutionHistory.push({
            ...evolutionReport,
            timestamp: Date.now()
        });
        
        // Keep history manageable
        if (this.evolutionHistory.length > 100) {
            this.evolutionHistory.shift();
        }
    }
    
    // Color utility methods
    shiftHue(color, degrees) {
        // Convert hex to HSL, shift hue, convert back
        const hsl = this.hexToHsl(color);
        hsl.h = (hsl.h + degrees) % 360;
        return this.hslToHex(hsl);
    }
    
    generateRelatedColor(color) {
        const hsl = this.hexToHsl(color);
        hsl.h = (hsl.h + 60 + Math.random() * 120) % 360; // Shift 60-180 degrees
        hsl.s = Math.max(0.2, Math.min(1, hsl.s + (Math.random() - 0.5) * 0.3));
        hsl.l = Math.max(0.2, Math.min(0.8, hsl.l + (Math.random() - 0.5) * 0.3));
        return this.hslToHex(hsl);
    }
    
    generateRandomColor() {
        const colors = [
            '#ff4444', '#44ff44', '#4444ff', '#ffff44', '#ff44ff', '#44ffff',
            '#ff8844', '#88ff44', '#4488ff', '#ff4488', '#88ff88', '#8844ff'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    generateComplementaryColor(color) {
        const hsl = this.hexToHsl(color);
        hsl.h = (hsl.h + 180) % 360;
        return this.hslToHex(hsl);
    }
    
    blendColors(color1, color2) {
        const rgb1 = this.hexToRgb(color1);
        const rgb2 = this.hexToRgb(color2);
        
        const blended = {
            r: Math.floor((rgb1.r + rgb2.r) / 2),
            g: Math.floor((rgb1.g + rgb2.g) / 2),
            b: Math.floor((rgb1.b + rgb2.b) / 2)
        };
        
        return this.rgbToHex(blended);
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 255, g: 255, b: 255 };
    }
    
    rgbToHex(rgb) {
        return "#" + ((1 << 24) + (rgb.r << 16) + (rgb.g << 8) + rgb.b).toString(16).slice(1);
    }
    
    hexToHsl(hex) {
        const rgb = this.hexToRgb(hex);
        const r = rgb.r / 255;
        const g = rgb.g / 255;
        const b = rgb.b / 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        
        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        
        return { h: h * 360, s: s, l: l };
    }
    
    hslToHex(hsl) {
        const h = hsl.h / 360;
        const s = hsl.s;
        const l = hsl.l;
        
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        
        let r, g, b;
        
        if (s === 0) {
            r = g = b = l;
        } else {
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        
        return this.rgbToHex({
            r: Math.round(r * 255),
            g: Math.round(g * 255),
            b: Math.round(b * 255)
        });
    }
}
