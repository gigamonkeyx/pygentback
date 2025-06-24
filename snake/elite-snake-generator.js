// Elite Snake Generator - Creates unique, detailed snakes with proper evolution and mutations
class EliteSnakeGenerator {
    constructor() {
        this.generation = 1;
        this.snakeCounter = 0;
        this.mutationChance = 0.15; // 15% chance for mutations
        
        // Rich naming system with more variety
        this.nameComponents = {
            prefixes: [
                'Crimson', 'Azure', 'Golden', 'Shadow', 'Frost', 'Ember', 'Storm', 'Void', 'Prismatic', 'Cosmic',
                'Neon', 'Crystal', 'Phantom', 'Viper', 'Cobra', 'Python', 'Mamba', 'Boa', 'Adder', 'Serpent',
                'Quantum', 'Plasma', 'Stellar', 'Nebula', 'Galaxy', 'Thunder', 'Lightning', 'Blaze', 'Glacier', 'Prism'
            ],
            cores: [
                'Strike', 'Fang', 'Coil', 'Slither', 'Hiss', 'Scale', 'Venom', 'Hunt', 'Swift', 'Sage',
                'Fury', 'Grace', 'Wrath', 'Spirit', 'Soul', 'Heart', 'Mind', 'Claw', 'Spine', 'Crown',
                'Wing', 'Blade', 'Storm', 'Fire', 'Ice', 'Light', 'Dark', 'Void', 'Star', 'Moon'
            ],
            suffixes: [
                'bringer', 'walker', 'rider', 'master', 'lord', 'queen', 'king', 'hunter', 'seeker', 'keeper',
                'weaver', 'dancer', 'singer', 'whisper', 'roar', 'cry', 'call', 'song', 'tale', 'legend'
            ]
        };

        // Comprehensive skill system with categories
        this.skillCategories = {
            combat: [
                'Iron Scales', 'Venom Strike', 'Lightning Reflexes', 'Berserker Rage', 'Coil Mastery',
                'Fang Sharpness', 'Battle Fury', 'Combat Veteran', 'Warrior Spirit', 'Death Dealer'
            ],
            defensive: [
                'Thick Skin', 'Regeneration', 'Shield Scales', 'Damage Reduction', 'Evasion Master',
                'Fortified Hide', 'Battle Scars', 'Survival Instinct', 'Last Stand', 'Phoenix Heart'
            ],
            mobility: [
                'Speed Boost', 'Agile Movement', 'Wall Crawler', 'Phase Step', 'Teleport',
                'Swift Strike', 'Momentum', 'Parkour', 'Slipstream', 'Wind Walker'
            ],
            sensory: [
                'Eagle Eyes', 'Danger Sense', 'Food Scanner', 'Threat Detection', 'Sixth Sense',
                'Heat Vision', 'Motion Tracker', 'Radar Sense', 'Psychic Awareness', 'True Sight'
            ],
            metabolic: [
                'Efficient Digestion', 'Power Absorption', 'Energy Vampire', 'Metabolic Boost',
                'Nutrient Optimizer', 'Calorie Burner', 'Fat Storage', 'Lean Machine', 'Power Core', 'Bio Reactor'
            ],
            special: [
                'Mutation Factor', 'Evolution Catalyst', 'Adaptation', 'Metamorphosis', 'Genome Shifter',
                'DNA Weaver', 'Genetic Memory', 'Ancestral Power', 'Bloodline Trait', 'Legacy Code'
            ]
        };

        // Color palettes for different snake types
        this.colorPalettes = {
            fire: ['#ff4444', '#ff6600', '#ff8800', '#ffaa00', '#ffcc00'],
            ice: ['#00ccff', '#0099ff', '#0066ff', '#4444ff', '#6666ff'],
            nature: ['#00ff00', '#44ff44', '#88ff88', '#00cc00', '#00aa00'],
            shadow: ['#333333', '#555555', '#666666', '#444444', '#222222'],
            cosmic: ['#9900ff', '#bb00ff', '#dd00ff', '#ff00cc', '#ff0099'],
            poison: ['#88ff00', '#aaff00', '#ccff00', '#99cc00', '#77aa00'],
            electric: ['#ffff00', '#ffcc00', '#ff9900', '#ffaa44', '#ffdd44'],
            royal: ['#ff00ff', '#cc00cc', '#9900cc', '#6600cc', '#3300cc'],
            crystal: ['#00ffff', '#44ffff', '#88ffff', '#aaffff', '#ccffff'],
            volcanic: ['#ff0000', '#cc0000', '#990000', '#660000', '#330000']
        };

        // Mutation types that can occur
        this.mutationTypes = {
            size: {
                name: 'Size Mutation',
                effects: ['Gigantism', 'Miniaturization', 'Length Extension', 'Compact Form'],
                rarity: 'common'
            },
            color: {
                name: 'Chromatic Shift',
                effects: ['Bioluminescence', 'Color Change', 'Pattern Shift', 'Iridescence'],
                rarity: 'common'
            },
            ability: {
                name: 'Ability Evolution',
                effects: ['Skill Enhancement', 'New Ability', 'Ability Fusion', 'Power Amplification'],
                rarity: 'uncommon'
            },
            physical: {
                name: 'Physical Adaptation',
                effects: ['Extra Segments', 'Armor Plating', 'Spine Growth', 'Membrane Development'],
                rarity: 'uncommon'
            },
            rare: {
                name: 'Rare Mutation',
                effects: ['Dual Consciousness', 'Temporal Awareness', 'Quantum Entanglement', 'Psychic Powers'],
                rarity: 'rare'
            },
            legendary: {
                name: 'Legendary Evolution',
                effects: ['Transcendence', 'God Mode', 'Reality Warping', 'Universal Awareness'],
                rarity: 'legendary'
            }
        };
    }

    generateName() {
        const prefix = this.nameComponents.prefixes[Math.floor(Math.random() * this.nameComponents.prefixes.length)];
        const core = this.nameComponents.cores[Math.floor(Math.random() * this.nameComponents.cores.length)];
        
        if (Math.random() < 0.3) {
            const suffix = this.nameComponents.suffixes[Math.floor(Math.random() * this.nameComponents.suffixes.length)];
            return `${prefix} ${core}${suffix}`;
        }
        
        return `${prefix} ${core}`;
    }

    generateStats() {
        return {
            strength: Math.floor(Math.random() * 100) + 1,
            speed: Math.floor(Math.random() * 100) + 1,
            agility: Math.floor(Math.random() * 100) + 1,
            intelligence: Math.floor(Math.random() * 100) + 1,
            endurance: Math.floor(Math.random() * 100) + 1,
            luck: Math.floor(Math.random() * 100) + 1,
            charisma: Math.floor(Math.random() * 100) + 1,
            wisdom: Math.floor(Math.random() * 100) + 1
        };
    }

    generateSkills(stats) {
        const skills = [];
        const numSkills = Math.floor(Math.random() * 3) + 2; // 2-4 skills
        
        // Weighted skill selection based on stats
        const skillPool = [];
        
        if (stats.strength > 70) skillPool.push(...this.skillCategories.combat);
        if (stats.endurance > 70) skillPool.push(...this.skillCategories.defensive);
        if (stats.speed > 70) skillPool.push(...this.skillCategories.mobility);
        if (stats.intelligence > 70) skillPool.push(...this.skillCategories.sensory);
        if (stats.wisdom > 70) skillPool.push(...this.skillCategories.metabolic);
        if (stats.luck > 80) skillPool.push(...this.skillCategories.special);

        // If no high stats, use all skills
        if (skillPool.length === 0) {
            Object.values(this.skillCategories).forEach(category => {
                skillPool.push(...category);
            });
        }

        // Select unique skills
        const usedSkills = new Set();
        for (let i = 0; i < numSkills && skillPool.length > 0; i++) {
            let attempts = 0;
            let skill;
            do {
                skill = skillPool[Math.floor(Math.random() * skillPool.length)];
                attempts++;
            } while (usedSkills.has(skill) && attempts < 20);
            
            if (!usedSkills.has(skill)) {
                skills.push(skill);
                usedSkills.add(skill);
            }
        }

        return skills;
    }

    generateAppearance(stats, skills) {
        // Choose color palette based on skills and stats
        let paletteType = 'nature'; // default
        
        if (skills.includes('Iron Scales') || skills.includes('Thick Skin')) {
            paletteType = 'shadow';
        } else if (skills.includes('Venom Strike') || skills.includes('Energy Vampire')) {
            paletteType = 'poison';
        } else if (skills.includes('Lightning Reflexes') || skills.includes('Speed Boost')) {
            paletteType = 'electric';
        } else if (skills.includes('Regeneration') || skills.includes('Phoenix Heart')) {
            paletteType = 'fire';
        } else if (stats.intelligence > 80) {
            paletteType = 'cosmic';
        } else if (stats.luck > 80) {
            paletteType = 'crystal';
        } else if (stats.strength > 80) {
            paletteType = 'volcanic';
        }

        const palette = this.colorPalettes[paletteType];
        const primaryColor = palette[Math.floor(Math.random() * palette.length)];
        const secondaryColor = palette[Math.floor(Math.random() * palette.length)];

        return {
            primaryColor,
            secondaryColor,
            paletteType,
            pattern: this.generatePattern(),
            size: this.calculateSize(stats),
            segments: Math.floor(Math.random() * 5) + 8, // 8-12 segments
            eyeColor: this.generateEyeColor(stats),
            specialEffects: this.generateSpecialEffects(skills)
        };
    }

    generatePattern() {
        const patterns = [
            'solid', 'stripes', 'spots', 'diamond', 'zigzag', 
            'gradient', 'scales', 'tribal', 'cosmic', 'crystalline'
        ];
        return patterns[Math.floor(Math.random() * patterns.length)];
    }

    calculateSize(stats) {
        const baseSize = 15;
        const strengthBonus = (stats.strength / 100) * 8;
        const enduranceBonus = (stats.endurance / 100) * 5;
        return Math.floor(baseSize + strengthBonus + enduranceBonus);
    }

    generateEyeColor(stats) {
        if (stats.intelligence > 80) return '#00ffff'; // Cyan for high intelligence
        if (stats.wisdom > 80) return '#ffd700'; // Gold for high wisdom
        if (stats.strength > 80) return '#ff0000'; // Red for high strength
        if (stats.speed > 80) return '#ffff00'; // Yellow for high speed
        return '#ffffff'; // Default white
    }

    generateSpecialEffects(skills) {
        const effects = [];
        
        if (skills.includes('Iron Scales')) effects.push('metallic');
        if (skills.includes('Venom Strike')) effects.push('poisonous');
        if (skills.includes('Lightning Reflexes')) effects.push('electric');
        if (skills.includes('Regeneration')) effects.push('healing');
        if (skills.includes('Phase Step')) effects.push('ghostly');
        if (skills.includes('Mutation Factor')) effects.push('unstable');
        
        return effects;
    }

    createSnake(parents = null) {
        this.snakeCounter++;
        
        let stats, skills, generation = 1;
        let mutations = [];
        
        if (parents && parents.length > 0) {
            // Evolved snake - inherit and potentially mutate
            stats = this.evolveStats(parents);
            skills = this.evolveSkills(parents);
            generation = Math.max(...parents.map(p => p.generation || 1)) + 1;
            
            // Check for mutations
            if (Math.random() < this.mutationChance) {
                mutations = this.generateMutations(stats, skills);
                this.applyMutations(stats, skills, mutations);
            }
        } else {
            // New random snake
            stats = this.generateStats();
            skills = this.generateSkills(stats);
        }

        const appearance = this.generateAppearance(stats, skills);
        
        // Calculate health based on Iron Scales and other defensive skills
        let maxHealth = 3;
        if (skills.includes('Iron Scales')) maxHealth = 6;
        if (skills.includes('Phoenix Heart')) maxHealth += 2;
        if (skills.includes('Thick Skin')) maxHealth += 1;

        const snake = {
            id: this.snakeCounter,
            name: this.generateName(),
            stats,
            skills,
            appearance,
            mutations,
            generation,
            health: maxHealth,
            maxHealth,
            score: 0,
            victories: 0,
            defeats: 0,
            powerupsCollected: 0,
            timeAlive: 0,
            birthTime: Date.now(),
            
            // Physical properties
            body: [],
            direction: 'right',
            speed: this.calculateSpeed(stats, skills),
            
            // Status effects
            invulnerable: false,
            invulnerableTime: 0,
            poisoned: false,
            poisonTime: 0,
            
            // Evolution tracking
            hasEvolved: parents !== null,
            parentNames: parents ? parents.map(p => p.name) : []
        };

        return snake;
    }

    evolveStats(parents) {
        const newStats = {};
        const statKeys = Object.keys(parents[0].stats);
        
        statKeys.forEach(key => {
            // Average parent stats with some variation
            const average = parents.reduce((sum, parent) => sum + parent.stats[key], 0) / parents.length;
            const variation = (Math.random() - 0.5) * 20; // ±10 variation
            newStats[key] = Math.max(1, Math.min(100, Math.floor(average + variation)));
        });
        
        return newStats;
    }

    evolveSkills(parents) {
        const inheritedSkills = new Set();
        
        // Inherit skills from parents with high probability
        parents.forEach(parent => {
            parent.skills.forEach(skill => {
                if (Math.random() < 0.8) { // 80% chance to inherit each skill
                    inheritedSkills.add(skill);
                }
            });
        });

        const skillsArray = Array.from(inheritedSkills);
        
        // Chance to gain new skill
        if (Math.random() < 0.3 && skillsArray.length < 6) {
            const allSkills = Object.values(this.skillCategories).flat();
            const availableSkills = allSkills.filter(skill => !inheritedSkills.has(skill));
            if (availableSkills.length > 0) {
                const newSkill = availableSkills[Math.floor(Math.random() * availableSkills.length)];
                skillsArray.push(newSkill);
            }
        }
        
        return skillsArray;
    }

    generateMutations(stats, skills) {
        const mutations = [];
        const mutationCount = Math.random() < 0.1 ? 2 : 1; // 10% chance for double mutation
        
        for (let i = 0; i < mutationCount; i++) {
            const mutationType = this.selectMutationType();
            const mutation = this.createMutation(mutationType, stats, skills);
            mutations.push(mutation);
        }
        
        return mutations;
    }

    selectMutationType() {
        const rand = Math.random();
        
        if (rand < 0.4) return 'size';      // 40%
        if (rand < 0.7) return 'color';     // 30%
        if (rand < 0.85) return 'ability';  // 15%
        if (rand < 0.95) return 'physical'; // 10%
        if (rand < 0.99) return 'rare';     // 4%
        return 'legendary';                  // 1%
    }

    createMutation(type, stats, skills) {
        const mutationInfo = this.mutationTypes[type];
        const effect = mutationInfo.effects[Math.floor(Math.random() * mutationInfo.effects.length)];
        
        return {
            type,
            name: mutationInfo.name,
            effect,
            rarity: mutationInfo.rarity,
            description: this.generateMutationDescription(type, effect),
            timestamp: Date.now()
        };
    }

    generateMutationDescription(type, effect) {
        const descriptions = {
            size: {
                'Gigantism': 'Snake grows to massive proportions, +50% size!',
                'Miniaturization': 'Snake shrinks but gains incredible speed!',
                'Length Extension': 'Snake can extend its body length dramatically!',
                'Compact Form': 'Snake becomes incredibly dense and resilient!'
            },
            color: {
                'Bioluminescence': 'Snake develops a brilliant, glowing aura!',
                'Color Change': 'Snake can shift colors like a chameleon!',
                'Pattern Shift': 'Intricate new patterns emerge across the scales!',
                'Iridescence': 'Scales shimmer with rainbow colors!'
            },
            ability: {
                'Skill Enhancement': 'Existing abilities become more powerful!',
                'New Ability': 'A completely new skill manifests!',
                'Ability Fusion': 'Two skills combine into something greater!',
                'Power Amplification': 'All abilities are amplified!'
            },
            physical: {
                'Extra Segments': 'Additional body segments provide more health!',
                'Armor Plating': 'Protective armor develops along the spine!',
                'Spine Growth': 'Defensive spines emerge from the scales!',
                'Membrane Development': 'Wing-like membranes aid in movement!'
            },
            rare: {
                'Dual Consciousness': 'Snake develops a second mind for strategy!',
                'Temporal Awareness': 'Snake can sense moments before they happen!',
                'Quantum Entanglement': 'Snake exists in multiple states simultaneously!',
                'Psychic Powers': 'Telepathic abilities emerge!'
            },
            legendary: {
                'Transcendence': 'Snake transcends physical limitations!',
                'God Mode': 'Near-invulnerability is achieved!',
                'Reality Warping': 'Snake can bend the rules of physics!',
                'Universal Awareness': 'Complete understanding of all things!'
            }
        };
        
        return descriptions[type] && descriptions[type][effect] ? descriptions[type][effect] : 'Unknown mutation effect!';
    }

    applyMutations(stats, skills, mutations) {
        mutations.forEach(mutation => {
            switch (mutation.type) {
                case 'size':
                    if (mutation.effect === 'Gigantism') {
                        stats.strength += 20;
                        stats.endurance += 15;
                    } else if (mutation.effect === 'Miniaturization') {
                        stats.speed += 25;
                        stats.agility += 20;
                    }
                    break;
                    
                case 'ability':
                    if (mutation.effect === 'New Ability' && skills.length < 8) {
                        const allSkills = Object.values(this.skillCategories).flat();
                        const availableSkills = allSkills.filter(skill => !skills.includes(skill));
                        if (availableSkills.length > 0) {
                            const newSkill = availableSkills[Math.floor(Math.random() * availableSkills.length)];
                            skills.push(newSkill);
                        }
                    } else if (mutation.effect === 'Power Amplification') {
                        Object.keys(stats).forEach(key => {
                            stats[key] = Math.min(100, stats[key] + 10);
                        });
                    }
                    break;
                    
                case 'rare':
                case 'legendary':
                    // These provide special abilities that are handled in gameplay
                    Object.keys(stats).forEach(key => {
                        stats[key] = Math.min(100, stats[key] + 5);
                    });
                    break;
            }
        });
    }

    calculateSpeed(stats, skills) {
        let speed = 100 + (stats.speed * 2); // Base speed + speed stat bonus
        
        if (skills.includes('Speed Boost')) speed += 50;
        if (skills.includes('Lightning Reflexes')) speed += 30;
        if (skills.includes('Swift Strike')) speed += 40;
        if (skills.includes('Wind Walker')) speed += 35;
        
        return Math.max(50, speed); // Minimum speed of 50ms
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EliteSnakeGenerator;
}
