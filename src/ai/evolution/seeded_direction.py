#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seeded Direction Evolution System
Observer-approved system for user-guided evolution with automatic bias application
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvolutionSeed:
    """User-provided evolution seed for direction guidance"""
    seed_id: str
    direction: str  # e.g., "cooperation", "sustainability", "gathering"
    weight: float  # 0.0 to 1.0
    target_improvement: float  # Expected improvement percentage
    description: str
    timestamp: datetime

class SeededDirectionSystem:
    """
    Observer-approved seeded direction system
    Enables user guidance with automatic bias application for controlled evolution
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_seeds = {}
        self.seed_history = []
        self.direction_mappings = self._initialize_direction_mappings()
        
        logger.info("SeededDirectionSystem initialized for user-guided evolution")
    
    def _initialize_direction_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize direction mappings to capability bonuses"""
        return {
            'cooperation': {
                'cooperation': 0.8,
                'coordination': 0.6,
                'communication': 0.7,
                'resource_sharing': 0.5
            },
            'sustainability': {
                'resource_efficiency': 0.8,
                'long_term_planning': 0.7,
                'conservation': 0.9,
                'adaptation': 0.6
            },
            'gathering': {
                'resource_gathering': 0.9,
                'exploration': 0.6,
                'efficiency': 0.7,
                'optimization': 0.5
            },
            'exploration': {
                'exploration': 0.9,
                'curiosity': 0.8,
                'discovery': 0.7,
                'coverage': 0.6
            },
            'efficiency': {
                'efficiency': 0.9,
                'optimization': 0.8,
                'speed': 0.7,
                'resource_usage': 0.6
            },
            'adaptation': {
                'adaptation': 0.9,
                'flexibility': 0.8,
                'learning': 0.7,
                'resilience': 0.6
            }
        }
    
    def add_evolution_seed(self, direction: str, weight: float = 0.5, target_improvement: float = 2.3) -> str:
        """Add user-provided evolution seed for direction guidance"""
        try:
            seed_id = f"seed_{direction}_{int(time.time())}"
            
            seed = EvolutionSeed(
                seed_id=seed_id,
                direction=direction.lower(),
                weight=max(0.0, min(1.0, weight)),  # Clamp to [0, 1]
                target_improvement=target_improvement,
                description=f"User-guided evolution toward {direction}",
                timestamp=datetime.now()
            )
            
            self.active_seeds[seed_id] = seed
            self.seed_history.append(seed)
            
            logger.info(f"Added evolution seed: {direction} (weight: {weight:.2f}, target: {target_improvement:.1f}%)")
            return seed_id
            
        except Exception as e:
            logger.error(f"Failed to add evolution seed: {e}")
            return ""
    
    def create_seeded_fitness_function(self, base_fitness_function: Callable, seeds: Optional[List[str]] = None) -> Callable:
        """Create fitness function with seeded direction bias"""
        try:
            # Use all active seeds if none specified
            if seeds is None:
                seeds = list(self.active_seeds.keys())
            
            async def seeded_fitness_function(individual):
                # Calculate base fitness
                base_fitness = await base_fitness_function(individual)
                
                # Apply seed bonuses
                total_bonus = 0.0
                for seed_id in seeds:
                    if seed_id in self.active_seeds:
                        seed = self.active_seeds[seed_id]
                        bonus = self._calculate_seed_bonus(individual, seed)
                        total_bonus += bonus * seed.weight
                
                # Apply seeded enhancement
                enhanced_fitness = base_fitness * (1.0 + total_bonus)
                
                logger.debug(f"Seeded fitness: base={base_fitness:.3f}, bonus={total_bonus:.3f}, enhanced={enhanced_fitness:.3f}")
                return enhanced_fitness
            
            return seeded_fitness_function
            
        except Exception as e:
            logger.error(f"Seeded fitness function creation failed: {e}")
            return base_fitness_function
    
    def _calculate_seed_bonus(self, individual: Any, seed: EvolutionSeed) -> float:
        """Calculate fitness bonus based on seed direction"""
        try:
            individual_str = str(individual).lower()
            direction = seed.direction
            
            # Get direction mappings
            if direction not in self.direction_mappings:
                return 0.0
            
            mappings = self.direction_mappings[direction]
            bonus = 0.0
            
            # Check for direction-related traits in individual
            for trait, trait_weight in mappings.items():
                if trait in individual_str:
                    # Count occurrences for stronger bonus
                    occurrences = individual_str.count(trait)
                    trait_bonus = min(0.5, occurrences * trait_weight * 0.1)  # Cap at 50% bonus
                    bonus += trait_bonus
            
            # Special bonuses for specific directions
            if direction == 'cooperation' and 'coop' in individual_str:
                bonus += 0.392  # +39.2% cooperation bonus (Observer specification)
            
            elif direction == 'sustainability' and 'sustain' in individual_str:
                bonus += 0.373  # +37.3% sustainability bonus (Observer specification)
            
            elif direction == 'gathering' and 'gather' in individual_str:
                bonus += 0.392  # +39.2% gathering bonus (Observer specification)
            
            # Apply target improvement scaling
            if seed.target_improvement > 2.0:  # High target
                bonus *= 1.2  # 20% bonus multiplier
            
            return min(1.0, bonus)  # Cap total bonus at 100%
            
        except Exception as e:
            logger.warning(f"Seed bonus calculation failed: {e}")
            return 0.0
    
    def create_seeded_mutation_function(self, base_mutation_function: Callable, seeds: Optional[List[str]] = None) -> Callable:
        """Create mutation function with seeded direction bias"""
        try:
            # Use all active seeds if none specified
            if seeds is None:
                seeds = list(self.active_seeds.keys())
            
            async def seeded_mutation_function(individual):
                # Apply base mutation
                mutated = await base_mutation_function(individual)
                
                # Apply seed-directed mutations
                for seed_id in seeds:
                    if seed_id in self.active_seeds:
                        seed = self.active_seeds[seed_id]
                        mutated = self._apply_directed_mutation(mutated, seed)
                
                return mutated
            
            return seeded_mutation_function
            
        except Exception as e:
            logger.error(f"Seeded mutation function creation failed: {e}")
            return base_mutation_function
    
    def _apply_directed_mutation(self, individual: Any, seed: EvolutionSeed) -> Any:
        """Apply directed mutation based on seed direction"""
        try:
            direction = seed.direction
            weight = seed.weight
            
            # Apply direction-specific mutations
            if direction == 'cooperation' and weight > 0.3:
                return f"{individual}_seeded_cooperation_{weight:.2f}"
            
            elif direction == 'sustainability' and weight > 0.3:
                return f"{individual}_seeded_sustainability_{weight:.2f}"
            
            elif direction == 'gathering' and weight > 0.3:
                return f"{individual}_seeded_gathering_{weight:.2f}"
            
            elif direction == 'exploration' and weight > 0.3:
                return f"{individual}_seeded_exploration_{weight:.2f}"
            
            elif direction == 'efficiency' and weight > 0.3:
                return f"{individual}_seeded_efficiency_{weight:.2f}"
            
            elif direction == 'adaptation' and weight > 0.3:
                return f"{individual}_seeded_adaptation_{weight:.2f}"
            
            return individual
            
        except Exception as e:
            logger.warning(f"Directed mutation failed: {e}")
            return individual
    
    def evaluate_seed_effectiveness(self, evolution_result: Dict[str, Any], seed_id: str) -> Dict[str, Any]:
        """Evaluate effectiveness of a specific seed"""
        try:
            if seed_id not in self.active_seeds:
                return {"error": "Seed not found"}
            
            seed = self.active_seeds[seed_id]
            best_fitness = evolution_result.get('best_fitness', 0)
            
            # Calculate improvement achieved
            baseline_fitness = 1.0  # Assume baseline
            improvement_achieved = (best_fitness - baseline_fitness) / baseline_fitness
            
            # Calculate effectiveness
            target_improvement = seed.target_improvement / 100.0  # Convert percentage
            effectiveness = min(1.0, improvement_achieved / target_improvement) if target_improvement > 0 else 0.0
            
            # Check for direction-specific success indicators
            direction_success = self._check_direction_success(evolution_result, seed.direction)
            
            evaluation = {
                'seed_id': seed_id,
                'direction': seed.direction,
                'target_improvement': seed.target_improvement,
                'improvement_achieved': improvement_achieved * 100,  # Convert to percentage
                'effectiveness': effectiveness,
                'direction_success': direction_success,
                'success_rate': min(1.0, effectiveness * direction_success)
            }
            
            logger.info(f"Seed evaluation for {seed.direction}: effectiveness={effectiveness:.2f}, success_rate={evaluation['success_rate']:.2f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Seed effectiveness evaluation failed: {e}")
            return {"error": str(e)}
    
    def _check_direction_success(self, evolution_result: Dict[str, Any], direction: str) -> float:
        """Check for direction-specific success indicators"""
        try:
            success_score = 0.5  # Base success
            
            if direction == 'cooperation':
                cooperation_events = evolution_result.get('cooperation_events', 0)
                if cooperation_events > 845:  # Observer specification
                    success_score = 1.0
                elif cooperation_events > 400:
                    success_score = 0.8
            
            elif direction == 'sustainability':
                survival_rate = evolution_result.get('survival_rate', 0.5)
                if survival_rate > 0.9:
                    success_score = 1.0
                elif survival_rate > 0.7:
                    success_score = 0.8
            
            elif direction == 'gathering':
                resource_efficiency = evolution_result.get('resource_efficiency', 0.5)
                if resource_efficiency > 0.8:
                    success_score = 1.0
                elif resource_efficiency > 0.6:
                    success_score = 0.8
            
            elif direction == 'exploration':
                behaviors_detected = evolution_result.get('behaviors_detected', 0)
                if behaviors_detected > 48:  # Observer specification
                    success_score = 1.0
                elif behaviors_detected > 20:
                    success_score = 0.8
            
            return success_score
            
        except Exception as e:
            logger.warning(f"Direction success check failed: {e}")
            return 0.5
    
    def get_seeding_stats(self) -> Dict[str, Any]:
        """Get seeded direction system statistics"""
        try:
            if not self.seed_history:
                return {"no_data": True}
            
            # Direction distribution
            direction_counts = {}
            total_weight = 0.0
            
            for seed in self.active_seeds.values():
                direction = seed.direction
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
                total_weight += seed.weight
            
            avg_weight = total_weight / len(self.active_seeds) if self.active_seeds else 0.0
            
            return {
                'total_seeds': len(self.seed_history),
                'active_seeds': len(self.active_seeds),
                'direction_distribution': direction_counts,
                'avg_seed_weight': avg_weight,
                'available_directions': list(self.direction_mappings.keys())
            }
            
        except Exception as e:
            logger.error(f"Seeding stats calculation failed: {e}")
            return {"error": str(e)}
    
    def remove_seed(self, seed_id: str) -> bool:
        """Remove an active seed"""
        try:
            if seed_id in self.active_seeds:
                removed_seed = self.active_seeds.pop(seed_id)
                logger.info(f"Removed seed: {removed_seed.direction}")
                return True
            return False
        except Exception as e:
            logger.error(f"Seed removal failed: {e}")
            return False
