#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Manager - Phase 2.3
Observer-approved dataset management for LoRA fine-tuning with RAG integration

Manages CodeAlpaca dataset loading and RAG output integration for hybrid training.
Implements efficient dataset preparation with quality filtering and augmentation.
"""

import asyncio
import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Core imports
try:
    from datasets import load_dataset, Dataset
    import requests
    DATASETS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dataset dependencies not available: {e}")
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    max_examples: int = 1000
    min_instruction_length: int = 10
    max_instruction_length: int = 500
    min_response_length: int = 20
    max_response_length: int = 2000
    quality_threshold: float = 0.7
    rag_integration_ratio: float = 0.3  # 30% RAG data, 70% CodeAlpaca
    seed: int = 42


class DatasetManager:
    """
    Dataset Manager for LoRA Fine-tuning
    
    Handles CodeAlpaca dataset loading, RAG output integration,
    and quality filtering for optimal training data preparation.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.codealpaca_data = []
        self.rag_data = []
        self.combined_dataset = []
        
        # Dataset statistics
        self.stats = {
            "codealpaca_loaded": 0,
            "codealpaca_filtered": 0,
            "rag_examples": 0,
            "combined_examples": 0,
            "quality_score": 0.0
        }
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        
        logger.info("Dataset Manager initialized")
    
    async def load_codealpaca_dataset(self) -> bool:
        """Load CodeAlpaca dataset for coding instruction tuning"""
        try:
            if not DATASETS_AVAILABLE:
                logger.warning("Datasets library not available, using mock data")
                return await self._create_mock_codealpaca_data()
            
            logger.info("Loading CodeAlpaca dataset...")
            
            # Load CodeAlpaca dataset
            try:
                dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
                raw_data = dataset.to_list()
                logger.info(f"Loaded {len(raw_data)} examples from CodeAlpaca")
            except Exception as e:
                logger.warning(f"Failed to load CodeAlpaca from HuggingFace: {e}")
                return await self._create_mock_codealpaca_data()
            
            # Filter and process data
            filtered_data = []
            for item in raw_data:
                if self._is_valid_example(item):
                    processed_item = self._process_codealpaca_example(item)
                    if processed_item:
                        filtered_data.append(processed_item)
            
            # Limit to max examples
            if len(filtered_data) > self.config.max_examples:
                filtered_data = random.sample(filtered_data, self.config.max_examples)
            
            self.codealpaca_data = filtered_data
            self.stats["codealpaca_loaded"] = len(raw_data)
            self.stats["codealpaca_filtered"] = len(filtered_data)
            
            logger.info(f"CodeAlpaca dataset processed: {len(filtered_data)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CodeAlpaca dataset: {e}")
            return await self._create_mock_codealpaca_data()
    
    async def _create_mock_codealpaca_data(self) -> bool:
        """Create mock CodeAlpaca data for testing"""
        try:
            logger.info("Creating mock CodeAlpaca dataset...")
            
            mock_examples = [
                {
                    "instruction": "Write a Python function to reverse a string",
                    "input": "",
                    "output": "def reverse_string(s):\n    return s[::-1]\n\n# Example usage\ntext = 'hello'\nreversed_text = reverse_string(text)\nprint(reversed_text)  # Output: olleh"
                },
                {
                    "instruction": "Create a function to calculate the factorial of a number",
                    "input": "n = 5",
                    "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial(5)\nprint(result)  # Output: 120"
                },
                {
                    "instruction": "Write a Python function to check if a number is prime",
                    "input": "",
                    "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\n# Example usage\nprint(is_prime(17))  # Output: True\nprint(is_prime(4))   # Output: False"
                },
                {
                    "instruction": "Create a function to find the maximum element in a list",
                    "input": "numbers = [3, 7, 2, 9, 1]",
                    "output": "def find_max(numbers):\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n\nresult = find_max([3, 7, 2, 9, 1])\nprint(result)  # Output: 9"
                },
                {
                    "instruction": "Write a function to implement binary search",
                    "input": "arr = [1, 3, 5, 7, 9, 11], target = 7",
                    "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1\n\nresult = binary_search([1, 3, 5, 7, 9, 11], 7)\nprint(result)  # Output: 3"
                }
            ]
            
            # Expand mock data to reach desired size
            expanded_data = []
            for i in range(min(self.config.max_examples, 50)):  # Limit mock data
                base_example = mock_examples[i % len(mock_examples)]
                expanded_data.append(base_example.copy())
            
            self.codealpaca_data = expanded_data
            self.stats["codealpaca_loaded"] = len(expanded_data)
            self.stats["codealpaca_filtered"] = len(expanded_data)
            
            logger.info(f"Mock CodeAlpaca dataset created: {len(expanded_data)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create mock CodeAlpaca data: {e}")
            return False
    
    def _is_valid_example(self, item: Dict[str, Any]) -> bool:
        """Check if an example meets quality criteria"""
        try:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            
            # Length checks
            if len(instruction) < self.config.min_instruction_length:
                return False
            if len(instruction) > self.config.max_instruction_length:
                return False
            if len(output) < self.config.min_response_length:
                return False
            if len(output) > self.config.max_response_length:
                return False
            
            # Content quality checks
            if not instruction.strip() or not output.strip():
                return False
            
            # Code-related keywords (for CodeAlpaca)
            code_keywords = ["function", "def", "class", "import", "return", "print", "if", "for", "while"]
            if not any(keyword in output.lower() for keyword in code_keywords):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Example validation failed: {e}")
            return False
    
    def _process_codealpaca_example(self, item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Process a CodeAlpaca example into training format"""
        try:
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            
            # Combine instruction and input if present
            if input_text:
                full_instruction = f"{instruction}\n\nInput: {input_text}"
            else:
                full_instruction = instruction
            
            return {
                "instruction": full_instruction,
                "response": output
            }
            
        except Exception as e:
            logger.debug(f"Example processing failed: {e}")
            return None
    
    def integrate_rag_outputs(self, rag_outputs: List[Dict[str, Any]]) -> bool:
        """Integrate RAG outputs into training dataset"""
        try:
            logger.info(f"Integrating {len(rag_outputs)} RAG outputs...")
            
            processed_rag_data = []
            for rag_output in rag_outputs:
                processed_item = self._process_rag_output(rag_output)
                if processed_item and self._is_valid_example(processed_item):
                    processed_rag_data.append(processed_item)
            
            self.rag_data = processed_rag_data
            self.stats["rag_examples"] = len(processed_rag_data)
            
            logger.info(f"RAG outputs integrated: {len(processed_rag_data)} examples")
            return True
            
        except Exception as e:
            logger.error(f"RAG integration failed: {e}")
            return False
    
    def _process_rag_output(self, rag_output: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Process a RAG output into training format"""
        try:
            query = rag_output.get("query", "")
            results = rag_output.get("results", "")
            methodology = rag_output.get("methodology", "")
            
            if not query or not results:
                return None
            
            # Create instruction from query
            instruction = f"Provide a comprehensive research response for: {query}"
            
            # Create response from results
            response = f"{results}"
            
            # Add methodology if available
            if methodology and "rag" in methodology.lower():
                response += f"\n\nMethodology: {methodology}"
            
            return {
                "instruction": instruction,
                "output": response  # Use 'output' for consistency with validation
            }
            
        except Exception as e:
            logger.debug(f"RAG output processing failed: {e}")
            return None
    
    def create_combined_dataset(self) -> List[Dict[str, str]]:
        """Create combined dataset from CodeAlpaca and RAG data"""
        try:
            logger.info("Creating combined dataset...")
            
            # Calculate target sizes
            total_target = min(self.config.max_examples, 
                             len(self.codealpaca_data) + len(self.rag_data))
            
            rag_target = int(total_target * self.config.rag_integration_ratio)
            codealpaca_target = total_target - rag_target
            
            # Sample data
            selected_rag = random.sample(self.rag_data, 
                                       min(rag_target, len(self.rag_data)))
            selected_codealpaca = random.sample(self.codealpaca_data, 
                                              min(codealpaca_target, len(self.codealpaca_data)))
            
            # Combine and shuffle
            combined = selected_rag + selected_codealpaca
            random.shuffle(combined)
            
            # Convert RAG data format to match CodeAlpaca
            formatted_combined = []
            for item in combined:
                if "response" in item:
                    # Already in correct format
                    formatted_combined.append(item)
                else:
                    # Convert from RAG format
                    formatted_item = {
                        "instruction": item.get("instruction", ""),
                        "response": item.get("output", "")
                    }
                    formatted_combined.append(formatted_item)
            
            self.combined_dataset = formatted_combined
            self.stats["combined_examples"] = len(formatted_combined)
            self.stats["quality_score"] = self._calculate_quality_score(formatted_combined)
            
            logger.info(f"Combined dataset created: {len(formatted_combined)} examples")
            logger.info(f"RAG examples: {len(selected_rag)}, CodeAlpaca examples: {len(selected_codealpaca)}")
            
            return formatted_combined
            
        except Exception as e:
            logger.error(f"Combined dataset creation failed: {e}")
            return []
    
    def _calculate_quality_score(self, dataset: List[Dict[str, str]]) -> float:
        """Calculate overall quality score for dataset"""
        try:
            if not dataset:
                return 0.0
            
            total_score = 0.0
            for item in dataset:
                instruction = item.get("instruction", "")
                response = item.get("response", "")
                
                # Length score (normalized)
                length_score = min(1.0, (len(instruction) + len(response)) / 1000)
                
                # Content diversity score (simplified)
                diversity_score = len(set(instruction.lower().split())) / max(1, len(instruction.split()))
                
                # Code quality score (for coding examples)
                code_score = 1.0 if any(keyword in response.lower() 
                                      for keyword in ["def", "class", "import", "return"]) else 0.5
                
                item_score = (length_score + diversity_score + code_score) / 3
                total_score += item_score
            
            return total_score / len(dataset)
            
        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            **self.stats,
            "config": self.config.__dict__,
            "datasets_available": DATASETS_AVAILABLE
        }
    
    async def prepare_training_dataset(self, rag_outputs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        """Complete dataset preparation pipeline"""
        try:
            logger.info("Starting dataset preparation pipeline...")
            
            # Step 1: Load CodeAlpaca dataset
            if not await self.load_codealpaca_dataset():
                logger.error("Failed to load CodeAlpaca dataset")
                return []
            
            # Step 2: Integrate RAG outputs if provided
            if rag_outputs:
                if not self.integrate_rag_outputs(rag_outputs):
                    logger.warning("RAG integration failed, continuing with CodeAlpaca only")
            
            # Step 3: Create combined dataset
            combined_dataset = self.create_combined_dataset()
            
            if not combined_dataset:
                logger.error("Failed to create combined dataset")
                return []
            
            logger.info(f"Dataset preparation complete: {len(combined_dataset)} examples ready for training")
            return combined_dataset
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return []


# Convenience functions
async def create_dataset_manager(config: Optional[DatasetConfig] = None) -> DatasetManager:
    """Create dataset manager"""
    return DatasetManager(config)


async def prepare_rag_enhanced_dataset(rag_outputs: List[Dict[str, Any]], 
                                     max_examples: int = 1000) -> List[Dict[str, str]]:
    """Convenience function to prepare RAG-enhanced dataset"""
    config = DatasetConfig(max_examples=max_examples)
    manager = DatasetManager(config)
    return await manager.prepare_training_dataset(rag_outputs)
