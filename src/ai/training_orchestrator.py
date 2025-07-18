#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Orchestrator - Phase 2.3
Observer-approved training orchestration for LoRA fine-tuning with RAG integration

Orchestrates the complete training pipeline: dataset preparation, LoRA fine-tuning,
and integration with Phase 2.2 RAG outputs for hybrid model enhancement.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from .fine_tune import LoRAFineTuner, LoRAConfig, LoRATrainingResult
from .dataset_manager import DatasetManager, DatasetConfig

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Training Orchestrator for Phase 2.3
    
    Coordinates dataset preparation, LoRA fine-tuning, and RAG integration
    to create hybrid RAG-LoRA enhanced models for doer agents.
    """
    
    def __init__(self, 
                 lora_config: Optional[LoRAConfig] = None,
                 dataset_config: Optional[DatasetConfig] = None):
        self.lora_config = lora_config or LoRAConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        
        self.fine_tuner = LoRAFineTuner(self.lora_config)
        self.dataset_manager = DatasetManager(self.dataset_config)
        
        # Training state
        self.training_history = []
        self.current_model_path = None
        
        # Performance tracking
        self.orchestration_stats = {
            "total_training_sessions": 0,
            "successful_sessions": 0,
            "average_session_time": 0.0,
            "best_accuracy_improvement": 0.0,
            "models_created": 0
        }
        
        logger.info("Training Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize training orchestrator components"""
        try:
            logger.info("Initializing training orchestrator...")
            
            # Initialize fine-tuner
            if not await self.fine_tuner.initialize():
                logger.error("Failed to initialize LoRA fine-tuner")
                return False
            
            logger.info("Training orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training orchestrator initialization failed: {e}")
            return False
    
    async def create_mock_rag_outputs(self) -> List[Dict[str, Any]]:
        """Create mock RAG outputs for testing (Phase 2.2 integration simulation)"""
        try:
            logger.info("Creating mock RAG outputs for training...")
            
            mock_rag_outputs = [
                {
                    "query": "How to implement a binary search algorithm in Python?",
                    "results": "Binary search is an efficient algorithm for finding an item from a sorted list. It works by repeatedly dividing the search interval in half. The algorithm compares the target value to the middle element of the array, and based on the comparison, eliminates half of the elements from consideration.",
                    "methodology": "RAG-enhanced research with academic sources",
                    "sources": ["Algorithm Design Manual", "Introduction to Algorithms"],
                    "confidence": 0.9
                },
                {
                    "query": "Explain object-oriented programming principles",
                    "results": "Object-oriented programming (OOP) is based on four main principles: Encapsulation (bundling data and methods), Inheritance (creating new classes based on existing ones), Polymorphism (objects of different types responding to the same interface), and Abstraction (hiding complex implementation details).",
                    "methodology": "RAG-enhanced multi-source synthesis",
                    "sources": ["Design Patterns", "Clean Code", "Effective Java"],
                    "confidence": 0.85
                },
                {
                    "query": "Best practices for error handling in Python",
                    "results": "Python error handling best practices include: Use specific exception types, implement proper try-except blocks, use finally for cleanup, avoid bare except clauses, log errors appropriately, and use custom exceptions for application-specific errors. The EAFP (Easier to Ask for Forgiveness than Permission) principle is preferred in Python.",
                    "methodology": "RAG-enhanced research with code examples",
                    "sources": ["Python Documentation", "Effective Python", "Python Tricks"],
                    "confidence": 0.88
                },
                {
                    "query": "How to optimize database queries for performance?",
                    "results": "Database query optimization involves several strategies: Use proper indexing, avoid SELECT *, use LIMIT for large datasets, optimize JOIN operations, use query execution plans, normalize database design appropriately, and consider caching strategies. Profiling and monitoring are essential for identifying bottlenecks.",
                    "methodology": "RAG-enhanced database research",
                    "sources": ["High Performance MySQL", "Database System Concepts"],
                    "confidence": 0.82
                },
                {
                    "query": "Machine learning model evaluation techniques",
                    "results": "Model evaluation techniques include: Cross-validation for robust performance estimation, confusion matrices for classification analysis, ROC curves and AUC for binary classification, precision-recall curves for imbalanced datasets, and various metrics like accuracy, precision, recall, and F1-score. Feature importance analysis helps understand model behavior.",
                    "methodology": "RAG-enhanced ML research synthesis",
                    "sources": ["Pattern Recognition and Machine Learning", "The Elements of Statistical Learning"],
                    "confidence": 0.91
                }
            ]
            
            logger.info(f"Created {len(mock_rag_outputs)} mock RAG outputs")
            return mock_rag_outputs
            
        except Exception as e:
            logger.error(f"Failed to create mock RAG outputs: {e}")
            return []
    
    async def run_training_session(self, 
                                 rag_outputs: Optional[List[Dict[str, Any]]] = None,
                                 model_name: str = "hybrid_rag_lora_model") -> Dict[str, Any]:
        """Run complete training session with dataset preparation and LoRA fine-tuning"""
        session_start_time = time.time()
        self.orchestration_stats["total_training_sessions"] += 1
        
        session_result = {
            "success": False,
            "model_path": "",
            "training_result": None,
            "dataset_stats": {},
            "session_time": 0.0,
            "error_message": ""
        }
        
        try:
            logger.info(f"Starting training session: {model_name}")
            
            # Step 1: Prepare dataset
            logger.info("Step 1: Preparing training dataset...")
            if rag_outputs is None:
                rag_outputs = await self.create_mock_rag_outputs()
            
            training_dataset = await self.dataset_manager.prepare_training_dataset(rag_outputs)
            
            if not training_dataset:
                raise Exception("Failed to prepare training dataset")
            
            dataset_stats = self.dataset_manager.get_stats()
            session_result["dataset_stats"] = dataset_stats
            
            logger.info(f"Dataset prepared: {len(training_dataset)} examples")
            
            # Step 2: Run LoRA fine-tuning
            logger.info("Step 2: Running LoRA fine-tuning...")
            training_result = await self.fine_tuner.fine_tune(training_dataset, model_name)
            
            if not training_result.success:
                raise Exception(f"LoRA fine-tuning failed: {training_result.error_message}")
            
            session_result["training_result"] = training_result
            session_result["model_path"] = training_result.model_path
            self.current_model_path = training_result.model_path
            
            # Step 3: Update statistics
            session_time = time.time() - session_start_time
            session_result["session_time"] = session_time
            session_result["success"] = True
            
            self._update_orchestration_stats(session_time, training_result.accuracy_improvement)
            self.orchestration_stats["successful_sessions"] += 1
            self.orchestration_stats["models_created"] += 1
            
            # Add to training history
            self.training_history.append({
                "timestamp": time.time(),
                "model_name": model_name,
                "session_result": session_result,
                "dataset_size": len(training_dataset),
                "accuracy_improvement": training_result.accuracy_improvement
            })
            
            logger.info(f"Training session completed successfully in {session_time:.2f}s")
            logger.info(f"Model saved to: {training_result.model_path}")
            logger.info(f"Accuracy improvement: {training_result.accuracy_improvement:.1f}%")
            
            return session_result
            
        except Exception as e:
            error_msg = f"Training session failed: {e}"
            logger.error(error_msg)
            
            session_result["error_message"] = error_msg
            session_result["session_time"] = time.time() - session_start_time
            
            return session_result
    
    def _update_orchestration_stats(self, session_time: float, accuracy_improvement: float):
        """Update orchestration statistics"""
        total = self.orchestration_stats["total_training_sessions"]
        
        # Update average session time
        current_avg_time = self.orchestration_stats["average_session_time"]
        self.orchestration_stats["average_session_time"] = (
            (current_avg_time * (total - 1) + session_time) / total
        )
        
        # Update best accuracy improvement
        if accuracy_improvement > self.orchestration_stats["best_accuracy_improvement"]:
            self.orchestration_stats["best_accuracy_improvement"] = accuracy_improvement
    
    async def test_model_generation(self, prompt: str) -> Dict[str, Any]:
        """Test generation with the current fine-tuned model"""
        try:
            if not self.current_model_path:
                return {
                    "success": False,
                    "error": "No trained model available"
                }
            
            logger.info("Testing model generation...")
            
            # Load the fine-tuned model
            if not await self.fine_tuner.load_fine_tuned_model(self.current_model_path):
                return {
                    "success": False,
                    "error": "Failed to load fine-tuned model"
                }
            
            # Generate response
            generated_text = await self.fine_tuner.generate(prompt)
            
            return {
                "success": True,
                "prompt": prompt,
                "generated_text": generated_text,
                "model_path": self.current_model_path
            }
            
        except Exception as e:
            logger.error(f"Model generation test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        return {
            **self.orchestration_stats,
            "success_rate": (
                self.orchestration_stats["successful_sessions"] / 
                max(1, self.orchestration_stats["total_training_sessions"])
            ),
            "current_model_path": self.current_model_path,
            "training_history_count": len(self.training_history),
            "fine_tuner_stats": self.fine_tuner.get_stats(),
            "dataset_manager_stats": self.dataset_manager.get_stats()
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history.copy()
    
    async def run_quick_training_demo(self) -> Dict[str, Any]:
        """Run a quick training demonstration for Phase 2.3 validation"""
        try:
            logger.info("Running quick training demonstration...")
            
            # Initialize if not already done
            if not await self.initialize():
                raise Exception("Failed to initialize training orchestrator")
            
            # Create optimized config for quick demo
            quick_config = LoRAConfig(
                max_steps=10,  # Very quick training
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=5e-4  # Higher learning rate for quick demo
            )
            
            # Update fine-tuner config
            self.fine_tuner.config = quick_config
            
            # Run training session
            session_result = await self.run_training_session(
                model_name="quick_demo_model"
            )
            
            if session_result["success"]:
                # Test generation
                test_result = await self.test_model_generation(
                    "Write a Python function to calculate fibonacci numbers:"
                )
                session_result["generation_test"] = test_result
            
            return session_result
            
        except Exception as e:
            logger.error(f"Quick training demo failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Convenience functions
async def create_training_orchestrator(lora_config: Optional[LoRAConfig] = None,
                                     dataset_config: Optional[DatasetConfig] = None) -> TrainingOrchestrator:
    """Create and initialize training orchestrator"""
    orchestrator = TrainingOrchestrator(lora_config, dataset_config)
    await orchestrator.initialize()
    return orchestrator


async def run_hybrid_rag_lora_training(rag_outputs: List[Dict[str, Any]], 
                                     model_name: str = "hybrid_rag_lora_model") -> Dict[str, Any]:
    """Convenience function to run hybrid RAG-LoRA training"""
    orchestrator = await create_training_orchestrator()
    return await orchestrator.run_training_session(rag_outputs, model_name)
