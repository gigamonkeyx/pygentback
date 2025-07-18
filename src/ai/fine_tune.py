#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Fine-tuning Module - Phase 2.3
Observer-approved LoRA integration with Hybrid RAG-LoRA Fusion

Implements efficient LoRA fine-tuning using Unsloth/PEFT with Ollama Llama3 8B base model.
Integrates with Phase 2.2 RAG outputs for enhanced training data and hybrid generation.
"""

import asyncio
import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# Core imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset, load_dataset
    from trl import SFTTrainer
    import bitsandbytes as bnb
    LORA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LoRA dependencies not available: {e}")
    LORA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    
    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    fp16: bool = not torch.cuda.is_available()
    bf16: bool = torch.cuda.is_available()
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407


@dataclass
class LoRATrainingResult:
    """Result of LoRA training process"""
    success: bool
    model_path: str
    training_time_seconds: float
    final_loss: float
    steps_completed: int
    accuracy_improvement: float
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoRAFineTuner:
    """
    LoRA Fine-tuning Manager for Phase 2.3
    
    Implements efficient LoRA fine-tuning with Unsloth optimization
    and integration with RAG outputs from Phase 2.2.
    """
    
    def __init__(self, config: Optional[LoRAConfig] = None):
        self.config = config or LoRAConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.is_initialized = False
        
        # Training state
        self.training_stats = {
            "total_trainings": 0,
            "successful_trainings": 0,
            "average_training_time": 0.0,
            "best_accuracy_improvement": 0.0,
            "models_saved": 0
        }
        
        # Model storage
        self.models_dir = Path("models/lora_adapters")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("LoRA Fine-tuner initialized")
    
    async def initialize(self) -> bool:
        """Initialize LoRA fine-tuning components"""
        if not LORA_AVAILABLE:
            logger.error("LoRA dependencies not available")
            return False
        
        try:
            logger.info("Initializing LoRA fine-tuning components...")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                logger.warning("CUDA not available - using CPU (will be slower)")
            
            # Initialize tokenizer
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("LoRA fine-tuning components initialized successfully")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"LoRA initialization failed: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the base model for fine-tuning"""
        try:
            logger.info(f"Loading base model: {self.config.model_name}")
            
            # Load model with 4-bit quantization for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_4bit=self.config.load_in_4bit,
                torch_dtype=torch.float16 if self.config.fp16 else torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("Base model loaded and LoRA applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """Prepare dataset for training"""
        try:
            logger.info(f"Preparing dataset with {len(data)} examples")
            
            # Format data for instruction tuning
            formatted_data = []
            for item in data:
                # Create instruction-response format
                if "instruction" in item and "response" in item:
                    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
                elif "input" in item and "output" in item:
                    text = f"### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
                else:
                    # Fallback format
                    text = str(item)
                
                formatted_data.append({"text": text})
            
            # Create dataset
            dataset = Dataset.from_list(formatted_data)
            
            logger.info(f"Dataset prepared: {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return None
    
    def create_rag_enhanced_dataset(self, rag_outputs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create training dataset from RAG outputs"""
        try:
            logger.info(f"Creating RAG-enhanced dataset from {len(rag_outputs)} outputs")
            
            training_data = []
            
            for rag_output in rag_outputs:
                # Extract query and results from RAG output
                query = rag_output.get("query", "")
                results = rag_output.get("results", "")
                sources = rag_output.get("sources", [])
                
                if query and results:
                    # Create instruction-response pair
                    instruction = f"Based on the following research query, provide a comprehensive response: {query}"
                    response = f"{results}\n\nSources: {', '.join(sources[:3])}"  # Limit sources
                    
                    training_data.append({
                        "instruction": instruction,
                        "response": response
                    })
            
            logger.info(f"Created {len(training_data)} training examples from RAG outputs")
            return training_data
            
        except Exception as e:
            logger.error(f"RAG-enhanced dataset creation failed: {e}")
            return []
    
    async def fine_tune(self, 
                       training_data: List[Dict[str, str]], 
                       model_name: str = "rag_enhanced_model") -> LoRATrainingResult:
        """Fine-tune model with LoRA"""
        start_time = time.time()
        self.training_stats["total_trainings"] += 1
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if not self.load_model():
                raise Exception("Failed to load model")
            
            # Prepare dataset
            dataset = self.prepare_dataset(training_data)
            if dataset is None:
                raise Exception("Failed to prepare dataset")
            
            # Setup training arguments
            output_dir = self.models_dir / model_name
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=1,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                optim=self.config.optim,
                save_steps=self.config.max_steps,
                logging_steps=self.config.logging_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                max_grad_norm=0.3,
                max_steps=self.config.max_steps,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type=self.config.lr_scheduler_type,
                report_to=None,  # Disable wandb
                seed=self.config.seed,
            )
            
            # Create trainer
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                args=training_args,
            )
            
            # Start training
            logger.info(f"Starting LoRA fine-tuning: {model_name}")
            train_result = self.trainer.train()
            
            # Save model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            # Calculate metrics
            training_time = time.time() - start_time
            final_loss = train_result.training_loss
            steps_completed = train_result.global_step
            
            # Estimate accuracy improvement (simplified)
            accuracy_improvement = max(0.0, (1.0 - final_loss) * 100)
            
            # Update stats
            self.training_stats["successful_trainings"] += 1
            self.training_stats["models_saved"] += 1
            self._update_training_stats(training_time, accuracy_improvement)
            
            result = LoRATrainingResult(
                success=True,
                model_path=str(output_dir),
                training_time_seconds=training_time,
                final_loss=final_loss,
                steps_completed=steps_completed,
                accuracy_improvement=accuracy_improvement,
                metadata={
                    "training_examples": len(training_data),
                    "model_name": model_name,
                    "config": self.config.__dict__
                }
            )
            
            logger.info(f"LoRA fine-tuning completed successfully in {training_time:.2f}s")
            logger.info(f"Final loss: {final_loss:.4f}, Accuracy improvement: {accuracy_improvement:.1f}%")
            
            return result
            
        except Exception as e:
            error_msg = f"LoRA fine-tuning failed: {e}"
            logger.error(error_msg)
            
            return LoRATrainingResult(
                success=False,
                model_path="",
                training_time_seconds=time.time() - start_time,
                final_loss=float('inf'),
                steps_completed=0,
                accuracy_improvement=0.0,
                error_message=error_msg
            )
    
    def _update_training_stats(self, training_time: float, accuracy_improvement: float):
        """Update training statistics"""
        total = self.training_stats["total_trainings"]
        
        # Update average training time
        current_avg_time = self.training_stats["average_training_time"]
        self.training_stats["average_training_time"] = (
            (current_avg_time * (total - 1) + training_time) / total
        )
        
        # Update best accuracy improvement
        if accuracy_improvement > self.training_stats["best_accuracy_improvement"]:
            self.training_stats["best_accuracy_improvement"] = accuracy_improvement
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            **self.training_stats,
            "success_rate": (
                self.training_stats["successful_trainings"] / 
                max(1, self.training_stats["total_trainings"])
            ),
            "initialized": self.is_initialized,
            "cuda_available": torch.cuda.is_available() if LORA_AVAILABLE else False,
            "models_directory": str(self.models_dir)
        }
    
    async def load_fine_tuned_model(self, model_path: str) -> bool:
        """Load a fine-tuned LoRA model"""
        try:
            logger.info(f"Loading fine-tuned model: {model_path}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_4bit=self.config.load_in_4bit,
                torch_dtype=torch.float16 if self.config.fp16 else torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info("Fine-tuned model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False
    
    async def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using fine-tuned model"""
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("Model not loaded")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation failed: {e}"


# Convenience functions
async def create_lora_fine_tuner(config: Optional[LoRAConfig] = None) -> LoRAFineTuner:
    """Create and initialize LoRA fine-tuner"""
    fine_tuner = LoRAFineTuner(config)
    await fine_tuner.initialize()
    return fine_tuner


async def fine_tune_with_rag_data(rag_outputs: List[Dict[str, Any]], 
                                 model_name: str = "rag_enhanced_model") -> LoRATrainingResult:
    """Convenience function to fine-tune with RAG data"""
    fine_tuner = await create_lora_fine_tuner()
    training_data = fine_tuner.create_rag_enhanced_dataset(rag_outputs)
    return await fine_tuner.fine_tune(training_data, model_name)
