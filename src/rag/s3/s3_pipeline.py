"""
S3 RAG Pipeline

Main pipeline that orchestrates the complete s3 RAG process:
search agent training, document retrieval, and response generation.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .models import S3Config, SearchState, GBRReward
from .search_agent import S3SearchAgent
from .gbr_reward import GBRRewardCalculator, NaiveRAG
from .rl_trainer import S3RLTrainer

logger = logging.getLogger(__name__)


@dataclass
class S3Result:
    """Result of s3 RAG pipeline execution"""
    query: str
    response: str
    search_state: SearchState
    gbr_reward: Optional[GBRReward] = None
    
    # Performance metrics
    total_time: float = 0.0
    search_time: float = 0.0
    generation_time: float = 0.0
    
    # Quality metrics
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    coherence_score: float = 0.0
    
    # Metadata
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'query': self.query,
            'response': self.response,
            'search_state': self.search_state.to_dict(),
            'gbr_reward': self.gbr_reward.to_dict() if self.gbr_reward else None,
            'total_time': self.total_time,
            'search_time': self.search_time,
            'generation_time': self.generation_time,
            'relevance_score': self.relevance_score,
            'diversity_score': self.diversity_score,
            'coherence_score': self.coherence_score,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class S3Pipeline:
    """
    Complete s3 RAG pipeline
    
    Integrates search agent, reward calculation, and generation
    to provide a full RAG system with minimal training data requirements.
    """
    
    def __init__(self, config: S3Config, retriever, generator, 
                 baseline_methods: Optional[List] = None):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        
        # Initialize components
        self.reward_calculator = GBRRewardCalculator(config, baseline_methods)
        self.search_agent = S3SearchAgent(config, retriever, generator, self.reward_calculator)
        self.rl_trainer = S3RLTrainer(config, self.search_agent, self.reward_calculator)
        
        # Pipeline state
        self.is_trained = False
        self.training_history = []
        self.inference_history = []
        
        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.total_inference_time = 0.0
        
        logger.info("S3 RAG pipeline initialized")
    
    async def train(self, training_data: List[Dict[str, Any]], 
                   validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Train the s3 search agent
        
        Args:
            training_data: List of training examples with queries and expected outputs
            validation_data: Optional validation data for evaluation
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting s3 training with {len(training_data)} examples")
        
        start_time = time.time()
        
        try:
            # Train the RL agent
            training_results = await self.rl_trainer.train(training_data, validation_data)
            
            # Update training state
            self.is_trained = True
            self.training_history.append({
                'timestamp': time.time(),
                'training_size': len(training_data),
                'validation_size': len(validation_data) if validation_data else 0,
                'results': training_results
            })
            
            training_time = time.time() - start_time
            
            logger.info(f"S3 training completed in {training_time:.2f}s")
            
            return {
                'success': True,
                'training_time': training_time,
                'training_episodes': training_results.get('episodes', 0),
                'final_reward': training_results.get('final_reward', 0.0),
                'convergence': training_results.get('converged', False),
                'validation_score': training_results.get('validation_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"S3 training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    async def query(self, query: str, return_details: bool = False) -> S3Result:
        """
        Process a query using the s3 RAG pipeline
        
        Args:
            query: Input query
            return_details: Whether to return detailed metrics
            
        Returns:
            S3Result with response and metrics
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Create temporary SearchState for initialization
        temp_search_state = SearchState()
        result = S3Result(query=query, response="", search_state=temp_search_state)
        
        try:
            # Phase 1: Search using s3 agent
            search_start = time.time()
            search_state = await self.search_agent.search(query)
            result.search_time = time.time() - search_start
            result.search_state = search_state
            
            # Phase 2: Generate response
            generation_start = time.time()
            response = await self._generate_response(query, search_state)
            result.generation_time = time.time() - generation_start
            result.response = response
            
            # Phase 3: Calculate rewards and metrics (if requested)
            if return_details:
                await self._calculate_detailed_metrics(result)
            
            result.total_time = time.time() - start_time
            result.success = True
            self.successful_queries += 1
            
            # Store in history
            self.inference_history.append(result)
            self.total_inference_time += result.total_time
            
            logger.info(f"Query processed successfully in {result.total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.total_time = time.time() - start_time
            return result
    
    async def batch_query(self, queries: List[str], 
                         return_details: bool = False) -> List[S3Result]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of input queries
            return_details: Whether to return detailed metrics
            
        Returns:
            List of S3Results
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        # Process queries concurrently (with some limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent queries
        
        async def process_query(query):
            async with semaphore:
                return await self.query(query, return_details)
        
        tasks = [process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = S3Result(
                    query=queries[i],
                    response="",
                    search_state=SearchState(),
                    success=False,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch processing completed: {successful}/{len(queries)} successful")
        
        return processed_results
    
    async def _generate_response(self, query: str, search_state: SearchState) -> str:
        """Generate response using retrieved documents"""
        
        if not search_state.documents:
            return "I couldn't find relevant information to answer your query."
        
        try:
            # Get top documents
            top_docs = search_state.get_top_documents(k=5)
            
            # Create context from documents
            context_parts = []
            for i, doc in enumerate(top_docs):
                content = doc.get('content', '')
                title = doc.get('title', f'Document {i+1}')
                context_parts.append(f"{title}: {content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response using the generator
            response = await self.generator.generate(query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I encountered an error while generating the response."
    
    async def _calculate_detailed_metrics(self, result: S3Result) -> None:
        """Calculate detailed quality metrics"""
        
        try:
            # Calculate GBR reward
            gbr_reward = await self.reward_calculator.calculate_gbr_reward(
                result.query, result.search_state, result.response
            )
            result.gbr_reward = gbr_reward
            
            # Extract component scores
            result.relevance_score = gbr_reward.relevance_score
            result.diversity_score = gbr_reward.diversity_score
            result.coherence_score = gbr_reward.coherence_score
            
            # Add metadata
            result.metadata.update({
                'search_iterations': result.search_state.iteration,
                'total_documents': result.search_state.total_documents,
                'unique_documents': result.search_state.unique_documents,
                'average_relevance': result.search_state.average_relevance,
                'search_efficiency': result.search_state.search_efficiency,
                'gbr_gain': gbr_reward.normalized_gain
            })
            
        except Exception as e:
            logger.error(f"Detailed metrics calculation failed: {e}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        # Basic stats
        success_rate = self.successful_queries / max(1, self.total_queries)
        avg_inference_time = self.total_inference_time / max(1, self.successful_queries)
        
        stats = {
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'success_rate': success_rate,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_inference_time,
            'is_trained': self.is_trained,
            'training_sessions': len(self.training_history)
        }
        
        # Add component stats
        stats['search_agent'] = self.search_agent.get_search_statistics()
        stats['reward_calculator'] = self.reward_calculator.get_reward_statistics()
        
        # Add recent performance metrics
        if self.inference_history:
            recent_results = self.inference_history[-100:]  # Last 100 queries
            
            successful_recent = [r for r in recent_results if r.success]
            if successful_recent:
                stats['recent_performance'] = {
                    'count': len(successful_recent),
                    'avg_total_time': sum(r.total_time for r in successful_recent) / len(successful_recent),
                    'avg_search_time': sum(r.search_time for r in successful_recent) / len(successful_recent),
                    'avg_generation_time': sum(r.generation_time for r in successful_recent) / len(successful_recent),
                    'avg_relevance': sum(r.relevance_score for r in successful_recent) / len(successful_recent),
                    'avg_diversity': sum(r.diversity_score for r in successful_recent) / len(successful_recent),
                    'avg_coherence': sum(r.coherence_score for r in successful_recent) / len(successful_recent)
                }
        
        return stats
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to disk"""
        try:
            # In a real implementation, this would save the RL policy
            import json
            
            model_data = {
                'config': self.config.__dict__,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'statistics': self.get_pipeline_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', [])
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate pipeline performance on test data
        
        Args:
            test_data: List of test examples with queries and expected outputs
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating pipeline on {len(test_data)} test examples")
        
        start_time = time.time()
        results = []
        
        for example in test_data:
            query = example['query']
            expected = example.get('expected_response', '')
            
            result = await self.query(query, return_details=True)
            
            # Calculate evaluation metrics
            eval_metrics = {
                'query': query,
                'expected': expected,
                'actual': result.response,
                'success': result.success,
                'relevance_score': result.relevance_score,
                'diversity_score': result.diversity_score,
                'coherence_score': result.coherence_score,
                'search_time': result.search_time,
                'generation_time': result.generation_time,
                'total_time': result.total_time
            }
            
            if result.gbr_reward:
                eval_metrics['gbr_gain'] = result.gbr_reward.normalized_gain
            
            results.append(eval_metrics)
        
        # Aggregate metrics
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            evaluation = {
                'total_examples': len(test_data),
                'successful_examples': len(successful_results),
                'success_rate': len(successful_results) / len(test_data),
                'average_relevance': sum(r['relevance_score'] for r in successful_results) / len(successful_results),
                'average_diversity': sum(r['diversity_score'] for r in successful_results) / len(successful_results),
                'average_coherence': sum(r['coherence_score'] for r in successful_results) / len(successful_results),
                'average_search_time': sum(r['search_time'] for r in successful_results) / len(successful_results),
                'average_generation_time': sum(r['generation_time'] for r in successful_results) / len(successful_results),
                'average_total_time': sum(r['total_time'] for r in successful_results) / len(successful_results),
                'evaluation_time': time.time() - start_time
            }
            
            # Add GBR metrics if available
            gbr_results = [r for r in successful_results if 'gbr_gain' in r]
            if gbr_results:
                evaluation['average_gbr_gain'] = sum(r['gbr_gain'] for r in gbr_results) / len(gbr_results)
        else:
            evaluation = {
                'total_examples': len(test_data),
                'successful_examples': 0,
                'success_rate': 0.0,
                'evaluation_time': time.time() - start_time,
                'error': 'No successful evaluations'
            }
        
        logger.info(f"Evaluation completed in {evaluation['evaluation_time']:.2f}s")
        return evaluation
