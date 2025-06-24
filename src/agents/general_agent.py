"""
General Agent - Direct Ollama Integration

Uses direct Ollama integration for general-purpose AI tasks.
Replaces the mock "general agent" with real AI-powered responses.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from core.agent import BaseAgent, AgentMessage, MessageType
from core.agent.config import AgentConfig
from ai.reasoning.tot.thought_generator import OllamaBackend

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """
    Real General Agent using direct Ollama integration
    
    This agent provides general-purpose AI assistance using direct
    communication with Ollama models for fast, reliable responses.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize Ollama backend
        self.model_name = config.get_custom_config("model_name", "")
        self.ollama_url = config.get_custom_config("ollama_url", "http://localhost:11434")
        
        self.ollama_backend = OllamaBackend(
            model_name=self.model_name,
            base_url=self.ollama_url
        )
        
        # Agent configuration
        self.max_tokens = config.get_custom_config("max_tokens", 500)
        self.temperature = config.get_custom_config("temperature", 0.7)
        
        # Agent state
        self.conversation_history = []
        self.total_generation_time = 0.0
        
        logger.info(f"GeneralAgent initialized with Ollama (model: {self.model_name})")

    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic"""
        # Test Ollama connection
        try:
            test_response = await self.ollama_backend.generate("Hello", max_tokens=5)
            if test_response:
                logger.info("Ollama connection verified")
            else:
                logger.warning("Ollama connection test failed")
        except Exception as e:
            logger.error(f"Ollama initialization error: {e}")

    async def _agent_shutdown(self) -> None:
        """Agent-specific shutdown logic"""
        logger.info("GeneralAgent shutting down")
        # Clean up resources if needed

    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a request message"""
        return await self.process_message(message)

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process message using direct Ollama generation
        
        Args:
            message: Incoming message for general AI assistance
            
        Returns:
            AgentMessage: AI-generated response
        """
        start_time = datetime.now()
        
        try:
            # Extract the user input from the message
            user_input = message.content.get("content", str(message.content))
            
            logger.info(f"Processing general query: {user_input[:100]}...")
            
            # Create enhanced prompt for better responses
            prompt = self._create_enhanced_prompt(user_input)
            
            # Generate response using Ollama
            ai_response = await self.ollama_backend.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            self.total_generation_time += generation_time
            
            if ai_response:
                # Process and format the response
                formatted_response = self._format_ai_response(ai_response, user_input, generation_time)
                
                # Store in conversation history
                self.conversation_history.append({
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "generation_time": generation_time,
                    "timestamp": start_time.isoformat()
                })
                
                logger.info(f"AI response generated in {generation_time:.2f}s")
                
            else:
                # Fallback if Ollama fails
                formatted_response = self._create_fallback_response(user_input)
                logger.warning("Ollama generation failed, using fallback")
            
            # Create response message
            response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content=formatted_response,
                correlation_id=message.id
            )
            
            # Update agent activity
            self.last_activity = datetime.utcnow()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in GeneralAgent.process_message: {e}")
            
            # Create error response
            error_response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content={
                    "type": "error",
                    "message": f"AI generation failed: {str(e)}",
                    "agent": "general"
                },
                correlation_id=message.id
            )
            
            return error_response
    
    def _create_enhanced_prompt(self, user_input: str) -> str:
        """Create an enhanced prompt for better AI responses"""
        
        # Add context and instructions for better responses
        enhanced_prompt = f"""You are a helpful AI assistant in the PyGent Factory system. Please provide a clear, informative, and helpful response to the following query.

User Query: {user_input}

Please provide a comprehensive response that:
- Directly addresses the user's question or request
- Includes relevant details and explanations
- Is well-structured and easy to understand
- Offers practical advice or next steps when appropriate

Response:"""
        
        return enhanced_prompt
    
    def _format_ai_response(self, ai_response: str, user_input: str, generation_time: float) -> Dict[str, Any]:
        """Format the AI response into a structured response"""
        
        # Clean up the response
        cleaned_response = ai_response.strip()
        
        # Estimate response quality (simple heuristic)
        quality_score = min(1.0, len(cleaned_response) / 100)  # Basic quality metric
        
        return {
            "type": "general_response",
            "response": cleaned_response,
            "user_query": user_input,
            "quality_score": quality_score,
            "metrics": {
                "generation_time": generation_time,
                "response_length": len(cleaned_response),
                "model_used": self.model_name
            },
            "agent": "general"
        }
    
    def _create_fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Create a fallback response when AI generation fails"""
        
        fallback_text = f"""I apologize, but I'm currently experiencing technical difficulties with my AI processing system. 

Your query: "{user_input}"

While I can't provide my usual AI-powered response right now, I'm designed to help with:
- General questions and information
- Problem-solving assistance  
- Explanations and guidance
- Creative tasks and brainstorming

Please try your request again, or contact support if the issue persists."""
        
        return {
            "type": "general_response",
            "response": fallback_text,
            "user_query": user_input,
            "quality_score": 0.3,
            "metrics": {
                "generation_time": 0.0,
                "response_length": len(fallback_text),
                "model_used": "fallback"
            },
            "agent": "general",
            "fallback": True
        }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Check the status of the Ollama model"""
        try:
            # Test generation with a simple prompt
            test_response = await self.ollama_backend.generate(
                "Hello", 
                max_tokens=10,
                temperature=0.1
            )
            
            return {
                "model_available": bool(test_response),
                "model_name": self.model_name,
                "ollama_url": self.ollama_url,
                "last_test": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "model_available": False,
                "model_name": self.model_name,
                "ollama_url": self.ollama_url,
                "error": str(e),
                "last_test": datetime.now().isoformat()
            }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics for this agent"""
        return {
            "total_conversations": len(self.conversation_history),
            "total_generation_time": self.total_generation_time,
            "average_generation_time": self.total_generation_time / max(1, len(self.conversation_history)),
            "model_name": self.model_name,
            "recent_conversations": self.conversation_history[-5:] if self.conversation_history else []
        }
