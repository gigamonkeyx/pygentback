"""
Coding Agent - Advanced Code Generation and Analysis

Uses Ollama integration for sophisticated code generation, analysis, debugging,
and optimization tasks. Provides real AI-powered coding assistance.
"""

import logging
import asyncio
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.agent import BaseAgent, AgentMessage, MessageType
from core.agent.config import AgentConfig
from ai.reasoning.tot.thought_generator import OllamaBackend

logger = logging.getLogger(__name__)


class CodingAgent(BaseAgent):
    """
    Real Coding Agent using Ollama for code generation and analysis
    
    This agent specializes in:
    - Code generation and completion
    - Code analysis and review
    - Debugging assistance
    - Code optimization suggestions
    - Documentation generation
    - Refactoring recommendations
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize Ollama backend optimized for coding
        self.model_name = config.get_custom_config("model_name", "deepseek-coder-v2:latest")  # Latest coding model
        self.ollama_url = config.get_custom_config("ollama_url", "http://localhost:11434")
        
        self.ollama_backend = OllamaBackend(
            model_name=self.model_name,
            base_url=self.ollama_url
        )
        
        # Coding-specific configuration
        self.max_tokens = config.get_custom_config("max_tokens", 1000)  # Longer for code
        self.temperature = config.get_custom_config("temperature", 0.3)  # Lower for more precise code
        
        # Supported programming languages
        self.supported_languages = [
            "python", "javascript", "typescript", "java", "c++", "c#", "go", 
            "rust", "php", "ruby", "swift", "kotlin", "scala", "r", "sql",
            "html", "css", "bash", "powershell", "yaml", "json", "xml"
        ]
        
        # Agent state
        self.coding_history = []
        self.total_coding_time = 0.0
        
        logger.info(f"CodingAgent initialized with Ollama (model: {self.model_name})")
    
    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic"""
        # Test Ollama connection with a coding task
        try:
            test_response = await self.ollama_backend.generate(
                "Write a simple Python function to add two numbers:",
                max_tokens=100,
                temperature=0.1
            )
            if test_response and "def" in test_response:
                logger.info("CodingAgent Ollama connection verified with coding test")
            else:
                logger.warning("CodingAgent Ollama connection test failed")
        except Exception as e:
            logger.error(f"CodingAgent Ollama initialization error: {e}")
    
    async def _agent_shutdown(self) -> None:
        """Agent-specific shutdown logic"""
        logger.info("CodingAgent shutting down")
    
    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a request message"""
        return await self.process_message(message)
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process coding-related messages
        
        Args:
            message: Incoming message with coding request
            
        Returns:
            AgentMessage: Response with code generation/analysis results
        """
        start_time = datetime.now()
        
        try:
            # Extract the coding request from the message
            user_input = message.content.get("content", str(message.content))
            
            logger.info(f"Processing coding request: {user_input[:100]}...")
            
            # Analyze the request type
            request_type = self._analyze_request_type(user_input)
            
            # Generate appropriate coding response
            if request_type == "code_generation":
                response_content = await self._handle_code_generation(user_input)
            elif request_type == "code_analysis":
                response_content = await self._handle_code_analysis(user_input)
            elif request_type == "debugging":
                response_content = await self._handle_debugging(user_input)
            elif request_type == "optimization":
                response_content = await self._handle_optimization(user_input)
            elif request_type == "documentation":
                response_content = await self._handle_documentation(user_input)
            else:
                response_content = await self._handle_general_coding(user_input)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_coding_time += processing_time
            
            # Store in coding history
            self.coding_history.append({
                "request": user_input,
                "request_type": request_type,
                "processing_time": processing_time,
                "timestamp": start_time.isoformat()
            })
            
            logger.info(f"Coding request processed in {processing_time:.2f}s")
            
            # Create response message
            response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content=response_content,
                correlation_id=message.id
            )
            
            # Update agent activity
            self.last_activity = datetime.utcnow()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in CodingAgent.process_message: {e}")
            
            # Create error response
            error_response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content={
                    "type": "error",
                    "message": f"Coding assistance failed: {str(e)}",
                    "agent": "coding"
                },
                correlation_id=message.id
            )
            
            return error_response
    
    def _analyze_request_type(self, user_input: str) -> str:
        """Analyze the type of coding request"""
        
        input_lower = user_input.lower()
        
        # Code generation keywords
        if any(keyword in input_lower for keyword in [
            "write", "create", "generate", "implement", "build", "make", "code"
        ]):
            return "code_generation"
        
        # Code analysis keywords
        elif any(keyword in input_lower for keyword in [
            "analyze", "review", "explain", "understand", "what does", "how does"
        ]):
            return "code_analysis"
        
        # Debugging keywords
        elif any(keyword in input_lower for keyword in [
            "debug", "fix", "error", "bug", "issue", "problem", "not working"
        ]):
            return "debugging"
        
        # Optimization keywords
        elif any(keyword in input_lower for keyword in [
            "optimize", "improve", "faster", "performance", "efficient", "refactor"
        ]):
            return "optimization"
        
        # Documentation keywords
        elif any(keyword in input_lower for keyword in [
            "document", "comment", "docstring", "readme", "documentation"
        ]):
            return "documentation"
        
        else:
            return "general_coding"
    
    async def _handle_code_generation(self, user_input: str) -> Dict[str, Any]:
        """Handle code generation requests"""
        
        # Detect programming language
        language = self._detect_language(user_input)
        
        # Create enhanced prompt for code generation
        prompt = f"""You are an expert programmer. Generate clean, efficient, and well-commented code for the following request:

Request: {user_input}

Programming Language: {language}

Please provide:
1. Complete, working code
2. Clear comments explaining the logic
3. Example usage if applicable
4. Any important notes or considerations

Code:"""
        
        try:
            code_response = await self.ollama_backend.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and format the code
            formatted_code = self._format_code_response(code_response, language)
            
            return {
                "type": "code_generation",
                "language": language,
                "code": formatted_code,
                "original_request": user_input,
                "quality_score": self._assess_code_quality(formatted_code),
                "metrics": {
                    "lines_of_code": len(formatted_code.split('\n')),
                    "estimated_complexity": "medium"
                },
                "agent": "coding"
            }
            
        except Exception as e:
            return self._create_fallback_response(user_input, f"Code generation failed: {e}")
    
    async def _handle_code_analysis(self, user_input: str) -> Dict[str, Any]:
        """Handle code analysis requests"""
        
        prompt = f"""You are an expert code reviewer. Analyze the following code or coding question:

{user_input}

Please provide:
1. Code explanation and functionality
2. Strengths and potential issues
3. Suggestions for improvement
4. Best practices recommendations

Analysis:"""
        
        try:
            analysis_response = await self.ollama_backend.generate(
                prompt,
                temperature=0.4,
                max_tokens=self.max_tokens
            )
            
            return {
                "type": "code_analysis",
                "analysis": analysis_response.strip(),
                "original_request": user_input,
                "confidence": 0.85,
                "agent": "coding"
            }
            
        except Exception as e:
            return self._create_fallback_response(user_input, f"Code analysis failed: {e}")
    
    async def _handle_debugging(self, user_input: str) -> Dict[str, Any]:
        """Handle debugging requests"""
        
        prompt = f"""You are an expert debugger. Help identify and fix the issue in the following code or error:

{user_input}

Please provide:
1. Identification of the problem
2. Explanation of why it's happening
3. Step-by-step solution
4. Corrected code if applicable
5. Prevention tips for the future

Debugging Solution:"""
        
        try:
            debug_response = await self.ollama_backend.generate(
                prompt,
                temperature=0.2,  # Lower temperature for more precise debugging
                max_tokens=self.max_tokens
            )
            
            return {
                "type": "debugging",
                "solution": debug_response.strip(),
                "original_request": user_input,
                "confidence": 0.8,
                "agent": "coding"
            }
            
        except Exception as e:
            return self._create_fallback_response(user_input, f"Debugging failed: {e}")
    
    async def _handle_optimization(self, user_input: str) -> Dict[str, Any]:
        """Handle code optimization requests"""
        
        prompt = f"""You are an expert in code optimization. Improve the following code for better performance, readability, and maintainability:

{user_input}

Please provide:
1. Optimized version of the code
2. Explanation of improvements made
3. Performance benefits expected
4. Any trade-offs to consider

Optimization:"""
        
        try:
            optimization_response = await self.ollama_backend.generate(
                prompt,
                temperature=0.3,
                max_tokens=self.max_tokens
            )
            
            return {
                "type": "optimization",
                "optimized_solution": optimization_response.strip(),
                "original_request": user_input,
                "confidence": 0.8,
                "agent": "coding"
            }
            
        except Exception as e:
            return self._create_fallback_response(user_input, f"Optimization failed: {e}")
    
    async def _handle_documentation(self, user_input: str) -> Dict[str, Any]:
        """Handle documentation generation requests"""
        
        prompt = f"""You are an expert technical writer. Generate comprehensive documentation for the following code:

{user_input}

Please provide:
1. Clear function/class descriptions
2. Parameter explanations
3. Return value descriptions
4. Usage examples
5. Any important notes or warnings

Documentation:"""
        
        try:
            doc_response = await self.ollama_backend.generate(
                prompt,
                temperature=0.4,
                max_tokens=self.max_tokens
            )
            
            return {
                "type": "documentation",
                "documentation": doc_response.strip(),
                "original_request": user_input,
                "confidence": 0.9,
                "agent": "coding"
            }
            
        except Exception as e:
            return self._create_fallback_response(user_input, f"Documentation generation failed: {e}")
    
    async def _handle_general_coding(self, user_input: str) -> Dict[str, Any]:
        """Handle general coding questions and requests"""
        
        prompt = f"""You are an expert programmer and coding mentor. Help with the following coding question or request:

{user_input}

Please provide a comprehensive, helpful response that includes:
1. Direct answer to the question
2. Code examples if relevant
3. Best practices and recommendations
4. Additional resources or next steps

Response:"""
        
        try:
            general_response = await self.ollama_backend.generate(
                prompt,
                temperature=0.5,
                max_tokens=self.max_tokens
            )
            
            return {
                "type": "general_coding",
                "response": general_response.strip(),
                "original_request": user_input,
                "confidence": 0.8,
                "agent": "coding"
            }
            
        except Exception as e:
            return self._create_fallback_response(user_input, f"General coding assistance failed: {e}")
    
    def _detect_language(self, user_input: str) -> str:
        """Detect programming language from user input"""
        
        input_lower = user_input.lower()
        
        # Check for explicit language mentions
        for lang in self.supported_languages:
            if lang in input_lower:
                return lang
        
        # Check for language-specific keywords
        if any(keyword in input_lower for keyword in ["def ", "import ", "python", ".py"]):
            return "python"
        elif any(keyword in input_lower for keyword in ["function", "const", "let", "var", "javascript", ".js"]):
            return "javascript"
        elif any(keyword in input_lower for keyword in ["class", "public", "private", "java", ".java"]):
            return "java"
        elif any(keyword in input_lower for keyword in ["#include", "int main", "c++", ".cpp"]):
            return "c++"
        elif any(keyword in input_lower for keyword in ["using", "namespace", "c#", ".cs"]):
            return "c#"
        
        # Default to Python if no language detected
        return "python"
    
    def _format_code_response(self, code_response: str, language: str) -> str:
        """Format and clean up the code response"""
        
        # Remove common prefixes/suffixes
        cleaned_code = code_response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_code.startswith("```"):
            lines = cleaned_code.split('\n')
            if len(lines) > 2:
                cleaned_code = '\n'.join(lines[1:-1])
        
        return cleaned_code
    
    def _assess_code_quality(self, code: str) -> float:
        """Simple code quality assessment"""
        
        score = 0.5  # Base score
        
        # Check for comments
        if '#' in code or '//' in code or '/*' in code:
            score += 0.2
        
        # Check for proper indentation
        lines = code.split('\n')
        if any(line.startswith('    ') or line.startswith('\t') for line in lines):
            score += 0.1
        
        # Check for function definitions
        if 'def ' in code or 'function' in code:
            score += 0.1
        
        # Check for error handling
        if any(keyword in code for keyword in ['try', 'catch', 'except', 'error']):
            score += 0.1
        
        return min(1.0, score)
    
    def _create_fallback_response(self, user_input: str, error_message: str) -> Dict[str, Any]:
        """Create a fallback response when AI generation fails"""
        
        return {
            "type": "coding_error",
            "response": f"I apologize, but I encountered an issue processing your coding request: {error_message}",
            "original_request": user_input,
            "confidence": 0.1,
            "agent": "coding",
            "fallback": True,
            "suggestions": [
                "Try rephrasing your request",
                "Be more specific about the programming language",
                "Break down complex requests into smaller parts"
            ]
        }
    
    def get_coding_stats(self) -> Dict[str, Any]:
        """Get coding statistics for this agent"""
        return {
            "total_coding_requests": len(self.coding_history),
            "total_coding_time": self.total_coding_time,
            "average_coding_time": self.total_coding_time / max(1, len(self.coding_history)),
            "supported_languages": self.supported_languages,
            "model_name": self.model_name,
            "recent_requests": self.coding_history[-5:] if self.coding_history else []
        }
