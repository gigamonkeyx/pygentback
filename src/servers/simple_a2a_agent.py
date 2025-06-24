#!/usr/bin/env python3
"""
Simple A2A Agent

A simple A2A-compatible agent that can receive and respond to messages.
This agent specializes in text processing and analysis tasks.
"""

import asyncio
import logging
import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# A2A SDK imports
from a2a.types import (
    AgentCard, AgentProvider, AgentCapabilities, AgentSkill,
    Task, TaskState, TaskStatus, Message, Part, TextPart
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simple_a2a_agent.log')
    ]
)

logger = logging.getLogger(__name__)


class SimpleA2AAgent:
    """Simple A2A Agent implementation"""
    
    def __init__(self, name: str = "Text Processor Agent", port: int = 8007):
        self.name = name
        self.port = port
        self.start_time = time.time()
        self.agent_id = f"text_processor_{uuid.uuid4().hex[:8]}"
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'tasks_completed': 0,
            'errors': 0
        }
        
        # Task storage
        self.active_tasks: Dict[str, Any] = {}
        
        # Initialize agent card
        self._initialize_agent_card()
        
        # Create FastAPI app
        self.app = FastAPI(
            title=f"{self.name}",
            description="Simple A2A-compatible text processing agent",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _initialize_agent_card(self):
        """Initialize the A2A agent card"""
        try:
            self.agent_card = AgentCard(
                name=self.name,
                description="A simple A2A agent specialized in text processing and analysis tasks",
                version="1.0.0",
                url=f"http://127.0.0.1:{self.port}",
                defaultInputModes=["text", "application/json"],
                defaultOutputModes=["text", "application/json"],
                provider=AgentProvider(
                    name="PyGent Factory",
                    organization="PyGent Factory",
                    description="Simple A2A agent for text processing",
                    url="https://github.com/gigamonkeyx/pygentback"
                ),
                capabilities=AgentCapabilities(
                    streaming=True,
                    pushNotifications=None,
                    stateTransitionHistory=None,
                    extensions=None
                ),
                skills=[
                    AgentSkill(
                        id="text_processing",
                        name="text_processing",
                        description="Process and analyze text content",
                        input_modalities=["text", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=["text", "processing", "analysis"],
                        examples=[
                            "Analyze sentiment of text",
                            "Extract key information from text",
                            "Summarize text content",
                            "Count words and characters"
                        ]
                    ),
                    AgentSkill(
                        id="text_transformation",
                        name="text_transformation",
                        description="Transform text in various ways",
                        input_modalities=["text"],
                        output_modalities=["text"],
                        tags=["text", "transformation", "formatting"],
                        examples=[
                            "Convert text to uppercase/lowercase",
                            "Reverse text",
                            "Remove special characters",
                            "Format text for display"
                        ]
                    )
                ]
            )
            
            logger.info(f"Agent card initialized for {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent card: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Return the agent card for A2A discovery"""
            return self.agent_card.dict() if hasattr(self.agent_card, 'dict') else self.agent_card
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            uptime = time.time() - self.start_time
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "agent_name": self.name,
                    "agent_id": self.agent_id,
                    "uptime_seconds": round(uptime, 2),
                    "messages_received": self.stats['messages_received'],
                    "messages_processed": self.stats['messages_processed'],
                    "tasks_completed": self.stats['tasks_completed'],
                    "active_tasks": len(self.active_tasks),
                    "error_count": self.stats['errors']
                }
            }
        
        @self.app.get("/")
        async def root():
            """Root endpoint with agent information"""
            return {
                "agent": self.name,
                "agent_id": self.agent_id,
                "description": "Simple A2A-compatible text processing agent",
                "version": "1.0.0",
                "capabilities": [
                    "Text processing and analysis",
                    "Text transformation",
                    "A2A protocol support"
                ],
                "endpoints": {
                    "agent_card": "/.well-known/agent.json",
                    "health": "/health",
                    "message": "/a2a/message/send"
                }
            }
        
        @self.app.post("/a2a/message/send")
        async def receive_a2a_message(request: dict):
            """Receive and process A2A messages"""
            try:
                logger.info(f"Received A2A message: {request}")
                
                # Extract message data
                message_data = request.get("params", {}).get("message", {})
                
                # Create task
                task_id = str(uuid.uuid4())
                context_id = message_data.get("contextId") or str(uuid.uuid4())
                
                # Create task object
                task = {
                    'id': task_id,
                    'context_id': context_id,
                    'status': {'state': TaskState.submitted.value},
                    'history': [message_data],
                    'artifacts': [],
                    'metadata': {'received_at': datetime.utcnow().isoformat()}
                }
                
                self.active_tasks[task_id] = task
                self.stats['messages_received'] += 1
                
                # Process the message
                result = await self._process_message(task, message_data)
                
                # Update task status
                task['status']['state'] = TaskState.completed.value
                self.stats['messages_processed'] += 1
                self.stats['tasks_completed'] += 1
                
                logger.info(f"Processed A2A message for task {task_id}")
                
                return {
                    "id": task_id,
                    "contextId": context_id,
                    "status": task['status'],
                    "artifacts": task['artifacts'],
                    "history": task['history'],
                    "kind": "task",
                    "metadata": task['metadata']
                }
                
            except Exception as e:
                logger.error(f"Failed to process A2A message: {str(e)}", exc_info=True)
                self.stats['errors'] += 1
                return {"error": str(e), "status": "failed"}
    
    async def _process_message(self, task: dict, message_data: dict) -> dict:
        """Process an incoming message and generate response"""
        try:
            # Extract text content
            text_content = ""
            parts = message_data.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    text_content += part.get("text", "") + "\n"
            
            text_content = text_content.strip()
            
            # Perform text processing
            analysis_result = self._analyze_text(text_content)
            
            # Create response artifact
            artifact = {
                "artifactId": str(uuid.uuid4()),
                "name": "text_analysis_result",
                "parts": [
                    {
                        "kind": "text",
                        "text": f"Text Analysis Results:\n\n{analysis_result}"
                    }
                ]
            }
            
            task['artifacts'].append(artifact)
            
            return {"success": True, "processed_text": text_content}
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise
    
    def _analyze_text(self, text: str) -> str:
        """Analyze text and return results"""
        if not text:
            return "No text provided for analysis."
        
        # Basic text analysis
        word_count = len(text.split())
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Simple sentiment analysis (very basic)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "Positive"
        elif negative_count > positive_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Generate analysis report
        analysis = f"""
Original Text: "{text}"

Statistics:
- Word Count: {word_count}
- Character Count: {char_count}
- Character Count (no spaces): {char_count_no_spaces}
- Estimated Sentence Count: {sentence_count}

Sentiment Analysis:
- Sentiment: {sentiment}
- Positive indicators: {positive_count}
- Negative indicators: {negative_count}

Text Transformations:
- Uppercase: {text.upper()}
- Lowercase: {text.lower()}
- Reversed: {text[::-1]}
- First 50 chars: {text[:50]}{'...' if len(text) > 50 else ''}

Processing completed by {self.name} at {datetime.utcnow().isoformat()}
        """.strip()
        
        return analysis


def main(host: str = "127.0.0.1", port: int = 8007):
    """Run the simple A2A agent"""
    agent = SimpleA2AAgent(port=port)
    
    logger.info(f"Starting {agent.name} on {host}:{port}")
    
    uvicorn.run(
        agent.app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8007
    
    main(host, port)
