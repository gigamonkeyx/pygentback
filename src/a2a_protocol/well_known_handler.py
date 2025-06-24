#!/usr/bin/env python3
"""
A2A Well-Known URL Handler

Implements the /.well-known/agent.json endpoint as required by the A2A specification.
Provides agent discovery and metadata according to Google A2A standards.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .agent_card_generator import AgentCard, A2AAgentCardGenerator

logger = logging.getLogger(__name__)


class A2AWellKnownHandler:
    """Handles A2A well-known URL endpoints for agent discovery"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.agent_cards: Dict[str, AgentCard] = {}
        self.card_generator = A2AAgentCardGenerator(base_url)
        
    def register_agent_card(self, agent_id: str, agent_card: AgentCard):
        """Register an agent card for well-known URL serving"""
        self.agent_cards[agent_id] = agent_card
        logger.info(f"Registered agent card for {agent_id} at well-known URL")
    
    def unregister_agent_card(self, agent_id: str):
        """Unregister an agent card"""
        if agent_id in self.agent_cards:
            del self.agent_cards[agent_id]
            logger.info(f"Unregistered agent card for {agent_id}")
    
    async def handle_well_known_agent_json(self, request: Request, agent_id: Optional[str] = None) -> JSONResponse:
        """Handle /.well-known/agent.json requests"""
        try:
            # If specific agent requested
            if agent_id:
                if agent_id not in self.agent_cards:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                agent_card = self.agent_cards[agent_id]
                return JSONResponse(
                    content=self.card_generator.to_json(agent_card),
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "public, max-age=300",  # 5 minutes cache
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization"
                    }
                )
            
            # Return list of all available agents
            agent_list = []
            for agent_id, agent_card in self.agent_cards.items():
                agent_list.append({
                    "id": agent_id,
                    "name": agent_card.name,
                    "description": agent_card.description,
                    "url": agent_card.url,
                    "version": agent_card.version,
                    "provider": {
                        "name": agent_card.provider.name,
                        "organization": agent_card.provider.organization
                    },
                    "capabilities": {
                        "streaming": agent_card.capabilities.streaming,
                        "pushNotifications": agent_card.capabilities.pushNotifications,
                        "stateTransitionHistory": agent_card.capabilities.stateTransitionHistory
                    },
                    "skills": [
                        {
                            "id": skill.id,
                            "name": skill.name,
                            "description": skill.description,
                            "tags": skill.tags
                        }
                        for skill in agent_card.skills
                    ]
                })
            
            return JSONResponse(
                content={
                    "agents": agent_list,
                    "total": len(agent_list),
                    "base_url": self.base_url,
                    "a2a_version": "1.0",
                    "timestamp": "2024-01-01T00:00:00Z"
                },
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "public, max-age=60",  # 1 minute cache for list
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling well-known agent request: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def handle_agent_card_request(self, agent_id: str, authenticated: bool = False) -> JSONResponse:
        """Handle individual agent card requests"""
        try:
            if agent_id not in self.agent_cards:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
            agent_card = self.agent_cards[agent_id]
            
            # Return extended card if authenticated and supported
            if authenticated and agent_card.supportsAuthenticatedExtendedCard:
                card_data = self.card_generator.to_json(agent_card)
                # Add extended information for authenticated requests
                card_data["extended"] = {
                    "internal_metrics": True,
                    "detailed_capabilities": True,
                    "configuration_options": True
                }
            else:
                # Return basic card
                card_data = self._get_basic_agent_card(agent_card)
            
            return JSONResponse(
                content=card_data,
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "public, max-age=300",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling agent card request for {agent_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _get_basic_agent_card(self, agent_card: AgentCard) -> Dict[str, Any]:
        """Get basic agent card without sensitive information"""
        return {
            "name": agent_card.name,
            "description": agent_card.description,
            "url": agent_card.url,
            "version": agent_card.version,
            "provider": {
                "name": agent_card.provider.name,
                "organization": agent_card.provider.organization,
                "url": agent_card.provider.url
            },
            "capabilities": {
                "streaming": agent_card.capabilities.streaming,
                "pushNotifications": agent_card.capabilities.pushNotifications,
                "stateTransitionHistory": agent_card.capabilities.stateTransitionHistory
            },
            "defaultInputModes": agent_card.defaultInputModes,
            "defaultOutputModes": agent_card.defaultOutputModes,
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                    "examples": skill.examples[:3],  # Limit examples for basic card
                    "inputModes": skill.inputModes,
                    "outputModes": skill.outputModes
                }
                for skill in agent_card.skills
            ],
            "documentationUrl": agent_card.documentationUrl,
            "supportsAuthenticatedExtendedCard": agent_card.supportsAuthenticatedExtendedCard
        }
    
    def setup_fastapi_routes(self, app: FastAPI):
        """Setup FastAPI routes for well-known URLs"""
        
        @app.get("/.well-known/agent.json")
        async def well_known_agent_list(request: Request):
            """Get list of all available agents"""
            return await self.handle_well_known_agent_json(request)
        
        @app.get("/.well-known/agent/{agent_id}.json")
        async def well_known_agent_card(request: Request, agent_id: str):
            """Get specific agent card"""
            return await self.handle_well_known_agent_json(request, agent_id)
        
        @app.get("/a2a/agents/{agent_id}/card")
        async def agent_card_endpoint(agent_id: str, request: Request):
            """Get agent card (basic or extended based on authentication)"""
            # Check for authentication
            auth_header = request.headers.get("Authorization")
            authenticated = bool(auth_header and auth_header.startswith("Bearer "))
            
            return await self.handle_agent_card_request(agent_id, authenticated)
        
        @app.options("/.well-known/agent.json")
        @app.options("/.well-known/agent/{agent_id}.json")
        @app.options("/a2a/agents/{agent_id}/card")
        async def options_handler():
            """Handle CORS preflight requests"""
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "86400"
                }
            )
        
        logger.info("A2A well-known URL routes registered with FastAPI")
    
    def get_agent_discovery_info(self) -> Dict[str, Any]:
        """Get agent discovery information for debugging"""
        return {
            "registered_agents": list(self.agent_cards.keys()),
            "total_agents": len(self.agent_cards),
            "base_url": self.base_url,
            "well_known_urls": [
                f"{self.base_url}/.well-known/agent.json",
                f"{self.base_url}/.well-known/agent/{{agent_id}}.json",
                f"{self.base_url}/a2a/agents/{{agent_id}}/card"
            ]
        }


# Global well-known handler instance
well_known_handler = A2AWellKnownHandler()
