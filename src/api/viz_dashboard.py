#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Visualization Dashboard
Observer-approved FastAPI dashboard with Plotly real-time visualizations
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)

class VizDashboard:
    """Observer-approved real-time visualization dashboard"""
    
    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI and Plotly required for visualization dashboard")
        
        self.app = FastAPI(title="Observer Production Dashboard", version="1.0.0")
        self.connected_clients = set()
        self.metrics_history = []
        self.max_history = 100
        
        # Setup routes
        self._setup_routes()
        
        logger.info("VizDashboard initialized with FastAPI and Plotly")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._generate_dashboard_html()
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics"""
            try:
                if not self.metrics_history:
                    return {"error": "No metrics available"}
                
                latest_metrics = self.metrics_history[-1]
                return {
                    "timestamp": latest_metrics.get("timestamp", datetime.now().isoformat()),
                    "metrics": latest_metrics,
                    "history_count": len(self.metrics_history)
                }
            except Exception as e:
                logger.error(f"Metrics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/viz/fitness")
        async def get_fitness_plot():
            """Get fitness evolution plot"""
            try:
                if len(self.metrics_history) < 2:
                    return {"error": "Insufficient data for plot"}
                
                # Extract fitness data
                generations = []
                fitness_values = []
                
                for i, metrics in enumerate(self.metrics_history):
                    generations.append(i + 1)
                    fitness_values.append(metrics.get('best_fitness', 0))
                
                # Create Plotly figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=generations,
                    y=fitness_values,
                    mode='lines+markers',
                    name='Best Fitness',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='Real-time Fitness Evolution',
                    xaxis_title='Generation',
                    yaxis_title='Fitness Score',
                    template='plotly_white',
                    height=400
                )
                
                return {"plot": json.loads(fig.to_json())}
                
            except Exception as e:
                logger.error(f"Fitness plot generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/viz/network")
        async def get_network_plot():
            """Get cooperation network plot"""
            try:
                if not self.metrics_history:
                    return {"error": "No data available"}
                
                latest_metrics = self.metrics_history[-1]
                
                # Create network visualization
                fig = go.Figure()
                
                # Add nodes (agents)
                agent_count = latest_metrics.get('agent_count', 10)
                cooperation_events = latest_metrics.get('cooperation_events', 0)
                
                # Generate circular layout for agents
                import math
                positions = []
                for i in range(min(agent_count, 20)):  # Limit for visualization
                    angle = 2 * math.pi * i / min(agent_count, 20)
                    x = math.cos(angle)
                    y = math.sin(angle)
                    positions.append((x, y))
                
                # Add agent nodes
                fig.add_trace(go.Scatter(
                    x=[pos[0] for pos in positions],
                    y=[pos[1] for pos in positions],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='lightblue',
                        line=dict(width=2, color='darkblue')
                    ),
                    text=[f'A{i}' for i in range(len(positions))],
                    textposition="middle center",
                    name='Agents'
                ))
                
                # Add cooperation connections
                cooperation_density = min(1.0, cooperation_events / 100.0)
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        if (i + j) % 3 == 0 and cooperation_density > 0.3:  # Sample connections
                            fig.add_trace(go.Scatter(
                                x=[positions[i][0], positions[j][0]],
                                y=[positions[i][1], positions[j][1]],
                                mode='lines',
                                line=dict(color='green', width=2, dash='dot'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                
                fig.update_layout(
                    title=f'Agent Cooperation Network (Events: {cooperation_events})',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                
                return {"plot": json.loads(fig.to_json())}
                
            except Exception as e:
                logger.error(f"Network plot generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/viz/dashboard")
        async def get_dashboard_plot():
            """Get comprehensive dashboard plot"""
            try:
                if len(self.metrics_history) < 2:
                    return {"error": "Insufficient data for dashboard"}
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Fitness Evolution', 'System Health', 'Cooperation Events', 'Performance Metrics'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Extract data
                generations = list(range(1, len(self.metrics_history) + 1))
                fitness_values = [m.get('best_fitness', 0) for m in self.metrics_history]
                health_values = [m.get('system_health', 0.5) for m in self.metrics_history]
                cooperation_values = [m.get('cooperation_events', 0) for m in self.metrics_history]
                performance_values = [m.get('performance_score', 0.5) for m in self.metrics_history]
                
                # Plot 1: Fitness Evolution
                fig.add_trace(
                    go.Scatter(x=generations, y=fitness_values, name='Fitness', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Plot 2: System Health
                fig.add_trace(
                    go.Scatter(x=generations, y=health_values, name='Health', line=dict(color='green')),
                    row=1, col=2
                )
                
                # Plot 3: Cooperation Events
                fig.add_trace(
                    go.Bar(x=generations, y=cooperation_values, name='Cooperation', marker_color='orange'),
                    row=2, col=1
                )
                
                # Plot 4: Performance Metrics
                fig.add_trace(
                    go.Scatter(x=generations, y=performance_values, name='Performance', line=dict(color='red')),
                    row=2, col=2
                )
                
                fig.update_layout(
                    title='Observer Production Dashboard',
                    height=600,
                    showlegend=False,
                    template='plotly_white'
                )
                
                return {"plot": json.loads(fig.to_json())}
                
            except Exception as e:
                logger.error(f"Dashboard plot generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    # Send latest metrics every 5 seconds
                    if self.metrics_history:
                        latest_metrics = self.metrics_history[-1]
                        await websocket.send_json({
                            "type": "metrics_update",
                            "data": latest_metrics,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    await asyncio.sleep(5)
                    
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML with Plotly integration"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Observer Production Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        .metric-label { color: #666; margin-top: 5px; }
        .plot-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .status.good { background-color: #d4edda; color: #155724; }
        .status.warning { background-color: #fff3cd; color: #856404; }
        .status.error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Observer Production Dashboard</h1>
        <p>Real-time monitoring of autonomous AI systems</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="fitness-value">--</div>
            <div class="metric-label">Best Fitness</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="agents-value">--</div>
            <div class="metric-label">Active Agents</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="cooperation-value">--</div>
            <div class="metric-label">Cooperation Events</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="health-value">--</div>
            <div class="metric-label">System Health</div>
        </div>
    </div>
    
    <div class="plot-container">
        <div id="dashboard-plot"></div>
    </div>
    
    <div id="status-container"></div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'metrics_update') {
                updateMetrics(data.data);
            }
        };
        
        function updateMetrics(metrics) {
            document.getElementById('fitness-value').textContent = (metrics.best_fitness || 0).toFixed(3);
            document.getElementById('agents-value').textContent = metrics.agent_count || 0;
            document.getElementById('cooperation-value').textContent = metrics.cooperation_events || 0;
            document.getElementById('health-value').textContent = ((metrics.system_health || 0.5) * 100).toFixed(1) + '%';
            
            // Update status
            updateStatus(metrics);
        }
        
        function updateStatus(metrics) {
            const container = document.getElementById('status-container');
            const fitness = metrics.best_fitness || 0;
            const health = metrics.system_health || 0.5;
            
            let statusClass = 'good';
            let statusText = 'System operating normally';
            
            if (fitness < 1.0 || health < 0.7) {
                statusClass = 'warning';
                statusText = 'System performance below optimal';
            }
            
            if (fitness < 0.5 || health < 0.5) {
                statusClass = 'error';
                statusText = 'System requires attention';
            }
            
            container.innerHTML = `<div class="status ${statusClass}">${statusText}</div>`;
        }
        
        // Load dashboard plot
        async function loadDashboard() {
            try {
                const response = await fetch('/api/viz/dashboard');
                const data = await response.json();
                
                if (data.plot) {
                    Plotly.newPlot('dashboard-plot', data.plot.data, data.plot.layout);
                }
            } catch (error) {
                console.error('Failed to load dashboard:', error);
            }
        }
        
        // Refresh dashboard every 10 seconds
        setInterval(loadDashboard, 10000);
        loadDashboard();
    </script>
</body>
</html>
        """
    
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics and notify connected clients"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Limit history size
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
            
            # Notify connected clients
            if self.connected_clients:
                message = {
                    "type": "metrics_update",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send to all connected clients
                disconnected_clients = set()
                for client in self.connected_clients:
                    try:
                        await client.send_json(message)
                    except Exception:
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                self.connected_clients -= disconnected_clients
            
            logger.info(f"Updated metrics: fitness={metrics.get('best_fitness', 0):.3f}, "
                       f"agents={metrics.get('agent_count', 0)}, "
                       f"cooperation={metrics.get('cooperation_events', 0)}")
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def get_app(self):
        """Get FastAPI app instance"""
        return self.app

# Global dashboard instance
dashboard = None

def get_dashboard() -> Optional[VizDashboard]:
    """Get global dashboard instance"""
    global dashboard
    if dashboard is None and FASTAPI_AVAILABLE:
        try:
            dashboard = VizDashboard()
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}")
    return dashboard

async def start_dashboard_server(host: str = "localhost", port: int = 8000):
    """Start the dashboard server"""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available - cannot start dashboard server")
        return False
    
    try:
        import uvicorn
        dashboard = get_dashboard()
        if dashboard:
            logger.info(f"Starting dashboard server on {host}:{port}")
            config = uvicorn.Config(dashboard.get_app(), host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
            return True
        else:
            logger.error("Failed to initialize dashboard")
            return False
    except ImportError:
        logger.error("Uvicorn not available - cannot start dashboard server")
        return False
    except Exception as e:
        logger.error(f"Dashboard server startup failed: {e}")
        return False
