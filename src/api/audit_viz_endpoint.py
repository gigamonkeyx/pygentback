#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Audit Visualization API Endpoint
Observer-approved interactive dashboard for real-time MCP audit monitoring
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template_string
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

logger = logging.getLogger(__name__)

class AuditVizEndpoint:
    """
    Observer-approved audit visualization endpoint
    Provides interactive Plotly dashboards for MCP audit monitoring
    """
    
    def __init__(self, app: Flask, audit_data_source):
        self.app = app
        self.audit_data_source = audit_data_source
        
        # Register routes
        self._register_routes()
        
        logger.info("Audit visualization endpoint initialized")
    
    def _register_routes(self):
        """Register Flask routes for audit visualization"""
        
        @self.app.route('/audit-viz')
        def audit_dashboard():
            """Main audit visualization dashboard"""
            return self._render_audit_dashboard()
        
        @self.app.route('/api/audit-data')
        def get_audit_data():
            """API endpoint for audit data"""
            return self._get_audit_data_api()
        
        @self.app.route('/api/audit-heatmap')
        def get_audit_heatmap():
            """API endpoint for audit heatmap"""
            return self._get_audit_heatmap_api()
        
        @self.app.route('/api/gaming-trends')
        def get_gaming_trends():
            """API endpoint for gaming trend analysis"""
            return self._get_gaming_trends_api()
        
        @self.app.route('/api/hash-verification')
        def get_hash_verification():
            """API endpoint for hash verification status"""
            return self._get_hash_verification_api()
    
    def _render_audit_dashboard(self) -> str:
        """Render the main audit dashboard HTML"""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Observer MCP Audit Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
                .chart-container { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .full-width { grid-column: 1 / -1; }
                .stats-panel { background-color: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
                .stat-item { display: inline-block; margin: 10px 20px; }
                .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .stat-label { font-size: 14px; color: #7f8c8d; }
                .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
                .status-good { background-color: #27ae60; }
                .status-warning { background-color: #f39c12; }
                .status-error { background-color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Observer MCP Audit Dashboard</h1>
                <p>Real-time monitoring of MCP usage, gaming detection, and enforcement effectiveness</p>
            </div>
            
            <div class="stats-panel" id="statsPanel">
                <div class="stat-item">
                    <div class="stat-value" id="totalAudits">-</div>
                    <div class="stat-label">Total Audits</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="successRate">-</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="gamingRate">-</div>
                    <div class="stat-label">Gaming Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="enforcementRate">-</div>
                    <div class="stat-label">Enforcement Rate</div>
                </div>
                <div class="stat-item">
                    <span class="status-indicator" id="systemStatus"></span>
                    <span id="systemStatusText">System Status</span>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="chart-container">
                    <h3>MCP Usage Over Generations</h3>
                    <div id="usageChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Gaming Detection Heatmap</h3>
                    <div id="heatmapChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Appropriateness Trends</h3>
                    <div id="appropriatenessChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Hash Verification Status</h3>
                    <div id="hashChart"></div>
                </div>
                
                <div class="chart-container full-width">
                    <h3>Real-time Gaming Trends</h3>
                    <div id="gamingTrendsChart"></div>
                </div>
            </div>
            
            <script>
                // Auto-refresh dashboard every 30 seconds
                setInterval(updateDashboard, 30000);
                
                // Initial load
                updateDashboard();
                
                async function updateDashboard() {
                    try {
                        // Fetch audit data
                        const auditResponse = await fetch('/api/audit-data');
                        const auditData = await auditResponse.json();
                        
                        // Update stats
                        updateStats(auditData);
                        
                        // Update charts
                        await updateUsageChart();
                        await updateHeatmapChart();
                        await updateAppropriatenessChart();
                        await updateHashChart();
                        await updateGamingTrendsChart();
                        
                    } catch (error) {
                        console.error('Dashboard update failed:', error);
                    }
                }
                
                function updateStats(data) {
                    document.getElementById('totalAudits').textContent = data.total_mcp_audits || 0;
                    document.getElementById('successRate').textContent = ((data.success_rate || 0) * 100).toFixed(1) + '%';
                    document.getElementById('gamingRate').textContent = ((data.gaming_rate || 0) * 100).toFixed(1) + '%';
                    document.getElementById('enforcementRate').textContent = ((data.enforcement_effectiveness || 0) * 100).toFixed(1) + '%';
                    
                    // Update system status
                    const statusIndicator = document.getElementById('systemStatus');
                    const statusText = document.getElementById('systemStatusText');
                    
                    if (data.enforcement_effectiveness >= 0.95) {
                        statusIndicator.className = 'status-indicator status-good';
                        statusText.textContent = 'Optimal';
                    } else if (data.enforcement_effectiveness >= 0.8) {
                        statusIndicator.className = 'status-indicator status-warning';
                        statusText.textContent = 'Good';
                    } else {
                        statusIndicator.className = 'status-indicator status-error';
                        statusText.textContent = 'Needs Attention';
                    }
                }
                
                async function updateUsageChart() {
                    const response = await fetch('/api/audit-heatmap');
                    const data = await response.json();
                    
                    if (data.generations) {
                        const trace1 = {
                            x: data.generations,
                            y: data.total_calls,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Total MCP Calls',
                            line: { color: '#3498db' }
                        };
                        
                        const trace2 = {
                            x: data.generations,
                            y: data.success_rates.map(rate => rate * Math.max(...data.total_calls)),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Success Rate (scaled)',
                            line: { color: '#27ae60' },
                            yaxis: 'y2'
                        };
                        
                        const layout = {
                            title: 'MCP Usage Trends',
                            xaxis: { title: 'Generation' },
                            yaxis: { title: 'MCP Calls' },
                            yaxis2: {
                                title: 'Success Rate',
                                overlaying: 'y',
                                side: 'right'
                            }
                        };
                        
                        Plotly.newPlot('usageChart', [trace1, trace2], layout);
                    }
                }
                
                async function updateHeatmapChart() {
                    const response = await fetch('/api/audit-heatmap');
                    const data = await response.json();
                    
                    if (data.generations) {
                        const trace = {
                            x: data.generations,
                            y: ['Gaming Rate'],
                            z: [data.gaming_rates],
                            type: 'heatmap',
                            colorscale: [
                                [0, '#27ae60'],
                                [0.5, '#f39c12'],
                                [1, '#e74c3c']
                            ]
                        };
                        
                        const layout = {
                            title: 'Gaming Detection Heatmap',
                            xaxis: { title: 'Generation' }
                        };
                        
                        Plotly.newPlot('heatmapChart', [trace], layout);
                    }
                }
                
                async function updateAppropriatenessChart() {
                    const response = await fetch('/api/audit-heatmap');
                    const data = await response.json();
                    
                    if (data.generations) {
                        const trace = {
                            x: data.generations,
                            y: data.appropriateness_scores,
                            type: 'scatter',
                            mode: 'lines+markers',
                            fill: 'tonexty',
                            name: 'Appropriateness Score',
                            line: { color: '#9b59b6' }
                        };
                        
                        const layout = {
                            title: 'Context Appropriateness Trends',
                            xaxis: { title: 'Generation' },
                            yaxis: { title: 'Appropriateness Score', range: [0, 1] }
                        };
                        
                        Plotly.newPlot('appropriatenessChart', [trace], layout);
                    }
                }
                
                async function updateHashChart() {
                    const response = await fetch('/api/hash-verification');
                    const data = await response.json();
                    
                    const trace = {
                        values: [data.verified_hashes || 0, data.pending_verification || 0, data.failed_verification || 0],
                        labels: ['Verified', 'Pending', 'Failed'],
                        type: 'pie',
                        marker: {
                            colors: ['#27ae60', '#f39c12', '#e74c3c']
                        }
                    };
                    
                    const layout = {
                        title: 'Hash Verification Status'
                    };
                    
                    Plotly.newPlot('hashChart', [trace], layout);
                }
                
                async function updateGamingTrendsChart() {
                    const response = await fetch('/api/gaming-trends');
                    const data = await response.json();
                    
                    if (data.timestamps) {
                        const trace1 = {
                            x: data.timestamps,
                            y: data.gaming_probabilities,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Gaming Probability',
                            line: { color: '#e74c3c' }
                        };
                        
                        const trace2 = {
                            x: data.timestamps,
                            y: data.enforcement_actions,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Enforcement Actions',
                            line: { color: '#3498db' },
                            yaxis: 'y2'
                        };
                        
                        const layout = {
                            title: 'Real-time Gaming Detection & Enforcement',
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Gaming Probability' },
                            yaxis2: {
                                title: 'Enforcement Actions',
                                overlaying: 'y',
                                side: 'right'
                            }
                        };
                        
                        Plotly.newPlot('gamingTrendsChart', [trace1, trace2], layout);
                    }
                }
            </script>
        </body>
        </html>
        """
        return dashboard_html
    
    def _get_audit_data_api(self) -> Dict[str, Any]:
        """Get audit data for API"""
        try:
            # Get audit summary from data source
            audit_summary = self.audit_data_source.get_mcp_audit_summary()
            
            return jsonify(audit_summary)
            
        except Exception as e:
            logger.error(f"Audit data API failed: {e}")
            return jsonify({"error": str(e)})
    
    def _get_audit_heatmap_api(self) -> Dict[str, Any]:
        """Get audit heatmap data for API"""
        try:
            # Get heatmap data from data source
            heatmap_data = self.audit_data_source.generate_audit_heatmap_data()
            
            return jsonify(heatmap_data)
            
        except Exception as e:
            logger.error(f"Audit heatmap API failed: {e}")
            return jsonify({"error": str(e)})
    
    def _get_gaming_trends_api(self) -> Dict[str, Any]:
        """Get gaming trends data for API"""
        try:
            # Simulate gaming trends data (would come from gaming predictor)
            current_time = datetime.now()
            timestamps = [(current_time - timedelta(hours=i)).isoformat() for i in range(24, 0, -1)]
            
            # Mock data - in real implementation, this would come from GamingPredictor
            gaming_probabilities = [0.1 + (i % 5) * 0.1 for i in range(24)]
            enforcement_actions = [1 if prob > 0.3 else 0 for prob in gaming_probabilities]
            
            return jsonify({
                'timestamps': timestamps,
                'gaming_probabilities': gaming_probabilities,
                'enforcement_actions': enforcement_actions
            })
            
        except Exception as e:
            logger.error(f"Gaming trends API failed: {e}")
            return jsonify({"error": str(e)})
    
    def _get_hash_verification_api(self) -> Dict[str, Any]:
        """Get hash verification status for API"""
        try:
            # Mock hash verification data - in real implementation, this would come from audit system
            return jsonify({
                'verified_hashes': 85,
                'pending_verification': 10,
                'failed_verification': 5,
                'total_hashes': 100,
                'verification_rate': 0.85
            })
            
        except Exception as e:
            logger.error(f"Hash verification API failed: {e}")
            return jsonify({"error": str(e)})


def create_audit_viz_app(audit_data_source) -> Flask:
    """Create Flask app with audit visualization endpoint"""
    app = Flask(__name__)
    
    # Initialize audit visualization endpoint
    audit_viz = AuditVizEndpoint(app, audit_data_source)
    
    return app


# Example usage
if __name__ == "__main__":
    # Mock audit data source for testing
    class MockAuditDataSource:
        def get_mcp_audit_summary(self):
            return {
                'total_mcp_audits': 150,
                'successful_audits': 135,
                'gaming_audits': 15,
                'success_rate': 0.9,
                'gaming_rate': 0.1,
                'enforcement_effectiveness': 0.95
            }
        
        def generate_audit_heatmap_data(self):
            return {
                'generations': list(range(1, 11)),
                'total_calls': [10 + i * 2 for i in range(10)],
                'success_rates': [0.8 + i * 0.01 for i in range(10)],
                'gaming_rates': [0.2 - i * 0.01 for i in range(10)],
                'appropriateness_scores': [0.6 + i * 0.03 for i in range(10)]
            }
    
    # Create and run app
    mock_source = MockAuditDataSource()
    app = create_audit_viz_app(mock_source)
    
    print("🎯 Observer MCP Audit Dashboard starting...")
    print("📊 Access dashboard at: http://localhost:5000/audit-viz")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
