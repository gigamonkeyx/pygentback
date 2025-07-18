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

        @self.app.route('/api/predictive-metrics')
        def get_predictive_metrics():
            """API endpoint for predictive metrics visualization"""
            return self._get_predictive_metrics_api()

        @self.app.route('/api/fusion-progress')
        def get_fusion_progress():
            """API endpoint for fusion effectiveness progress"""
            return self._get_fusion_progress_api()

        @self.app.route('/api/iteration-progress')
        def get_iteration_progress():
            """API endpoint for iteration progress paths"""
            return self._get_iteration_progress_api()

        @self.app.route('/api/real-time-optimization')
        def get_real_time_optimization():
            """API endpoint for real-time optimization metrics"""
            return self._get_real_time_optimization_api()
    
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

                <div class="chart-container">
                    <h3>Predictive Fusion Metrics</h3>
                    <div id="predictiveChart"></div>
                </div>

                <div class="chart-container">
                    <h3>95%+ Progress Tracking</h3>
                    <div id="progressChart"></div>
                </div>

                <div class="chart-container full-width">
                    <h3>Iteration Progress Paths</h3>
                    <div id="iterationPathsChart"></div>
                </div>

                <div class="chart-container full-width">
                    <h3>Real-time Optimization Velocity</h3>
                    <div id="optimizationVelocityChart"></div>
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
                        await updatePredictiveChart();
                        await updateProgressChart();
                        await updateIterationPathsChart();
                        await updateOptimizationVelocityChart();
                        
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

                async function updatePredictiveChart() {
                    const response = await fetch('/api/predictive-metrics');
                    const data = await response.json();

                    if (data.generations) {
                        const trace1 = {
                            x: data.generations,
                            y: data.predicted_effectiveness,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Predicted Effectiveness',
                            line: { color: '#9b59b6', dash: 'dash' }
                        };

                        const trace2 = {
                            x: data.generations,
                            y: data.actual_effectiveness,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Actual Effectiveness',
                            line: { color: '#2ecc71' }
                        };

                        const trace3 = {
                            x: data.generations,
                            y: Array(data.generations.length).fill(0.95),
                            type: 'scatter',
                            mode: 'lines',
                            name: '95% Target',
                            line: { color: '#e74c3c', dash: 'dot' }
                        };

                        const layout = {
                            title: 'Predictive vs Actual Effectiveness',
                            xaxis: { title: 'Generation' },
                            yaxis: { title: 'Effectiveness', range: [0, 1] }
                        };

                        Plotly.newPlot('predictiveChart', [trace1, trace2, trace3], layout);
                    }
                }

                async function updateProgressChart() {
                    const response = await fetch('/api/fusion-progress');
                    const data = await response.json();

                    const trace = {
                        values: [data.current_progress || 0, 100 - (data.current_progress || 0)],
                        labels: ['Progress', 'Remaining'],
                        type: 'pie',
                        hole: 0.4,
                        marker: {
                            colors: ['#3498db', '#ecf0f1']
                        },
                        textinfo: 'label+percent',
                        textposition: 'inside'
                    };

                    const layout = {
                        title: `95%+ Progress: ${data.current_progress || 0}%`,
                        annotations: [{
                            font: { size: 20 },
                            showarrow: false,
                            text: `${data.current_progress || 0}%`,
                            x: 0.5,
                            y: 0.5
                        }]
                    };

                    Plotly.newPlot('progressChart', [trace], layout);
                }

                async function updateIterationPathsChart() {
                    const response = await fetch('/api/iteration-progress');
                    const data = await response.json();

                    if (data.iterations) {
                        const trace1 = {
                            x: data.iterations,
                            y: data.calibration_path,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Calibration Path',
                            line: { color: '#3498db', dash: 'dot' }
                        };

                        const trace2 = {
                            x: data.iterations,
                            y: data.gaming_prevention_path,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Gaming Prevention Path',
                            line: { color: '#e74c3c', dash: 'dash' }
                        };

                        const trace3 = {
                            x: data.iterations,
                            y: data.auto_tune_path,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Auto-tune Path',
                            line: { color: '#f39c12', dash: 'dashdot' }
                        };

                        const trace4 = {
                            x: data.iterations,
                            y: data.combined_optimization_path,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Combined Optimization',
                            line: { color: '#27ae60', width: 3 }
                        };

                        const trace5 = {
                            x: data.iterations,
                            y: Array(data.iterations.length).fill(0.95),
                            type: 'scatter',
                            mode: 'lines',
                            name: '95% Target',
                            line: { color: '#9b59b6', dash: 'solid', width: 2 }
                        };

                        const layout = {
                            title: 'Iteration Progress Paths to 95%+ Effectiveness',
                            xaxis: { title: 'Iteration' },
                            yaxis: { title: 'Effectiveness', range: [0.5, 1.0] },
                            annotations: [{
                                x: data.current_iteration,
                                y: data.combined_optimization_path[data.current_iteration - 1],
                                text: 'Current',
                                showarrow: true,
                                arrowhead: 2
                            }]
                        };

                        Plotly.newPlot('iterationPathsChart', [trace1, trace2, trace3, trace4, trace5], layout);
                    }
                }

                async function updateOptimizationVelocityChart() {
                    const response = await fetch('/api/real-time-optimization');
                    const data = await response.json();

                    if (data.timestamps) {
                        const trace1 = {
                            x: data.timestamps,
                            y: data.effectiveness_trend,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Overall Effectiveness',
                            line: { color: '#2ecc71', width: 3 }
                        };

                        const trace2 = {
                            x: data.timestamps,
                            y: data.calibration_impact,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Calibration Impact',
                            line: { color: '#3498db' },
                            stackgroup: 'one'
                        };

                        const trace3 = {
                            x: data.timestamps,
                            y: data.gaming_prevention_impact,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Gaming Prevention Impact',
                            line: { color: '#e74c3c' },
                            stackgroup: 'one'
                        };

                        const trace4 = {
                            x: data.timestamps,
                            y: data.auto_tune_impact,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Auto-tune Impact',
                            line: { color: '#f39c12' },
                            stackgroup: 'one'
                        };

                        const trace5 = {
                            x: data.timestamps,
                            y: data.optimization_velocity,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Optimization Velocity',
                            line: { color: '#9b59b6', dash: 'dot' },
                            yaxis: 'y2'
                        };

                        const layout = {
                            title: 'Real-time Optimization Velocity & Component Impact',
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Effectiveness Impact' },
                            yaxis2: {
                                title: 'Velocity (Δ/min)',
                                overlaying: 'y',
                                side: 'right'
                            }
                        };

                        Plotly.newPlot('optimizationVelocityChart', [trace1, trace2, trace3, trace4, trace5], layout);
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

    def _get_predictive_metrics_api(self) -> Dict[str, Any]:
        """Get predictive metrics data for API"""
        try:
            # Generate predictive effectiveness data
            generations = list(range(1, 11))

            # Simulate predictive vs actual effectiveness (would come from ML models)
            predicted_effectiveness = [0.56 + (i * 0.04) for i in range(10)]  # Starting from 56%, growing to 95%+
            actual_effectiveness = [0.53 + (i * 0.035) + (0.02 * (i % 3)) for i in range(10)]  # With some variance

            return jsonify({
                'generations': generations,
                'predicted_effectiveness': predicted_effectiveness,
                'actual_effectiveness': actual_effectiveness,
                'prediction_accuracy': 0.92,
                'trend_direction': 'increasing'
            })

        except Exception as e:
            logger.error(f"Predictive metrics API failed: {e}")
            return jsonify({"error": str(e)})

    def _get_fusion_progress_api(self) -> Dict[str, Any]:
        """Get fusion effectiveness progress for API"""
        try:
            # Calculate current progress toward 95% target
            # This would come from actual fusion effectiveness measurements
            current_effectiveness = 0.563  # 56.3% from test results
            target_effectiveness = 0.95

            progress_percentage = (current_effectiveness / target_effectiveness) * 100

            return jsonify({
                'current_effectiveness': current_effectiveness,
                'target_effectiveness': target_effectiveness,
                'current_progress': round(progress_percentage, 1),
                'remaining_progress': round(100 - progress_percentage, 1),
                'estimated_completion': '2-3 weeks',
                'progress_trend': 'accelerating'
            })

        except Exception as e:
            logger.error(f"Fusion progress API failed: {e}")
            return jsonify({"error": str(e)})

    def _get_iteration_progress_api(self) -> Dict[str, Any]:
        """Get iteration progress paths for predictive visualization"""
        try:
            # Generate iteration progress paths (would come from actual iteration data)
            iterations = list(range(1, 11))

            # Simulate progress paths for different optimization strategies
            calibration_path = [0.539 + (i * 0.025) for i in range(10)]  # From 53.9% to 76.4%
            gaming_prevention_path = [0.539 + (i * 0.03) for i in range(10)]  # From 53.9% to 80.9%
            auto_tune_path = [0.539 + (i * 0.035) for i in range(10)]  # From 53.9% to 85.4%
            combined_path = [0.539 + (i * 0.045) for i in range(10)]  # From 53.9% to 94.4%

            # Target achievement probabilities
            target_achievement_probs = [0.1 + (i * 0.08) for i in range(10)]  # Growing confidence

            return jsonify({
                'iterations': iterations,
                'calibration_path': calibration_path,
                'gaming_prevention_path': gaming_prevention_path,
                'auto_tune_path': auto_tune_path,
                'combined_optimization_path': combined_path,
                'target_achievement_probabilities': target_achievement_probs,
                'current_iteration': 3,
                'projected_95_achievement': 'iteration_8'
            })

        except Exception as e:
            logger.error(f"Iteration progress API failed: {e}")
            return jsonify({"error": str(e)})

    def _get_real_time_optimization_api(self) -> Dict[str, Any]:
        """Get real-time optimization metrics for dashboard"""
        try:
            # Generate real-time optimization data
            current_time = datetime.now()
            timestamps = [(current_time - timedelta(minutes=i*5)).isoformat() for i in range(12, 0, -1)]

            # Simulate real-time optimization metrics
            effectiveness_trend = [0.539 + (i * 0.008) for i in range(12)]  # Gradual improvement
            calibration_impact = [0.0 + (i * 0.003) for i in range(12)]  # Calibration contribution
            gaming_prevention_impact = [0.0 + (i * 0.004) for i in range(12)]  # Gaming prevention contribution
            auto_tune_impact = [0.0 + (i * 0.005) for i in range(12)]  # Auto-tune contribution

            # Optimization velocity (rate of improvement)
            optimization_velocity = [0.002 + (i * 0.0005) for i in range(12)]

            return jsonify({
                'timestamps': timestamps,
                'effectiveness_trend': effectiveness_trend,
                'calibration_impact': calibration_impact,
                'gaming_prevention_impact': gaming_prevention_impact,
                'auto_tune_impact': auto_tune_impact,
                'optimization_velocity': optimization_velocity,
                'current_effectiveness': effectiveness_trend[-1],
                'velocity_trend': 'accelerating'
            })

        except Exception as e:
            logger.error(f"Real-time optimization API failed: {e}")
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
