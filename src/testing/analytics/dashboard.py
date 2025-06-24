"""
Performance Dashboard

Real-time performance dashboard for recipe testing with metrics visualization,
monitoring capabilities, and interactive reporting.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard metrics data"""
    timestamp: datetime
    total_tests: int
    successful_tests: int
    failed_tests: int
    average_execution_time: float
    success_rate: float
    throughput_per_hour: float
    active_recipes: int
    system_health_score: float


@dataclass
class PerformanceWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str
    title: str
    data_source: str
    refresh_interval: int = 30
    config: Dict[str, Any] = field(default_factory=dict)


class PerformanceDashboard:
    """
    Performance Dashboard System.
    
    Provides real-time monitoring and visualization of recipe testing
    performance with customizable widgets and reporting capabilities.
    """
    
    def __init__(self, update_interval: float = 30.0):
        self.update_interval = update_interval
        
        # Dashboard state
        self.is_running = False
        self.update_task: Optional[asyncio.Task] = None
        
        # Metrics storage
        self.current_metrics: Optional[DashboardMetrics] = None
        self.metrics_history: List[DashboardMetrics] = []
        self.max_history_size = 1000
        
        # Widgets
        self.widgets: Dict[str, PerformanceWidget] = {}
        self.widget_data: Dict[str, Any] = {}
        
        # Data sources
        self.data_sources: Dict[str, Any] = {}
        self.data_collectors: Dict[str, callable] = {}
        
        # Dashboard configuration
        self.config = {
            'theme': 'light',
            'auto_refresh': True,
            'show_alerts': True,
            'chart_animation': True,
            'export_enabled': True
        }
        
        # Initialize default widgets
        self._initialize_default_widgets()
    
    async def start(self):
        """Start the dashboard"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Performance dashboard started")
    
    async def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
        logger.info("Performance dashboard stopped")
    
    def register_data_source(self, source_name: str, data_collector: callable):
        """Register a data source for the dashboard"""
        self.data_collectors[source_name] = data_collector
        logger.debug(f"Registered data source: {source_name}")
    
    def add_widget(self, widget: PerformanceWidget):
        """Add a widget to the dashboard"""
        self.widgets[widget.widget_id] = widget
        self.widget_data[widget.widget_id] = {}
        logger.debug(f"Added widget: {widget.title}")
    
    def remove_widget(self, widget_id: str):
        """Remove a widget from the dashboard"""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
        if widget_id in self.widget_data:
            del self.widget_data[widget_id]
        logger.debug(f"Removed widget: {widget_id}")
    
    async def _update_loop(self):
        """Main dashboard update loop"""
        while self.is_running:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Update widgets
                await self._update_widgets()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_metrics(self):
        """Collect current metrics"""
        try:
            # Collect data from all sources
            collected_data = {}
            for source_name, collector in self.data_collectors.items():
                try:
                    if asyncio.iscoroutinefunction(collector):
                        data = await collector()
                    else:
                        data = collector()
                    collected_data[source_name] = data
                except Exception as e:
                    logger.error(f"Data collection error for {source_name}: {e}")
                    collected_data[source_name] = {}
            
            # Create dashboard metrics
            metrics = self._create_dashboard_metrics(collected_data)
            
            # Store metrics
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Trim history
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size//2:]
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    def _create_dashboard_metrics(self, collected_data: Dict[str, Any]) -> DashboardMetrics:
        """Create dashboard metrics from collected data"""
        # Extract metrics from collected data
        test_data = collected_data.get('test_results', {})
        system_data = collected_data.get('system_health', {})
        
        total_tests = test_data.get('total_tests', 0)
        successful_tests = test_data.get('successful_tests', 0)
        failed_tests = total_tests - successful_tests
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        avg_execution_time = test_data.get('average_execution_time', 0.0)
        
        # Calculate throughput (tests per hour)
        if len(self.metrics_history) > 1:
            time_diff = (datetime.utcnow() - self.metrics_history[-1].timestamp).total_seconds()
            if time_diff > 0:
                test_diff = total_tests - self.metrics_history[-1].total_tests
                throughput_per_hour = (test_diff / time_diff) * 3600
            else:
                throughput_per_hour = 0.0
        else:
            throughput_per_hour = 0.0
        
        return DashboardMetrics(
            timestamp=datetime.utcnow(),
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            average_execution_time=avg_execution_time,
            success_rate=success_rate,
            throughput_per_hour=throughput_per_hour,
            active_recipes=test_data.get('active_recipes', 0),
            system_health_score=system_data.get('health_score', 1.0)
        )
    
    async def _update_widgets(self):
        """Update all dashboard widgets"""
        for widget_id, widget in self.widgets.items():
            try:
                # Get data for widget
                widget_data = await self._get_widget_data(widget)
                self.widget_data[widget_id] = widget_data
                
            except Exception as e:
                logger.error(f"Widget update error for {widget_id}: {e}")
    
    async def _get_widget_data(self, widget: PerformanceWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        if widget.data_source in self.data_sources:
            return self.data_sources[widget.data_source]
        
        # Generate widget-specific data based on type
        if widget.widget_type == "line_chart":
            return self._get_line_chart_data(widget)
        elif widget.widget_type == "bar_chart":
            return self._get_bar_chart_data(widget)
        elif widget.widget_type == "gauge":
            return self._get_gauge_data(widget)
        elif widget.widget_type == "table":
            return self._get_table_data(widget)
        elif widget.widget_type == "metric":
            return self._get_metric_data(widget)
        else:
            return {}
    
    def _get_line_chart_data(self, widget: PerformanceWidget) -> Dict[str, Any]:
        """Get line chart data"""
        if not self.metrics_history:
            return {'labels': [], 'datasets': []}
        
        # Get recent metrics
        recent_metrics = self.metrics_history[-50:]  # Last 50 data points
        
        labels = [m.timestamp.strftime('%H:%M:%S') for m in recent_metrics]
        
        datasets = []
        
        if 'success_rate' in widget.config.get('metrics', []):
            datasets.append({
                'label': 'Success Rate (%)',
                'data': [m.success_rate * 100 for m in recent_metrics],
                'borderColor': 'rgb(75, 192, 192)',
                'tension': 0.1
            })
        
        if 'execution_time' in widget.config.get('metrics', []):
            datasets.append({
                'label': 'Avg Execution Time (s)',
                'data': [m.average_execution_time for m in recent_metrics],
                'borderColor': 'rgb(255, 99, 132)',
                'tension': 0.1
            })
        
        if 'throughput' in widget.config.get('metrics', []):
            datasets.append({
                'label': 'Throughput (tests/hour)',
                'data': [m.throughput_per_hour for m in recent_metrics],
                'borderColor': 'rgb(54, 162, 235)',
                'tension': 0.1
            })
        
        return {'labels': labels, 'datasets': datasets}
    
    def _get_bar_chart_data(self, widget: PerformanceWidget) -> Dict[str, Any]:
        """Get bar chart data"""
        if not self.current_metrics:
            return {'labels': [], 'data': []}
        
        return {
            'labels': ['Successful', 'Failed'],
            'data': [self.current_metrics.successful_tests, self.current_metrics.failed_tests],
            'backgroundColor': ['rgb(75, 192, 192)', 'rgb(255, 99, 132)']
        }
    
    def _get_gauge_data(self, widget: PerformanceWidget) -> Dict[str, Any]:
        """Get gauge data"""
        if not self.current_metrics:
            return {'value': 0, 'max': 100}
        
        metric_name = widget.config.get('metric', 'success_rate')
        
        if metric_name == 'success_rate':
            return {
                'value': self.current_metrics.success_rate * 100,
                'max': 100,
                'unit': '%',
                'color': 'green' if self.current_metrics.success_rate > 0.8 else 'orange'
            }
        elif metric_name == 'system_health':
            return {
                'value': self.current_metrics.system_health_score * 100,
                'max': 100,
                'unit': '%',
                'color': 'green' if self.current_metrics.system_health_score > 0.8 else 'red'
            }
        else:
            return {'value': 0, 'max': 100}
    
    def _get_table_data(self, widget: PerformanceWidget) -> Dict[str, Any]:
        """Get table data"""
        if not self.current_metrics:
            return {'headers': [], 'rows': []}
        
        headers = ['Metric', 'Value', 'Status']
        rows = [
            ['Total Tests', str(self.current_metrics.total_tests), 'info'],
            ['Success Rate', f"{self.current_metrics.success_rate:.1%}", 
             'success' if self.current_metrics.success_rate > 0.8 else 'warning'],
            ['Avg Execution Time', f"{self.current_metrics.average_execution_time:.2f}s", 'info'],
            ['Throughput', f"{self.current_metrics.throughput_per_hour:.1f}/hour", 'info'],
            ['System Health', f"{self.current_metrics.system_health_score:.1%}", 
             'success' if self.current_metrics.system_health_score > 0.8 else 'danger']
        ]
        
        return {'headers': headers, 'rows': rows}
    
    def _get_metric_data(self, widget: PerformanceWidget) -> Dict[str, Any]:
        """Get single metric data"""
        if not self.current_metrics:
            return {'value': 0, 'label': 'No Data'}
        
        metric_name = widget.config.get('metric', 'total_tests')
        
        if metric_name == 'total_tests':
            return {
                'value': self.current_metrics.total_tests,
                'label': 'Total Tests',
                'trend': self._calculate_trend('total_tests')
            }
        elif metric_name == 'success_rate':
            return {
                'value': f"{self.current_metrics.success_rate:.1%}",
                'label': 'Success Rate',
                'trend': self._calculate_trend('success_rate')
            }
        elif metric_name == 'throughput':
            return {
                'value': f"{self.current_metrics.throughput_per_hour:.1f}",
                'label': 'Tests/Hour',
                'trend': self._calculate_trend('throughput_per_hour')
            }
        else:
            return {'value': 0, 'label': 'Unknown Metric'}
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric"""
        if len(self.metrics_history) < 2:
            return 'stable'
        
        recent_values = [getattr(m, metric_name) for m in self.metrics_history[-10:]]
        
        if len(recent_values) < 2:
            return 'stable'
        
        # Simple trend calculation
        first_half = statistics.mean(recent_values[:len(recent_values)//2])
        second_half = statistics.mean(recent_values[len(recent_values)//2:])
        
        if second_half > first_half * 1.05:
            return 'up'
        elif second_half < first_half * 0.95:
            return 'down'
        else:
            return 'stable'
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        # Keep only recent metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    def _initialize_default_widgets(self):
        """Initialize default dashboard widgets"""
        # Performance trend chart
        self.add_widget(PerformanceWidget(
            widget_id="performance_trend",
            widget_type="line_chart",
            title="Performance Trends",
            data_source="metrics",
            config={'metrics': ['success_rate', 'execution_time']}
        ))
        
        # Test results bar chart
        self.add_widget(PerformanceWidget(
            widget_id="test_results",
            widget_type="bar_chart",
            title="Test Results",
            data_source="metrics"
        ))
        
        # Success rate gauge
        self.add_widget(PerformanceWidget(
            widget_id="success_rate_gauge",
            widget_type="gauge",
            title="Success Rate",
            data_source="metrics",
            config={'metric': 'success_rate'}
        ))
        
        # Metrics table
        self.add_widget(PerformanceWidget(
            widget_id="metrics_table",
            widget_type="table",
            title="Current Metrics",
            data_source="metrics"
        ))
        
        # Total tests metric
        self.add_widget(PerformanceWidget(
            widget_id="total_tests_metric",
            widget_type="metric",
            title="Total Tests",
            data_source="metrics",
            config={'metric': 'total_tests'}
        ))
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            'current_metrics': self.current_metrics.__dict__ if self.current_metrics else {},
            'widgets': {
                widget_id: {
                    'config': widget.__dict__,
                    'data': self.widget_data.get(widget_id, {})
                }
                for widget_id, widget in self.widgets.items()
            },
            'config': self.config,
            'last_updated': datetime.utcnow().isoformat(),
            'is_running': self.is_running
        }
    
    def export_data(self, format: str = "json", time_range_hours: int = 24) -> str:
        """Export dashboard data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        export_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'time_range_hours': time_range_hours,
            'metrics_count': len(export_metrics),
            'metrics': [m.__dict__ for m in export_metrics],
            'dashboard_config': self.config
        }
        
        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_summary_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        success_rates = [m.success_rate for m in recent_metrics]
        execution_times = [m.average_execution_time for m in recent_metrics]
        throughputs = [m.throughput_per_hour for m in recent_metrics]
        
        return {
            'time_period_hours': hours,
            'data_points': len(recent_metrics),
            'success_rate': {
                'avg': statistics.mean(success_rates),
                'min': min(success_rates),
                'max': max(success_rates),
                'std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0
            },
            'execution_time': {
                'avg': statistics.mean(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            },
            'throughput': {
                'avg': statistics.mean(throughputs),
                'min': min(throughputs),
                'max': max(throughputs),
                'std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
            }
        }
