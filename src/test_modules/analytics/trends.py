"""
Trend Analysis System

Advanced trend analysis for recipe testing performance with time-series analysis,
pattern detection, forecasting, and anomaly detection capabilities.
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction indicators"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class TrendStrength(Enum):
    """Trend strength indicators"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TrendPoint:
    """Single data point for trend analysis"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Result of trend analysis"""
    metric_name: str
    direction: TrendDirection
    strength: TrendStrength
    slope: float
    correlation: float
    confidence: float
    data_points: int
    time_span_hours: float
    start_value: float
    end_value: float
    change_percentage: float
    volatility: float
    anomalies_detected: int


@dataclass
class SeasonalPattern:
    """Detected seasonal pattern"""
    pattern_type: str  # daily, weekly, monthly
    period_hours: float
    amplitude: float
    phase_offset: float
    confidence: float


@dataclass
class Forecast:
    """Trend forecast"""
    metric_name: str
    forecast_horizon_hours: float
    predicted_values: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]]
    forecast_accuracy: float
    model_type: str


class TrendAnalyzer:
    """
    Advanced Trend Analysis System.
    
    Analyzes time-series data for trends, patterns, seasonality,
    and provides forecasting capabilities for recipe testing metrics.
    """
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
        
        # Data storage
        self.metric_data: Dict[str, List[TrendPoint]] = {}
        self.trend_history: Dict[str, List[TrendAnalysis]] = {}
        
        # Analysis configuration
        self.trend_window_hours = 24
        self.volatility_threshold = 0.2
        self.anomaly_threshold = 2.0  # Standard deviations
        self.confidence_threshold = 0.7
        
        # Pattern detection
        self.seasonal_patterns: Dict[str, List[SeasonalPattern]] = {}
        self.detected_anomalies: Dict[str, List[TrendPoint]] = {}
    
    def add_data_point(self, metric_name: str, value: float, 
                      timestamp: Optional[datetime] = None,
                      metadata: Dict[str, Any] = None):
        """Add a data point for trend analysis"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        point = TrendPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata or {}
        )
        
        if metric_name not in self.metric_data:
            self.metric_data[metric_name] = []
        
        self.metric_data[metric_name].append(point)
        
        # Keep only recent data
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.metric_data[metric_name] = [
            p for p in self.metric_data[metric_name]
            if p.timestamp > cutoff_time
        ]
        
        # Sort by timestamp
        self.metric_data[metric_name].sort(key=lambda x: x.timestamp)
    
    def analyze_trend(self, metric_name: str, 
                     window_hours: Optional[float] = None) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric"""
        if metric_name not in self.metric_data:
            return None
        
        window_hours = window_hours or self.trend_window_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        
        # Get data points in window
        data_points = [
            p for p in self.metric_data[metric_name]
            if p.timestamp > cutoff_time
        ]
        
        if len(data_points) < self.min_data_points:
            return None
        
        # Calculate trend metrics
        values = [p.value for p in data_points]
        timestamps = [p.timestamp for p in data_points]
        
        # Convert timestamps to numeric values (hours from start)
        start_time = timestamps[0]
        time_values = [(t - start_time).total_seconds() / 3600 for t in timestamps]
        
        # Calculate linear regression
        slope, correlation = self._calculate_linear_regression(time_values, values)
        
        # Determine trend direction and strength
        direction = self._determine_trend_direction(slope, correlation)
        strength = self._determine_trend_strength(correlation)
        
        # Calculate additional metrics
        volatility = self._calculate_volatility(values)
        confidence = self._calculate_confidence(correlation, len(data_points))
        
        # Detect anomalies
        anomalies = self._detect_anomalies(data_points)
        
        # Calculate change percentage
        start_value = values[0]
        end_value = values[-1]
        change_percentage = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0.0
        
        analysis = TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            strength=strength,
            slope=slope,
            correlation=correlation,
            confidence=confidence,
            data_points=len(data_points),
            time_span_hours=window_hours,
            start_value=start_value,
            end_value=end_value,
            change_percentage=change_percentage,
            volatility=volatility,
            anomalies_detected=len(anomalies)
        )
        
        # Store analysis
        if metric_name not in self.trend_history:
            self.trend_history[metric_name] = []
        self.trend_history[metric_name].append(analysis)
        
        # Keep only recent analyses
        if len(self.trend_history[metric_name]) > 100:
            self.trend_history[metric_name] = self.trend_history[metric_name][-50:]
        
        return analysis
    
    def _calculate_linear_regression(self, x_values: List[float], 
                                   y_values: List[float]) -> Tuple[float, float]:
        """Calculate linear regression slope and correlation"""
        n = len(x_values)
        if n < 2:
            return 0.0, 0.0
        
        # Calculate means
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        # Calculate slope and correlation
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        y_variance = sum((y - y_mean) ** 2 for y in y_values)
        
        if x_variance == 0:
            return 0.0, 0.0
        
        slope = numerator / x_variance
        
        if y_variance == 0:
            correlation = 0.0
        else:
            correlation = numerator / math.sqrt(x_variance * y_variance)
        
        return slope, correlation
    
    def _determine_trend_direction(self, slope: float, correlation: float) -> TrendDirection:
        """Determine trend direction from slope and correlation"""
        if abs(correlation) < 0.3:
            return TrendDirection.STABLE
        elif abs(correlation) < 0.5 and abs(slope) > 0.1:
            return TrendDirection.VOLATILE
        elif slope > 0:
            return TrendDirection.INCREASING
        elif slope < 0:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE
    
    def _determine_trend_strength(self, correlation: float) -> TrendStrength:
        """Determine trend strength from correlation"""
        abs_correlation = abs(correlation)
        
        if abs_correlation >= 0.8:
            return TrendStrength.VERY_STRONG
        elif abs_correlation >= 0.6:
            return TrendStrength.STRONG
        elif abs_correlation >= 0.4:
            return TrendStrength.MODERATE
        else:
            return TrendStrength.WEAK
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (coefficient of variation)"""
        if len(values) < 2:
            return 0.0
        
        mean_value = statistics.mean(values)
        if mean_value == 0:
            return 0.0
        
        std_dev = statistics.stdev(values)
        return std_dev / abs(mean_value)
    
    def _calculate_confidence(self, correlation: float, data_points: int) -> float:
        """Calculate confidence in trend analysis"""
        # Base confidence on correlation strength and data points
        correlation_confidence = abs(correlation)
        data_confidence = min(1.0, data_points / 50.0)  # Full confidence at 50+ points
        
        return (correlation_confidence + data_confidence) / 2.0
    
    def _detect_anomalies(self, data_points: List[TrendPoint]) -> List[TrendPoint]:
        """Detect anomalies in data points"""
        if len(data_points) < 5:
            return []
        
        values = [p.value for p in data_points]
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if std_dev == 0:
            return []
        
        anomalies = []
        for point in data_points:
            z_score = abs(point.value - mean_value) / std_dev
            if z_score > self.anomaly_threshold:
                anomalies.append(point)
        
        return anomalies
    
    def detect_seasonal_patterns(self, metric_name: str) -> List[SeasonalPattern]:
        """Detect seasonal patterns in metric data"""
        if metric_name not in self.metric_data:
            return []
        
        data_points = self.metric_data[metric_name]
        if len(data_points) < 50:  # Need sufficient data for pattern detection
            return []
        
        patterns = []
        
        # Check for daily patterns (24-hour cycle)
        daily_pattern = self._detect_periodic_pattern(data_points, 24.0)
        if daily_pattern:
            patterns.append(daily_pattern)
        
        # Check for weekly patterns (168-hour cycle)
        weekly_pattern = self._detect_periodic_pattern(data_points, 168.0)
        if weekly_pattern:
            patterns.append(weekly_pattern)
        
        # Store detected patterns
        self.seasonal_patterns[metric_name] = patterns
        
        return patterns
    
    def _detect_periodic_pattern(self, data_points: List[TrendPoint], 
                                period_hours: float) -> Optional[SeasonalPattern]:
        """Detect a specific periodic pattern"""
        if len(data_points) < period_hours * 2:  # Need at least 2 cycles
            return None
        
        # Group data points by phase within the period
        phase_groups = {}
        start_time = data_points[0].timestamp
        
        for point in data_points:
            hours_from_start = (point.timestamp - start_time).total_seconds() / 3600
            phase = hours_from_start % period_hours
            phase_bucket = int(phase / (period_hours / 24))  # 24 buckets per period
            
            if phase_bucket not in phase_groups:
                phase_groups[phase_bucket] = []
            phase_groups[phase_bucket].append(point.value)
        
        # Calculate average value for each phase
        phase_averages = {}
        for phase, values in phase_groups.items():
            if len(values) >= 2:  # Need multiple samples
                phase_averages[phase] = statistics.mean(values)
        
        if len(phase_averages) < 12:  # Need sufficient phase coverage
            return None
        
        # Calculate pattern strength
        all_values = [avg for avg in phase_averages.values()]
        overall_mean = statistics.mean(all_values)
        amplitude = (max(all_values) - min(all_values)) / 2.0
        
        # Calculate confidence based on consistency
        variance = statistics.variance(all_values) if len(all_values) > 1 else 0.0
        confidence = amplitude / (math.sqrt(variance) + 0.001)  # Avoid division by zero
        confidence = min(1.0, confidence / 5.0)  # Normalize to 0-1
        
        if confidence < 0.3:  # Minimum confidence threshold
            return None
        
        return SeasonalPattern(
            pattern_type=f"{period_hours:.0f}h_cycle",
            period_hours=period_hours,
            amplitude=amplitude,
            phase_offset=0.0,  # Simplified - could calculate actual phase offset
            confidence=confidence
        )
    
    def forecast_trend(self, metric_name: str, 
                      forecast_hours: float = 24.0) -> Optional[Forecast]:
        """Generate trend forecast"""
        analysis = self.analyze_trend(metric_name)
        if not analysis or analysis.confidence < self.confidence_threshold:
            return None
        
        if metric_name not in self.metric_data:
            return None
        
        # Get recent data for forecasting
        recent_data = self.metric_data[metric_name][-50:]  # Last 50 points
        
        if len(recent_data) < self.min_data_points:
            return None
        
        # Simple linear extrapolation
        values = [p.value for p in recent_data]
        timestamps = [p.timestamp for p in recent_data]
        
        # Convert to hours from start
        start_time = timestamps[0]
        time_values = [(t - start_time).total_seconds() / 3600 for t in timestamps]
        
        slope, _ = self._calculate_linear_regression(time_values, values)
        
        # Generate forecast points
        last_time = timestamps[-1]
        last_value = values[-1]
        
        forecast_points = []
        confidence_intervals = []
        
        # Calculate prediction error for confidence intervals
        predicted_values = [last_value + slope * (t - time_values[-1]) for t in time_values]
        errors = [abs(actual - predicted) for actual, predicted in zip(values, predicted_values)]
        avg_error = statistics.mean(errors) if errors else 0.0
        
        for i in range(int(forecast_hours)):
            future_time = last_time + timedelta(hours=i + 1)
            predicted_value = last_value + slope * (i + 1)
            
            forecast_points.append((future_time, predicted_value))
            
            # Simple confidence interval based on historical error
            error_margin = avg_error * (1 + i * 0.1)  # Increasing uncertainty
            confidence_intervals.append((
                predicted_value - error_margin,
                predicted_value + error_margin
            ))
        
        return Forecast(
            metric_name=metric_name,
            forecast_horizon_hours=forecast_hours,
            predicted_values=forecast_points,
            confidence_intervals=confidence_intervals,
            forecast_accuracy=max(0.0, 1.0 - avg_error / abs(last_value)) if last_value != 0 else 0.0,
            model_type="linear_regression"
        )
    
    def get_trend_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive trend summary for a metric"""
        if metric_name not in self.metric_data:
            return {}
        
        # Current trend analysis
        current_trend = self.analyze_trend(metric_name)
        
        # Seasonal patterns
        patterns = self.detect_seasonal_patterns(metric_name)
        
        # Forecast
        forecast = self.forecast_trend(metric_name)
        
        # Recent anomalies
        recent_anomalies = self.detected_anomalies.get(metric_name, [])
        
        return {
            'metric_name': metric_name,
            'data_points': len(self.metric_data[metric_name]),
            'current_trend': current_trend.__dict__ if current_trend else None,
            'seasonal_patterns': [p.__dict__ for p in patterns],
            'forecast': forecast.__dict__ if forecast else None,
            'recent_anomalies': len(recent_anomalies),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def get_all_trends_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics"""
        summaries = {}
        
        for metric_name in self.metric_data.keys():
            summaries[metric_name] = self.get_trend_summary(metric_name)
        
        return {
            'total_metrics': len(self.metric_data),
            'metrics': summaries,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old data to manage memory"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
        
        for metric_name in self.metric_data:
            self.metric_data[metric_name] = [
                p for p in self.metric_data[metric_name]
                if p.timestamp > cutoff_time
            ]
        
        # Clean up empty metrics
        empty_metrics = [name for name, data in self.metric_data.items() if not data]
        for name in empty_metrics:
            del self.metric_data[name]
            if name in self.trend_history:
                del self.trend_history[name]
            if name in self.seasonal_patterns:
                del self.seasonal_patterns[name]
            if name in self.detected_anomalies:
                del self.detected_anomalies[name]
