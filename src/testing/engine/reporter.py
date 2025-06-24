"""
Test Reporter

Generates comprehensive test reports in multiple formats including HTML, JSON, and console output.
Provides detailed analysis of test results, performance metrics, and recommendations.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    JSON = "json"
    CONSOLE = "console"
    MARKDOWN = "markdown"
    CSV = "csv"


@dataclass
class TestSummary:
    """Summary of test execution"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time_seconds: float
    success_rate_percent: float
    start_time: datetime
    end_time: datetime
    test_categories: Dict[str, int] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    test_name: str
    category: str
    status: str  # passed, failed, skipped, error
    execution_time_seconds: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    output_directory: str
    include_performance_charts: bool = True
    include_detailed_logs: bool = True
    include_recommendations: bool = True
    group_by_category: bool = True
    show_stack_traces: bool = True
    max_log_entries: int = 1000
    chart_width: int = 800
    chart_height: int = 400


class TestReporter:
    """
    Comprehensive Test Reporting System.
    
    Generates detailed test reports in multiple formats with performance analysis,
    trend tracking, and actionable recommendations for test improvement.
    """
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report data
        self.test_results: List[TestResult] = []
        self.test_summary: Optional[TestSummary] = None
        self.report_metadata: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_history: List[TestSummary] = []
        self.trend_data: Dict[str, List[float]] = {}
    
    def add_test_result(self, result: TestResult):
        """Add a test result to the report"""
        self.test_results.append(result)
        logger.debug(f"Added test result: {result.test_name} - {result.status}")
    
    def add_test_results(self, results: List[TestResult]):
        """Add multiple test results to the report"""
        self.test_results.extend(results)
        logger.info(f"Added {len(results)} test results to report")
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set report metadata"""
        self.report_metadata.update(metadata)
    
    def generate_summary(self) -> TestSummary:
        """Generate test execution summary"""
        if not self.test_results:
            return TestSummary(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time_seconds=0.0,
                success_rate_percent=0.0,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
        
        # Calculate basic metrics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "passed"])
        failed_tests = len([r for r in self.test_results if r.status == "failed"])
        skipped_tests = len([r for r in self.test_results if r.status == "skipped"])
        
        # Calculate timing
        start_time = min(r.start_time for r in self.test_results)
        end_time = max(r.end_time for r in self.test_results)
        total_execution_time = sum(r.execution_time_seconds for r in self.test_results)
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
        
        # Group by category
        categories = {}
        for result in self.test_results:
            category = result.category
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        # Calculate performance metrics
        performance_metrics = {
            "average_test_time": statistics.mean([r.execution_time_seconds for r in self.test_results]),
            "median_test_time": statistics.median([r.execution_time_seconds for r in self.test_results]),
            "max_test_time": max([r.execution_time_seconds for r in self.test_results]),
            "min_test_time": min([r.execution_time_seconds for r in self.test_results])
        }
        
        self.test_summary = TestSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time_seconds=total_execution_time,
            success_rate_percent=success_rate,
            start_time=start_time,
            end_time=end_time,
            test_categories=categories,
            performance_metrics=performance_metrics
        )
        
        return self.test_summary
    
    def generate_report(self, formats: List[ReportFormat], filename_prefix: str = "test_report") -> Dict[str, str]:
        """
        Generate test reports in specified formats.
        
        Args:
            formats: List of report formats to generate
            filename_prefix: Prefix for output files
            
        Returns:
            Dict mapping format to output file path
        """
        if not self.test_summary:
            self.generate_summary()
        
        generated_files = {}
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            try:
                if format_type == ReportFormat.HTML:
                    filepath = self._generate_html_report(filename_prefix, timestamp)
                elif format_type == ReportFormat.JSON:
                    filepath = self._generate_json_report(filename_prefix, timestamp)
                elif format_type == ReportFormat.CONSOLE:
                    filepath = self._generate_console_report(filename_prefix, timestamp)
                elif format_type == ReportFormat.MARKDOWN:
                    filepath = self._generate_markdown_report(filename_prefix, timestamp)
                elif format_type == ReportFormat.CSV:
                    filepath = self._generate_csv_report(filename_prefix, timestamp)
                else:
                    logger.warning(f"Unsupported report format: {format_type}")
                    continue
                
                generated_files[format_type.value] = filepath
                logger.info(f"Generated {format_type.value} report: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to generate {format_type.value} report: {e}")
        
        return generated_files
    
    def _generate_html_report(self, prefix: str, timestamp: str) -> str:
        """Generate HTML report"""
        filename = f"{prefix}_{timestamp}.html"
        filepath = self.output_dir / filename
        
        html_content = self._create_html_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def _generate_json_report(self, prefix: str, timestamp: str) -> str:
        """Generate JSON report"""
        filename = f"{prefix}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        report_data = {
            "summary": asdict(self.test_summary) if self.test_summary else {},
            "test_results": [asdict(result) for result in self.test_results],
            "metadata": self.report_metadata,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Convert datetime objects to strings
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=datetime_converter)
        
        return str(filepath)
    
    def _generate_console_report(self, prefix: str, timestamp: str) -> str:
        """Generate console-formatted report"""
        filename = f"{prefix}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        console_content = self._create_console_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(console_content)
        
        return str(filepath)
    
    def _generate_markdown_report(self, prefix: str, timestamp: str) -> str:
        """Generate Markdown report"""
        filename = f"{prefix}_{timestamp}.md"
        filepath = self.output_dir / filename
        
        markdown_content = self._create_markdown_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(filepath)
    
    def _generate_csv_report(self, prefix: str, timestamp: str) -> str:
        """Generate CSV report"""
        filename = f"{prefix}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        csv_content = self._create_csv_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        return str(filepath)
    
    def _create_html_content(self) -> str:
        """Create HTML report content"""
        if not self.test_summary:
            return "<html><body><h1>No test data available</h1></body></html>"
        
        # Basic HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {self.test_summary.start_time.strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-passed {{ background-color: #d4edda; }}
        .status-failed {{ background-color: #f8d7da; }}
        .status-skipped {{ background-color: #fff3cd; }}
    </style>
</head>
<body>
    <h1>Test Execution Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {self.test_summary.total_tests}</p>
        <p><strong>Passed:</strong> <span class="passed">{self.test_summary.passed_tests}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{self.test_summary.failed_tests}</span></p>
        <p><strong>Skipped:</strong> <span class="skipped">{self.test_summary.skipped_tests}</span></p>
        <p><strong>Success Rate:</strong> {self.test_summary.success_rate_percent:.1f}%</p>
        <p><strong>Execution Time:</strong> {self.test_summary.execution_time_seconds:.2f} seconds</p>
        <p><strong>Start Time:</strong> {self.test_summary.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>End Time:</strong> {self.test_summary.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Category</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Error Message</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for result in self.test_results:
            status_class = f"status-{result.status}"
            error_msg = result.error_message or ""
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            
            html += f"""
            <tr class="{status_class}">
                <td>{result.test_name}</td>
                <td>{result.category}</td>
                <td>{result.status.upper()}</td>
                <td>{result.execution_time_seconds:.3f}</td>
                <td>{error_msg}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
</body>
</html>
"""
        return html
    
    def _create_console_content(self) -> str:
        """Create console report content"""
        if not self.test_summary:
            return "No test data available\n"
        
        content = []
        content.append("=" * 80)
        content.append("TEST EXECUTION REPORT")
        content.append("=" * 80)
        content.append("")
        
        # Summary
        content.append("SUMMARY:")
        content.append(f"  Total Tests: {self.test_summary.total_tests}")
        content.append(f"  Passed:      {self.test_summary.passed_tests}")
        content.append(f"  Failed:      {self.test_summary.failed_tests}")
        content.append(f"  Skipped:     {self.test_summary.skipped_tests}")
        content.append(f"  Success Rate: {self.test_summary.success_rate_percent:.1f}%")
        content.append(f"  Duration:    {self.test_summary.execution_time_seconds:.2f}s")
        content.append("")
        
        # Failed tests
        failed_tests = [r for r in self.test_results if r.status == "failed"]
        if failed_tests:
            content.append("FAILED TESTS:")
            for result in failed_tests:
                content.append(f"  ❌ {result.test_name} ({result.category})")
                if result.error_message:
                    content.append(f"     Error: {result.error_message}")
            content.append("")
        
        # Performance summary
        if self.test_summary.performance_metrics:
            content.append("PERFORMANCE METRICS:")
            for metric, value in self.test_summary.performance_metrics.items():
                content.append(f"  {metric}: {value:.3f}s")
            content.append("")
        
        content.append("=" * 80)
        
        return "\n".join(content)
    
    def _create_markdown_content(self) -> str:
        """Create Markdown report content"""
        if not self.test_summary:
            return "# Test Report\n\nNo test data available\n"
        
        content = []
        content.append("# Test Execution Report")
        content.append("")
        content.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Summary
        content.append("## Summary")
        content.append("")
        content.append(f"- **Total Tests:** {self.test_summary.total_tests}")
        content.append(f"- **Passed:** {self.test_summary.passed_tests}")
        content.append(f"- **Failed:** {self.test_summary.failed_tests}")
        content.append(f"- **Skipped:** {self.test_summary.skipped_tests}")
        content.append(f"- **Success Rate:** {self.test_summary.success_rate_percent:.1f}%")
        content.append(f"- **Duration:** {self.test_summary.execution_time_seconds:.2f}s")
        content.append("")
        
        # Test results table
        content.append("## Test Results")
        content.append("")
        content.append("| Test Name | Category | Status | Duration (s) | Error |")
        content.append("|-----------|----------|--------|--------------|-------|")
        
        for result in self.test_results:
            status_emoji = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(result.status, "❓")
            error_msg = (result.error_message or "")[:50]
            if len(result.error_message or "") > 50:
                error_msg += "..."
            
            content.append(f"| {result.test_name} | {result.category} | {status_emoji} {result.status} | {result.execution_time_seconds:.3f} | {error_msg} |")
        
        content.append("")
        
        return "\n".join(content)
    
    def _create_csv_content(self) -> str:
        """Create CSV report content"""
        content = []
        content.append("test_name,category,status,duration_seconds,start_time,end_time,error_message")
        
        for result in self.test_results:
            error_msg = (result.error_message or "").replace('"', '""')
            content.append(f'"{result.test_name}","{result.category}","{result.status}",{result.execution_time_seconds},"{result.start_time.isoformat()}","{result.end_time.isoformat()}","{error_msg}"')
        
        return "\n".join(content)
    
    def print_summary(self):
        """Print summary to console"""
        if not self.test_summary:
            self.generate_summary()
        
        print(self._create_console_content())
    
    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_summary:
            return recommendations
        
        # Performance recommendations
        if self.test_summary.performance_metrics.get("average_test_time", 0) > 5.0:
            recommendations.append("Consider optimizing slow tests - average test time is over 5 seconds")
        
        # Failure rate recommendations
        if self.test_summary.success_rate_percent < 90:
            recommendations.append("Success rate is below 90% - investigate failing tests")
        
        # Category-specific recommendations
        failed_by_category = {}
        for result in self.test_results:
            if result.status == "failed":
                category = result.category
                failed_by_category[category] = failed_by_category.get(category, 0) + 1
        
        for category, count in failed_by_category.items():
            if count > 3:
                recommendations.append(f"Multiple failures in {category} category - review test setup")
        
        return recommendations
