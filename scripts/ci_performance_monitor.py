#!/usr/bin/env python3
"""
CI/CD Performance Monitoring Script - Observer Enhancement
Tracks build times, success rates, and performance baselines
"""

import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

class CIPerformanceMonitor:
    """Monitor CI/CD performance metrics and baselines"""
    
    def __init__(self, metrics_file: str = "ci_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.baseline_targets = {
            'total_build_time': 600,  # 10 minutes max
            'test_execution_time': 300,  # 5 minutes max
            'dependency_install_time': 120,  # 2 minutes max
            'success_rate_threshold': 0.95,  # 95% success rate
        }
        
    def load_metrics(self) -> Dict:
        """Load existing metrics or create new structure"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            'builds': [],
            'baselines': self.baseline_targets,
            'last_updated': datetime.now().isoformat()
        }
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to file"""
        metrics['last_updated'] = datetime.now().isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def record_build(self, build_data: Dict):
        """Record a new build execution"""
        metrics = self.load_metrics()
        
        build_record = {
            'timestamp': datetime.now().isoformat(),
            'build_id': build_data.get('build_id', f"build_{int(time.time())}"),
            'branch': build_data.get('branch', 'unknown'),
            'commit_sha': build_data.get('commit_sha', 'unknown'),
            'total_time': build_data.get('total_time', 0),
            'test_time': build_data.get('test_time', 0),
            'install_time': build_data.get('install_time', 0),
            'success': build_data.get('success', False),
            'failure_reason': build_data.get('failure_reason', None),
            'dependency_conflicts': build_data.get('dependency_conflicts', 0),
        }
        
        metrics['builds'].append(build_record)
        
        # Keep only last 100 builds
        if len(metrics['builds']) > 100:
            metrics['builds'] = metrics['builds'][-100:]
        
        self.save_metrics(metrics)
        return build_record
    
    def analyze_performance(self) -> Dict:
        """Analyze current performance against baselines"""
        metrics = self.load_metrics()
        builds = metrics['builds']
        
        if not builds:
            return {'status': 'no_data', 'message': 'No build data available'}
        
        # Recent builds (last 10)
        recent_builds = builds[-10:]
        
        # Calculate averages
        avg_total_time = sum(b['total_time'] for b in recent_builds) / len(recent_builds)
        avg_test_time = sum(b['test_time'] for b in recent_builds) / len(recent_builds)
        avg_install_time = sum(b['install_time'] for b in recent_builds) / len(recent_builds)
        
        # Success rate
        successful_builds = sum(1 for b in recent_builds if b['success'])
        success_rate = successful_builds / len(recent_builds)
        
        # Dependency conflicts
        avg_conflicts = sum(b['dependency_conflicts'] for b in recent_builds) / len(recent_builds)
        
        # Check against baselines
        performance_status = {
            'total_time': {
                'current': avg_total_time,
                'baseline': self.baseline_targets['total_build_time'],
                'status': 'pass' if avg_total_time <= self.baseline_targets['total_build_time'] else 'fail'
            },
            'test_time': {
                'current': avg_test_time,
                'baseline': self.baseline_targets['test_execution_time'],
                'status': 'pass' if avg_test_time <= self.baseline_targets['test_execution_time'] else 'fail'
            },
            'install_time': {
                'current': avg_install_time,
                'baseline': self.baseline_targets['dependency_install_time'],
                'status': 'pass' if avg_install_time <= self.baseline_targets['dependency_install_time'] else 'fail'
            },
            'success_rate': {
                'current': success_rate,
                'baseline': self.baseline_targets['success_rate_threshold'],
                'status': 'pass' if success_rate >= self.baseline_targets['success_rate_threshold'] else 'fail'
            },
            'dependency_conflicts': {
                'current': avg_conflicts,
                'baseline': 0,
                'status': 'pass' if avg_conflicts == 0 else 'warning'
            }
        }
        
        # Overall status
        failed_metrics = [k for k, v in performance_status.items() if v['status'] == 'fail']
        warning_metrics = [k for k, v in performance_status.items() if v['status'] == 'warning']
        
        overall_status = 'pass'
        if failed_metrics:
            overall_status = 'fail'
        elif warning_metrics:
            overall_status = 'warning'
        
        return {
            'overall_status': overall_status,
            'failed_metrics': failed_metrics,
            'warning_metrics': warning_metrics,
            'performance_status': performance_status,
            'recent_builds_count': len(recent_builds),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_report(self) -> str:
        """Generate human-readable performance report"""
        analysis = self.analyze_performance()
        
        if analysis.get('status') == 'no_data':
            return "❌ No build data available for analysis"
        
        report = ["=== CI/CD PERFORMANCE REPORT ===\n"]
        
        overall = analysis['overall_status']
        status_emoji = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}
        report.append(f"Overall Status: {status_emoji[overall]} {overall.upper()}\n")
        
        report.append("Performance Metrics:")
        for metric, data in analysis['performance_status'].items():
            status = data['status']
            emoji = status_emoji.get(status, '❓')
            current = data['current']
            baseline = data['baseline']
            
            if metric in ['total_time', 'test_time', 'install_time']:
                report.append(f"  {emoji} {metric}: {current:.1f}s (baseline: {baseline}s)")
            elif metric == 'success_rate':
                report.append(f"  {emoji} {metric}: {current:.1%} (baseline: {baseline:.1%})")
            else:
                report.append(f"  {emoji} {metric}: {current:.1f} (baseline: {baseline})")
        
        if analysis['failed_metrics']:
            report.append(f"\n❌ Failed Metrics: {', '.join(analysis['failed_metrics'])}")
        
        if analysis['warning_metrics']:
            report.append(f"\n⚠️ Warning Metrics: {', '.join(analysis['warning_metrics'])}")
        
        report.append(f"\nAnalysis based on {analysis['recent_builds_count']} recent builds")
        report.append(f"Generated: {analysis['analysis_timestamp']}")
        
        return "\n".join(report)

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CI/CD Performance Monitor')
    parser.add_argument('--record', action='store_true', help='Record a new build')
    parser.add_argument('--analyze', action='store_true', help='Analyze performance')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--build-time', type=float, help='Total build time in seconds')
    parser.add_argument('--test-time', type=float, help='Test execution time in seconds')
    parser.add_argument('--install-time', type=float, help='Dependency install time in seconds')
    parser.add_argument('--success', action='store_true', help='Mark build as successful')
    parser.add_argument('--conflicts', type=int, default=0, help='Number of dependency conflicts')
    
    args = parser.parse_args()
    
    monitor = CIPerformanceMonitor()
    
    if args.record:
        build_data = {
            'total_time': args.build_time or 0,
            'test_time': args.test_time or 0,
            'install_time': args.install_time or 0,
            'success': args.success,
            'dependency_conflicts': args.conflicts,
            'build_id': os.environ.get('GITHUB_RUN_ID', f"local_{int(time.time())}"),
            'branch': os.environ.get('GITHUB_REF_NAME', 'local'),
            'commit_sha': os.environ.get('GITHUB_SHA', 'unknown')
        }
        
        record = monitor.record_build(build_data)
        print(f"✅ Recorded build: {record['build_id']}")
    
    if args.analyze or args.report:
        if args.analyze:
            analysis = monitor.analyze_performance()
            print(json.dumps(analysis, indent=2))
        
        if args.report:
            report = monitor.generate_report()
            print(report)
    
    if not any([args.record, args.analyze, args.report]):
        parser.print_help()

if __name__ == '__main__':
    main()
