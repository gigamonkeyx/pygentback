#!/usr/bin/env python3
"""
Final MCP Ecosystem Assessment

Comprehensive production readiness assessment for PyGent Factory MCP ecosystem.
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any

class FinalMCPAssessment:
    """Final production readiness assessment"""
    
    def __init__(self):
        self.servers = {
            'embedding': {
                'port': 8002,
                'name': 'Embedding MCP Server',
                'category': 'Core AI Services',
                'criticality': 'High',
                'features': [
                    'Multi-provider embeddings (SentenceTransformer, OpenAI, OpenRouter, Ollama)',
                    'OpenAI SDK compatibility',
                    'Batch processing optimization',
                    'Performance monitoring'
                ]
            },
            'document-processing': {
                'port': 8003,
                'name': 'Document Processing MCP Server',
                'category': 'Document Services',
                'criticality': 'High',
                'features': [
                    'PDF download and processing',
                    'Multi-method text extraction (PyMuPDF, OCR)',
                    'AI-powered content analysis',
                    'Quality assessment and metadata generation'
                ]
            },
            'vector-search': {
                'port': 8004,
                'name': 'Vector Search MCP Server',
                'category': 'Search & Retrieval',
                'criticality': 'High',
                'features': [
                    'Semantic search and similarity matching',
                    'Collection management',
                    'In-memory vector storage',
                    'Similarity threshold filtering'
                ]
            },
            'agent-orchestration': {
                'port': 8005,
                'name': 'Agent Orchestration MCP Server',
                'category': 'Agent Management',
                'criticality': 'Critical',
                'features': [
                    'Agent lifecycle management',
                    'Task dispatching and scheduling',
                    'Priority-based execution',
                    'Multi-agent coordination'
                ]
            }
        }
    
    def assess_server_production_readiness(self, server_id: str, server_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess production readiness of a single server"""
        port = server_info['port']
        assessment = {
            'server_id': server_id,
            'name': server_info['name'],
            'category': server_info['category'],
            'criticality': server_info['criticality'],
            'status': 'unknown',
            'uptime': 0,
            'performance_score': 0,
            'feature_completeness': 0,
            'reliability_score': 0,
            'production_ready': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Test basic connectivity
            root_response = requests.get(f"http://localhost:{port}/", timeout=5)
            if root_response.status_code != 200:
                assessment['issues'].append(f"Root endpoint not accessible (HTTP {root_response.status_code})")
                return assessment
            
            root_data = root_response.json()
            
            # Test health endpoint
            health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if health_response.status_code != 200:
                assessment['issues'].append(f"Health endpoint not accessible (HTTP {health_response.status_code})")
                return assessment
            
            health_data = health_response.json()
            assessment['status'] = health_data.get('status', 'unknown')
            
            if assessment['status'] != 'healthy':
                assessment['issues'].append(f"Server reports unhealthy status: {assessment['status']}")
                return assessment
            
            # Extract performance metrics
            performance = health_data.get('performance', {})
            assessment['uptime'] = performance.get('uptime_seconds', 0)
            
            # Performance scoring
            response_time = None
            start_time = time.time()
            test_response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if test_response.status_code == 200:
                response_time = time.time() - start_time
            
            # Score performance (0-100)
            if response_time:
                if response_time < 0.1:  # Under 100ms
                    assessment['performance_score'] = 100
                elif response_time < 0.5:  # Under 500ms
                    assessment['performance_score'] = 80
                elif response_time < 1.0:  # Under 1s
                    assessment['performance_score'] = 60
                elif response_time < 2.0:  # Under 2s
                    assessment['performance_score'] = 40
                else:
                    assessment['performance_score'] = 20
                    assessment['issues'].append(f"Slow response time: {response_time*1000:.0f}ms")
            
            # Feature completeness (based on available endpoints)
            endpoints = root_data.get('endpoints', {})
            expected_endpoints = len(server_info['features'])
            actual_endpoints = len(endpoints)
            assessment['feature_completeness'] = min(100, (actual_endpoints / max(expected_endpoints, 1)) * 100)
            
            # Reliability scoring (based on uptime and error rates)
            uptime_hours = assessment['uptime'] / 3600
            error_rate = performance.get('error_rate', 0)
            
            if uptime_hours > 1 and error_rate < 5:  # Over 1 hour uptime, under 5% error rate
                assessment['reliability_score'] = 100
            elif uptime_hours > 0.5 and error_rate < 10:  # Over 30 min uptime, under 10% error rate
                assessment['reliability_score'] = 80
            elif uptime_hours > 0.1 and error_rate < 20:  # Over 6 min uptime, under 20% error rate
                assessment['reliability_score'] = 60
            else:
                assessment['reliability_score'] = 40
                if uptime_hours < 0.1:
                    assessment['issues'].append(f"Low uptime: {uptime_hours*60:.1f} minutes")
                if error_rate > 10:
                    assessment['issues'].append(f"High error rate: {error_rate:.1f}%")
            
            # Overall production readiness
            overall_score = (
                assessment['performance_score'] * 0.3 +
                assessment['feature_completeness'] * 0.3 +
                assessment['reliability_score'] * 0.4
            )
            
            assessment['production_ready'] = overall_score >= 75 and len(assessment['issues']) == 0
            
            # Generate recommendations
            if assessment['performance_score'] < 80:
                assessment['recommendations'].append("Optimize response times for better performance")
            if assessment['feature_completeness'] < 100:
                assessment['recommendations'].append("Complete implementation of all planned features")
            if assessment['reliability_score'] < 90:
                assessment['recommendations'].append("Improve error handling and stability")
            
        except Exception as e:
            assessment['issues'].append(f"Assessment failed: {str(e)}")
            assessment['status'] = 'failed'
        
        return assessment
    
    def generate_final_assessment(self) -> Dict[str, Any]:
        """Generate final ecosystem assessment"""
        assessment = {
            'timestamp': datetime.utcnow().isoformat(),
            'ecosystem_name': 'PyGent Factory MCP Ecosystem',
            'version': '1.0.0',
            'assessment_type': 'Production Readiness',
            'servers': {},
            'summary': {
                'total_servers': len(self.servers),
                'production_ready': 0,
                'critical_issues': 0,
                'overall_score': 0,
                'production_status': 'unknown',
                'deployment_recommendation': 'unknown'
            }
        }
        
        print("🔍 Final MCP Ecosystem Production Readiness Assessment")
        print("=" * 70)
        
        total_score = 0
        critical_issues = 0
        
        for server_id, server_info in self.servers.items():
            print(f"\n📊 Assessing {server_info['name']}...")
            server_assessment = self.assess_server_production_readiness(server_id, server_info)
            assessment['servers'][server_id] = server_assessment
            
            # Print server status
            status_emoji = "✅" if server_assessment['production_ready'] else "❌"
            print(f"   {status_emoji} Status: {server_assessment['status'].upper()}")
            print(f"   📈 Performance Score: {server_assessment['performance_score']}/100")
            print(f"   🔧 Feature Completeness: {server_assessment['feature_completeness']:.0f}%")
            print(f"   🛡️ Reliability Score: {server_assessment['reliability_score']}/100")
            print(f"   ⏱️ Uptime: {server_assessment['uptime']/3600:.1f} hours")
            
            if server_assessment['issues']:
                print(f"   ⚠️ Issues: {len(server_assessment['issues'])}")
                for issue in server_assessment['issues']:
                    print(f"      - {issue}")
            
            if server_assessment['recommendations']:
                print(f"   💡 Recommendations: {len(server_assessment['recommendations'])}")
                for rec in server_assessment['recommendations']:
                    print(f"      - {rec}")
            
            # Update summary
            if server_assessment['production_ready']:
                assessment['summary']['production_ready'] += 1
            
            if server_assessment['issues'] and server_info['criticality'] in ['Critical', 'High']:
                critical_issues += len(server_assessment['issues'])
            
            # Calculate weighted score (critical servers count more)
            weight = 2 if server_info['criticality'] == 'Critical' else 1
            server_score = (
                server_assessment['performance_score'] * 0.3 +
                server_assessment['feature_completeness'] * 0.3 +
                server_assessment['reliability_score'] * 0.4
            )
            total_score += server_score * weight
        
        # Calculate overall metrics
        total_weight = sum(2 if info['criticality'] == 'Critical' else 1 for info in self.servers.values())
        assessment['summary']['overall_score'] = total_score / total_weight
        assessment['summary']['critical_issues'] = critical_issues
        
        # Determine production status
        if assessment['summary']['overall_score'] >= 90 and critical_issues == 0:
            assessment['summary']['production_status'] = 'EXCELLENT'
            assessment['summary']['deployment_recommendation'] = 'DEPLOY IMMEDIATELY'
        elif assessment['summary']['overall_score'] >= 75 and critical_issues <= 2:
            assessment['summary']['production_status'] = 'GOOD'
            assessment['summary']['deployment_recommendation'] = 'DEPLOY WITH MONITORING'
        elif assessment['summary']['overall_score'] >= 60:
            assessment['summary']['production_status'] = 'FAIR'
            assessment['summary']['deployment_recommendation'] = 'DEPLOY TO STAGING FIRST'
        else:
            assessment['summary']['production_status'] = 'POOR'
            assessment['summary']['deployment_recommendation'] = 'DO NOT DEPLOY'
        
        return assessment
    
    def print_final_summary(self, assessment: Dict[str, Any]):
        """Print final summary report"""
        summary = assessment['summary']
        
        print("\n" + "="*70)
        print("🎯 FINAL PRODUCTION READINESS ASSESSMENT")
        print("="*70)
        
        print(f"📊 OVERALL METRICS")
        print(f"   Total Servers: {summary['total_servers']}")
        print(f"   Production Ready: {summary['production_ready']}/{summary['total_servers']}")
        print(f"   Overall Score: {summary['overall_score']:.1f}/100")
        print(f"   Critical Issues: {summary['critical_issues']}")
        
        print(f"\n🎯 PRODUCTION STATUS: {summary['production_status']}")
        print(f"📋 DEPLOYMENT RECOMMENDATION: {summary['deployment_recommendation']}")
        
        if summary['production_status'] == 'EXCELLENT':
            print("\n✅ ECOSYSTEM ASSESSMENT: PRODUCTION READY")
            print("   All systems are operational and performing excellently.")
            print("   The MCP ecosystem is ready for immediate production deployment.")
        elif summary['production_status'] == 'GOOD':
            print("\n⚠️ ECOSYSTEM ASSESSMENT: MOSTLY READY")
            print("   Most systems are operational with minor issues.")
            print("   Deploy with enhanced monitoring and issue tracking.")
        elif summary['production_status'] == 'FAIR':
            print("\n❌ ECOSYSTEM ASSESSMENT: NEEDS IMPROVEMENT")
            print("   Significant issues detected that need resolution.")
            print("   Deploy to staging environment first for further testing.")
        else:
            print("\n❌ ECOSYSTEM ASSESSMENT: NOT READY")
            print("   Critical issues prevent production deployment.")
            print("   Resolve all issues before considering deployment.")
        
        print("="*70)


def main():
    """Main execution"""
    assessor = FinalMCPAssessment()
    
    print("🔍 Starting Final MCP Ecosystem Assessment...")
    print("This comprehensive assessment will evaluate production readiness.\n")
    
    # Generate assessment
    assessment = assessor.generate_final_assessment()
    
    # Print summary
    assessor.print_final_summary(assessment)
    
    # Save detailed assessment
    with open('final_mcp_ecosystem_assessment.json', 'w') as f:
        json.dump(assessment, f, indent=2)
    
    print(f"\n📄 Detailed assessment saved to: final_mcp_ecosystem_assessment.json")
    
    # Return exit code based on production readiness
    if assessment['summary']['production_status'] in ['EXCELLENT', 'GOOD']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
