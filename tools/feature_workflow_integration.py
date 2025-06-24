#!/usr/bin/env python3
"""
Development Workflow Integration for Feature Registry

This script integrates the feature registry into the development workflow
to prevent the "forgotten features" problem by:

1. Running feature discovery on git commits
2. Updating documentation automatically
3. Alerting developers to orphaned or undocumented features
4. Generating feature reports for code reviews

Run this as part of your CI/CD pipeline or git hooks.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from src.feature_registry.core import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureWorkflowIntegration:
    """Integrates feature registry with development workflow"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.registry = FeatureRegistry(project_root)
        self.report_path = self.project_root / "feature_workflow_report.json"
    
    async def run_pre_commit_check(self) -> bool:
        """
        Run feature registry checks before commit
        Returns True if all checks pass, False otherwise
        """
        logger.info("Running pre-commit feature registry checks...")
        
        # Discover current features
        await self.registry.discover_all_features()
        
        # Analyze health
        health_analysis = await self.registry.analyze_feature_health()
        
        # Generate report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_type": "pre_commit",
            "total_features": len(self.registry.features),
            "health_analysis": health_analysis,
            "warnings": [],
            "errors": [],
            "passed": True
        }
        
        # Check for critical issues
        critical_issues = []
        
        # Check for too many undocumented features
        undoc_count = len(health_analysis.get('undocumented_features', []))
        if undoc_count > 10:  # Threshold
            critical_issues.append(f"{undoc_count} undocumented features exceed threshold of 10")
        
        # Check for too many missing tests
        missing_tests = len(health_analysis.get('missing_tests', []))
        if missing_tests > 5:  # Threshold
            critical_issues.append(f"{missing_tests} features missing tests exceed threshold of 5")
        
        if critical_issues:
            report["errors"] = critical_issues
            report["passed"] = False
            logger.error("Pre-commit checks failed:")
            for issue in critical_issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("Pre-commit checks passed")
        
        # Add warnings for non-critical issues
        if undoc_count > 0:
            report["warnings"].append(f"{undoc_count} features need documentation")
        
        if missing_tests > 0:
            report["warnings"].append(f"{missing_tests} features need tests")
        
        # Save report
        await self._save_report(report)
        
        return report["passed"]
    
    async def run_post_commit_update(self) -> None:
        """
        Update feature registry after commit
        """
        logger.info("Running post-commit feature registry update...")
        
        # Full feature discovery
        await self.registry.discover_all_features()
        
        # Save updated registry
        await self.registry.save_registry()
        
        # Update documentation
        await self._update_documentation()
        
        # Generate report
        health_analysis = await self.registry.analyze_feature_health()
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_type": "post_commit",
            "total_features": len(self.registry.features),
            "health_analysis": health_analysis,
            "documentation_updated": True
        }
        
        await self._save_report(report)
        logger.info("Post-commit update completed")
    
    async def run_daily_audit(self) -> Dict[str, Any]:
        """
        Daily comprehensive feature audit
        """
        logger.info("Running daily feature audit...")
        
        # Full discovery and analysis
        await self.registry.discover_all_features()
        health_analysis = await self.registry.analyze_feature_health()
        
        # Save registry and documentation
        await self.registry.save_registry()
        await self._update_documentation()
        
        # Detailed analysis
        audit_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_type": "daily_audit",
            "total_features": len(self.registry.features),
            "health_analysis": health_analysis,
            "feature_breakdown": self._analyze_feature_breakdown(),
            "recommendations": self._generate_recommendations(health_analysis),
            "trend_analysis": await self._analyze_trends()
        }
        
        await self._save_report(audit_report)
        
        # Print summary
        self._print_audit_summary(audit_report)
        
        return audit_report
    
    def _analyze_feature_breakdown(self) -> Dict[str, Any]:
        """Analyze breakdown of features by various dimensions"""
        breakdown = {
            "by_type": {},
            "by_status": {},
            "by_age": {"new": 0, "recent": 0, "mature": 0, "stale": 0},
            "by_documentation": {"documented": 0, "undocumented": 0},
            "by_testing": {"tested": 0, "untested": 0}
        }
        
        from datetime import timedelta
        now = datetime.utcnow()
        
        for feature in self.registry.features.values():
            # By type
            type_key = feature.type.value
            breakdown["by_type"][type_key] = breakdown["by_type"].get(type_key, 0) + 1
            
            # By status
            status_key = feature.status.value
            breakdown["by_status"][status_key] = breakdown["by_status"].get(status_key, 0) + 1
            
            # By age
            age = now - feature.last_modified
            if age < timedelta(days=7):
                breakdown["by_age"]["new"] += 1
            elif age < timedelta(days=30):
                breakdown["by_age"]["recent"] += 1
            elif age < timedelta(days=90):
                breakdown["by_age"]["mature"] += 1
            else:
                breakdown["by_age"]["stale"] += 1
            
            # By documentation
            if feature.documentation_path:
                breakdown["by_documentation"]["documented"] += 1
            else:
                breakdown["by_documentation"]["undocumented"] += 1
            
            # By testing
            if feature.tests_path:
                breakdown["by_testing"]["tested"] += 1
            else:
                breakdown["by_testing"]["untested"] += 1
        
        return breakdown
    
    def _generate_recommendations(self, health_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Documentation recommendations
        undoc_count = len(health_analysis.get('undocumented_features', []))
        if undoc_count > 0:
            recommendations.append(
                f"📚 Add documentation for {undoc_count} undocumented features. "
                f"Start with API endpoints and MCP servers as they have the highest impact."
            )
        
        # Testing recommendations
        missing_tests = len(health_analysis.get('missing_tests', []))
        if missing_tests > 0:
            recommendations.append(
                f"🧪 Add tests for {missing_tests} features. "
                f"Prioritize API endpoints and critical UI components."
            )
        
        # Maintenance recommendations
        stale_count = len(health_analysis.get('stale_features', []))
        if stale_count > 0:
            recommendations.append(
                f"🔧 Review {stale_count} stale features. "
                f"Consider deprecating unused features or updating documentation."
            )
        
        # Orphaned feature recommendations
        orphaned_count = len(health_analysis.get('orphaned_features', []))
        if orphaned_count > 0:
            recommendations.append(
                f"🔗 Review {orphaned_count} orphaned features. "
                f"Add dependencies or remove if no longer needed."
            )
        
        return recommendations
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends over time (requires historical data)"""
        # This would compare with previous reports to show trends
        # For now, return basic trend structure
        return {
            "feature_growth": "stable",  # Would be calculated from historical data
            "documentation_coverage": "improving",
            "test_coverage": "stable",
            "note": "Trend analysis requires historical data from multiple runs"
        }
    
    async def _update_documentation(self) -> None:
        """Update project documentation"""
        try:
            # Generate feature documentation
            documentation = await self.registry.generate_documentation()
            
            # Save to docs directory
            doc_path = self.project_root / "docs" / "COMPLETE_FEATURE_REGISTRY.md"
            doc_path.parent.mkdir(exist_ok=True)
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            logger.info(f"Documentation updated: {doc_path}")
            
        except Exception as e:
            logger.error(f"Error updating documentation: {e}")
    
    async def _save_report(self, report: Dict[str, Any]) -> None:
        """Save workflow report"""
        try:
            with open(self.report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def _print_audit_summary(self, audit_report: Dict[str, Any]) -> None:
        """Print audit summary to console"""
        health = audit_report["health_analysis"]
        breakdown = audit_report["feature_breakdown"]
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                   PyGent Factory Feature Audit                                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Total Features: {audit_report['total_features']:3d}                                                           ║
║                                                                                                      ║
║ Documentation Coverage: {breakdown['by_documentation']['documented']:3d} documented, {breakdown['by_documentation']['undocumented']:3d} undocumented                  ║
║ Test Coverage:         {breakdown['by_testing']['tested']:3d} tested, {breakdown['by_testing']['untested']:3d} untested                          ║
║                                                                                                      ║
║ Health Issues:                                                                                       ║
║   • Undocumented:      {len(health.get('undocumented_features', [])):3d}                                                   ║
║   • Missing Tests:     {len(health.get('missing_tests', [])):3d}                                                   ║
║   • Stale Features:    {len(health.get('stale_features', [])):3d}                                                   ║
║   • Orphaned:          {len(health.get('orphaned_features', [])):3d}                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Recommendations:                                                                                     ║""")
        
        for i, rec in enumerate(audit_report["recommendations"][:3], 1):  # Show top 3
            print(f"║ {i}. {rec[:85]}{'...' if len(rec) > 85 else '':90} ║")
        
        print("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")


async def main():
    """Main function for different workflow commands"""
    if len(sys.argv) < 2:
        print("Usage: python feature_workflow_integration.py [pre-commit|post-commit|daily-audit]")
        sys.exit(1)
    
    command = sys.argv[1]
    workflow = FeatureWorkflowIntegration()
    
    if command == "pre-commit":
        success = await workflow.run_pre_commit_check()
        sys.exit(0 if success else 1)
    
    elif command == "post-commit":
        await workflow.run_post_commit_update()
    
    elif command == "daily-audit":
        await workflow.run_daily_audit()
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: pre-commit, post-commit, daily-audit")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
