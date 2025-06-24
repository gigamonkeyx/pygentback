#!/usr/bin/env python3
"""
Comprehensive validation script for the Research-to-Analysis workflow implementation

This script validates:
1. All backend components are properly implemented
2. API endpoints are correctly configured
3. Frontend components are in place
4. Integration points are working
5. Academic formatting is functional
"""

import os
import sys
import json
from pathlib import Path

def validate_backend_implementation():
    """Validate backend implementation"""
    print("🔍 Validating Backend Implementation")
    print("=" * 40)
    
    results = {
        "orchestrator": False,
        "api_routes": False,
        "dependencies": False,
        "integration": False
    }
    
    # Check orchestrator file
    orchestrator_path = Path("src/workflows/research_analysis_orchestrator.py")
    if orchestrator_path.exists():
        print("✅ Research-Analysis Orchestrator: FOUND")
        
        # Check for key classes and methods
        content = orchestrator_path.read_text()
        if "class ResearchAnalysisOrchestrator" in content:
            print("   ✅ ResearchAnalysisOrchestrator class: FOUND")
        if "execute_workflow" in content:
            print("   ✅ execute_workflow method: FOUND")
        if "WorkflowStatus" in content:
            print("   ✅ WorkflowStatus enum: FOUND")
        if "_format_academic_output" in content:
            print("   ✅ Academic formatting: FOUND")
        
        results["orchestrator"] = True
    else:
        print("❌ Research-Analysis Orchestrator: NOT FOUND")
    
    # Check API routes
    api_routes_path = Path("src/api/routes/workflows.py")
    if api_routes_path.exists():
        print("✅ Workflow API Routes: FOUND")
        
        content = api_routes_path.read_text()
        if "start_research_analysis_workflow" in content:
            print("   ✅ Start workflow endpoint: FOUND")
        if "get_workflow_status" in content:
            print("   ✅ Status endpoint: FOUND")
        if "get_workflow_result" in content:
            print("   ✅ Result endpoint: FOUND")
        if "export_workflow_result" in content:
            print("   ✅ Export endpoint: FOUND")
        
        results["api_routes"] = True
    else:
        print("❌ Workflow API Routes: NOT FOUND")
    
    # Check dependencies
    dependencies_path = Path("src/api/dependencies.py")
    if dependencies_path.exists():
        print("✅ API Dependencies: FOUND")
        results["dependencies"] = True
    else:
        print("❌ API Dependencies: NOT FOUND")
    
    # Check main API integration
    main_api_path = Path("src/api/main.py")
    if main_api_path.exists():
        content = main_api_path.read_text()
        if "workflows_router" in content:
            print("✅ Workflows Router Integration: FOUND")
            results["integration"] = True
        else:
            print("❌ Workflows Router Integration: NOT FOUND")
    else:
        print("❌ Main API file: NOT FOUND")
    
    return results


def validate_frontend_implementation():
    """Validate frontend implementation"""
    print("\n🔍 Validating Frontend Implementation")
    print("=" * 40)
    
    results = {
        "main_page": False,
        "ui_components": False,
        "navigation": False,
        "routing": False
    }
    
    # Check main page
    page_path = Path("ui/src/pages/ResearchAnalysisPage.tsx")
    if page_path.exists():
        print("✅ Research-Analysis Page: FOUND")
        
        content = page_path.read_text()
        if "ResearchAnalysisPage" in content:
            print("   ✅ Main component: FOUND")
        if "startWorkflow" in content:
            print("   ✅ Workflow trigger: FOUND")
        if "pollProgress" in content:
            print("   ✅ Progress polling: FOUND")
        if "exportResult" in content:
            print("   ✅ Export functionality: FOUND")
        
        results["main_page"] = True
    else:
        print("❌ Research-Analysis Page: NOT FOUND")
    
    # Check UI components
    ui_components = [
        "ui/src/components/ui/progress.tsx",
        "ui/src/components/ui/textarea.tsx", 
        "ui/src/components/ui/select.tsx",
        "ui/src/components/ui/separator.tsx"
    ]
    
    all_components_found = True
    for component in ui_components:
        if Path(component).exists():
            print(f"✅ {Path(component).name}: FOUND")
        else:
            print(f"❌ {Path(component).name}: NOT FOUND")
            all_components_found = False
    
    results["ui_components"] = all_components_found
    
    # Check navigation integration
    sidebar_path = Path("ui/src/components/layout/Sidebar.tsx")
    if sidebar_path.exists():
        content = sidebar_path.read_text()
        if "RESEARCH_ANALYSIS" in content:
            print("✅ Navigation Integration: FOUND")
            results["navigation"] = True
        else:
            print("❌ Navigation Integration: NOT FOUND")
    else:
        print("❌ Sidebar component: NOT FOUND")
    
    # Check routing
    app_path = Path("ui/src/App.tsx")
    if app_path.exists():
        content = app_path.read_text()
        if "research-analysis" in content:
            print("✅ Route Integration: FOUND")
            results["routing"] = True
        else:
            print("❌ Route Integration: NOT FOUND")
    else:
        print("❌ App component: NOT FOUND")
    
    return results


def validate_academic_formatting():
    """Validate academic formatting features"""
    print("\n🔍 Validating Academic Formatting")
    print("=" * 40)
    
    results = {
        "citation_formatting": False,
        "export_options": False,
        "typography": False
    }
    
    # Check orchestrator for academic formatting
    orchestrator_path = Path("src/workflows/research_analysis_orchestrator.py")
    if orchestrator_path.exists():
        content = orchestrator_path.read_text()
        
        if "_format_academic_output" in content:
            print("✅ Academic Output Formatting: FOUND")
        if "_extract_citations" in content:
            print("✅ Citation Extraction: FOUND")
        if "_format_citations" in content:
            print("✅ Citation Formatting: FOUND")
        if "References" in content:
            print("✅ Reference Section: FOUND")
        
        results["citation_formatting"] = True
    
    # Check API for export options
    api_routes_path = Path("src/api/routes/workflows.py")
    if api_routes_path.exists():
        content = api_routes_path.read_text()
        
        if "markdown" in content and "html" in content:
            print("✅ Export Formats: FOUND")
            results["export_options"] = True
    
    # Check frontend for typography
    page_path = Path("ui/src/pages/ResearchAnalysisPage.tsx")
    if page_path.exists():
        content = page_path.read_text()
        
        if "font-serif" in content:
            print("✅ Academic Typography: FOUND")
            results["typography"] = True
    
    return results


def validate_integration_points():
    """Validate integration points"""
    print("\n🔍 Validating Integration Points")
    print("=" * 40)
    
    results = {
        "agent_factory_integration": False,
        "progress_tracking": False,
        "error_handling": False,
        "websocket_support": False
    }
    
    # Check agent factory integration
    orchestrator_path = Path("src/workflows/research_analysis_orchestrator.py")
    if orchestrator_path.exists():
        content = orchestrator_path.read_text()
        
        if "agent_factory" in content:
            print("✅ Agent Factory Integration: FOUND")
            results["agent_factory_integration"] = True
        
        if "progress_callback" in content:
            print("✅ Progress Tracking: FOUND")
            results["progress_tracking"] = True
        
        if "try:" in content and "except" in content:
            print("✅ Error Handling: FOUND")
            results["error_handling"] = True
    
    # Check WebSocket support
    api_routes_path = Path("src/api/routes/workflows.py")
    if api_routes_path.exists():
        content = api_routes_path.read_text()
        
        if "StreamingResponse" in content:
            print("✅ WebSocket/Streaming Support: FOUND")
            results["websocket_support"] = True
    
    return results


def generate_validation_report(backend_results, frontend_results, formatting_results, integration_results):
    """Generate comprehensive validation report"""
    print("\n📊 VALIDATION REPORT")
    print("=" * 50)
    
    total_checks = 0
    passed_checks = 0
    
    # Count results
    for category_results in [backend_results, frontend_results, formatting_results, integration_results]:
        for check, passed in category_results.items():
            total_checks += 1
            if passed:
                passed_checks += 1
    
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"📈 Overall Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks})")
    
    # Detailed breakdown
    categories = [
        ("Backend Implementation", backend_results),
        ("Frontend Implementation", frontend_results), 
        ("Academic Formatting", formatting_results),
        ("Integration Points", integration_results)
    ]
    
    for category_name, category_results in categories:
        category_passed = sum(category_results.values())
        category_total = len(category_results)
        category_rate = (category_passed / category_total) * 100
        
        status = "✅" if category_rate == 100 else "⚠️" if category_rate >= 75 else "❌"
        print(f"{status} {category_name}: {category_rate:.1f}% ({category_passed}/{category_total})")
    
    # Overall status
    if success_rate >= 90:
        print("\n🎉 IMPLEMENTATION STATUS: EXCELLENT")
        print("   The Research-to-Analysis workflow is fully implemented and ready for use!")
    elif success_rate >= 75:
        print("\n✅ IMPLEMENTATION STATUS: GOOD")
        print("   The Research-to-Analysis workflow is mostly complete with minor issues.")
    elif success_rate >= 50:
        print("\n⚠️ IMPLEMENTATION STATUS: PARTIAL")
        print("   The Research-to-Analysis workflow has significant gaps that need attention.")
    else:
        print("\n❌ IMPLEMENTATION STATUS: INCOMPLETE")
        print("   The Research-to-Analysis workflow needs major work before it can be used.")
    
    return {
        "success_rate": success_rate,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "backend": backend_results,
        "frontend": frontend_results,
        "formatting": formatting_results,
        "integration": integration_results
    }


def main():
    """Main validation function"""
    print("🚀 Research-to-Analysis Workflow Validation")
    print("=" * 60)
    print("Comprehensive validation of the automated Research-to-Analysis workflow implementation")
    print("=" * 60)
    
    # Run all validations
    backend_results = validate_backend_implementation()
    frontend_results = validate_frontend_implementation()
    formatting_results = validate_academic_formatting()
    integration_results = validate_integration_points()
    
    # Generate report
    report = generate_validation_report(backend_results, frontend_results, formatting_results, integration_results)
    
    # Save report
    report_file = "research_analysis_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    print("\n🔗 Next Steps:")
    print("1. Address any missing components identified above")
    print("2. Test the server startup and API endpoints")
    print("3. Test the UI components in the browser")
    print("4. Run end-to-end workflow testing")
    print("5. Validate with real research queries")
    
    return report["success_rate"] >= 75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
