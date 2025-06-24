#!/usr/bin/env python3
"""
A2A Compliance Validation

Implements comprehensive validation to ensure all A2A components meet the Google A2A specification requirements.
"""

import asyncio
import json
import logging
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """A2A compliance levels"""
    CRITICAL = "critical"
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationCheck:
    """Individual validation check"""
    id: str
    name: str
    description: str
    compliance_level: ComplianceLevel
    category: str
    specification_reference: str
    
    def __post_init__(self):
        """Validate check configuration"""
        if not self.id or not self.name:
            raise ValueError("Check ID and name are required")


@dataclass
class ValidationIssue:
    """Validation issue details"""
    check_id: str
    severity: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    skipped_checks: int = 0
    compliance_score: float = 0.0
    issues: List[ValidationIssue] = field(default_factory=list)
    check_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class A2AComplianceValidator:
    """A2A Protocol Compliance Validator"""
    
    def __init__(self):
        self.checks: Dict[str, ValidationCheck] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_validation_checks()
    
    def _setup_validation_checks(self):
        """Setup all A2A compliance validation checks"""
        
        # Agent Card Compliance Checks
        self.checks["agent_card_structure"] = ValidationCheck(
            id="agent_card_structure",
            name="Agent Card Structure",
            description="Validate agent card has required fields according to A2A spec",
            compliance_level=ComplianceLevel.CRITICAL,
            category="agent_card",
            specification_reference="A2A Spec Section 3.1"
        )
        
        self.checks["agent_card_well_known"] = ValidationCheck(
            id="agent_card_well_known",
            name="Well-known URL Support",
            description="Validate /.well-known/agent.json endpoint availability",
            compliance_level=ComplianceLevel.CRITICAL,
            category="agent_card",
            specification_reference="A2A Spec Section 3.2"
        )
        
        # JSON-RPC Transport Checks
        self.checks["jsonrpc_format"] = ValidationCheck(
            id="jsonrpc_format",
            name="JSON-RPC 2.0 Format",
            description="Validate JSON-RPC 2.0 request/response format compliance",
            compliance_level=ComplianceLevel.CRITICAL,
            category="transport",
            specification_reference="JSON-RPC 2.0 Specification"
        )
        
        self.checks["a2a_methods"] = ValidationCheck(
            id="a2a_methods",
            name="A2A Method Support",
            description="Validate support for required A2A methods",
            compliance_level=ComplianceLevel.CRITICAL,
            category="transport",
            specification_reference="A2A Spec Section 4.1"
        )
        
        # Task Management Checks
        self.checks["task_lifecycle"] = ValidationCheck(
            id="task_lifecycle",
            name="Task Lifecycle States",
            description="Validate proper task state transitions",
            compliance_level=ComplianceLevel.CRITICAL,
            category="task_management",
            specification_reference="A2A Spec Section 5.1"
        )
        
        self.checks["task_context"] = ValidationCheck(
            id="task_context",
            name="Task Context Management",
            description="Validate task context preservation and sharing",
            compliance_level=ComplianceLevel.REQUIRED,
            category="task_management",
            specification_reference="A2A Spec Section 5.2"
        )
        
        # Message Structure Checks
        self.checks["message_parts"] = ValidationCheck(
            id="message_parts",
            name="Message Part Types",
            description="Validate support for TextPart, FilePart, and DataPart",
            compliance_level=ComplianceLevel.CRITICAL,
            category="messaging",
            specification_reference="A2A Spec Section 6.1"
        )
        
        # Security Checks
        self.checks["authentication"] = ValidationCheck(
            id="authentication",
            name="Authentication Support",
            description="Validate authentication mechanisms (JWT, API keys)",
            compliance_level=ComplianceLevel.REQUIRED,
            category="security",
            specification_reference="A2A Spec Section 7.1"
        )
        
        self.checks["security_schemes"] = ValidationCheck(
            id="security_schemes",
            name="Security Scheme Declaration",
            description="Validate proper security scheme declaration in agent cards",
            compliance_level=ComplianceLevel.REQUIRED,
            category="security",
            specification_reference="A2A Spec Section 7.2"
        )
        
        # Error Handling Checks
        self.checks["error_codes"] = ValidationCheck(
            id="error_codes",
            name="A2A Error Codes",
            description="Validate proper A2A error code usage",
            compliance_level=ComplianceLevel.CRITICAL,
            category="error_handling",
            specification_reference="A2A Spec Section 8.1"
        )
        
        # Streaming Checks
        self.checks["streaming_support"] = ValidationCheck(
            id="streaming_support",
            name="Streaming Support",
            description="Validate Server-Sent Events streaming capability",
            compliance_level=ComplianceLevel.RECOMMENDED,
            category="streaming",
            specification_reference="A2A Spec Section 9.1"
        )
        
        # Discovery Checks
        self.checks["agent_discovery"] = ValidationCheck(
            id="agent_discovery",
            name="Agent Discovery",
            description="Validate agent discovery and capability matching",
            compliance_level=ComplianceLevel.RECOMMENDED,
            category="discovery",
            specification_reference="A2A Spec Section 10.1"
        )
    
    async def validate_agent_card_structure(self, agent_card: Dict[str, Any]) -> Tuple[ValidationResult, List[ValidationIssue]]:
        """Validate agent card structure compliance"""
        issues = []
        
        # Required fields according to A2A spec
        required_fields = ["name", "description", "url", "capabilities", "skills", "provider"]
        
        for field in required_fields:
            if field not in agent_card:
                issues.append(ValidationIssue(
                    check_id="agent_card_structure",
                    severity="critical",
                    message=f"Missing required field: {field}",
                    suggestion=f"Add '{field}' field to agent card"
                ))
        
        # Validate capabilities structure
        if "capabilities" in agent_card:
            capabilities = agent_card["capabilities"]
            if not isinstance(capabilities, dict):
                issues.append(ValidationIssue(
                    check_id="agent_card_structure",
                    severity="critical",
                    message="Capabilities must be an object",
                    suggestion="Change capabilities to object format"
                ))
            else:
                # Check for standard capability fields
                expected_caps = ["streaming", "pushNotifications", "stateTransitionHistory"]
                for cap in expected_caps:
                    if cap not in capabilities:
                        issues.append(ValidationIssue(
                            check_id="agent_card_structure",
                            severity="warning",
                            message=f"Missing capability declaration: {cap}",
                            suggestion=f"Add '{cap}' capability declaration"
                        ))
        
        # Validate skills structure
        if "skills" in agent_card:
            skills = agent_card["skills"]
            if not isinstance(skills, list):
                issues.append(ValidationIssue(
                    check_id="agent_card_structure",
                    severity="critical",
                    message="Skills must be an array",
                    suggestion="Change skills to array format"
                ))
            else:
                for i, skill in enumerate(skills):
                    if not isinstance(skill, dict):
                        issues.append(ValidationIssue(
                            check_id="agent_card_structure",
                            severity="critical",
                            message=f"Skill {i} must be an object",
                            suggestion=f"Change skill {i} to object format"
                        ))
                    elif "id" not in skill:
                        issues.append(ValidationIssue(
                            check_id="agent_card_structure",
                            severity="critical",
                            message=f"Skill {i} missing required 'id' field",
                            suggestion=f"Add 'id' field to skill {i}"
                        ))
        
        # Validate provider structure
        if "provider" in agent_card:
            provider = agent_card["provider"]
            if not isinstance(provider, dict):
                issues.append(ValidationIssue(
                    check_id="agent_card_structure",
                    severity="critical",
                    message="Provider must be an object",
                    suggestion="Change provider to object format"
                ))
            elif "name" not in provider:
                issues.append(ValidationIssue(
                    check_id="agent_card_structure",
                    severity="critical",
                    message="Provider missing required 'name' field",
                    suggestion="Add 'name' field to provider"
                ))
        
        result = ValidationResult.PASS if not any(issue.severity == "critical" for issue in issues) else ValidationResult.FAIL
        return result, issues
    
    async def validate_well_known_endpoint(self, base_url: str) -> Tuple[ValidationResult, List[ValidationIssue]]:
        """Validate well-known endpoint availability"""
        issues = []
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        well_known_url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        
        try:
            async with self.session.get(well_known_url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' not in content_type:
                        issues.append(ValidationIssue(
                            check_id="agent_card_well_known",
                            severity="warning",
                            message=f"Well-known endpoint returns non-JSON content type: {content_type}",
                            suggestion="Set Content-Type to application/json"
                        ))
                    
                    try:
                        data = await response.json()
                        # Validate the returned agent card
                        card_result, card_issues = await self.validate_agent_card_structure(data)
                        issues.extend(card_issues)
                    except json.JSONDecodeError:
                        issues.append(ValidationIssue(
                            check_id="agent_card_well_known",
                            severity="critical",
                            message="Well-known endpoint returns invalid JSON",
                            suggestion="Fix JSON format in well-known endpoint response"
                        ))
                else:
                    issues.append(ValidationIssue(
                        check_id="agent_card_well_known",
                        severity="critical",
                        message=f"Well-known endpoint returned status {response.status}",
                        suggestion="Ensure /.well-known/agent.json endpoint returns 200 OK"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                check_id="agent_card_well_known",
                severity="critical",
                message=f"Failed to access well-known endpoint: {str(e)}",
                suggestion="Ensure well-known endpoint is accessible"
            ))
        
        result = ValidationResult.PASS if not any(issue.severity == "critical" for issue in issues) else ValidationResult.FAIL
        return result, issues
    
    async def validate_jsonrpc_compliance(self, endpoint_url: str) -> Tuple[ValidationResult, List[ValidationIssue]]:
        """Validate JSON-RPC 2.0 compliance"""
        issues = []
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Test basic JSON-RPC request
        test_request = {
            "jsonrpc": "2.0",
            "method": "agent/card",
            "id": "test-compliance-check"
        }
        
        try:
            async with self.session.post(endpoint_url, json=test_request) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        
                        # Validate JSON-RPC response format
                        if "jsonrpc" not in data:
                            issues.append(ValidationIssue(
                                check_id="jsonrpc_format",
                                severity="critical",
                                message="Response missing 'jsonrpc' field",
                                suggestion="Add 'jsonrpc': '2.0' to all responses"
                            ))
                        elif data["jsonrpc"] != "2.0":
                            issues.append(ValidationIssue(
                                check_id="jsonrpc_format",
                                severity="critical",
                                message=f"Invalid jsonrpc version: {data['jsonrpc']}",
                                suggestion="Set 'jsonrpc' to '2.0'"
                            ))
                        
                        if "id" not in data:
                            issues.append(ValidationIssue(
                                check_id="jsonrpc_format",
                                severity="critical",
                                message="Response missing 'id' field",
                                suggestion="Include request 'id' in response"
                            ))
                        elif data["id"] != test_request["id"]:
                            issues.append(ValidationIssue(
                                check_id="jsonrpc_format",
                                severity="critical",
                                message="Response 'id' doesn't match request 'id'",
                                suggestion="Ensure response 'id' matches request 'id'"
                            ))
                        
                        # Should have either 'result' or 'error'
                        if "result" not in data and "error" not in data:
                            issues.append(ValidationIssue(
                                check_id="jsonrpc_format",
                                severity="critical",
                                message="Response missing both 'result' and 'error' fields",
                                suggestion="Include either 'result' or 'error' in response"
                            ))
                        
                    except json.JSONDecodeError:
                        issues.append(ValidationIssue(
                            check_id="jsonrpc_format",
                            severity="critical",
                            message="Endpoint returns invalid JSON",
                            suggestion="Fix JSON format in endpoint responses"
                        ))
                else:
                    issues.append(ValidationIssue(
                        check_id="jsonrpc_format",
                        severity="warning",
                        message=f"Endpoint returned status {response.status}",
                        suggestion="Ensure endpoint returns appropriate HTTP status codes"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                check_id="jsonrpc_format",
                severity="critical",
                message=f"Failed to test JSON-RPC endpoint: {str(e)}",
                suggestion="Ensure JSON-RPC endpoint is accessible"
            ))
        
        result = ValidationResult.PASS if not any(issue.severity == "critical" for issue in issues) else ValidationResult.FAIL
        return result, issues
    
    async def validate_a2a_methods(self, endpoint_url: str) -> Tuple[ValidationResult, List[ValidationIssue]]:
        """Validate A2A method support"""
        issues = []
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Required A2A methods
        required_methods = [
            "agent/card",
            "message/send",
            "tasks/get"
        ]
        
        for method in required_methods:
            test_request = {
                "jsonrpc": "2.0",
                "method": method,
                "id": f"test-{method.replace('/', '-')}"
            }
            
            try:
                async with self.session.post(endpoint_url, json=test_request) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            
                            # Check if method is supported (not "method not found" error)
                            if "error" in data and data["error"].get("code") == -32601:
                                issues.append(ValidationIssue(
                                    check_id="a2a_methods",
                                    severity="critical",
                                    message=f"Required A2A method not supported: {method}",
                                    suggestion=f"Implement {method} method handler"
                                ))
                        
                        except json.JSONDecodeError:
                            issues.append(ValidationIssue(
                                check_id="a2a_methods",
                                severity="warning",
                                message=f"Invalid JSON response for method {method}",
                                suggestion="Fix JSON format in method responses"
                            ))
            
            except Exception as e:
                issues.append(ValidationIssue(
                    check_id="a2a_methods",
                    severity="warning",
                    message=f"Failed to test method {method}: {str(e)}",
                    suggestion=f"Ensure {method} method is accessible"
                ))
        
        result = ValidationResult.PASS if not any(issue.severity == "critical" for issue in issues) else ValidationResult.FAIL
        return result, issues
    
    async def run_full_validation(self, agent_url: str) -> ValidationReport:
        """Run complete A2A compliance validation"""
        report = ValidationReport()
        
        try:
            # Initialize HTTP session
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Run all validation checks
            validation_methods = [
                ("agent_card_well_known", self.validate_well_known_endpoint),
                ("jsonrpc_format", self.validate_jsonrpc_compliance),
                ("a2a_methods", self.validate_a2a_methods)
            ]
            
            for check_id, method in validation_methods:
                try:
                    if check_id == "agent_card_well_known":
                        result, issues = await method(agent_url)
                    else:
                        # Assume JSON-RPC endpoint is at /a2a/v1
                        endpoint_url = f"{agent_url.rstrip('/')}/a2a/v1"
                        result, issues = await method(endpoint_url)
                    
                    report.check_results[check_id] = {
                        "result": result.value,
                        "issues": len(issues),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    report.issues.extend(issues)
                    report.total_checks += 1
                    
                    if result == ValidationResult.PASS:
                        report.passed_checks += 1
                    elif result == ValidationResult.FAIL:
                        report.failed_checks += 1
                    elif result == ValidationResult.WARNING:
                        report.warning_checks += 1
                    else:
                        report.skipped_checks += 1
                
                except Exception as e:
                    logger.error(f"Error running validation check {check_id}: {e}")
                    report.check_results[check_id] = {
                        "result": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    report.total_checks += 1
                    report.failed_checks += 1
            
            # Calculate compliance score
            if report.total_checks > 0:
                report.compliance_score = (report.passed_checks / report.total_checks) * 100
            
            # Generate summary
            report.summary = {
                "compliance_score": report.compliance_score,
                "critical_issues": len([issue for issue in report.issues if issue.severity == "critical"]),
                "warning_issues": len([issue for issue in report.issues if issue.severity == "warning"]),
                "recommendations": self._generate_recommendations(report.issues)
            }
            
        finally:
            if self.session:
                await self.session.close()
                self.session = None
        
        return report
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on validation issues"""
        recommendations = []
        
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        if critical_issues:
            recommendations.append("Address all critical compliance issues before deployment")
        
        # Group issues by category
        issue_categories = {}
        for issue in issues:
            check = self.checks.get(issue.check_id)
            if check:
                category = check.category
                if category not in issue_categories:
                    issue_categories[category] = []
                issue_categories[category].append(issue)
        
        # Generate category-specific recommendations
        for category, category_issues in issue_categories.items():
            if len(category_issues) > 2:
                recommendations.append(f"Review and improve {category} implementation")
        
        return recommendations


# Global compliance validator instance
compliance_validator = A2AComplianceValidator()
