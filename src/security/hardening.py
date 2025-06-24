#!/usr/bin/env python3
"""
A2A Security Hardening

Production-grade security hardening for A2A multi-agent system.
"""

import os
import re
import time
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    details: Dict[str, Any]
    blocked: bool = False


class SecurityHardening:
    """A2A Security Hardening Manager"""
    
    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = {}
        self.security_events: List[SecurityEvent] = []
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # Load security patterns
        self.malicious_patterns = self._load_malicious_patterns()
        self.rate_limits = {}
        
    def _load_malicious_patterns(self) -> List[re.Pattern]:
        """Load patterns for detecting malicious input"""
        patterns = [
            # SQL Injection patterns
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(--|#|/\*|\*/)",
            
            # XSS patterns
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:|vbscript:|onload=|onerror=)",
            r"(<iframe|<object|<embed|<applet)",
            
            # Command injection patterns
            r"(\b(exec|eval|system|shell_exec|passthru)\s*\()",
            r"(\||&|;|\$\(|\`)",
            
            # Path traversal patterns
            r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
            
            # LDAP injection patterns
            r"(\*|\(|\)|&|\|)",
            
            # XXE patterns
            r"(<!ENTITY|<!DOCTYPE|SYSTEM|PUBLIC)",
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def validate_input(self, data: Any, source_ip: str = "unknown") -> tuple[bool, Optional[str]]:
        """Validate input for security threats"""
        
        if isinstance(data, str):
            return self._validate_string_input(data, source_ip)
        elif isinstance(data, dict):
            return self._validate_dict_input(data, source_ip)
        elif isinstance(data, list):
            return self._validate_list_input(data, source_ip)
        
        return True, None
    
    def _validate_string_input(self, text: str, source_ip: str) -> tuple[bool, Optional[str]]:
        """Validate string input for malicious patterns"""
        
        # Check length limits
        if len(text) > 50000:  # 50KB limit
            self._record_security_event(
                "oversized_input",
                ThreatLevel.MEDIUM,
                source_ip,
                {"length": len(text)}
            )
            return False, "Input too large"
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if pattern.search(text):
                self._record_security_event(
                    "malicious_pattern_detected",
                    ThreatLevel.HIGH,
                    source_ip,
                    {"pattern": pattern.pattern, "text": text[:100]}
                )
                return False, "Potentially malicious input detected"
        
        # Check for excessive special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_char_count > len(text) * 0.3:  # More than 30% special chars
            self._record_security_event(
                "suspicious_character_ratio",
                ThreatLevel.MEDIUM,
                source_ip,
                {"special_char_ratio": special_char_count / len(text)}
            )
        
        return True, None
    
    def _validate_dict_input(self, data: dict, source_ip: str) -> tuple[bool, Optional[str]]:
        """Validate dictionary input recursively"""
        
        # Check nesting depth
        if self._get_dict_depth(data) > 10:
            self._record_security_event(
                "excessive_nesting",
                ThreatLevel.MEDIUM,
                source_ip,
                {"depth": self._get_dict_depth(data)}
            )
            return False, "Excessive nesting depth"
        
        # Check number of keys
        if len(data) > 1000:
            self._record_security_event(
                "excessive_keys",
                ThreatLevel.MEDIUM,
                source_ip,
                {"key_count": len(data)}
            )
            return False, "Too many keys"
        
        # Validate each value recursively
        for key, value in data.items():
            # Validate key
            if isinstance(key, str):
                valid, error = self._validate_string_input(key, source_ip)
                if not valid:
                    return False, f"Invalid key: {error}"
            
            # Validate value
            valid, error = self.validate_input(value, source_ip)
            if not valid:
                return False, error
        
        return True, None
    
    def _validate_list_input(self, data: list, source_ip: str) -> tuple[bool, Optional[str]]:
        """Validate list input recursively"""
        
        # Check list size
        if len(data) > 10000:
            self._record_security_event(
                "oversized_list",
                ThreatLevel.MEDIUM,
                source_ip,
                {"length": len(data)}
            )
            return False, "List too large"
        
        # Validate each item
        for item in data:
            valid, error = self.validate_input(item, source_ip)
            if not valid:
                return False, error
        
        return True, None
    
    def _get_dict_depth(self, data: dict, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of dictionary"""
        if not isinstance(data, dict):
            return current_depth
        
        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self._get_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def check_rate_limiting(self, client_id: str, endpoint: str, limit: int = 100, window: int = 60) -> bool:
        """Enhanced rate limiting with per-endpoint limits"""
        
        key = f"{client_id}:{endpoint}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Clean old requests
        self.rate_limits[key] = [
            req_time for req_time in self.rate_limits[key]
            if req_time > now - window
        ]
        
        # Check limit
        if len(self.rate_limits[key]) >= limit:
            self._record_security_event(
                "rate_limit_exceeded",
                ThreatLevel.MEDIUM,
                client_id,
                {"endpoint": endpoint, "requests": len(self.rate_limits[key])}
            )
            return False
        
        # Add current request
        self.rate_limits[key].append(now)
        return True
    
    def detect_brute_force(self, client_ip: str, failed: bool = False) -> bool:
        """Detect brute force attacks"""
        
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        now = datetime.utcnow()
        
        # Clean old attempts (last 15 minutes)
        cutoff = now - timedelta(minutes=15)
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if attempt > cutoff
        ]
        
        if failed:
            self.failed_attempts[client_ip].append(now)
        
        # Check for brute force pattern
        recent_failures = len(self.failed_attempts[client_ip])
        
        if recent_failures >= 10:  # 10 failures in 15 minutes
            self._record_security_event(
                "brute_force_detected",
                ThreatLevel.HIGH,
                client_ip,
                {"failed_attempts": recent_failures}
            )
            self.block_ip(client_ip, "Brute force attack detected")
            return True
        
        return False
    
    def block_ip(self, ip_address: str, reason: str):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        
        self._record_security_event(
            "ip_blocked",
            ThreatLevel.HIGH,
            ip_address,
            {"reason": reason}
        )
        
        logger.warning(f"Blocked IP {ip_address}: {reason}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def analyze_request_anomalies(self, request_data: Dict[str, Any], source_ip: str) -> List[str]:
        """Analyze request for anomalies"""
        anomalies = []
        
        # Check request size anomalies
        if "content_length" in request_data:
            size = request_data["content_length"]
            if size > 10 * 1024 * 1024:  # 10MB
                anomalies.append("oversized_request")
            elif size == 0 and request_data.get("method") == "POST":
                anomalies.append("empty_post_request")
        
        # Check unusual headers
        headers = request_data.get("headers", {})
        suspicious_headers = [
            "x-forwarded-for", "x-real-ip", "x-originating-ip",
            "x-cluster-client-ip", "x-forwarded", "forwarded-for"
        ]
        
        for header in suspicious_headers:
            if header in headers:
                anomalies.append(f"suspicious_header_{header}")
        
        # Check user agent anomalies
        user_agent = headers.get("user-agent", "")
        if not user_agent:
            anomalies.append("missing_user_agent")
        elif len(user_agent) > 1000:
            anomalies.append("oversized_user_agent")
        elif any(bot in user_agent.lower() for bot in ["bot", "crawler", "spider", "scraper"]):
            anomalies.append("bot_user_agent")
        
        # Check for automation patterns
        if "automation" in user_agent.lower() or "selenium" in user_agent.lower():
            anomalies.append("automation_detected")
        
        # Record anomalies
        if anomalies:
            self._record_security_event(
                "request_anomalies",
                ThreatLevel.LOW,
                source_ip,
                {"anomalies": anomalies}
            )
        
        return anomalies
    
    def _record_security_event(self, event_type: str, threat_level: ThreatLevel, source_ip: str, details: Dict[str, Any]):
        """Record security event"""
        
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            details=details
        )
        
        self.security_events.append(event)
        
        # Keep only last 10000 events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        # Log high-priority events
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.warning(f"Security event: {event_type} from {source_ip} - {details}")
        
        # Auto-block for critical threats
        if threat_level == ThreatLevel.CRITICAL:
            self.block_ip(source_ip, f"Critical security event: {event_type}")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period"""
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [
            event for event in self.security_events
            if event.timestamp > cutoff
        ]
        
        # Count events by type and threat level
        event_counts = {}
        threat_counts = {level.value: 0 for level in ThreatLevel}
        
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            threat_counts[event.threat_level.value] += 1
        
        # Get top source IPs
        ip_counts = {}
        for event in recent_events:
            ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "events_by_type": event_counts,
            "events_by_threat_level": threat_counts,
            "top_source_ips": top_ips,
            "blocked_ips_count": len(self.blocked_ips),
            "active_rate_limits": len(self.rate_limits)
        }
    
    def export_security_report(self) -> str:
        """Export comprehensive security report"""
        
        summary = self.get_security_summary(24)
        
        report = f"""
A2A SECURITY REPORT
Generated: {datetime.utcnow().isoformat()}

SUMMARY (Last 24 Hours):
- Total Security Events: {summary['total_events']}
- Critical Events: {summary['events_by_threat_level']['critical']}
- High Priority Events: {summary['events_by_threat_level']['high']}
- Blocked IPs: {summary['blocked_ips_count']}

TOP SECURITY EVENTS:
"""
        
        for event_type, count in sorted(summary['events_by_type'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"- {event_type}: {count}\n"
        
        report += f"""
TOP SOURCE IPs:
"""
        
        for ip, count in summary['top_source_ips']:
            status = "BLOCKED" if ip in self.blocked_ips else "ACTIVE"
            report += f"- {ip}: {count} events ({status})\n"
        
        return report


# Global security hardening instance
security_hardening = SecurityHardening()
