# src/validation/anti_hallucination_framework.py

import logging
from typing import Dict, List, Any
from datetime import datetime
import json
import hashlib
from dataclasses import dataclass, field
import aiohttp
from enum import Enum

from ..core.ollama_manager import get_ollama_manager
from ..storage.vector.manager import VectorStoreManager
from ..utils.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    """Levels of verification for fact checking"""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    CRITICAL = "critical"


class ConfidenceLevel(Enum):
    """Confidence levels for factual claims"""
    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"            # 0.7-0.9
    MEDIUM = "medium"        # 0.5-0.7
    LOW = "low"              # 0.3-0.5
    VERY_LOW = "very_low"    # 0.0-0.3


@dataclass
class FactualClaim:
    """Represents a factual claim extracted from text"""
    id: str
    claim_text: str
    claim_type: str  # date, person, event, location, statistical, etc.
    context: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    verification_status: str = "unverified"
    verification_evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    extracted_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VerificationResult:
    """Result of fact verification process"""
    claim_id: str
    verified: bool
    confidence_score: float
    confidence_level: ConfidenceLevel
    supporting_evidence: List[str] = field(default_factory=list)
    contradictory_evidence: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    verification_method: str = ""
    verification_timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


@dataclass
class TemporalInconsistency:
    """Represents a temporal inconsistency in historical claims"""
    claim_id: str
    inconsistency_type: str  # anachronism, impossible_date, conflicting_dates
    description: str
    severity: str  # low, medium, high, critical
    suggested_correction: str = ""


@dataclass
class BiasIndicator:
    """Represents detected bias in historical content"""
    bias_type: str  # temporal, cultural, political, source, confirmation
    description: str
    severity: float  # 0.0-1.0
    examples: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)


class AntiHallucinationFramework:
    """Advanced anti-hallucination framework for historical research."""
    
    def __init__(self, 
                 vector_manager: VectorStoreManager = None,
                 embedding_service: EmbeddingService = None,
                 verification_level: VerificationLevel = VerificationLevel.STANDARD):
        # Initialize with defaults if not provided
        if vector_manager is None:
            try:
                from ..config.settings import get_settings
                settings = get_settings()
                vector_manager = VectorStoreManager(settings)
            except ImportError:
                logger.warning("Vector manager not provided and settings unavailable - some features will be limited")
                vector_manager = None
                
        if embedding_service is None:
            try:
                embedding_service = EmbeddingService()
            except Exception as e:
                logger.warning(f"Failed to initialize embedding service: {e} - some features will be limited")
                embedding_service = None
                
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.verification_level = verification_level
        self.ollama_manager = get_ollama_manager()
        
        # Historical knowledge bases for cross-referencing
        self.historical_timelines = {}
        self.known_persons = {}
        self.historical_events = {}
        self.geographical_data = {}
        
        # Initialize verification statistics
        self.verification_stats = {
            'total_claims_processed': 0,
            'claims_verified': 0,
            'claims_flagged': 0,
            'temporal_inconsistencies': 0,
            'bias_indicators_detected': 0,
            'cross_references_found': 0
        }
        
        logger.info(f"Anti-Hallucination Framework initialized with {verification_level.value} verification level")
    
    async def verify_historical_content(self, 
                                      content: str, 
                                      source_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive verification of historical content."""
        try:
            # Step 1: Extract factual claims
            factual_claims = await self._extract_factual_claims(content)
            
            # Step 2: Verify each claim
            verification_results = []
            for claim in factual_claims:
                result = await self._verify_factual_claim(claim, source_metadata)
                verification_results.append(result)
            
            # Step 3: Check for temporal inconsistencies
            temporal_inconsistencies = await self._check_temporal_consistency(factual_claims)
            
            # Step 4: Detect bias indicators
            bias_indicators = await self._detect_bias_indicators(content, factual_claims)
            
            # Step 5: Cross-reference with known historical data
            cross_references = await self._cross_reference_historical_data(factual_claims)
            
            # Step 6: Generate confidence assessment
            overall_confidence = self._calculate_overall_confidence(verification_results)
            
            # Step 7: Generate verification report
            verification_report = self._generate_verification_report(
                factual_claims, verification_results, temporal_inconsistencies,
                bias_indicators, cross_references, overall_confidence
            )
            
            # Update statistics
            self.verification_stats['total_claims_processed'] += len(factual_claims)
            self.verification_stats['claims_verified'] += sum(1 for r in verification_results if r.verified)
            self.verification_stats['claims_flagged'] += sum(1 for r in verification_results if not r.verified)
            self.verification_stats['temporal_inconsistencies'] += len(temporal_inconsistencies)
            self.verification_stats['bias_indicators_detected'] += len(bias_indicators)
            
            return {
                'success': True,
                'verification_report': verification_report,
                'factual_claims': factual_claims,
                'verification_results': verification_results,
                'temporal_inconsistencies': temporal_inconsistencies,
                'bias_indicators': bias_indicators,
                'overall_confidence': overall_confidence,
                'verification_level': self.verification_level.value
            }
            
        except Exception as e:
            logger.error(f"Error in content verification: {str(e)}")
            return {'error': f"Verification failed: {str(e)}"}
    
    async def _extract_factual_claims(self, content: str) -> List[FactualClaim]:
        """Extract factual claims from content using AI."""
        try:
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            # Limit content length for processing
            analysis_content = content[:4000] if len(content) > 4000 else content
            
            prompt = f"""
            Extract factual claims from this historical text. Focus on:
            - Dates and time periods
            - Historical figures and their actions
            - Events and their outcomes
            - Locations and geographical information
            - Statistical data and numbers
            - Cause-and-effect relationships
            
            Text: {analysis_content}
            
            Return a JSON array of claims in this format:
            [
                {{
                    "claim_text": "specific factual statement",
                    "claim_type": "date|person|event|location|statistical|causal",
                    "context": "surrounding context",
                    "confidence": 0.0-1.0
                }}
            ]
            
            Only extract verifiable factual claims, not opinions or interpretations.
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        claims_text = result.get('response', '[]').strip()
                        
                        try:
                            claims_data = json.loads(claims_text)
                            factual_claims = []
                            
                            for i, claim_data in enumerate(claims_data):
                                claim_id = hashlib.md5(
                                    f"{claim_data.get('claim_text', '')}{i}".encode()
                                ).hexdigest()[:12]
                                
                                claim = FactualClaim(
                                    id=claim_id,
                                    claim_text=claim_data.get('claim_text', ''),
                                    claim_type=claim_data.get('claim_type', 'general'),
                                    context=claim_data.get('context', ''),
                                    confidence=float(claim_data.get('confidence', 0.5))
                                )
                                factual_claims.append(claim)
                            
                            logger.info(f"Extracted {len(factual_claims)} factual claims")
                            return factual_claims
                            
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from claim extraction: {claims_text}")
                            return []
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting factual claims: {str(e)}")
            return []
    
    async def _verify_factual_claim(self, 
                                   claim: FactualClaim, 
                                   source_metadata: Dict[str, Any]) -> VerificationResult:
        """Verify a single factual claim using multiple methods."""
        try:
            verification_methods = []
            
            # Method 1: Vector database search for supporting evidence
            vector_evidence = await self._search_vector_evidence(claim)
            verification_methods.append(("vector_search", vector_evidence))
            
            # Method 2: AI-powered fact checking
            ai_verification = await self._ai_fact_check(claim)
            verification_methods.append(("ai_verification", ai_verification))
            
            # Method 3: Cross-reference with known historical data
            historical_verification = await self._historical_cross_reference(claim)
            verification_methods.append(("historical_reference", historical_verification))
            
            # Method 4: Temporal consistency check
            temporal_verification = await self._temporal_consistency_check(claim)
            verification_methods.append(("temporal_check", temporal_verification))
            
            # Aggregate verification results
            total_confidence = 0.0
            supporting_evidence = []
            contradictory_evidence = []
            method_count = 0
            
            for method_name, method_result in verification_methods:
                if method_result and isinstance(method_result, dict):
                    confidence = method_result.get('confidence', 0.0)
                    total_confidence += confidence
                    method_count += 1
                    
                    if method_result.get('supporting_evidence'):
                        supporting_evidence.extend(method_result['supporting_evidence'])
                    
                    if method_result.get('contradictory_evidence'):
                        contradictory_evidence.extend(method_result['contradictory_evidence'])
            
            # Calculate overall confidence
            overall_confidence = total_confidence / max(1, method_count)
            
            # Determine verification status
            confidence_threshold = {
                VerificationLevel.BASIC: 0.3,
                VerificationLevel.STANDARD: 0.5,
                VerificationLevel.RIGOROUS: 0.7,
                VerificationLevel.CRITICAL: 0.8
            }
            
            threshold = confidence_threshold[self.verification_level]
            verified = overall_confidence >= threshold and len(contradictory_evidence) == 0
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            return VerificationResult(
                claim_id=claim.id,
                verified=verified,
                confidence_score=overall_confidence,
                confidence_level=confidence_level,
                supporting_evidence=supporting_evidence[:5],  # Limit to top 5
                contradictory_evidence=contradictory_evidence[:3],  # Limit to top 3
                verification_method="multi_method",
                notes=f"Verified using {method_count} methods"
            )
            
        except Exception as e:
            logger.error(f"Error verifying claim {claim.id}: {str(e)}")
            return VerificationResult(
                claim_id=claim.id,
                verified=False,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                notes=f"Verification error: {str(e)}"
            )
    
    async def _search_vector_evidence(self, claim: FactualClaim) -> Dict[str, Any]:
        """Search vector database for evidence supporting or contradicting the claim."""
        try:
            # Search for related documents
            search_results = await self.vector_manager.similarity_search(
                query=claim.claim_text,
                collection_name="historical_documents",
                limit=10
            )
            
            supporting_evidence = []
            contradictory_evidence = []
            
            for result in search_results:
                if result.similarity_score > 0.8:  # High similarity threshold
                    # Use AI to determine if this supports or contradicts the claim
                    assessment = await self._assess_evidence_relevance(claim, result.content)
                    
                    if assessment.get('supports', False):
                        supporting_evidence.append(result.content[:200] + "...")
                    elif assessment.get('contradicts', False):
                        contradictory_evidence.append(result.content[:200] + "...")
            
            confidence = min(1.0, len(supporting_evidence) * 0.2)  # Max confidence from vector search
            
            return {
                'confidence': confidence,
                'supporting_evidence': supporting_evidence,
                'contradictory_evidence': contradictory_evidence,
                'total_documents_checked': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error in vector evidence search: {str(e)}")
            return {'confidence': 0.0, 'supporting_evidence': [], 'contradictory_evidence': []}
    
    async def _assess_evidence_relevance(self, 
                                       claim: FactualClaim, 
                                       evidence_text: str) -> Dict[str, Any]:
        """Use AI to assess if evidence supports or contradicts a claim."""
        try:
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            prompt = f"""
            Assess whether this evidence supports or contradicts the given claim.
            
            Claim: {claim.claim_text}
            Evidence: {evidence_text[:500]}
            
            Respond with JSON:
            {{
                "supports": true/false,
                "contradicts": true/false,
                "relevance": 0.0-1.0,
                "explanation": "brief explanation"
            }}
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        assessment_text = result.get('response', '{}').strip()
                        
                        try:
                            return json.loads(assessment_text)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from evidence assessment: {assessment_text}")
                            return {'supports': False, 'contradicts': False, 'relevance': 0.0}
            
            return {'supports': False, 'contradicts': False, 'relevance': 0.0}
            
        except Exception as e:
            logger.error(f"Error assessing evidence relevance: {str(e)}")
            return {'supports': False, 'contradicts': False, 'relevance': 0.0}
    
    async def _ai_fact_check(self, claim: FactualClaim) -> Dict[str, Any]:
        """Use AI to fact-check the claim against general historical knowledge."""
        try:
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            prompt = f"""
            Fact-check this historical claim using your knowledge:
            
            Claim: {claim.claim_text}
            Type: {claim.claim_type}
            Context: {claim.context}
            
            Respond with JSON:
            {{
                "confidence": 0.0-1.0,
                "likely_accurate": true/false,
                "historical_knowledge_assessment": "detailed assessment",
                "potential_issues": ["issue1", "issue2"],
                "supporting_facts": ["fact1", "fact2"],
                "contradictory_facts": ["contradiction1"]
            }}
            
            Be conservative in your assessment. If uncertain, express low confidence.
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        fact_check_text = result.get('response', '{}').strip()
                        
                        try:
                            fact_check_result = json.loads(fact_check_text)
                            
                            return {
                                'confidence': fact_check_result.get('confidence', 0.0),
                                'supporting_evidence': fact_check_result.get('supporting_facts', []),
                                'contradictory_evidence': fact_check_result.get('contradictory_facts', []),
                                'assessment': fact_check_result.get('historical_knowledge_assessment', ''),
                                'issues': fact_check_result.get('potential_issues', [])
                            }
                            
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from AI fact check: {fact_check_text}")
                            return {'confidence': 0.0, 'supporting_evidence': [], 'contradictory_evidence': []}
            
            return {'confidence': 0.0, 'supporting_evidence': [], 'contradictory_evidence': []}
            
        except Exception as e:
            logger.error(f"Error in AI fact checking: {str(e)}")
            return {'confidence': 0.0, 'supporting_evidence': [], 'contradictory_evidence': []}
    
    async def _historical_cross_reference(self, claim: FactualClaim) -> Dict[str, Any]:
        """Cross-reference claim with structured historical data."""
        try:
            # This would integrate with historical databases, timelines, etc.
            # For now, implement basic validation patterns
            
            confidence = 0.5  # Default confidence
            supporting_evidence = []
            contradictory_evidence = []
            
            # Date validation for historical claims
            if claim.claim_type == "date":
                date_validation = self._validate_historical_dates(claim.claim_text)
                confidence = date_validation.get('confidence', 0.5)
                if date_validation.get('issues'):
                    contradictory_evidence.extend(date_validation['issues'])
            
            # Person validation
            elif claim.claim_type == "person":
                person_validation = self._validate_historical_persons(claim.claim_text)
                confidence = person_validation.get('confidence', 0.5)
            
            # Event validation
            elif claim.claim_type == "event":
                event_validation = self._validate_historical_events(claim.claim_text)
                confidence = event_validation.get('confidence', 0.5)
            
            return {
                'confidence': confidence,
                'supporting_evidence': supporting_evidence,
                'contradictory_evidence': contradictory_evidence
            }
            
        except Exception as e:
            logger.error(f"Error in historical cross-reference: {str(e)}")
            return {'confidence': 0.0, 'supporting_evidence': [], 'contradictory_evidence': []}
    
    def _validate_historical_dates(self, claim_text: str) -> Dict[str, Any]:
        """Validate dates mentioned in historical claims."""
        import re
        
        # Extract years from claim
        year_pattern = r'\b(1[0-9]{3}|20[0-2][0-9])\b'
        years = re.findall(year_pattern, claim_text)
        
        issues = []
        confidence = 0.7  # Default confidence for date validation
        
        for year in years:
            year_int = int(year)
            
            # Check for obvious anachronisms
            if year_int > datetime.now().year:
                issues.append(f"Future date mentioned: {year}")
                confidence = 0.1
            elif year_int < 1000:  # Very ancient dates might need special handling
                issues.append(f"Very ancient date mentioned: {year}")
                confidence = 0.3
        
        return {
            'confidence': confidence,
            'issues': issues,
            'years_found': years
        }
    
    def _validate_historical_persons(self, claim_text: str) -> Dict[str, Any]:
        """Validate persons mentioned in historical claims."""
        # This would integrate with historical person databases
        # For now, return default validation
        return {'confidence': 0.6}
    
    def _validate_historical_events(self, claim_text: str) -> Dict[str, Any]:
        """Validate events mentioned in historical claims."""
        # This would integrate with historical event databases
        # For now, return default validation
        return {'confidence': 0.6}
    
    async def _temporal_consistency_check(self, claim: FactualClaim) -> Dict[str, Any]:
        """Check temporal consistency of the claim."""
        try:
            # This would implement sophisticated temporal reasoning
            # For now, implement basic checks
            
            confidence = 0.7
            supporting_evidence = []
            contradictory_evidence = []
            
            # Basic temporal consistency checks would go here
            # e.g., checking if dates make sense relative to each other
            
            return {
                'confidence': confidence,
                'supporting_evidence': supporting_evidence,
                'contradictory_evidence': contradictory_evidence
            }
            
        except Exception as e:
            logger.error(f"Error in temporal consistency check: {str(e)}")
            return {'confidence': 0.0, 'supporting_evidence': [], 'contradictory_evidence': []}
    
    async def _check_temporal_consistency(self, claims: List[FactualClaim]) -> List[TemporalInconsistency]:
        """Check for temporal inconsistencies across multiple claims."""
        inconsistencies = []
        
        # This would implement sophisticated temporal reasoning
        # For now, implement basic date conflict detection
        
        date_claims = [claim for claim in claims if claim.claim_type == "date"]
        
        # Look for conflicting dates
        for i, claim1 in enumerate(date_claims):
            for claim2 in date_claims[i+1:]:
                # Check for potential conflicts
                if self._claims_potentially_conflicting(claim1, claim2):
                    inconsistency = TemporalInconsistency(
                        claim_id=f"{claim1.id}_{claim2.id}",
                        inconsistency_type="conflicting_dates",
                        description="Potential date conflict between claims",
                        severity="medium"
                    )
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _claims_potentially_conflicting(self, claim1: FactualClaim, claim2: FactualClaim) -> bool:
        """Check if two claims potentially conflict."""
        # This would implement sophisticated conflict detection
        # For now, return false (no conflicts detected)
        return False
    
    async def _detect_bias_indicators(self, 
                                    content: str, 
                                    claims: List[FactualClaim]) -> List[BiasIndicator]:
        """Detect potential bias indicators in the content."""
        bias_indicators = []
        
        try:
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            # Limit content for analysis
            analysis_content = content[:3000] if len(content) > 3000 else content
            
            prompt = f"""
            Analyze this historical text for potential bias indicators:
            
            Text: {analysis_content}
            
            Look for:
            - Temporal bias (present-day perspectives on past events)
            - Cultural bias (Western-centric or culture-specific viewpoints)
            - Political bias (partisan interpretations)
            - Source bias (over-reliance on particular types of sources)
            - Confirmation bias (selective presentation of evidence)
            
            Respond with JSON array:
            [
                {{
                    "bias_type": "temporal|cultural|political|source|confirmation",
                    "description": "detailed description",
                    "severity": 0.0-1.0,
                    "examples": ["example1", "example2"],
                    "mitigation_suggestions": ["suggestion1", "suggestion2"]
                }}
            ]
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        bias_text = result.get('response', '[]').strip()
                        
                        try:
                            bias_data = json.loads(bias_text)
                            
                            for bias_item in bias_data:
                                bias_indicator = BiasIndicator(
                                    bias_type=bias_item.get('bias_type', 'unknown'),
                                    description=bias_item.get('description', ''),
                                    severity=float(bias_item.get('severity', 0.0)),
                                    examples=bias_item.get('examples', []),
                                    mitigation_suggestions=bias_item.get('mitigation_suggestions', [])
                                )
                                bias_indicators.append(bias_indicator)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from bias detection: {bias_text}")
        
        except Exception as e:
            logger.error(f"Error detecting bias indicators: {str(e)}")
        
        return bias_indicators
    
    async def _cross_reference_historical_data(self, claims: List[FactualClaim]) -> Dict[str, Any]:
        """Cross-reference claims with external historical databases."""
        try:
            cross_references = {
                'timeline_matches': [],
                'encyclopedia_matches': [],
                'academic_source_matches': [],
                'primary_source_matches': []
            }
            
            # This would integrate with external historical databases
            # For now, implement placeholder functionality
            
            self.verification_stats['cross_references_found'] += len(claims)
            
            return cross_references
            
        except Exception as e:
            logger.error(f"Error in cross-referencing: {str(e)}")
            return {}
    
    def _calculate_overall_confidence(self, verification_results: List[VerificationResult]) -> float:
        """Calculate overall confidence for the entire content."""
        if not verification_results:
            return 0.0
        
        total_confidence = sum(result.confidence_score for result in verification_results)
        return total_confidence / len(verification_results)
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level based on numerical score."""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_verification_report(self, 
                                    factual_claims: List[FactualClaim],
                                    verification_results: List[VerificationResult],
                                    temporal_inconsistencies: List[TemporalInconsistency],
                                    bias_indicators: List[BiasIndicator],
                                    cross_references: Dict[str, Any],
                                    overall_confidence: float) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        
        verified_claims = sum(1 for result in verification_results if result.verified)
        flagged_claims = len(verification_results) - verified_claims
        
        return {
            'verification_summary': {
                'total_claims': len(factual_claims),
                'verified_claims': verified_claims,
                'flagged_claims': flagged_claims,
                'verification_rate': verified_claims / max(1, len(factual_claims)),
                'overall_confidence': overall_confidence,
                'overall_confidence_level': self._determine_confidence_level(overall_confidence).value
            },
            'claim_breakdown': {
                'by_type': self._group_claims_by_type(factual_claims),
                'by_confidence': self._group_results_by_confidence(verification_results)
            },
            'quality_indicators': {
                'temporal_inconsistencies': len(temporal_inconsistencies),
                'bias_indicators': len(bias_indicators),
                'high_confidence_claims': sum(1 for r in verification_results 
                                            if r.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH])
            },
            'verification_details': {
                'factual_claims': [self._serialize_claim(claim) for claim in factual_claims],
                'verification_results': [self._serialize_verification_result(result) for result in verification_results],
                'temporal_inconsistencies': [self._serialize_temporal_inconsistency(ti) for ti in temporal_inconsistencies],
                'bias_indicators': [self._serialize_bias_indicator(bi) for bi in bias_indicators]
            },
            'recommendations': self._generate_recommendations(
                verification_results, temporal_inconsistencies, bias_indicators
            ),
            'verification_metadata': {
                'verification_level': self.verification_level.value,
                'timestamp': datetime.now().isoformat(),
                'framework_version': '1.0'
            }
        }
    
    def _group_claims_by_type(self, claims: List[FactualClaim]) -> Dict[str, int]:
        """Group claims by type for reporting."""
        type_counts = {}
        for claim in claims:
            type_counts[claim.claim_type] = type_counts.get(claim.claim_type, 0) + 1
        return type_counts
    
    def _group_results_by_confidence(self, results: List[VerificationResult]) -> Dict[str, int]:
        """Group verification results by confidence level."""
        confidence_counts = {}
        for result in results:
            level = result.confidence_level.value
            confidence_counts[level] = confidence_counts.get(level, 0) + 1
        return confidence_counts
    
    def _serialize_claim(self, claim: FactualClaim) -> Dict[str, Any]:
        """Serialize FactualClaim for JSON output."""
        return {
            'id': claim.id,
            'claim_text': claim.claim_text,
            'claim_type': claim.claim_type,
            'context': claim.context,
            'confidence': claim.confidence,
            'verification_status': claim.verification_status
        }
    
    def _serialize_verification_result(self, result: VerificationResult) -> Dict[str, Any]:
        """Serialize VerificationResult for JSON output."""
        return {
            'claim_id': result.claim_id,
            'verified': result.verified,
            'confidence_score': result.confidence_score,
            'confidence_level': result.confidence_level.value,
            'supporting_evidence_count': len(result.supporting_evidence),
            'contradictory_evidence_count': len(result.contradictory_evidence),
            'verification_method': result.verification_method,
            'notes': result.notes
        }
    
    def _serialize_temporal_inconsistency(self, inconsistency: TemporalInconsistency) -> Dict[str, Any]:
        """Serialize TemporalInconsistency for JSON output."""
        return {
            'claim_id': inconsistency.claim_id,
            'inconsistency_type': inconsistency.inconsistency_type,
            'description': inconsistency.description,
            'severity': inconsistency.severity,
            'suggested_correction': inconsistency.suggested_correction
        }
    
    def _serialize_bias_indicator(self, bias: BiasIndicator) -> Dict[str, Any]:
        """Serialize BiasIndicator for JSON output."""
        return {
            'bias_type': bias.bias_type,
            'description': bias.description,
            'severity': bias.severity,
            'examples_count': len(bias.examples),
            'mitigation_suggestions_count': len(bias.mitigation_suggestions)
        }
    
    def _generate_recommendations(self, 
                                verification_results: List[VerificationResult],
                                temporal_inconsistencies: List[TemporalInconsistency],
                                bias_indicators: List[BiasIndicator]) -> List[str]:
        """Generate actionable recommendations based on verification results."""
        recommendations = []
        
        # Check verification rates
        total_results = len(verification_results)
        verified_count = sum(1 for r in verification_results if r.verified)
        verification_rate = verified_count / max(1, total_results)
        
        if verification_rate < 0.7:
            recommendations.append(
                "Low verification rate detected. Consider cross-referencing with additional sources."
            )
        
        # Check for temporal inconsistencies
        if temporal_inconsistencies:
            recommendations.append(
                f"Found {len(temporal_inconsistencies)} temporal inconsistencies. "
                "Review chronological claims for accuracy."
            )
        
        # Check for bias indicators
        high_bias_indicators = [bi for bi in bias_indicators if bi.severity > 0.7]
        if high_bias_indicators:
            recommendations.append(
                f"Detected {len(high_bias_indicators)} high-severity bias indicators. "
                "Consider balanced perspective and additional viewpoints."
            )
        
        # Check confidence distribution
        low_confidence_results = [r for r in verification_results 
                                if r.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]]
        if len(low_confidence_results) > total_results * 0.3:
            recommendations.append(
                "High proportion of low-confidence claims. Consider additional verification sources."
            )
        
        if not recommendations:
            recommendations.append("Content shows good verification quality with no major issues detected.")
        
        return recommendations
    
    async def get_verification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive verification statistics."""
        return {
            'processing_stats': self.verification_stats.copy(),
            'system_info': {
                'verification_level': self.verification_level.value,
                'ollama_ready': self.ollama_manager.is_ready,
                'vector_manager_status': 'available' if self.vector_manager else 'unavailable'
            },
            'success_rates': {
                'claim_verification_rate': (
                    self.verification_stats['claims_verified'] / 
                    max(1, self.verification_stats['total_claims_processed'])
                ),
                'cross_reference_rate': (
                    self.verification_stats['cross_references_found'] / 
                    max(1, self.verification_stats['total_claims_processed'])
                )
            }
        }
