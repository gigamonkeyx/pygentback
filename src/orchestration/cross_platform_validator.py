"""
Cross-Platform Historical Research Validation System
Validates sources across multiple major research databases and platforms
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.orchestration.research_models import ResearchSource
from src.orchestration.coordination_models import OrchestrationConfig
from src.orchestration.hathitrust_integration import HathiTrustBibliographicAPI

logger = logging.getLogger(__name__)


class ValidationPlatform(Enum):
    """Supported validation platforms"""
    HATHITRUST = "hathitrust"
    INTERNET_ARCHIVE = "internet_archive"
    LIBRARY_OF_CONGRESS = "library_of_congress"
    GALLICA = "gallica"
    JSTOR = "jstor"
    GOOGLE_BOOKS = "google_books"
    WORLDCAT = "worldcat"


@dataclass
class ValidationResult:
    """Result of cross-platform validation"""
    platform: ValidationPlatform
    confirmed: bool
    credibility_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsolidatedValidation:
    """Consolidated validation result across all platforms"""
    source: ResearchSource
    platform_results: List[ValidationResult]
    overall_credibility: float
    consensus_score: float
    verification_count: int
    authority_score: float
    recommendation: str
    validation_summary: Dict[str, Any] = field(default_factory=dict)


class CrossPlatformValidator:
    """
    Cross-platform validation system for historical research sources
    Validates sources across multiple major databases and platforms
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config
        self.enabled_platforms = set()
        self.platform_integrations = {}
        
        # Initialize available platforms
        self._initialize_platforms()
        
        # Validation thresholds
        self.credibility_threshold = 0.70
        self.consensus_threshold = 0.75
        self.authority_threshold = 0.80
        
        # Weighting for different platforms
        self.platform_weights = {
            ValidationPlatform.HATHITRUST: 0.25,
            ValidationPlatform.LIBRARY_OF_CONGRESS: 0.20,
            ValidationPlatform.INTERNET_ARCHIVE: 0.15,
            ValidationPlatform.JSTOR: 0.15,
            ValidationPlatform.GALLICA: 0.10,
            ValidationPlatform.GOOGLE_BOOKS: 0.10,
            ValidationPlatform.WORLDCAT: 0.05
        }
    
    def _initialize_platforms(self):
        """Initialize available platform integrations"""
        try:
            # Initialize HathiTrust integration
            self.platform_integrations[ValidationPlatform.HATHITRUST] = None  # Will be created async
            self.enabled_platforms.add(ValidationPlatform.HATHITRUST)
            
            # Add other platforms as they become available
            logger.info(f"Initialized {len(self.enabled_platforms)} validation platforms")
            
        except Exception as e:
            logger.error(f"Platform initialization failed: {e}")
    
    async def validate_sources(self, sources: List[ResearchSource]) -> List[ConsolidatedValidation]:
        """
        Validate a list of research sources across all available platforms
        """
        validated_sources = []
        
        for source in sources:
            try:
                consolidated = await self.validate_single_source(source)
                validated_sources.append(consolidated)
                
            except Exception as e:
                logger.error(f"Source validation failed: {e}")
                # Create minimal validation result
                minimal_validation = ConsolidatedValidation(
                    source=source,
                    platform_results=[],
                    overall_credibility=0.5,
                    consensus_score=0.0,
                    verification_count=0,
                    authority_score=0.0,
                    recommendation="validation_failed"
                )
                validated_sources.append(minimal_validation)
        
        return validated_sources
    
    async def validate_single_source(self, source: ResearchSource) -> ConsolidatedValidation:
        """
        Validate a single source across all available platforms
        """
        platform_results = []
        
        # Validate across all enabled platforms
        validation_tasks = []
        for platform in self.enabled_platforms:
            task = self._validate_on_platform(source, platform)
            validation_tasks.append(task)
        
        # Execute validations in parallel
        if validation_tasks:
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ValidationResult):
                    platform_results.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Platform validation failed: {result}")
        
        # Consolidate results
        consolidated = self._consolidate_validation_results(source, platform_results)
        
        return consolidated
    
    async def _validate_on_platform(self, source: ResearchSource, platform: ValidationPlatform) -> ValidationResult:
        """
        Validate source on a specific platform
        """
        try:
            if platform == ValidationPlatform.HATHITRUST:
                return await self._validate_on_hathitrust(source)
            elif platform == ValidationPlatform.INTERNET_ARCHIVE:
                return await self._validate_on_internet_archive(source)
            elif platform == ValidationPlatform.LIBRARY_OF_CONGRESS:
                return await self._validate_on_loc(source)
            elif platform == ValidationPlatform.GALLICA:
                return await self._validate_on_gallica(source)
            elif platform == ValidationPlatform.JSTOR:
                return await self._validate_on_jstor(source)
            else:
                # Default validation for unsupported platforms
                return ValidationResult(
                    platform=platform,
                    confirmed=False,
                    credibility_score=0.0,
                    metadata={'status': 'platform_not_implemented'}
                )
                
        except Exception as e:
            logger.error(f"Platform {platform} validation failed: {e}")
            return ValidationResult(
                platform=platform,
                confirmed=False,
                credibility_score=0.0,
                metadata={'status': 'validation_error', 'error': str(e)}
            )
    
    async def _validate_on_hathitrust(self, source: ResearchSource) -> ValidationResult:
        """
        Validate source using HathiTrust integration
        """
        try:
            # Initialize HathiTrust integration if not already done
            if self.platform_integrations[ValidationPlatform.HATHITRUST] is None:
                self.platform_integrations[ValidationPlatform.HATHITRUST] = HathiTrustBibliographicAPI(self.config)
            
            integration = self.platform_integrations[ValidationPlatform.HATHITRUST]
            
            # Use HathiTrust's verify_source method
            verification_result = await integration.verify_source(source)
            
            return ValidationResult(
                platform=ValidationPlatform.HATHITRUST,
                confirmed=verification_result.get('confirmed', False),
                credibility_score=verification_result.get('credibility_score', 0.0),
                metadata={
                    'platform_source': 'HathiTrust Digital Library',
                    'institutional_authority': verification_result.get('institutional_authority', False),
                    'verification_method': 'api_verification'
                },
                verification_details=verification_result
            )
            
        except Exception as e:
            logger.error(f"HathiTrust validation failed: {e}")
            return ValidationResult(
                platform=ValidationPlatform.HATHITRUST,                confirmed=False,
                credibility_score=0.0,
                metadata={'status': 'validation_error', 'error': str(e)}
            )
    
    async def _validate_on_internet_archive(self, source: ResearchSource) -> ValidationResult:
        """
        Validate source using Internet Archive
        """
        # Real implementation would integrate with Internet Archive's API
        # For now, return negative result - integration pending API access
        
        return ValidationResult(
            platform=ValidationPlatform.INTERNET_ARCHIVE,
            found=False,
            validation_score=0.0,
            confidence_level=0.0,
            matched_fields=[],
            api_response_time=0.0,            metadata={"status": "not_implemented"}
        )
    
    async def _validate_on_loc(self, source: ResearchSource) -> ValidationResult:
        """
        Validate source using Library of Congress
        """
        # Real implementation would integrate with LC's API
        # For now, return negative result - integration pending API access
        
        return ValidationResult(
            platform=ValidationPlatform.LIBRARY_OF_CONGRESS,
            found=False,
            validation_score=0.0,
            confidence_level=0.0,
            matched_fields=[],
            api_response_time=0.0,            metadata={"status": "not_implemented"}
        )
    
    async def _validate_on_gallica(self, source: ResearchSource) -> ValidationResult:
        """
        Validate source using Gallica (Bibliothèque nationale de France)
        """
        # Real implementation would integrate with Gallica's API
        # For now, return negative result - integration pending API access
        
        return ValidationResult(
            platform=ValidationPlatform.GALLICA,
            found=False,
            validation_score=0.0,
            confidence_level=0.0,
            matched_fields=[],
            api_response_time=0.0,            metadata={"status": "not_implemented"}
        )
    
    async def _validate_on_jstor(self, source: ResearchSource) -> ValidationResult:
        """
        Validate source using JSTOR
        """
        # Real implementation would integrate with JSTOR's API
        # For now, return negative result - integration pending API access
        
        return ValidationResult(
            platform=ValidationPlatform.JSTOR,
            found=False,
            validation_score=0.0,
            confidence_level=0.0,
            matched_fields=[],
            api_response_time=0.0,            metadata={"status": "not_implemented"}
        )
    
    def _consolidate_validation_results(self, source: ResearchSource, platform_results: List[ValidationResult]) -> ConsolidatedValidation:
        """
        Consolidate validation results from multiple platforms
        """
        if not platform_results:
            return ConsolidatedValidation(
                source=source,
                platform_results=[],
                overall_credibility=0.0,
                consensus_score=0.0,
                verification_count=0,
                authority_score=0.0,
                recommendation="no_validation_data"
            )
        
        # Calculate overall metrics
        confirmed_count = sum(1 for result in platform_results if result.confirmed)
        total_count = len(platform_results)
        
        # Weighted credibility score
        weighted_credibility = 0.0
        total_weight = 0.0
        
        for result in platform_results:
            weight = self.platform_weights.get(result.platform, 0.05)
            weighted_credibility += result.credibility_score * weight
            total_weight += weight
        
        overall_credibility = weighted_credibility / total_weight if total_weight > 0 else 0.0
        
        # Consensus score (percentage of platforms that confirmed)
        consensus_score = confirmed_count / total_count
        
        # Authority score (based on institutional authority indicators)
        authority_indicators = sum(1 for result in platform_results 
                                 if result.metadata.get('institutional_authority', False) or 
                                    result.metadata.get('national_authority', False) or
                                    result.metadata.get('academic_authority', False))
        authority_score = authority_indicators / total_count if total_count > 0 else 0.0
        
        # Generate recommendation
        recommendation = self._generate_recommendation(overall_credibility, consensus_score, authority_score)
        
        # Create validation summary
        validation_summary = {
            'platforms_checked': [result.platform.value for result in platform_results],
            'confirmed_platforms': [result.platform.value for result in platform_results if result.confirmed],
            'highest_credibility': max(result.credibility_score for result in platform_results) if platform_results else 0.0,
            'institutional_authorities': authority_indicators,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        return ConsolidatedValidation(
            source=source,
            platform_results=platform_results,
            overall_credibility=overall_credibility,
            consensus_score=consensus_score,
            verification_count=confirmed_count,
            authority_score=authority_score,
            recommendation=recommendation,
            validation_summary=validation_summary
        )
    
    def _generate_recommendation(self, credibility: float, consensus: float, authority: float) -> str:
        """
        Generate validation recommendation based on scores
        """
        if credibility >= self.authority_threshold and consensus >= self.consensus_threshold and authority >= 0.5:
            return "highly_recommended"
        elif credibility >= self.credibility_threshold and consensus >= 0.5:
            return "recommended"
        elif credibility >= 0.5 and consensus >= 0.3:
            return "acceptable"
        elif credibility >= 0.3:
            return "use_with_caution"
        else:
            return "not_recommended"
    
    async def get_validation_summary(self, validated_sources: List[ConsolidatedValidation]) -> Dict[str, Any]:
        """
        Generate summary statistics for validation results
        """
        if not validated_sources:
            return {
                'total_sources': 0,
                'validation_summary': 'No sources validated'
            }
        
        total_sources = len(validated_sources)
        highly_recommended = sum(1 for v in validated_sources if v.recommendation == "highly_recommended")
        recommended = sum(1 for v in validated_sources if v.recommendation == "recommended")
        acceptable = sum(1 for v in validated_sources if v.recommendation == "acceptable")
        
        avg_credibility = sum(v.overall_credibility for v in validated_sources) / total_sources
        avg_consensus = sum(v.consensus_score for v in validated_sources) / total_sources
        avg_authority = sum(v.authority_score for v in validated_sources) / total_sources
        
        # Platform coverage analysis
        platform_coverage = {}
        for validation in validated_sources:
            for result in validation.platform_results:
                platform = result.platform.value
                if platform not in platform_coverage:
                    platform_coverage[platform] = {'total': 0, 'confirmed': 0}
                platform_coverage[platform]['total'] += 1
                if result.confirmed:
                    platform_coverage[platform]['confirmed'] += 1
        
        return {
            'total_sources': total_sources,
            'highly_recommended': highly_recommended,
            'recommended': recommended,
            'acceptable': acceptable,
            'not_recommended': total_sources - highly_recommended - recommended - acceptable,
            'average_credibility': round(avg_credibility, 3),
            'average_consensus': round(avg_consensus, 3),
            'average_authority': round(avg_authority, 3),
            'platform_coverage': platform_coverage,
            'validation_quality': 'excellent' if avg_credibility > 0.8 else 'good' if avg_credibility > 0.6 else 'moderate'
        }


# Factory function for easy integration
async def create_cross_platform_validator(config: OrchestrationConfig) -> CrossPlatformValidator:
    """
    Factory function to create and initialize cross-platform validator
    """
    validator = CrossPlatformValidator(config)
    return validator
