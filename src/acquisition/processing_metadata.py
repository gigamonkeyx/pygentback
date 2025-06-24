"""
Document processing metadata system for tracking extraction methods and statistics.
"""
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStatistics:
    """Statistics for document processing operations."""
    total_documents: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    ocr_used_count: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    extraction_methods_used: Dict[str, int] = None
    
    def __post_init__(self):
        if self.extraction_methods_used is None:
            self.extraction_methods_used = {}

@dataclass
class DocumentProcessingMetadata:
    """Comprehensive metadata for document processing."""
    document_id: str
    processing_timestamp: str
    extraction_method: str  # text, dict, html, xml, ocr
    extraction_success: bool
    quality_score: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    text_length: Optional[int] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    ocr_confidence: Optional[float] = None
    language_detected: Optional[str] = None
    error_message: Optional[str] = None
    fallback_methods_tried: List[str] = None
    
    def __post_init__(self):
        if self.fallback_methods_tried is None:
            self.fallback_methods_tried = []
        if not self.processing_timestamp:
            self.processing_timestamp = datetime.utcnow().isoformat()

class DocumentProcessingTracker:
    """Track and manage document processing metadata and statistics."""
    
    def __init__(self, storage_system):
        self.storage_system = storage_system
        self.processing_dir = storage_system.directories['metadata'] / "processing"
        self.processing_dir.mkdir(exist_ok=True)
        
        self.stats_file = self.processing_dir / "processing_statistics.json"
        self._ensure_stats_file_exists()
    
    def _ensure_stats_file_exists(self):
        """Ensure processing statistics file exists."""
        if not self.stats_file.exists():
            initial_stats = ProcessingStatistics()
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(initial_stats), f, indent=2)
    
    def record_processing_result(self, metadata: DocumentProcessingMetadata):
        """Record the result of a document processing operation."""
        try:
            # Save individual processing metadata
            processing_file = self.processing_dir / f"{metadata.document_id}_processing.json"
            with open(processing_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
            
            # Update overall statistics
            self._update_statistics(metadata)
            
            logger.info(f"Recorded processing metadata for {metadata.document_id}")
            
        except Exception as e:
            logger.error(f"Failed to record processing metadata for {metadata.document_id}: {e}")
    
    def _update_statistics(self, metadata: DocumentProcessingMetadata):
        """Update overall processing statistics."""
        try:
            # Load current statistics
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
            
            stats = ProcessingStatistics(**stats_data)
            
            # Update counters
            stats.total_documents += 1
            
            if metadata.extraction_success:
                stats.successful_extractions += 1
            else:
                stats.failed_extractions += 1
            
            if metadata.extraction_method == 'ocr':
                stats.ocr_used_count += 1
            
            # Update method usage tracking
            method = metadata.extraction_method
            stats.extraction_methods_used[method] = stats.extraction_methods_used.get(method, 0) + 1
            
            # Update averages (running average)
            if metadata.processing_time_seconds is not None:
                current_avg_time = stats.average_processing_time
                new_count = stats.total_documents
                stats.average_processing_time = (
                    (current_avg_time * (new_count - 1) + metadata.processing_time_seconds) / new_count
                )
            
            if metadata.quality_score is not None:
                current_avg_quality = stats.average_quality_score
                new_count = stats.successful_extractions
                if new_count > 0:
                    stats.average_quality_score = (
                        (current_avg_quality * (new_count - 1) + metadata.quality_score) / new_count
                    )
            
            # Save updated statistics
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(stats), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to update processing statistics: {e}")
    
    def get_processing_metadata(self, document_id: str) -> Optional[DocumentProcessingMetadata]:
        """Retrieve processing metadata for a specific document."""
        processing_file = self.processing_dir / f"{document_id}_processing.json"
        
        if not processing_file.exists():
            return None
        
        try:
            with open(processing_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return DocumentProcessingMetadata(**data)
            
        except Exception as e:
            logger.error(f"Failed to load processing metadata for {document_id}: {e}")
            return None
    
    def get_statistics(self) -> ProcessingStatistics:
        """Get current processing statistics."""
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ProcessingStatistics(**data)
            
        except Exception as e:
            logger.error(f"Failed to load processing statistics: {e}")
            return ProcessingStatistics()
    
    def get_method_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance analysis by extraction method."""
        try:
            method_stats = {}
            
            # Analyze all processing files
            for processing_file in self.processing_dir.glob("*_processing.json"):
                try:
                    with open(processing_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metadata = DocumentProcessingMetadata(**data)
                    method = metadata.extraction_method
                    
                    if method not in method_stats:
                        method_stats[method] = {
                            'count': 0,
                            'success_count': 0,
                            'total_processing_time': 0,
                            'total_quality_score': 0,
                            'success_rate': 0.0,
                            'average_processing_time': 0.0,
                            'average_quality_score': 0.0
                        }
                    
                    stats = method_stats[method]
                    stats['count'] += 1
                    
                    if metadata.extraction_success:
                        stats['success_count'] += 1
                    
                    if metadata.processing_time_seconds:
                        stats['total_processing_time'] += metadata.processing_time_seconds
                    
                    if metadata.quality_score:
                        stats['total_quality_score'] += metadata.quality_score
                        
                except Exception as e:
                    logger.warning(f"Failed to process {processing_file}: {e}")
                    continue
            
            # Calculate averages and rates
            for method, stats in method_stats.items():
                if stats['count'] > 0:
                    stats['success_rate'] = stats['success_count'] / stats['count']
                    stats['average_processing_time'] = stats['total_processing_time'] / stats['count']
                    
                    if stats['success_count'] > 0:
                        stats['average_quality_score'] = stats['total_quality_score'] / stats['success_count']
            
            return method_stats
            
        except Exception as e:
            logger.error(f"Failed to analyze method performance: {e}")
            return {}
    
    def get_recent_processing_activity(self, days: int = 7) -> List[DocumentProcessingMetadata]:
        """Get recent processing activity within specified days."""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_activity = []
        
        try:
            for processing_file in self.processing_dir.glob("*_processing.json"):
                try:
                    with open(processing_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metadata = DocumentProcessingMetadata(**data)
                    
                    # Parse timestamp and check if within range
                    processing_time = datetime.fromisoformat(metadata.processing_timestamp.replace('Z', '+00:00'))
                    if processing_time >= cutoff_date:
                        recent_activity.append(metadata)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {processing_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            recent_activity.sort(key=lambda x: x.processing_timestamp, reverse=True)
            return recent_activity
            
        except Exception as e:
            logger.error(f"Failed to get recent processing activity: {e}")
            return []
    
    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        try:
            overall_stats = self.get_statistics()
            method_performance = self.get_method_performance()
            recent_activity = self.get_recent_processing_activity()
            
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'overall_statistics': asdict(overall_stats),
                'method_performance': method_performance,
                'recent_activity_count': len(recent_activity),
                'recent_failures': [
                    {
                        'document_id': activity.document_id,
                        'timestamp': activity.processing_timestamp,
                        'method': activity.extraction_method,
                        'error': activity.error_message
                    }
                    for activity in recent_activity
                    if not activity.extraction_success
                ][:10],  # Last 10 failures
                'recommendations': self._generate_recommendations(overall_stats, method_performance)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate processing report: {e}")
            return {}
    
    def _generate_recommendations(self, 
                                overall_stats: ProcessingStatistics, 
                                method_performance: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on processing statistics."""
        recommendations = []
        
        try:
            # Check success rate
            if overall_stats.total_documents > 0:
                success_rate = overall_stats.successful_extractions / overall_stats.total_documents
                if success_rate < 0.8:
                    recommendations.append(
                        f"Success rate is {success_rate:.1%}. Consider improving text extraction methods."
                    )
            
            # Check OCR usage
            if overall_stats.total_documents > 0:
                ocr_rate = overall_stats.ocr_used_count / overall_stats.total_documents
                if ocr_rate > 0.3:
                    recommendations.append(
                        f"OCR is used in {ocr_rate:.1%} of cases. Consider optimizing standard extraction."
                    )
            
            # Check method performance
            for method, stats in method_performance.items():
                if stats['count'] > 5 and stats['success_rate'] < 0.7:
                    recommendations.append(
                        f"Method '{method}' has low success rate ({stats['success_rate']:.1%}). Review implementation."
                    )
            
            # Check processing time
            if overall_stats.average_processing_time > 30:
                recommendations.append(
                    f"Average processing time is {overall_stats.average_processing_time:.1f}s. Consider optimization."
                )
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
