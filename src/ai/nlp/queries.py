"""
Query Processing Module

Modular components for processing natural language queries and intent classification.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .core import NLPProcessor, PatternMatcher, ConfidenceCalculator, ProcessingResult
from .models import QueryType, QueryResponse, QueryContext, Intent, IntentPrediction

logger = logging.getLogger(__name__)


class QueryPatterns:
    """Query classification patterns"""
    
    QUERY_TYPE_PATTERNS = {
        QueryType.RECIPE_SEARCH: [
            r"find.*recipe", r"search.*recipe", r"recipe.*for", r"how.*to.*create",
            r"show.*me.*recipe", r"list.*recipes", r"recipe.*that"
        ],
        QueryType.SERVER_STATUS: [
            r"server.*status", r"health.*check", r"server.*up", r"server.*down",
            r"status.*of.*server", r"check.*server", r"server.*running"
        ],
        QueryType.PERFORMANCE_QUERY: [
            r"performance", r"how.*fast", r"speed", r"latency", r"response.*time",
            r"memory.*usage", r"cpu.*usage", r"benchmark"
        ],
        QueryType.CAPABILITY_QUERY: [
            r"what.*can", r"capabilities", r"features", r"support", r"able.*to",
            r"functions", r"tools.*available", r"what.*does"
        ],
        QueryType.HELP_REQUEST: [
            r"help", r"how.*do", r"explain", r"what.*is", r"tutorial",
            r"guide", r"documentation", r"instructions"
        ],
        QueryType.CONFIGURATION: [
            r"config", r"settings", r"configure", r"setup", r"parameters",
            r"options", r"preferences", r"customize"
        ],
        QueryType.TROUBLESHOOTING: [
            r"error", r"problem", r"issue", r"not.*working", r"failed",
            r"broken", r"fix", r"troubleshoot", r"debug"
        ]
    }
    
    INTENT_KEYWORDS = {
        'search': ['find', 'search', 'look', 'get', 'show', 'list'],
        'create': ['create', 'make', 'build', 'generate', 'new'],
        'modify': ['change', 'update', 'edit', 'modify', 'alter'],
        'delete': ['delete', 'remove', 'clear', 'clean'],
        'analyze': ['analyze', 'check', 'examine', 'review', 'inspect'],
        'configure': ['configure', 'setup', 'set', 'adjust', 'customize']
    }
    
    ENTITY_PATTERNS = {
        'recipe_name': r"recipe\s+['\"]?([^'\"]+)['\"]?",
        'server_name': r"server\s+['\"]?([^'\"]+)['\"]?",
        'file_path': r"file\s+['\"]?([^'\"]+)['\"]?",
        'number': r"(\d+(?:\.\d+)?)",
        'percentage': r"(\d+(?:\.\d+)?)%",
        'time_duration': r"(\d+)\s*(seconds?|minutes?|hours?|days?)"
    }


class IntentClassifier:
    """
    Classifies user intents from natural language queries.
    """
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self._setup_patterns()
        
        # Intent scoring weights
        self.intent_weights = {
            'keyword_match': 0.4,
            'pattern_match': 0.3,
            'context_match': 0.2,
            'entity_presence': 0.1
        }
    
    def _setup_patterns(self):
        """Setup pattern matcher for intent classification"""
        # Add query type patterns
        for query_type, patterns in QueryPatterns.QUERY_TYPE_PATTERNS.items():
            for i, pattern in enumerate(patterns):
                self.pattern_matcher.add_pattern(f"{query_type.value}_{i}", pattern)
        
        # Add entity patterns
        for entity_type, pattern in QueryPatterns.ENTITY_PATTERNS.items():
            self.pattern_matcher.add_pattern(f"entity_{entity_type}", pattern)
    
    def classify_intent(self, query: str, context: Optional[QueryContext] = None) -> IntentPrediction:
        """Classify intent from query"""
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Detect action intent
        action_intent = self._detect_action_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Calculate confidence
        confidence = self._calculate_intent_confidence(query, query_type, action_intent, entities)
        
        # Create primary intent
        primary_intent = Intent(
            intent_type=f"{query_type.value}_{action_intent}",
            confidence=confidence,
            parameters={
                'query_type': query_type.value,
                'action': action_intent,
                'entities': entities
            }
        )
        
        # Generate alternative intents
        alternatives = self._generate_alternative_intents(query_lower, query_type)
        
        return IntentPrediction(
            primary_intent=primary_intent,
            alternative_intents=alternatives,
            prediction_confidence=confidence,
            processing_metadata={
                'query_type': query_type.value,
                'action_intent': action_intent,
                'entities_found': len(entities)
            }
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from text"""
        type_scores = {}
        
        for query_type in QueryType:
            score = 0.0
            type_patterns = [name for name in self.pattern_matcher.compiled_patterns.keys() 
                           if name.startswith(query_type.value)]
            
            for pattern_name in type_patterns:
                matches = self.pattern_matcher.match_pattern(query, pattern_name)
                score += len(matches)
            
            if score > 0:
                type_scores[query_type] = score
        
        if type_scores:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        else:
            return QueryType.HELP_REQUEST  # Default
    
    def _detect_action_intent(self, query: str) -> str:
        """Detect action intent from query"""
        for action, keywords in QueryPatterns.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    return action
        
        return 'search'  # Default action
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {}
        
        entity_patterns = [name for name in self.pattern_matcher.compiled_patterns.keys() 
                         if name.startswith("entity_")]
        
        for pattern_name in entity_patterns:
            matches = self.pattern_matcher.match_pattern(query, pattern_name)
            if matches:
                entity_type = pattern_name.replace("entity_", "")
                entities[entity_type] = matches
        
        return entities
    
    def _calculate_intent_confidence(self, query: str, query_type: QueryType, 
                                   action: str, entities: Dict[str, List[str]]) -> float:
        """Calculate confidence in intent classification"""
        factors = []
        
        # Keyword match factor
        action_keywords = QueryPatterns.INTENT_KEYWORDS.get(action, [])
        keyword_matches = sum(1 for keyword in action_keywords if keyword in query.lower())
        keyword_factor = min(1.0, keyword_matches / max(len(action_keywords), 1))
        factors.append(keyword_factor)
        
        # Pattern match factor
        pattern_score = self.pattern_matcher.score_pattern_match(query, f"{query_type.value}_0")
        factors.append(pattern_score)
        
        # Entity presence factor
        entity_factor = min(1.0, len(entities) / 3.0)  # Normalize to 0-1
        factors.append(entity_factor)
        
        # Query clarity factor
        clarity_factor = ConfidenceCalculator.length_confidence(query, 5, 50)
        factors.append(clarity_factor)
        
        return ConfidenceCalculator.combined_confidence(
            factors, [0.3, 0.3, 0.2, 0.2]
        )
    
    def _generate_alternative_intents(self, query: str, primary_type: QueryType) -> List[Intent]:
        """Generate alternative intent interpretations"""
        alternatives = []
        
        # Check other query types with lower confidence
        for query_type in QueryType:
            if query_type == primary_type:
                continue
            
            score = self.pattern_matcher.score_pattern_match(query, f"{query_type.value}_0")
            if score > 0.3:  # Threshold for alternatives
                intent = Intent(
                    intent_type=query_type.value,
                    confidence=score * 0.8,  # Lower confidence for alternatives
                    parameters={'query_type': query_type.value}
                )
                alternatives.append(intent)
        
        # Sort by confidence
        alternatives.sort(key=lambda x: x.confidence, reverse=True)
        
        return alternatives[:3]  # Return top 3 alternatives


class QueryProcessor(NLPProcessor):
    """
    Processes natural language queries and generates responses.
    """
    
    def __init__(self):
        super().__init__("QueryProcessor")
        self.intent_classifier = IntentClassifier()
        
        # Response templates
        self.response_templates = {
            QueryType.RECIPE_SEARCH: "Found {count} recipes matching your criteria.",
            QueryType.SERVER_STATUS: "Server status: {status}. Uptime: {uptime}.",
            QueryType.PERFORMANCE_QUERY: "Performance metrics: {metrics}.",
            QueryType.CAPABILITY_QUERY: "Available capabilities: {capabilities}.",
            QueryType.HELP_REQUEST: "Here's help with {topic}: {help_content}.",
            QueryType.CONFIGURATION: "Configuration options: {options}.",
            QueryType.TROUBLESHOOTING: "Troubleshooting {issue}: {solution}."
        }
    
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process query and return response"""
        start_time = time.time()
        
        try:
            # Create query context
            query_context = self._create_query_context(context)
            
            # Process the query
            response = await self.process_query(text, query_context)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.update_stats(True, response.confidence, processing_time)
            
            return ProcessingResult(
                success=True,
                confidence=response.confidence,
                processing_time_ms=processing_time,
                metadata={'response': response.to_dict()}
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.update_stats(False, 0.0, processing_time)
            
            logger.error(f"Query processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                confidence=0.0,
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )
    
    def _create_query_context(self, context: Optional[Dict[str, Any]]) -> QueryContext:
        """Create query context from input"""
        if not context:
            return QueryContext()
        
        return QueryContext(
            user_id=context.get('user_id'),
            session_id=context.get('session_id'),
            previous_queries=context.get('previous_queries', []),
            current_state=context.get('current_state', {}),
            preferences=context.get('preferences', {})
        )
    
    async def process_query(self, query: str, context: QueryContext) -> QueryResponse:
        """Process a natural language query"""
        # Classify intent
        intent_prediction = self.intent_classifier.classify_intent(query, context)
        primary_intent = intent_prediction.primary_intent
        
        # Extract query type
        query_type_str = primary_intent.parameters.get('query_type', 'help_request')
        query_type = QueryType(query_type_str)
        
        # Generate response based on query type
        response_text, data = await self._generate_response(query, query_type, primary_intent, context)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query_type, primary_intent)
        
        # Generate follow-up questions
        follow_ups = self._generate_follow_ups(query_type, primary_intent)
        
        # Check if clarification is needed
        needs_clarification, clarification_questions = self._check_clarification_needed(
            query, primary_intent, intent_prediction.alternative_intents
        )
        
        return QueryResponse(
            query_type=query_type,
            response_text=response_text,
            data=data,
            suggestions=suggestions,
            follow_up_questions=follow_ups,
            confidence=primary_intent.confidence,
            processing_time_ms=0.0,  # Will be set by caller
            requires_clarification=needs_clarification,
            clarification_questions=clarification_questions
        )
    
    async def _generate_response(self, query: str, query_type: QueryType, 
                               intent: Intent, context: QueryContext) -> Tuple[str, Dict[str, Any]]:
        """Generate response text and data"""
        entities = intent.parameters.get('entities', {})
        
        if query_type == QueryType.RECIPE_SEARCH:
            # Simulate recipe search
            recipe_name = entities.get('recipe_name', [''])[0] if 'recipe_name' in entities else ''
            count = 5  # Simulated count
            
            response_text = f"Found {count} recipes"
            if recipe_name:
                response_text += f" related to '{recipe_name}'"
            response_text += "."
            
            data = {
                'recipes': [f"Recipe {i+1}" for i in range(count)],
                'search_term': recipe_name,
                'total_count': count
            }
        
        elif query_type == QueryType.SERVER_STATUS:
            server_name = entities.get('server_name', ['default'])[0] if 'server_name' in entities else 'default'
            
            response_text = f"Server '{server_name}' is currently online with 99.5% uptime."
            data = {
                'server_name': server_name,
                'status': 'online',
                'uptime_percentage': 99.5,
                'last_check': datetime.utcnow().isoformat()
            }
        
        elif query_type == QueryType.PERFORMANCE_QUERY:
            response_text = "Current system performance: Average response time 150ms, CPU usage 45%, Memory usage 60%."
            data = {
                'response_time_ms': 150,
                'cpu_usage_percent': 45,
                'memory_usage_percent': 60,
                'status': 'good'
            }
        
        elif query_type == QueryType.CAPABILITY_QUERY:
            response_text = "Available capabilities include file processing, data analysis, API integration, and automation workflows."
            data = {
                'capabilities': [
                    'file_processing',
                    'data_analysis', 
                    'api_integration',
                    'automation_workflows'
                ],
                'total_count': 4
            }
        
        elif query_type == QueryType.HELP_REQUEST:
            topic = entities.get('recipe_name', ['general'])[0] if 'recipe_name' in entities else 'general'
            response_text = f"Here's help with {topic}: You can create recipes using natural language descriptions."
            data = {
                'topic': topic,
                'help_sections': ['getting_started', 'examples', 'advanced_features']
            }
        
        elif query_type == QueryType.CONFIGURATION:
            response_text = "Configuration options include server settings, performance thresholds, and notification preferences."
            data = {
                'config_sections': ['server', 'performance', 'notifications'],
                'editable': True
            }
        
        elif query_type == QueryType.TROUBLESHOOTING:
            issue = entities.get('recipe_name', ['general issue'])[0] if 'recipe_name' in entities else 'general issue'
            response_text = f"For troubleshooting {issue}, check logs and verify configuration settings."
            data = {
                'issue': issue,
                'suggested_steps': ['check_logs', 'verify_config', 'restart_service']
            }
        
        else:
            response_text = "I can help you with recipes, server status, performance queries, and more."
            data = {'available_query_types': [qt.value for qt in QueryType]}
        
        return response_text, data
    
    def _generate_suggestions(self, query_type: QueryType, intent: Intent) -> List[str]:
        """Generate helpful suggestions"""
        suggestions = []
        
        if query_type == QueryType.RECIPE_SEARCH:
            suggestions = [
                "Try searching by capability (e.g., 'file processing recipes')",
                "Use specific keywords for better results",
                "Browse recipes by category"
            ]
        elif query_type == QueryType.SERVER_STATUS:
            suggestions = [
                "Check specific server by name",
                "View detailed health metrics",
                "Set up status notifications"
            ]
        elif query_type == QueryType.PERFORMANCE_QUERY:
            suggestions = [
                "View historical performance trends",
                "Compare performance across servers",
                "Set performance alerts"
            ]
        elif query_type == QueryType.HELP_REQUEST:
            suggestions = [
                "Check the documentation",
                "Try the interactive tutorial",
                "Browse example recipes"
            ]
        
        return suggestions
    
    def _generate_follow_ups(self, query_type: QueryType, intent: Intent) -> List[str]:
        """Generate follow-up questions"""
        follow_ups = []
        
        if query_type == QueryType.RECIPE_SEARCH:
            follow_ups = [
                "Would you like to see recipe details?",
                "Do you want to create a new recipe?",
                "Should I filter by complexity level?"
            ]
        elif query_type == QueryType.SERVER_STATUS:
            follow_ups = [
                "Would you like to see detailed metrics?",
                "Do you want to check other servers?",
                "Should I set up monitoring alerts?"
            ]
        elif query_type == QueryType.PERFORMANCE_QUERY:
            follow_ups = [
                "Would you like historical data?",
                "Do you want to see bottleneck analysis?",
                "Should I suggest optimizations?"
            ]
        
        return follow_ups
    
    def _check_clarification_needed(self, query: str, primary_intent: Intent, 
                                  alternatives: List[Intent]) -> Tuple[bool, List[str]]:
        """Check if clarification is needed"""
        needs_clarification = False
        questions = []
        
        # Check if confidence is low
        if primary_intent.confidence < 0.6:
            needs_clarification = True
            questions.append("Could you please rephrase your question?")
        
        # Check if there are strong alternative intents
        strong_alternatives = [alt for alt in alternatives if alt.confidence > 0.5]
        if strong_alternatives:
            needs_clarification = True
            alt_types = [alt.intent_type for alt in strong_alternatives[:2]]
            questions.append(f"Did you mean to ask about {' or '.join(alt_types)}?")
        
        # Check if query is very short
        if len(query.split()) < 3:
            needs_clarification = True
            questions.append("Could you provide more details about what you're looking for?")
        
        return needs_clarification, questions
