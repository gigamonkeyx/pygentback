"""
Recipe Parser

Natural language parsing system for converting human-readable recipe descriptions
into structured recipe definitions and extracting recipe intents.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class RecipeIntent(Enum):
    """Types of recipe intents"""
    CREATE_CONTENT = "create_content"
    ANALYZE_DATA = "analyze_data"
    PROCESS_FILES = "process_files"
    INTEGRATE_APIS = "integrate_apis"
    AUTOMATE_WORKFLOW = "automate_workflow"
    GENERATE_REPORTS = "generate_reports"
    TRANSFORM_DATA = "transform_data"
    VALIDATE_INPUT = "validate_input"
    MONITOR_SYSTEM = "monitor_system"
    COMMUNICATE = "communicate"
    SEARCH_INFORMATION = "search_information"
    EXECUTE_CODE = "execute_code"


class ActionType(Enum):
    """Types of actions in recipes"""
    INPUT = "input"
    OUTPUT = "output"
    PROCESSING = "processing"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    INTEGRATION = "integration"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"


@dataclass
class ParsedAction:
    """Parsed action from natural language"""
    action_type: ActionType
    description: str
    tool_suggestions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ParsedRecipe:
    """Parsed recipe from natural language description"""
    name: str
    description: str
    intent: RecipeIntent
    actions: List[ParsedAction] = field(default_factory=list)
    
    # Recipe metadata
    complexity_level: str = "medium"  # simple, medium, complex
    estimated_duration: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    
    # Parsing metadata
    parsing_confidence: float = 0.0
    ambiguities: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Source information
    original_text: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)


class RecipeParser:
    """
    Natural language parser for recipe descriptions.
    
    Converts human-readable recipe descriptions into structured
    recipe definitions with extracted intents and actions.
    """
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Action detection patterns
        self.action_patterns = self._initialize_action_patterns()
        
        # Tool suggestion mappings
        self.tool_mappings = self._initialize_tool_mappings()
        
        # Parameter extraction patterns
        self.parameter_patterns = self._initialize_parameter_patterns()
        
        # Parsing statistics
        self.parsing_stats = {
            'total_parsed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'avg_confidence': 0.0
        }
    
    def _initialize_intent_patterns(self) -> Dict[RecipeIntent, List[str]]:
        """Initialize patterns for intent detection"""
        return {
            RecipeIntent.CREATE_CONTENT: [
                r"create|generate|produce|write|compose|build|make",
                r"content|text|document|article|report|story",
                r"blog|post|email|letter|summary"
            ],
            RecipeIntent.ANALYZE_DATA: [
                r"analyze|examine|study|investigate|review",
                r"data|dataset|information|statistics|metrics",
                r"trend|pattern|insight|correlation"
            ],
            RecipeIntent.PROCESS_FILES: [
                r"process|handle|manage|organize|sort",
                r"file|document|image|video|audio",
                r"upload|download|convert|compress|extract"
            ],
            RecipeIntent.INTEGRATE_APIS: [
                r"integrate|connect|link|sync|fetch",
                r"api|service|endpoint|webhook|rest",
                r"third.?party|external|remote"
            ],
            RecipeIntent.AUTOMATE_WORKFLOW: [
                r"automate|schedule|trigger|execute|run",
                r"workflow|process|task|job|routine",
                r"batch|pipeline|sequence|chain"
            ],
            RecipeIntent.GENERATE_REPORTS: [
                r"generate|create|produce|compile",
                r"report|dashboard|chart|graph|visualization",
                r"summary|overview|analysis|metrics"
            ],
            RecipeIntent.TRANSFORM_DATA: [
                r"transform|convert|modify|change|format",
                r"data|information|content|structure",
                r"csv|json|xml|excel|database"
            ],
            RecipeIntent.VALIDATE_INPUT: [
                r"validate|verify|check|confirm|ensure",
                r"input|data|format|structure|schema",
                r"correct|valid|proper|accurate"
            ],
            RecipeIntent.MONITOR_SYSTEM: [
                r"monitor|watch|track|observe|check",
                r"system|server|service|application|health",
                r"status|performance|uptime|availability"
            ],
            RecipeIntent.COMMUNICATE: [
                r"send|notify|alert|message|communicate",
                r"email|sms|slack|discord|webhook",
                r"notification|announcement|update"
            ],
            RecipeIntent.SEARCH_INFORMATION: [
                r"search|find|lookup|query|retrieve",
                r"information|data|content|knowledge",
                r"web|database|index|catalog"
            ],
            RecipeIntent.EXECUTE_CODE: [
                r"execute|run|compile|interpret|process",
                r"code|script|program|function|command",
                r"python|javascript|shell|sql"
            ]
        }
    
    def _initialize_action_patterns(self) -> Dict[ActionType, List[str]]:
        """Initialize patterns for action detection"""
        return {
            ActionType.INPUT: [
                r"input|receive|get|fetch|read|load|import",
                r"from|source|origin|upload|provide"
            ],
            ActionType.OUTPUT: [
                r"output|return|send|save|export|write|store",
                r"to|destination|target|download|result"
            ],
            ActionType.PROCESSING: [
                r"process|handle|execute|run|perform|apply",
                r"operation|function|method|algorithm"
            ],
            ActionType.VALIDATION: [
                r"validate|verify|check|confirm|ensure|test",
                r"valid|correct|proper|accurate|compliant"
            ],
            ActionType.TRANSFORMATION: [
                r"transform|convert|modify|change|format|parse",
                r"into|to|as|from|structure|shape"
            ],
            ActionType.INTEGRATION: [
                r"integrate|connect|link|sync|merge|combine",
                r"with|to|from|api|service|system"
            ],
            ActionType.COMMUNICATION: [
                r"send|notify|alert|message|communicate|broadcast",
                r"email|sms|webhook|notification|signal"
            ],
            ActionType.ANALYSIS: [
                r"analyze|examine|study|calculate|compute|evaluate",
                r"pattern|trend|insight|statistic|metric"
            ]
        }
    
    def _initialize_tool_mappings(self) -> Dict[str, List[str]]:
        """Initialize tool suggestions for different operations"""
        return {
            "file_operations": ["read_file", "write_file", "list_directory", "file_info"],
            "data_processing": ["parse_csv", "parse_json", "transform_data", "filter_data"],
            "web_scraping": ["scrape_webpage", "extract_links", "get_page_content"],
            "api_integration": ["make_api_request", "call_rest_api", "send_webhook"],
            "text_processing": ["analyze_sentiment", "extract_entities", "summarize_text"],
            "communication": ["send_email", "send_sms", "post_message", "create_notification"],
            "code_execution": ["execute_python", "run_script", "compile_code"],
            "database": ["execute_query", "connect_database", "insert_data"],
            "image_processing": ["resize_image", "detect_objects", "apply_filter"],
            "validation": ["validate_schema", "check_format", "verify_data"]
        }
    
    def _initialize_parameter_patterns(self) -> Dict[str, str]:
        """Initialize patterns for parameter extraction"""
        return {
            "file_path": r"(?:file|path|location):\s*['\"]?([^'\"\\s]+)['\"]?",
            "url": r"(?:url|link|address):\s*['\"]?(https?://[^'\"\\s]+)['\"]?",
            "format": r"(?:format|type|extension):\s*['\"]?([a-zA-Z0-9]+)['\"]?",
            "timeout": r"(?:timeout|wait|delay):\s*([0-9]+)(?:\\s*(?:seconds?|ms|minutes?))?",
            "count": r"(?:count|number|limit):\s*([0-9]+)",
            "threshold": r"(?:threshold|limit|max|min):\s*([0-9.]+)",
            "email": r"(?:email|to|recipient):\s*['\"]?([^'\"\\s@]+@[^'\"\\s]+)['\"]?",
            "api_key": r"(?:api.?key|token|auth):\s*['\"]?([^'\"\\s]+)['\"]?"
        }
    
    async def parse_recipe(self, text: str, context: Optional[Dict[str, Any]] = None) -> ParsedRecipe:
        """
        Parse natural language recipe description.
        
        Args:
            text: Natural language recipe description
            context: Optional context for parsing
            
        Returns:
            Parsed recipe with structured information
        """
        try:
            logger.info(f"Parsing recipe from text: {text[:100]}...")
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract recipe name and description
            name, description = self._extract_name_and_description(cleaned_text)
            
            # Detect recipe intent
            intent = self._detect_intent(cleaned_text)
            
            # Parse actions
            actions = await self._parse_actions(cleaned_text, intent)
            
            # Extract metadata
            complexity = self._estimate_complexity(actions)
            duration = self._estimate_duration(actions)
            capabilities = self._extract_capabilities(actions)
            
            # Calculate parsing confidence
            confidence = self._calculate_parsing_confidence(text, intent, actions)
            
            # Identify ambiguities and suggestions
            ambiguities = self._identify_ambiguities(text, actions)
            suggestions = self._generate_suggestions(text, actions)
            
            # Create parsed recipe
            parsed_recipe = ParsedRecipe(
                name=name,
                description=description,
                intent=intent,
                actions=actions,
                complexity_level=complexity,
                estimated_duration=duration,
                required_capabilities=capabilities,
                parsing_confidence=confidence,
                ambiguities=ambiguities,
                suggestions=suggestions,
                original_text=text
            )
            
            # Update statistics
            self._update_parsing_stats(True, confidence)
            
            logger.info(f"Successfully parsed recipe '{name}' with {len(actions)} actions "
                       f"(confidence: {confidence:.3f})")
            
            return parsed_recipe
            
        except Exception as e:
            logger.error(f"Recipe parsing failed: {e}")
            self._update_parsing_stats(False, 0.0)
            
            # Return minimal parsed recipe
            return ParsedRecipe(
                name="Unknown Recipe",
                description=f"Parsing failed: {e}",
                intent=RecipeIntent.CREATE_CONTENT,
                original_text=text,
                parsing_confidence=0.0,
                ambiguities=[f"Parsing error: {e}"]
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for parsing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Convert to lowercase for pattern matching
        return text.lower()
    
    def _extract_name_and_description(self, text: str) -> Tuple[str, str]:
        """Extract recipe name and description"""
        # Look for explicit name patterns
        name_patterns = [
            r"recipe\s+(?:name|title):\s*([^.!?]+)",
            r"(?:create|build|make)\s+(?:a\s+)?([^.!?]+?)(?:\s+recipe)?",
            r"^([^.!?]+?)(?:\s+recipe|\s+workflow|\s+process)?[.!?]"
        ]
        
        name = "Untitled Recipe"
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()
                break
        
        # Use first sentence as description if no explicit description
        sentences = re.split(r'[.!?]+', text)
        description = sentences[0].strip() if sentences else text[:200]
        
        return name, description
    
    def _detect_intent(self, text: str) -> RecipeIntent:
        """Detect the primary intent of the recipe"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            # Return intent with highest score
            return max(intent_scores.keys(), key=lambda k: intent_scores[k])
        else:
            # Default intent
            return RecipeIntent.CREATE_CONTENT
    
    async def _parse_actions(self, text: str, intent: RecipeIntent) -> List[ParsedAction]:
        """Parse actions from text"""
        actions = []
        
        # Split text into sentences for action detection
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Detect action type
            action_type = self._detect_action_type(sentence)
            
            # Extract action description
            description = sentence.strip()
            
            # Suggest tools
            tool_suggestions = self._suggest_tools(sentence, action_type, intent)
            
            # Extract parameters
            parameters = self._extract_parameters(sentence)
            
            # Detect dependencies
            dependencies = self._detect_dependencies(sentence, i, sentences)
            
            # Extract conditions
            conditions = self._extract_conditions(sentence)
            
            # Calculate confidence
            confidence = self._calculate_action_confidence(sentence, action_type, tool_suggestions)
            
            action = ParsedAction(
                action_type=action_type,
                description=description,
                tool_suggestions=tool_suggestions,
                parameters=parameters,
                dependencies=dependencies,
                conditions=conditions,
                confidence=confidence
            )
            
            actions.append(action)
        
        return actions
    
    def _detect_action_type(self, sentence: str) -> ActionType:
        """Detect action type from sentence"""
        action_scores = {}
        
        for action_type, patterns in self.action_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, sentence, re.IGNORECASE))
                score += matches
            
            if score > 0:
                action_scores[action_type] = score
        
        if action_scores:
            return max(action_scores.keys(), key=lambda k: action_scores[k])
        else:
            return ActionType.PROCESSING  # Default
    
    def _suggest_tools(self, sentence: str, action_type: ActionType, intent: RecipeIntent) -> List[str]:
        """Suggest tools based on sentence content"""
        suggestions = []
        
        # Map action types to tool categories
        action_tool_mapping = {
            ActionType.INPUT: ["file_operations", "api_integration"],
            ActionType.OUTPUT: ["file_operations", "communication"],
            ActionType.PROCESSING: ["data_processing", "text_processing"],
            ActionType.VALIDATION: ["validation"],
            ActionType.TRANSFORMATION: ["data_processing"],
            ActionType.INTEGRATION: ["api_integration"],
            ActionType.COMMUNICATION: ["communication"],
            ActionType.ANALYSIS: ["text_processing", "data_processing"]
        }
        
        # Get tool categories for action type
        categories = action_tool_mapping.get(action_type, [])
        
        for category in categories:
            if category in self.tool_mappings:
                suggestions.extend(self.tool_mappings[category])
        
        # Add intent-specific suggestions
        if intent == RecipeIntent.PROCESS_FILES:
            suggestions.extend(self.tool_mappings.get("file_operations", []))
        elif intent == RecipeIntent.INTEGRATE_APIS:
            suggestions.extend(self.tool_mappings.get("api_integration", []))
        elif intent == RecipeIntent.ANALYZE_DATA:
            suggestions.extend(self.tool_mappings.get("data_processing", []))
        
        # Remove duplicates and limit suggestions
        return list(set(suggestions))[:5]
    
    def _extract_parameters(self, sentence: str) -> Dict[str, Any]:
        """Extract parameters from sentence"""
        parameters = {}
        
        for param_name, pattern in self.parameter_patterns.items():
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                value = match.group(1)
                
                # Type conversion
                if param_name in ["timeout", "count"]:
                    try:
                        parameters[param_name] = int(value)
                    except ValueError:
                        parameters[param_name] = value
                elif param_name == "threshold":
                    try:
                        parameters[param_name] = float(value)
                    except ValueError:
                        parameters[param_name] = value
                else:
                    parameters[param_name] = value
        
        return parameters
    
    def _detect_dependencies(self, sentence: str, index: int, all_sentences: List[str]) -> List[str]:
        """Detect dependencies between actions"""
        dependencies = []
        
        # Look for dependency keywords
        dependency_patterns = [
            r"after|following|once|when|if",
            r"depends on|requires|needs",
            r"then|next|subsequently"
        ]
        
        for pattern in dependency_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # If dependency keyword found, previous action is likely a dependency
                if index > 0:
                    dependencies.append(f"action_{index - 1}")
        
        return dependencies
    
    def _extract_conditions(self, sentence: str) -> List[str]:
        """Extract conditions from sentence"""
        conditions = []
        
        # Look for conditional patterns
        condition_patterns = [
            r"if\s+([^,]+)",
            r"when\s+([^,]+)",
            r"unless\s+([^,]+)",
            r"provided\s+(?:that\s+)?([^,]+)"
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            conditions.extend([match.strip() for match in matches])
        
        return conditions
    
    def _calculate_action_confidence(self, sentence: str, action_type: ActionType, 
                                   tool_suggestions: List[str]) -> float:
        """Calculate confidence for action parsing"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if action type patterns match strongly
        patterns = self.action_patterns.get(action_type, [])
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                confidence += 0.1
        
        # Increase confidence if tool suggestions are available
        if tool_suggestions:
            confidence += min(0.3, len(tool_suggestions) * 0.1)
        
        # Increase confidence if sentence is clear and specific
        if len(sentence.split()) > 5:  # Detailed description
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _estimate_complexity(self, actions: List[ParsedAction]) -> str:
        """Estimate recipe complexity"""
        if len(actions) <= 3:
            return "simple"
        elif len(actions) <= 8:
            return "medium"
        else:
            return "complex"
    
    def _estimate_duration(self, actions: List[ParsedAction]) -> str:
        """Estimate recipe duration"""
        action_count = len(actions)
        
        if action_count <= 2:
            return "< 1 minute"
        elif action_count <= 5:
            return "1-5 minutes"
        elif action_count <= 10:
            return "5-15 minutes"
        else:
            return "> 15 minutes"
    
    def _extract_capabilities(self, actions: List[ParsedAction]) -> List[str]:
        """Extract required capabilities from actions"""
        capabilities = set()
        
        for action in actions:
            # Map action types to capabilities
            if action.action_type == ActionType.INPUT:
                capabilities.add("file_operations")
            elif action.action_type == ActionType.OUTPUT:
                capabilities.add("file_operations")
            elif action.action_type == ActionType.PROCESSING:
                capabilities.add("data_processing")
            elif action.action_type == ActionType.VALIDATION:
                capabilities.add("validation")
            elif action.action_type == ActionType.TRANSFORMATION:
                capabilities.add("data_processing")
            elif action.action_type == ActionType.INTEGRATION:
                capabilities.add("api_integration")
            elif action.action_type == ActionType.COMMUNICATION:
                capabilities.add("communication")
            elif action.action_type == ActionType.ANALYSIS:
                capabilities.add("analytics")
        
        return list(capabilities)
    
    def _calculate_parsing_confidence(self, text: str, intent: RecipeIntent, 
                                    actions: List[ParsedAction]) -> float:
        """Calculate overall parsing confidence"""
        confidence = 0.0
        
        # Base confidence from text clarity
        word_count = len(text.split())
        if word_count > 10:
            confidence += 0.2
        
        # Confidence from intent detection
        intent_patterns = self.intent_patterns.get(intent, [])
        intent_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in intent_patterns)
        if intent_matches > 0:
            confidence += min(0.3, intent_matches * 0.1)
        
        # Confidence from action parsing
        if actions:
            avg_action_confidence = sum(action.confidence for action in actions) / len(actions)
            confidence += avg_action_confidence * 0.4
        
        # Confidence from structure
        if len(actions) > 1:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _identify_ambiguities(self, text: str, actions: List[ParsedAction]) -> List[str]:
        """Identify ambiguities in the parsed recipe"""
        ambiguities = []
        
        # Check for vague language
        vague_patterns = [
            r"somehow|maybe|perhaps|possibly|might",
            r"some|any|various|different|several",
            r"appropriate|suitable|relevant|proper"
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                ambiguities.append(f"Vague language detected: '{pattern}'")
        
        # Check for missing parameters
        for i, action in enumerate(actions):
            if not action.parameters and action.action_type in [ActionType.INPUT, ActionType.OUTPUT]:
                ambiguities.append(f"Action {i+1} missing specific parameters")
        
        # Check for unclear dependencies
        dependency_count = sum(len(action.dependencies) for action in actions)
        if len(actions) > 3 and dependency_count == 0:
            ambiguities.append("Action sequence unclear - dependencies not specified")
        
        return ambiguities
    
    def _generate_suggestions(self, text: str, actions: List[ParsedAction]) -> List[str]:
        """Generate suggestions for improving the recipe"""
        suggestions = []
        
        # Suggest adding parameters
        actions_without_params = [i for i, action in enumerate(actions) 
                                if not action.parameters and action.action_type in 
                                [ActionType.INPUT, ActionType.OUTPUT, ActionType.INTEGRATION]]
        
        if actions_without_params:
            suggestions.append("Consider adding specific parameters (file paths, URLs, etc.)")
        
        # Suggest error handling
        if len(actions) > 2 and not any("error" in action.description.lower() for action in actions):
            suggestions.append("Consider adding error handling steps")
        
        # Suggest validation
        if not any(action.action_type == ActionType.VALIDATION for action in actions):
            suggestions.append("Consider adding validation steps")
        
        # Suggest documentation
        if len(actions) > 5:
            suggestions.append("Consider breaking down into smaller sub-recipes")
        
        return suggestions
    
    def _update_parsing_stats(self, success: bool, confidence: float):
        """Update parsing statistics"""
        self.parsing_stats['total_parsed'] += 1
        
        if success:
            self.parsing_stats['successful_parses'] += 1
        else:
            self.parsing_stats['failed_parses'] += 1
        
        # Update average confidence
        total_confidence = (self.parsing_stats['avg_confidence'] * 
                          (self.parsing_stats['total_parsed'] - 1) + confidence)
        self.parsing_stats['avg_confidence'] = total_confidence / self.parsing_stats['total_parsed']
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        stats = self.parsing_stats.copy()
        
        if stats['total_parsed'] > 0:
            stats['success_rate'] = stats['successful_parses'] / stats['total_parsed']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported recipe intents"""
        return [intent.value for intent in RecipeIntent]
    
    def get_supported_actions(self) -> List[str]:
        """Get list of supported action types"""
        return [action.value for action in ActionType]
