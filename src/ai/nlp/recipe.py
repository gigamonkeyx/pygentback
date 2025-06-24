"""
Recipe Processing Module

Modular components for parsing and analyzing recipe descriptions.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .core import NLPProcessor, PatternMatcher, ConfidenceCalculator, ProcessingResult
from .models import ParsedRecipe, ParsedAction, RecipeIntent, ActionType

logger = logging.getLogger(__name__)


class RecipePatterns:
    """Recipe-specific pattern definitions"""
    
    INTENT_PATTERNS = {
        RecipeIntent.CREATE_CONTENT: [
            r"create|generate|produce|write|compose|build|make",
            r"content|text|document|article|report|story"
        ],
        RecipeIntent.ANALYZE_DATA: [
            r"analyze|examine|study|investigate|review",
            r"data|dataset|information|statistics|metrics"
        ],
        RecipeIntent.PROCESS_FILES: [
            r"process|handle|manage|organize|sort",
            r"file|document|image|video|audio"
        ],
        RecipeIntent.INTEGRATE_APIS: [
            r"integrate|connect|link|sync|fetch",
            r"api|service|endpoint|webhook|rest"
        ],
        RecipeIntent.AUTOMATE_WORKFLOW: [
            r"automate|schedule|trigger|execute|run",
            r"workflow|process|task|job|routine"
        ]
    }
    
    ACTION_PATTERNS = {
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
        ]
    }
    
    PARAMETER_PATTERNS = {
        "file_path": r"(?:file|path|location):\s*['\"]?([^'\"\\s]+)['\"]?",
        "url": r"(?:url|link|address):\s*['\"]?(https?://[^'\"\\s]+)['\"]?",
        "format": r"(?:format|type|extension):\s*['\"]?([a-zA-Z0-9]+)['\"]?",
        "timeout": r"(?:timeout|wait|delay):\s*([0-9]+)(?:\\s*(?:seconds?|ms|minutes?))?",
        "email": r"(?:email|to|recipient):\s*['\"]?([^'\"\\s@]+@[^'\"\\s]+)['\"]?"
    }


class RecipeAnalyzer:
    """
    Analyzes recipe structure and provides insights.
    """
    
    def __init__(self):
        self.complexity_weights = {
            'action_count': 0.3,
            'parameter_count': 0.2,
            'dependency_count': 0.2,
            'condition_count': 0.2,
            'tool_diversity': 0.1
        }
    
    def analyze_complexity(self, recipe: ParsedRecipe) -> Dict[str, Any]:
        """Analyze recipe complexity"""
        analysis = {
            'action_count': len(recipe.actions),
            'parameter_count': sum(len(action.parameters) for action in recipe.actions),
            'dependency_count': sum(len(action.dependencies) for action in recipe.actions),
            'condition_count': sum(len(action.conditions) for action in recipe.actions),
            'tool_diversity': len(set(tool for action in recipe.actions for tool in action.tool_suggestions))
        }
        
        # Calculate weighted complexity score
        complexity_score = 0.0
        for factor, weight in self.complexity_weights.items():
            normalized_value = min(1.0, analysis[factor] / 10.0)  # Normalize to 0-1
            complexity_score += normalized_value * weight
        
        analysis['complexity_score'] = complexity_score
        analysis['complexity_level'] = self._categorize_complexity(complexity_score)
        
        return analysis
    
    def _categorize_complexity(self, score: float) -> str:
        """Categorize complexity score"""
        if score < 0.3:
            return "simple"
        elif score < 0.7:
            return "medium"
        else:
            return "complex"
    
    def estimate_duration(self, recipe: ParsedRecipe) -> str:
        """Estimate recipe execution duration"""
        action_count = len(recipe.actions)
        
        # Base time estimates per action type
        time_estimates = {
            ActionType.INPUT: 30,      # seconds
            ActionType.OUTPUT: 20,
            ActionType.PROCESSING: 60,
            ActionType.VALIDATION: 15,
            ActionType.TRANSFORMATION: 45,
            ActionType.INTEGRATION: 90,
            ActionType.COMMUNICATION: 30,
            ActionType.ANALYSIS: 120
        }
        
        total_seconds = 0
        for action in recipe.actions:
            base_time = time_estimates.get(action.action_type, 60)
            
            # Adjust for parameters (more parameters = more time)
            param_multiplier = 1.0 + (len(action.parameters) * 0.1)
            
            # Adjust for dependencies (dependencies add coordination time)
            dep_multiplier = 1.0 + (len(action.dependencies) * 0.2)
            
            action_time = base_time * param_multiplier * dep_multiplier
            total_seconds += action_time
        
        return self._format_duration(total_seconds)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form"""
        if seconds < 60:
            return f"< 1 minute"
        elif seconds < 300:  # 5 minutes
            return f"{int(seconds // 60)} minutes"
        elif seconds < 900:  # 15 minutes
            return f"{int(seconds // 60)} minutes"
        else:
            return f"> 15 minutes"
    
    def identify_bottlenecks(self, recipe: ParsedRecipe) -> List[str]:
        """Identify potential bottlenecks in recipe"""
        bottlenecks = []
        
        # Check for sequential dependencies
        for i, action in enumerate(recipe.actions):
            if len(action.dependencies) > 2:
                bottlenecks.append(f"Action {i+1} has many dependencies")
        
        # Check for integration actions (typically slower)
        integration_actions = [
            i for i, action in enumerate(recipe.actions)
            if action.action_type == ActionType.INTEGRATION
        ]
        
        if len(integration_actions) > 3:
            bottlenecks.append("Multiple API integrations may cause delays")
        
        # Check for complex processing
        complex_actions = [
            i for i, action in enumerate(recipe.actions)
            if action.action_type == ActionType.ANALYSIS and len(action.parameters) > 5
        ]
        
        if complex_actions:
            bottlenecks.append("Complex analysis steps may be time-consuming")
        
        return bottlenecks


class RecipeParser(NLPProcessor):
    """
    Modular recipe parser for natural language descriptions.
    """
    
    def __init__(self):
        super().__init__("RecipeParser")
        
        self.analyzer = RecipeAnalyzer()
        self.patterns = RecipePatterns()
        
        # Initialize pattern matcher
        self._setup_patterns()
        
        # Tool suggestion mappings
        self.tool_mappings = {
            "file_operations": ["read_file", "write_file", "list_directory"],
            "data_processing": ["parse_csv", "parse_json", "transform_data"],
            "api_integration": ["make_api_request", "call_rest_api"],
            "communication": ["send_email", "send_sms", "post_message"],
            "validation": ["validate_schema", "check_format"]
        }
    
    def _setup_patterns(self):
        """Setup pattern matcher with recipe patterns"""
        # Add intent patterns
        for intent, patterns in self.patterns.INTENT_PATTERNS.items():
            for i, pattern in enumerate(patterns):
                self.pattern_matcher.add_pattern(f"{intent.value}_{i}", pattern)
        
        # Add action patterns
        for action_type, patterns in self.patterns.ACTION_PATTERNS.items():
            for i, pattern in enumerate(patterns):
                self.pattern_matcher.add_pattern(f"{action_type.value}_{i}", pattern)
        
        # Add parameter patterns
        for param_name, pattern in self.patterns.PARAMETER_PATTERNS.items():
            self.pattern_matcher.add_pattern(f"param_{param_name}", pattern)
    
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process recipe text and return parsing result"""
        start_time = time.time()
        
        try:
            # Parse the recipe
            recipe = await self.parse_recipe(text, context)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.update_stats(True, recipe.parsing_confidence, processing_time)
            
            return ProcessingResult(
                success=True,
                confidence=recipe.parsing_confidence,
                processing_time_ms=processing_time,
                metadata={'recipe': recipe.to_dict()}
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.update_stats(False, 0.0, processing_time)
            
            logger.error(f"Recipe parsing failed: {e}")
            
            return ProcessingResult(
                success=False,
                confidence=0.0,
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )
    
    async def parse_recipe(self, text: str, context: Optional[Dict[str, Any]] = None) -> ParsedRecipe:
        """Parse natural language recipe description"""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Extract basic information
        name, description = self._extract_name_and_description(cleaned_text)
        
        # Detect intent
        intent = self._detect_intent(cleaned_text)
        
        # Parse actions
        actions = await self._parse_actions(cleaned_text, intent)
        
        # Analyze recipe
        complexity_analysis = self.analyzer.analyze_complexity(
            ParsedRecipe(name=name, description=description, intent=intent, actions=actions)
        )
        
        # Estimate duration
        estimated_duration = self.analyzer.estimate_duration(
            ParsedRecipe(name=name, description=description, intent=intent, actions=actions)
        )
        
        # Calculate confidence
        confidence = self._calculate_parsing_confidence(text, intent, actions)
        
        # Create recipe
        recipe = ParsedRecipe(
            name=name,
            description=description,
            intent=intent,
            actions=actions,
            complexity_level=complexity_analysis['complexity_level'],
            estimated_duration=estimated_duration,
            required_capabilities=self._extract_capabilities(actions),
            parsing_confidence=confidence,
            ambiguities=self._identify_ambiguities(text, actions),
            suggestions=self._generate_suggestions(actions),
            original_text=text
        )
        
        return recipe
    
    def _extract_name_and_description(self, text: str) -> Tuple[str, str]:
        """Extract recipe name and description"""
        sentences = self.text_processor.extract_sentences(text)
        
        # Use first sentence as name/description
        if sentences:
            first_sentence = sentences[0]
            # Simple heuristic: if short, use as name; if long, use as description
            if len(first_sentence.split()) <= 8:
                name = first_sentence.title()
                description = text[:200] if len(text) > 200 else text
            else:
                name = "Recipe"  # Default name
                description = first_sentence
        else:
            name = "Untitled Recipe"
            description = text[:200] if len(text) > 200 else text
        
        return name, description
    
    def _detect_intent(self, text: str) -> RecipeIntent:
        """Detect recipe intent from text"""
        intent_scores = {}
        
        for intent in RecipeIntent:
            score = 0.0
            intent_patterns = [name for name in self.pattern_matcher.compiled_patterns.keys() 
                             if name.startswith(intent.value)]
            
            for pattern_name in intent_patterns:
                matches = self.pattern_matcher.match_pattern(text, pattern_name)
                score += len(matches)
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score, or default
        if intent_scores:
            return max(intent_scores.keys(), key=lambda k: intent_scores[k])
        else:
            return RecipeIntent.CREATE_CONTENT
    
    async def _parse_actions(self, text: str, intent: RecipeIntent) -> List[ParsedAction]:
        """Parse actions from text"""
        actions = []
        sentences = self.text_processor.extract_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Detect action type
            action_type = self._detect_action_type(sentence)
            
            # Create action
            action = ParsedAction(
                action_type=action_type,
                description=sentence,
                tool_suggestions=self._suggest_tools(sentence, action_type),
                parameters=self._extract_parameters(sentence),
                dependencies=self._detect_dependencies(i, sentences),
                conditions=self._extract_conditions(sentence),
                confidence=self._calculate_action_confidence(sentence, action_type)
            )
            
            actions.append(action)
        
        return actions
    
    def _detect_action_type(self, sentence: str) -> ActionType:
        """Detect action type from sentence"""
        action_scores = {}
        
        for action_type in ActionType:
            score = 0.0
            action_patterns = [name for name in self.pattern_matcher.compiled_patterns.keys() 
                             if name.startswith(action_type.value)]
            
            for pattern_name in action_patterns:
                matches = self.pattern_matcher.match_pattern(sentence, pattern_name)
                score += len(matches)
            
            if score > 0:
                action_scores[action_type] = score
        
        if action_scores:
            return max(action_scores.keys(), key=lambda k: action_scores[k])
        else:
            return ActionType.PROCESSING
    
    def _suggest_tools(self, sentence: str, action_type: ActionType) -> List[str]:
        """Suggest tools for action"""
        suggestions = []
        
        # Map action types to tool categories
        action_tool_mapping = {
            ActionType.INPUT: ["file_operations", "api_integration"],
            ActionType.OUTPUT: ["file_operations", "communication"],
            ActionType.PROCESSING: ["data_processing"],
            ActionType.VALIDATION: ["validation"],
            ActionType.INTEGRATION: ["api_integration"],
            ActionType.COMMUNICATION: ["communication"]
        }
        
        categories = action_tool_mapping.get(action_type, [])
        for category in categories:
            if category in self.tool_mappings:
                suggestions.extend(self.tool_mappings[category])
        
        return list(set(suggestions))[:3]  # Limit suggestions
    
    def _extract_parameters(self, sentence: str) -> Dict[str, Any]:
        """Extract parameters from sentence"""
        parameters = {}
        
        param_patterns = [name for name in self.pattern_matcher.compiled_patterns.keys() 
                         if name.startswith("param_")]
        
        for pattern_name in param_patterns:
            matches = self.pattern_matcher.match_pattern(sentence, pattern_name)
            if matches:
                param_name = pattern_name.replace("param_", "")
                parameters[param_name] = matches[0]  # Take first match
        
        return parameters
    
    def _detect_dependencies(self, index: int, sentences: List[str]) -> List[str]:
        """Detect dependencies between actions"""
        dependencies = []
        
        if index > 0:
            current_sentence = sentences[index].lower()
            dependency_keywords = ["after", "following", "once", "when", "then"]
            
            for keyword in dependency_keywords:
                if keyword in current_sentence:
                    dependencies.append(f"action_{index - 1}")
                    break
        
        return dependencies
    
    def _extract_conditions(self, sentence: str) -> List[str]:
        """Extract conditions from sentence"""
        conditions = []
        condition_keywords = ["if", "when", "unless", "provided"]
        
        for keyword in condition_keywords:
            if keyword in sentence.lower():
                # Simple extraction - could be more sophisticated
                parts = sentence.lower().split(keyword)
                if len(parts) > 1:
                    condition = parts[1].split(',')[0].strip()
                    conditions.append(condition)
        
        return conditions
    
    def _calculate_action_confidence(self, sentence: str, action_type: ActionType) -> float:
        """Calculate confidence for action parsing"""
        factors = []
        
        # Pattern match confidence
        pattern_score = self.pattern_matcher.score_pattern_match(sentence, f"{action_type.value}_0")
        factors.append(pattern_score)
        
        # Length confidence
        length_confidence = ConfidenceCalculator.length_confidence(sentence, 5, 30)
        factors.append(length_confidence)
        
        # Specificity confidence (presence of specific terms)
        specific_terms = len(self.text_processor.extract_numbers(sentence))
        total_terms = len(self.text_processor.extract_words(sentence))
        specificity = ConfidenceCalculator.specificity_confidence(specific_terms, total_terms)
        factors.append(specificity)
        
        return ConfidenceCalculator.combined_confidence(factors)
    
    def _extract_capabilities(self, actions: List[ParsedAction]) -> List[str]:
        """Extract required capabilities from actions"""
        capabilities = set()
        
        for action in actions:
            if action.action_type in [ActionType.INPUT, ActionType.OUTPUT]:
                capabilities.add("file_operations")
            elif action.action_type == ActionType.PROCESSING:
                capabilities.add("data_processing")
            elif action.action_type == ActionType.VALIDATION:
                capabilities.add("validation")
            elif action.action_type == ActionType.INTEGRATION:
                capabilities.add("api_integration")
            elif action.action_type == ActionType.COMMUNICATION:
                capabilities.add("communication")
        
        return list(capabilities)
    
    def _calculate_parsing_confidence(self, text: str, intent: RecipeIntent, 
                                    actions: List[ParsedAction]) -> float:
        """Calculate overall parsing confidence"""
        factors = []
        
        # Text quality
        text_confidence = ConfidenceCalculator.length_confidence(text, 20, 200)
        factors.append(text_confidence)
        
        # Intent detection confidence
        intent_score = self.pattern_matcher.score_pattern_match(text, f"{intent.value}_0")
        factors.append(intent_score)
        
        # Action parsing confidence
        if actions:
            avg_action_confidence = sum(action.confidence for action in actions) / len(actions)
            factors.append(avg_action_confidence)
        
        return ConfidenceCalculator.combined_confidence(factors, [0.3, 0.3, 0.4])
    
    def _identify_ambiguities(self, text: str, actions: List[ParsedAction]) -> List[str]:
        """Identify ambiguities in parsing"""
        ambiguities = []
        
        # Check for vague language
        vague_words = ["somehow", "maybe", "perhaps", "some", "any"]
        for word in vague_words:
            if word in text.lower():
                ambiguities.append(f"Vague language: '{word}'")
        
        # Check for missing parameters
        for i, action in enumerate(actions):
            if not action.parameters and action.action_type in [ActionType.INPUT, ActionType.OUTPUT]:
                ambiguities.append(f"Action {i+1} missing parameters")
        
        return ambiguities
    
    def _generate_suggestions(self, actions: List[ParsedAction]) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []
        
        # Suggest adding parameters
        if any(not action.parameters for action in actions):
            suggestions.append("Add specific parameters for better clarity")
        
        # Suggest error handling
        if len(actions) > 2 and not any("error" in action.description.lower() for action in actions):
            suggestions.append("Consider adding error handling")
        
        # Suggest validation
        if not any(action.action_type == ActionType.VALIDATION for action in actions):
            suggestions.append("Consider adding validation steps")
        
        return suggestions
