"""
Tests for NLP core components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.ai.nlp.core import NLPProcessor, TextProcessor, PatternMatcher
from src.ai.nlp.models import ParsedRecipe, RecipeIntent, RecipeIntentResult, TestResult, QueryResponse
from tests.utils.helpers import create_test_recipe, assert_prediction_valid


class TestNLPProcessor:
    """Test cases for NLPProcessor."""
    
    @pytest.fixture
    def nlp_processor(self):
        """Create NLP processor instance."""
        # Use unittest.mock instead of importing from production
        processor = Mock(spec=NLPProcessor)
        processor.initialize = AsyncMock()
        processor.analyze_content = AsyncMock()
        processor.parse_recipe = AsyncMock()
        processor.interpret_results = AsyncMock()
        return processor
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_processor_initialization(self, nlp_processor):
        """Test NLP processor initialization."""
        assert nlp_processor.is_initialized is False
        
        # Test initialization
        success = await nlp_processor.initialize()
        assert success is True
        assert nlp_processor.is_initialized is True
        
        # Test status
        status = nlp_processor.get_processor_status()
        assert status["is_initialized"] is True
        assert "models_loaded" in status
        assert "capabilities" in status
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_process_text(self, nlp_processor):
        """Test text processing functionality."""
        await nlp_processor.initialize()
        
        test_text = "Create a recipe for testing web scraping with Python"
        
        result = await nlp_processor.process_text(test_text)
        
        assert "processed_text" in result
        assert "tokens" in result
        assert "entities" in result
        assert "intent" in result
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_extract_intent(self, nlp_processor):
        """Test intent extraction."""
        await nlp_processor.initialize()
        
        test_texts = [
            "Create a recipe for data analysis",
            "How do I optimize this recipe?",
            "What went wrong with my test?",
            "Generate documentation for this workflow"
        ]
        
        for text in test_texts:
            intent = await nlp_processor.extract_intent(text)
            
            assert isinstance(intent, RecipeIntentResult)
            assert intent.intent_type in ["create", "optimize", "analyze", "document", "query"]
            assert 0.0 <= intent.confidence <= 1.0
            assert isinstance(intent.entities, dict)
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_parse_recipe_text(self, nlp_processor):
        """Test recipe text parsing."""
        await nlp_processor.initialize()
        
        recipe_text = """
        Recipe: Web Scraping Analysis
        Description: Scrape data from websites and analyze results
        
        Steps:
        1. Initialize web scraper with target URLs
        2. Extract data using BeautifulSoup
        3. Clean and process the data
        4. Perform analysis and generate insights
        5. Save results to database
        
        Requirements:
        - Python requests library
        - BeautifulSoup4
        - pandas for data processing
        """
        
        parsed = await nlp_processor.parse_recipe_text(recipe_text)
        
        assert isinstance(parsed, ParsedRecipe)
        assert parsed.name == "Web Scraping Analysis"
        assert len(parsed.steps) == 5
        assert len(parsed.requirements) > 0
        assert parsed.complexity > 0
        assert parsed.estimated_duration > 0
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_interpret_test_results(self, nlp_processor, sample_test_results):
        """Test test result interpretation."""
        await nlp_processor.initialize()
        
        interpretation = await nlp_processor.interpret_test_results(sample_test_results)
        
        assert "summary" in interpretation
        assert "insights" in interpretation
        assert "recommendations" in interpretation
        assert "confidence" in interpretation
        assert isinstance(interpretation["insights"], list)
        assert isinstance(interpretation["recommendations"], list)
        assert 0.0 <= interpretation["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_error_handling(self, nlp_processor):
        """Test error handling in NLP processor."""
        await nlp_processor.initialize()
        
        # Test with invalid input
        with pytest.raises(ValueError):
            await nlp_processor.process_text("")
        
        with pytest.raises(TypeError):
            await nlp_processor.process_text(None)
        
        # Test with malformed recipe text
        malformed_text = "This is not a valid recipe format"
        parsed = await nlp_processor.parse_recipe_text(malformed_text)
        
        # Should handle gracefully and return basic structure
        assert isinstance(parsed, ParsedRecipe)
        assert parsed.name is not None
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_shutdown(self, nlp_processor):
        """Test NLP processor shutdown."""
        await nlp_processor.initialize()
        assert nlp_processor.is_initialized is True
        
        await nlp_processor.shutdown()
        assert nlp_processor.is_initialized is False


class TestTextProcessor:
    """Test cases for TextProcessor."""
    
    @pytest.fixture
    def text_processor(self):
        """Create text processor instance."""
        return TextProcessor()
    
    @pytest.mark.nlp
    def test_tokenize(self, text_processor):
        """Test text tokenization."""
        text = "Create a recipe for web scraping with Python and BeautifulSoup"
        
        tokens = text_processor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "recipe" in tokens
        assert "Python" in tokens
    
    @pytest.mark.nlp
    def test_extract_entities(self, text_processor):
        """Test entity extraction."""
        text = "Use Python with pandas and numpy for data analysis"
        
        entities = text_processor.extract_entities(text)
        
        assert isinstance(entities, dict)
        assert "technologies" in entities or "tools" in entities
        # Should identify Python, pandas, numpy as technologies/tools
    
    @pytest.mark.nlp
    def test_clean_text(self, text_processor):
        """Test text cleaning."""
        dirty_text = "  This is a TEST with    extra spaces and CAPS!!! "
        
        cleaned = text_processor.clean_text(dirty_text)
        
        assert cleaned.strip() == cleaned  # No leading/trailing spaces
        assert "  " not in cleaned  # No double spaces
        assert cleaned.islower() or cleaned.isupper()  # Consistent case
    
    @pytest.mark.nlp
    def test_extract_keywords(self, text_processor):
        """Test keyword extraction."""
        text = """
        This recipe demonstrates web scraping using Python.
        We will use requests for HTTP calls and BeautifulSoup for parsing.
        The data will be processed with pandas and stored in a database.
        """
        
        keywords = text_processor.extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should extract relevant technical terms
        tech_keywords = [kw for kw in keywords if kw.lower() in 
                        ["python", "requests", "beautifulsoup", "pandas", "database"]]
        assert len(tech_keywords) > 0


class TestPatternMatcher:
    """Test cases for PatternMatcher."""
    
    @pytest.fixture
    def pattern_matcher(self):
        """Create pattern matcher instance."""
        return PatternMatcher()
    
    @pytest.mark.nlp
    def test_match_recipe_patterns(self, pattern_matcher):
        """Test recipe pattern matching."""
        recipe_texts = [
            "Step 1: Initialize the environment",
            "First, install the required packages",
            "Next, configure the database connection",
            "Finally, run the analysis script"
        ]
        
        for text in recipe_texts:
            patterns = pattern_matcher.match_recipe_patterns(text)
            
            assert isinstance(patterns, list)
            # Should identify step indicators, action words, etc.
    
    @pytest.mark.nlp
    def test_match_requirement_patterns(self, pattern_matcher):
        """Test requirement pattern matching."""
        requirement_texts = [
            "Requires Python 3.8 or higher",
            "Dependencies: requests, beautifulsoup4, pandas",
            "System requirements: 4GB RAM, 2 CPU cores",
            "External services: PostgreSQL database"
        ]
        
        for text in requirement_texts:
            patterns = pattern_matcher.match_requirement_patterns(text)
            
            assert isinstance(patterns, list)
            # Should identify version numbers, package names, system specs
    
    @pytest.mark.nlp
    def test_match_error_patterns(self, pattern_matcher):
        """Test error pattern matching."""
        error_texts = [
            "ConnectionError: Failed to connect to database",
            "TimeoutError: Request timed out after 30 seconds",
            "ValidationError: Invalid parameter value",
            "ImportError: Module 'requests' not found"
        ]
        
        for text in error_texts:
            patterns = pattern_matcher.match_error_patterns(text)
            
            assert isinstance(patterns, list)
            # Should identify error types, causes, etc.
    
    @pytest.mark.nlp
    def test_extract_code_blocks(self, pattern_matcher):
        """Test code block extraction."""
        text_with_code = """
        Here's how to implement the solution:
        
        ```python
        import requests
        from bs4 import BeautifulSoup
        
        def scrape_data(url):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.find_all('div', class_='data')
        ```
        
        This function will extract the data elements.
        """
        
        code_blocks = pattern_matcher.extract_code_blocks(text_with_code)
        
        assert isinstance(code_blocks, list)
        assert len(code_blocks) > 0
        assert "import requests" in code_blocks[0]
        assert "def scrape_data" in code_blocks[0]


@pytest.mark.nlp
@pytest.mark.integration
class TestNLPIntegration:
    """Integration tests for NLP components."""
    
    @pytest.mark.asyncio
    async def test_full_recipe_processing_pipeline(self):
        """Test complete recipe processing pipeline."""
        # Create mock processor using unittest.mock
        processor = Mock(spec=NLPProcessor)
        processor.initialize = AsyncMock()
        processor.analyze_content = AsyncMock()
        processor.parse_recipe = AsyncMock()
        processor.interpret_results = AsyncMock()
        await processor.initialize()
        
        recipe_text = """
        Recipe: Automated Testing Pipeline
        Description: Set up automated testing for Python projects
        
        Steps:
        1. Install pytest and coverage tools
        2. Create test directory structure
        3. Write unit tests for core functions
        4. Configure CI/CD pipeline
        5. Set up code coverage reporting
        
        Requirements:
        - Python 3.8+
        - pytest framework
        - GitHub Actions or similar CI
        
        Expected Output:
        - Test reports
        - Coverage statistics
        - CI/CD pipeline status
        """
        
        # Parse the recipe
        parsed = await processor.parse_recipe_text(recipe_text)
        
        # Verify parsing results
        assert parsed.name == "Automated Testing Pipeline"
        assert len(parsed.steps) == 5
        assert "pytest" in str(parsed.requirements).lower()
        assert parsed.complexity > 0
        
        # Test intent extraction
        intent = await processor.extract_intent(recipe_text)
        assert intent.intent_type in ["create", "setup", "configure"]
        
        await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_analysis_pipeline(self):
        """Test error analysis and recommendation pipeline."""
        # Create mock processor using unittest.mock
        processor = Mock(spec=NLPProcessor)
        processor.initialize = AsyncMock()
        processor.analyze_content = AsyncMock()
        processor.parse_recipe = AsyncMock()
        processor.interpret_results = AsyncMock()
        await processor.initialize()
        
        error_results = {
            "test_id": "test_001",
            "success": False,
            "errors": [
                {
                    "step": 2,
                    "error_type": "ImportError",
                    "message": "Module 'requests' not found",
                    "severity": "error"
                },
                {
                    "step": 4,
                    "error_type": "TimeoutError", 
                    "message": "Connection timed out after 30 seconds",
                    "severity": "warning"
                }
            ],
            "performance_metrics": {
                "execution_time": 45.2,
                "memory_usage": 512.0
            }
        }
        
        interpretation = await processor.interpret_test_results(error_results)
        
        assert "summary" in interpretation
        assert "recommendations" in interpretation
        assert len(interpretation["recommendations"]) > 0
        
        # Should provide actionable recommendations
        recommendations = interpretation["recommendations"]
        assert any("install" in rec.lower() or "dependency" in rec.lower() 
                  for rec in recommendations)
        
        await processor.shutdown()
