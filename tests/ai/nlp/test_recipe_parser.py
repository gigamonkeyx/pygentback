"""
Tests for NLP recipe parsing components.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.ai.nlp.recipe import RecipeParser, RecipeAnalyzer
from src.ai.nlp.models import ParsedRecipe, RecipeIntent
from tests.utils.helpers import create_test_recipe


class TestRecipeParser:
    """Test cases for RecipeParser."""
    
    @pytest.fixture
    def recipe_parser(self):
        """Create recipe parser instance."""
        return RecipeParser()
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    @pytest.mark.recipe_parser
    async def test_parser_initialization(self, recipe_parser):
        """Test recipe parser initialization."""
        assert recipe_parser.is_initialized is False
        
        success = await recipe_parser.initialize()
        assert success is True
        assert recipe_parser.is_initialized is True
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    @pytest.mark.recipe_parser
    async def test_parse_simple_recipe(self, recipe_parser):
        """Test parsing a simple recipe."""
        await recipe_parser.initialize()
        
        recipe_text = """
        Recipe: Simple Data Processing
        Description: Process CSV data and generate reports
        
        Steps:
        1. Load CSV file using pandas
        2. Clean and validate data
        3. Generate summary statistics
        4. Create visualization charts
        5. Export results to Excel
        
        Requirements:
        - pandas
        - matplotlib
        - openpyxl
        """
        
        parsed = await recipe_parser.parse_recipe(recipe_text)
        
        assert isinstance(parsed, ParsedRecipe)
        assert parsed.name == "Simple Data Processing"
        assert "Process CSV data" in parsed.description
        assert len(parsed.steps) == 5
        assert len(parsed.requirements) >= 3
        assert parsed.complexity > 0
        assert parsed.estimated_duration > 0
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    @pytest.mark.recipe_parser
    async def test_parse_complex_recipe(self, recipe_parser):
        """Test parsing a complex recipe with multiple sections."""
        await recipe_parser.initialize()
        
        complex_recipe = """
        Recipe: Machine Learning Pipeline
        Description: Complete ML pipeline for classification tasks
        Category: machine_learning
        Difficulty: advanced
        
        Prerequisites:
        - Python 3.8+
        - Basic ML knowledge
        - GPU recommended
        
        Steps:
        1. Data Collection and Preparation
           - Gather training data from multiple sources
           - Clean and preprocess the data
           - Split into train/validation/test sets
           
        2. Feature Engineering
           - Extract relevant features
           - Apply feature scaling and normalization
           - Handle categorical variables
           
        3. Model Training
           - Try multiple algorithms (RF, SVM, Neural Networks)
           - Perform hyperparameter tuning
           - Use cross-validation for model selection
           
        4. Model Evaluation
           - Calculate performance metrics
           - Generate confusion matrices
           - Analyze feature importance
           
        5. Deployment
           - Save trained model
           - Create prediction API
           - Set up monitoring
        
        Requirements:
        - scikit-learn >= 1.0
        - pandas >= 1.3
        - numpy >= 1.21
        - matplotlib >= 3.5
        - seaborn >= 0.11
        - fastapi (for API)
        - uvicorn (for serving)
        
        Expected Outputs:
        - Trained model file (.pkl)
        - Performance report (PDF)
        - API endpoint for predictions
        - Monitoring dashboard
        
        Estimated Duration: 4-6 hours
        Resource Requirements:
        - CPU: 4+ cores
        - RAM: 8+ GB
        - Storage: 2+ GB
        """
        
        parsed = await recipe_parser.parse_recipe(complex_recipe)
        
        assert isinstance(parsed, ParsedRecipe)
        assert parsed.name == "Machine Learning Pipeline"
        assert parsed.category == "machine_learning"
        assert parsed.difficulty == "advanced"
        assert len(parsed.steps) == 5
        assert len(parsed.requirements) >= 7
        assert len(parsed.expected_outputs) >= 4
        assert parsed.complexity >= 7  # Should be high for complex recipe
        assert parsed.estimated_duration >= 240  # 4+ hours in minutes
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    @pytest.mark.recipe_parser
    async def test_parse_recipe_with_code_blocks(self, recipe_parser):
        """Test parsing recipe with embedded code blocks."""
        await recipe_parser.initialize()
        
        recipe_with_code = """
        Recipe: API Testing Automation
        Description: Automated testing for REST APIs
        
        Steps:
        1. Setup test environment
           ```python
           import requests
           import pytest
           
           BASE_URL = "https://api.example.com"
           ```
        
        2. Create test fixtures
           ```python
           @pytest.fixture
           def api_client():
               return requests.Session()
           ```
        
        3. Write test cases
           ```python
           def test_get_users(api_client):
               response = api_client.get(f"{BASE_URL}/users")
               assert response.status_code == 200
               assert len(response.json()) > 0
           ```
        
        Requirements:
        - requests
        - pytest
        """
        
        parsed = await recipe_parser.parse_recipe(recipe_with_code)
        
        assert isinstance(parsed, ParsedRecipe)
        assert parsed.name == "API Testing Automation"
        assert len(parsed.steps) == 3
        assert any("import requests" in str(step) for step in parsed.steps)
        assert any("pytest.fixture" in str(step) for step in parsed.steps)
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    @pytest.mark.recipe_parser
    async def test_parse_malformed_recipe(self, recipe_parser):
        """Test parsing malformed or incomplete recipes."""
        await recipe_parser.initialize()
        
        malformed_recipes = [
            "This is not a recipe at all",
            "Recipe: \nNo description or steps",
            "Steps: 1. Do something\n2. Do something else\n(No recipe name)",
            ""
        ]
        
        for malformed_text in malformed_recipes:
            parsed = await recipe_parser.parse_recipe(malformed_text)
            
            # Should handle gracefully
            assert isinstance(parsed, ParsedRecipe)
            assert parsed.name is not None  # Should provide default name
            assert isinstance(parsed.steps, list)
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    @pytest.mark.recipe_parser
    async def test_extract_recipe_metadata(self, recipe_parser):
        """Test extraction of recipe metadata."""
        await recipe_parser.initialize()
        
        recipe_with_metadata = """
        Recipe: Data Visualization Dashboard
        Description: Create interactive dashboard with Plotly
        Author: John Doe
        Version: 2.1
        Created: 2024-01-15
        Tags: visualization, dashboard, plotly, interactive
        License: MIT
        
        Steps:
        1. Install required packages
        2. Load and prepare data
        3. Create dashboard layout
        4. Add interactive components
        5. Deploy to web server
        
        Requirements:
        - plotly
        - dash
        - pandas
        """
        
        parsed = await recipe_parser.parse_recipe(recipe_with_metadata)
        
        assert parsed.metadata["author"] == "John Doe"
        assert parsed.metadata["version"] == "2.1"
        assert "visualization" in parsed.metadata["tags"]
        assert "dashboard" in parsed.metadata["tags"]
        assert parsed.metadata["license"] == "MIT"


class TestRecipeAnalyzer:
    """Test cases for RecipeAnalyzer."""
    
    @pytest.fixture
    def recipe_analyzer(self):
        """Create recipe analyzer instance."""
        return RecipeAnalyzer()
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_analyzer_initialization(self, recipe_analyzer):
        """Test recipe analyzer initialization."""
        success = await recipe_analyzer.initialize()
        assert success is True
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_analyze_complexity(self, recipe_analyzer):
        """Test recipe complexity analysis."""
        await recipe_analyzer.initialize()
        
        # Simple recipe
        simple_recipe = create_test_recipe("simple", complexity=3)
        simple_analysis = await recipe_analyzer.analyze_complexity(simple_recipe)
        
        assert simple_analysis["complexity_score"] <= 5
        assert simple_analysis["difficulty"] in ["basic", "intermediate"]
        
        # Complex recipe
        complex_recipe = create_test_recipe("complex", complexity=8)
        complex_analysis = await recipe_analyzer.analyze_complexity(complex_recipe)
        
        assert complex_analysis["complexity_score"] >= 6
        assert complex_analysis["difficulty"] in ["advanced", "expert"]
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_estimate_duration(self, recipe_analyzer):
        """Test recipe duration estimation."""
        await recipe_analyzer.initialize()
        
        test_recipe = create_test_recipe("test", complexity=5)
        
        duration_estimate = await recipe_analyzer.estimate_duration(test_recipe)
        
        assert "estimated_minutes" in duration_estimate
        assert "confidence" in duration_estimate
        assert duration_estimate["estimated_minutes"] > 0
        assert 0.0 <= duration_estimate["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_identify_dependencies(self, recipe_analyzer):
        """Test dependency identification."""
        await recipe_analyzer.initialize()
        
        recipe_with_deps = {
            "name": "Web Scraping Recipe",
            "steps": [
                {"description": "Install requests and beautifulsoup4"},
                {"description": "Use pandas for data processing"},
                {"description": "Save to PostgreSQL database"}
            ],
            "requirements": ["requests", "beautifulsoup4", "pandas", "psycopg2"]
        }
        
        dependencies = await recipe_analyzer.identify_dependencies(recipe_with_deps)
        
        assert "python_packages" in dependencies
        assert "external_services" in dependencies
        assert "requests" in dependencies["python_packages"]
        assert "pandas" in dependencies["python_packages"]
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_analyze_resource_requirements(self, recipe_analyzer):
        """Test resource requirement analysis."""
        await recipe_analyzer.initialize()
        
        resource_intensive_recipe = {
            "name": "ML Training Recipe",
            "description": "Train deep learning model on large dataset",
            "steps": [
                {"description": "Load 10GB dataset into memory"},
                {"description": "Train neural network for 100 epochs"},
                {"description": "Evaluate on test set"}
            ],
            "requirements": ["tensorflow", "numpy", "pandas"]
        }
        
        resources = await recipe_analyzer.analyze_resource_requirements(resource_intensive_recipe)
        
        assert "cpu_cores" in resources
        assert "memory_gb" in resources
        assert "storage_gb" in resources
        assert "gpu_required" in resources
        assert resources["memory_gb"] >= 8  # Should detect high memory need
    
    @pytest.mark.asyncio
    @pytest.mark.nlp
    async def test_suggest_improvements(self, recipe_analyzer):
        """Test recipe improvement suggestions."""
        await recipe_analyzer.initialize()
        
        improvable_recipe = {
            "name": "Basic Recipe",
            "description": "Simple task",
            "steps": [
                {"description": "Do something"},
                {"description": "Do something else"}
            ],
            "requirements": []
        }
        
        suggestions = await recipe_analyzer.suggest_improvements(improvable_recipe)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest adding more details, requirements, etc.
        suggestion_text = " ".join(suggestions).lower()
        assert any(keyword in suggestion_text for keyword in 
                  ["detail", "requirement", "step", "description", "parameter"])


@pytest.mark.nlp
@pytest.mark.recipe_parser
@pytest.mark.integration
class TestRecipeParsingIntegration:
    """Integration tests for recipe parsing components."""
    
    @pytest.mark.asyncio
    async def test_full_parsing_and_analysis_pipeline(self):
        """Test complete recipe parsing and analysis pipeline."""
        parser = RecipeParser()
        analyzer = RecipeAnalyzer()
        
        await parser.initialize()
        await analyzer.initialize()
        
        comprehensive_recipe = """
        Recipe: E-commerce Data Pipeline
        Description: Complete data pipeline for e-commerce analytics
        Category: data_engineering
        Difficulty: advanced
        
        Steps:
        1. Extract data from multiple sources (APIs, databases, files)
        2. Transform and clean the data using pandas and dask
        3. Load data into data warehouse (Snowflake/BigQuery)
        4. Create data models and aggregations
        5. Build real-time dashboards with Tableau/PowerBI
        6. Set up automated monitoring and alerting
        
        Requirements:
        - Python 3.9+
        - pandas >= 1.4
        - dask >= 2022.1
        - sqlalchemy >= 1.4
        - snowflake-connector-python
        - apache-airflow >= 2.3
        
        Resource Requirements:
        - CPU: 8+ cores
        - RAM: 16+ GB
        - Storage: 100+ GB
        - Network: High bandwidth for data transfer
        
        Expected Duration: 2-3 days
        """
        
        # Parse the recipe
        parsed = await parser.parse_recipe(comprehensive_recipe)
        
        # Analyze the parsed recipe
        complexity = await analyzer.analyze_complexity(parsed.to_dict())
        duration = await analyzer.estimate_duration(parsed.to_dict())
        dependencies = await analyzer.identify_dependencies(parsed.to_dict())
        resources = await analyzer.analyze_resource_requirements(parsed.to_dict())
        
        # Verify results
        assert parsed.name == "E-commerce Data Pipeline"
        assert parsed.category == "data_engineering"
        assert parsed.difficulty == "advanced"
        
        assert complexity["complexity_score"] >= 7
        assert complexity["difficulty"] == "advanced"
        
        assert duration["estimated_minutes"] >= 2880  # 2+ days
        
        assert "pandas" in dependencies["python_packages"]
        assert "dask" in dependencies["python_packages"]
        
        assert resources["cpu_cores"] >= 8
        assert resources["memory_gb"] >= 16
        
        await parser.shutdown()
        await analyzer.shutdown()
