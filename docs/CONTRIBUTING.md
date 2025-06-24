# Contributing to PyGent Factory

We welcome contributions to PyGent Factory! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- Node.js 18 or higher
- PostgreSQL 12 or higher
- Redis 6 or higher
- Git

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/pygentback.git
   cd pygentback
   ```

2. **Set up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up Frontend**
   ```bash
   cd src/ui
   npm install
   cd ../..
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize Database**
   ```bash
   python scripts/init_db.py
   ```

6. **Run Tests**
   ```bash
   pytest tests/
   npm test
   ```

## 📋 Development Guidelines

### Code Style

#### Python
- Follow PEP 8 style guide
- Use Black for code formatting: `black src/`
- Use isort for import sorting: `isort src/`
- Use flake8 for linting: `flake8 src/`
- Type hints are required for all functions

#### TypeScript/JavaScript
- Use ESLint and Prettier
- Follow React best practices
- Use TypeScript strict mode
- Document components with JSDoc

#### Example Python Code Style
```python
from typing import List, Optional
from pydantic import BaseModel


class AgentRequest(BaseModel):
    """Request model for agent creation."""
    
    name: str
    capabilities: List[str]
    config: Optional[dict] = None


async def create_agent(request: AgentRequest) -> Agent:
    """Create a new agent with the specified configuration.
    
    Args:
        request: The agent creation request
        
    Returns:
        The created agent instance
        
    Raises:
        ValueError: If the request is invalid
    """
    # Implementation here
    pass
```

### Git Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write tests first (TDD approach preferred)
   - Implement the feature
   - Update documentation

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new agent capability"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We use [Conventional Commits](https://conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(mcp): add new server registry functionality
fix(api): resolve authentication token validation
docs(readme): update installation instructions
test(agents): add unit tests for agent factory
```

## 🧪 Testing

### Running Tests

```bash
# Python tests
pytest tests/ -v

# Frontend tests
cd src/ui && npm test

# Integration tests
python tests/integration/test_full_system.py

# Performance tests
python tests/performance/test_load.py
```

### Test Coverage

- Aim for 80%+ test coverage
- Write unit tests for all new functions
- Write integration tests for API endpoints
- Write E2E tests for critical user workflows

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

from src.agents.factory import AgentFactory


class TestAgentFactory:
    """Test suite for AgentFactory."""
    
    @pytest.fixture
    def factory(self):
        """Create an AgentFactory instance for testing."""
        return AgentFactory()
    
    def test_create_agent_success(self, factory):
        """Test successful agent creation."""
        # Arrange
        config = {"name": "test-agent", "type": "general"}
        
        # Act
        agent = factory.create_agent(config)
        
        # Assert
        assert agent.name == "test-agent"
        assert agent.type == "general"
    
    def test_create_agent_invalid_config(self, factory):
        """Test agent creation with invalid config."""
        # Arrange
        config = {}
        
        # Act & Assert
        with pytest.raises(ValueError, match="Agent name is required"):
            factory.create_agent(config)
```

## 📝 Documentation

### Code Documentation

- Use docstrings for all classes and functions
- Follow Google-style docstrings
- Include examples in docstrings when helpful

### API Documentation

- Update OpenAPI schemas for API changes
- Test API documentation with real requests
- Include request/response examples

### User Documentation

- Update README.md for user-facing changes
- Add guides to the `docs/` directory
- Update CHANGELOG.md for releases

## 🔧 Architecture Guidelines

### Adding New Features

1. **Design First**
   - Create an issue describing the feature
   - Discuss the approach with maintainers
   - Create a design document for complex features

2. **Implementation**
   - Follow existing patterns
   - Maintain backward compatibility
   - Add configuration options when appropriate

3. **Integration**
   - Update relevant documentation
   - Add monitoring/logging
   - Consider performance implications

### MCP Server Development

When adding new MCP servers:

1. Follow the MCP specification
2. Add comprehensive error handling
3. Include timeout and retry logic
4. Document all tools and resources
5. Add integration tests

### Agent Development

When creating new agent types:

1. Inherit from `BaseAgent`
2. Define clear capabilities
3. Implement proper error handling
4. Add evaluation metrics
5. Document agent behavior

## 🚦 Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] Feature branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project standards
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits before merge (if requested)

## 🐛 Bug Reports

### Before Reporting

1. Search existing issues
2. Try latest version
3. Check documentation
4. Reproduce the issue

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g. Windows 10]
- Python: [e.g. 3.11.2]
- PyGent Factory: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Describe the problem this solves

**Proposed Solution**
How you envision this working

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other context or screenshots
```

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Check the docs first
- **Code Review**: Ask for feedback on your approach

## 🏆 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor statistics

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

Thank you for contributing to PyGent Factory! 🎉
