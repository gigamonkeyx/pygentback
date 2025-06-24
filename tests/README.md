# PyGent Factory Test Suite

This directory contains all testing code for the PyGent Factory project, organized by test type and scope.

## Directory Structure

### `/unit/`
Unit tests for individual components and modules. These tests focus on testing single functions, classes, or small units of code in isolation.

### `/integration/`
Integration tests that verify the interaction between multiple components or systems working together.

### `/system/`
System-level tests that validate the entire PyGent Factory system end-to-end, including comprehensive validation suites.

### `/validation/`
Production validation tests that verify system readiness, performance, and compliance with requirements.

### `/phase_tests/`
Phase-specific validation tests for the mock elimination project, organized by development phases.

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# System tests only
pytest tests/system/

# Validation tests only
pytest tests/validation/
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Organization Principles

1. **Separation of Concerns**: Each test type has its own directory
2. **Production Ready**: All tests are designed for production validation
3. **Zero Mock Dependencies**: Tests validate real implementations, not mock code
4. **Comprehensive Coverage**: Tests cover all critical system functionality

## Contributing

When adding new tests:
1. Place unit tests in `/unit/`
2. Place integration tests in `/integration/`
3. Place system-wide tests in `/system/`
4. Place production validation in `/validation/`
5. Follow existing naming conventions
6. Ensure tests are independent and can run in any order
