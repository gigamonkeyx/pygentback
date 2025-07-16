# CI/CD Troubleshooting Runbook - Observer Enhanced

## Quick Reference

### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Dependency Conflicts | `pip check` failures, import errors | [Dependency Resolution](#dependency-conflicts) |
| UTF-8 Encoding | Unicode errors, character encoding issues | [UTF-8 Configuration](#utf-8-encoding-issues) |
| Test Timeouts | Tests hanging, timeout errors | [Timeout Management](#test-timeout-issues) |
| Build Context Too Large | Slow Docker builds, context transfer issues | [Docker Optimization](#docker-build-issues) |
| Memory Issues | OOM errors, resource exhaustion | [Resource Management](#memory-and-resource-issues) |

## Dependency Conflicts

### Symptoms
- `pip check` reports version conflicts
- Import errors for critical packages
- Binary compatibility issues (numpy, torch)

### Diagnosis
```bash
# Check for conflicts
pip check

# Validate critical imports
python -c "
critical_imports = ['torch', 'networkx', 'fastapi', 'numpy', 'pandas']
for module in critical_imports:
    try:
        __import__(module)
        print(f'✅ {module}: OK')
    except ImportError as e:
        print(f'❌ {module}: {e}')
"
```

### Resolution
1. **Pin Critical Dependencies** (Observer-approved versions):
   ```
   torch==1.13.1  # CI/CD compatible
   networkx==2.8.8  # Stable version
   fastapi==0.115.9  # Minimum required
   numpy==1.26.4  # Binary compatibility
   ```

2. **Clean Install**:
   ```bash
   pip uninstall -y torch networkx fastapi
   pip install -r requirements.txt
   pip check
   ```

3. **Virtual Environment Reset**:
   ```bash
   rm -rf venv/
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

## UTF-8 Encoding Issues

### Symptoms
- `UnicodeEncodeError` in logs
- Character encoding failures
- Non-ASCII character issues

### Diagnosis
```bash
# Check environment variables
echo $PYTHONIOENCODING
echo $LANG
echo $LC_ALL

# Test UTF-8 handling
python -c "
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Test UTF-8: café, naïve, résumé')
"
```

### Resolution
1. **Set Environment Variables**:
   ```bash
   export PYTHONIOENCODING=utf-8
   export LANG=en_US.UTF-8
   export LC_ALL=en_US.UTF-8
   ```

2. **CI/CD Configuration** (already implemented):
   ```yaml
   env:
     PYTHONIOENCODING: utf-8
     LANG: en_US.UTF-8
     LC_ALL: en_US.UTF-8
   ```

3. **Python Code Fix**:
   ```python
   import sys
   import locale
   
   # Force UTF-8 encoding
   if sys.stdout.encoding != 'utf-8':
       sys.stdout.reconfigure(encoding='utf-8')
   ```

## Test Timeout Issues

### Symptoms
- Tests hanging indefinitely
- Timeout errors in CI
- Resource exhaustion during tests

### Diagnosis
```bash
# Check pytest configuration
cat pytest.ini | grep timeout

# Run with timeout debugging
pytest --timeout=60 --timeout-method=thread -v tests/
```

### Resolution
1. **Pytest Configuration** (already implemented):
   ```ini
   [pytest]
   addopts = --timeout=300 --maxfail=5 -n auto
   ```

2. **Parallel Testing**:
   ```bash
   pip install pytest-xdist
   pytest -n auto --dist=worksteal
   ```

3. **Resource Monitoring**:
   ```python
   import psutil
   
   def monitor_test_resources():
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024 / 1024
       cpu_percent = process.cpu_percent()
       print(f"Memory: {memory_mb:.1f}MB, CPU: {cpu_percent}%")
   ```

## Docker Build Issues

### Symptoms
- Slow Docker builds
- Large build context
- Build timeouts

### Diagnosis
```bash
# Check build context size
docker build --progress=plain . 2>&1 | grep "load build context"

# Analyze .dockerignore effectiveness
du -sh . && echo "Total project size"
du -sh .git tests node_modules 2>/dev/null || echo "Large dirs excluded"
```

### Resolution
1. **Optimize .dockerignore** (already implemented):
   ```
   .git/
   tests/
   node_modules/
   *.log
   __pycache__/
   ```

2. **Multi-stage Builds** (already implemented):
   ```dockerfile
   FROM python:3.11-slim as base
   # ... base stage
   
   FROM base as production
   # ... production stage
   ```

3. **Build Context Monitoring**:
   ```bash
   # Monitor build context transfer
   docker build --progress=plain . 2>&1 | head -20
   ```

## Memory and Resource Issues

### Symptoms
- Out of memory errors
- Resource exhaustion
- Slow test execution

### Diagnosis
```bash
# System resources
free -h
df -h
top -p $(pgrep python)

# Python memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```

### Resolution
1. **Resource Limits**:
   ```yaml
   # CI/CD job limits
   timeout-minutes: 15
   ```

2. **Memory Optimization**:
   ```python
   import gc
   
   # Force garbage collection
   gc.collect()
   
   # Monitor memory usage
   import psutil
   memory_percent = psutil.virtual_memory().percent
   ```

3. **Parallel Test Optimization**:
   ```bash
   # Limit parallel workers
   pytest -n 2  # Instead of -n auto
   ```

## Health Check Procedures

### Daily Checks
1. **Dependency Status**:
   ```bash
   pip check
   python scripts/test_dependency_validation.py
   ```

2. **Performance Metrics**:
   ```bash
   python scripts/ci_performance_monitor.py --report
   ```

3. **Build Success Rate**:
   ```bash
   # Check recent CI runs
   gh run list --limit 10
   ```

### Weekly Maintenance
1. **Dependency Updates**:
   ```bash
   pip list --outdated
   # Review and update requirements.txt
   ```

2. **Performance Baseline Review**:
   ```bash
   python scripts/ci_performance_monitor.py --analyze
   ```

3. **Log Analysis**:
   ```bash
   # Review CI logs for patterns
   grep -i "error\|warning\|timeout" ci_logs.txt
   ```

## Emergency Procedures

### CI/CD Complete Failure
1. **Immediate Actions**:
   - Check GitHub Actions status page
   - Verify repository access
   - Review recent commits for breaking changes

2. **Rollback Strategy**:
   ```bash
   git revert HEAD  # Revert last commit
   git push origin main
   ```

3. **Manual Validation**:
   ```bash
   # Local testing
   python scripts/test_utf8_validation.py
   python scripts/test_dependency_validation.py
   python scripts/test_resource_management.py
   ```

### Contact Information
- **Observer Supervision**: Monitor RIPER-Ω protocol compliance
- **Repository Owner**: Check GitHub repository settings
- **CI/CD Platform**: GitHub Actions status and documentation

---

**Observer Compliance**: This runbook maintains RIPER-Ω protocol standards for systematic troubleshooting and maintains zero-hallucination accuracy in all procedures.
