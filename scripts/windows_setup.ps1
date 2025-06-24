# Windows PowerShell Setup Script for PyGent Factory
# Fixes common Windows/PowerShell issues that cause Python import hanging

Write-Host "Setting up PyGent Factory for Windows..." -ForegroundColor Green

# 1. Set PowerShell Execution Policy
Write-Host "Setting PowerShell execution policy..." -ForegroundColor Yellow
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "✅ PowerShell execution policy set to RemoteSigned" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not set execution policy: $_" -ForegroundColor Yellow
}

# 2. Set Windows Console Mode
Write-Host "Configuring Windows console..." -ForegroundColor Yellow
try {
    # Enable ANSI escape sequences
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    
    # Disable Windows console legacy mode
    if (Get-Command "reg" -ErrorAction SilentlyContinue) {
        reg add "HKCU\Console" /v "VirtualTerminalLevel" /t REG_DWORD /d 1 /f | Out-Null
    }
    Write-Host "✅ Console configuration updated" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not configure console: $_" -ForegroundColor Yellow
}

# 3. Set Python Environment Variables
Write-Host "Setting Python environment variables..." -ForegroundColor Yellow
$env:PYTHONDONTWRITEBYTECODE = "1"  # Prevent .pyc files
$env:PYTHONUNBUFFERED = "1"         # Unbuffered output
$env:PYTHONPATH = (Get-Location).Path + "\src"  # Add src to Python path

Write-Host "✅ Python environment configured" -ForegroundColor Green

# 4. Check Python Installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    exit 1
}

# 5. Install/Update Required Packages
Write-Host "Checking required packages..." -ForegroundColor Yellow
$packages = @("psutil", "asyncio", "pydantic", "fastapi", "uvicorn")

foreach ($package in $packages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $package is installed" -ForegroundColor Green
        } else {
            Write-Host "Installing $package..." -ForegroundColor Yellow
            pip install $package
        }
    } catch {
        Write-Host "⚠️ Could not check $package" -ForegroundColor Yellow
    }
}

# 6. Create Windows-Specific Launch Script
Write-Host "Creating Windows launch script..." -ForegroundColor Yellow
$launchScript = @"
@echo off
REM Windows Launch Script for PyGent Factory
REM Prevents common hanging issues

REM Set environment variables
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONPATH=%~dp0src

REM Change to script directory
cd /d "%~dp0"

REM Launch with proper Windows settings
python %*
"@

$launchScript | Out-File -FilePath "run_windows.bat" -Encoding ASCII
Write-Host "✅ Created run_windows.bat launcher" -ForegroundColor Green

# 7. Test Basic Import
Write-Host "Testing basic Python imports..." -ForegroundColor Yellow
try {
    $testResult = python -c "
import sys
import os
print('Python version:', sys.version)
print('Platform:', sys.platform)
print('Working directory:', os.getcwd())

# Test problematic imports
try:
    import psutil
    print('✅ psutil imported successfully')
except Exception as e:
    print('❌ psutil failed:', e)

try:
    import asyncio
    print('✅ asyncio imported successfully')
except Exception as e:
    print('❌ asyncio failed:', e)

print('Basic imports test completed')
" 2>&1

    Write-Host $testResult -ForegroundColor Cyan
} catch {
    Write-Host "❌ Basic import test failed: $_" -ForegroundColor Red
}

# 8. Create Test Script for Imports
Write-Host "Creating import test script..." -ForegroundColor Yellow
$testScript = @"
# Test script for PyGent Factory imports
import sys
import os
import time

print("Testing PyGent Factory imports...")
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Working directory: {os.getcwd()}")

# Test basic imports first
try:
    import psutil
    print("✅ psutil imported")
except Exception as e:
    print(f"❌ psutil failed: {e}")

try:
    import asyncio
    print("✅ asyncio imported")
except Exception as e:
    print(f"❌ asyncio failed: {e}")

# Test our modules with timeout
def test_import_with_timeout(module_name, timeout=10):
    import signal
    import importlib
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Import of {module_name} timed out")
    
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        start_time = time.time()
        importlib.import_module(module_name)
        duration = time.time() - start_time
        print(f"✅ {module_name} imported in {duration:.2f}s")
        return True
    except TimeoutError:
        print(f"❌ {module_name} import timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")
        return False
    finally:
        if sys.platform != 'win32':
            signal.alarm(0)

# Test our modules
modules_to_test = [
    'src.integration.workflows',
    'src.integration.monitoring', 
    'src.integration.events',
    'src.testing.analytics.dashboard',
    'src.testing.analytics.trends'
]

for module in modules_to_test:
    test_import_with_timeout(module)

print("Import test completed!")
"@

$testScript | Out-File -FilePath "test_imports.py" -Encoding UTF8
Write-Host "✅ Created test_imports.py" -ForegroundColor Green

# 9. Final Instructions
Write-Host "`n🎉 Windows setup completed!" -ForegroundColor Green
Write-Host "`nTo run PyGent Factory on Windows:" -ForegroundColor Cyan
Write-Host "1. Use: .\run_windows.bat <your_python_script>" -ForegroundColor White
Write-Host "2. Or: python test_imports.py (to test imports)" -ForegroundColor White
Write-Host "3. Or: Set environment variables manually and use python normally" -ForegroundColor White

Write-Host "`nEnvironment variables set for this session:" -ForegroundColor Cyan
Write-Host "PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE" -ForegroundColor White
Write-Host "PYTHONUNBUFFERED=$env:PYTHONUNBUFFERED" -ForegroundColor White
Write-Host "PYTHONIOENCODING=$env:PYTHONIOENCODING" -ForegroundColor White
Write-Host "PYTHONUTF8=$env:PYTHONUTF8" -ForegroundColor White
Write-Host "PYTHONPATH=$env:PYTHONPATH" -ForegroundColor White
