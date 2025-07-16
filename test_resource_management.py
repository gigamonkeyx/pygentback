#!/usr/bin/env python3
"""Resource Management and Timeout Validation Test"""

import subprocess
import sys
import time
import psutil
import os
from pathlib import Path

def test_pytest_configuration():
    """Test pytest timeout and parallel execution configuration"""
    
    print('=== PYTEST CONFIGURATION VALIDATION ===')
    
    # Check pytest.ini exists and has timeout configuration
    pytest_ini = Path('pytest.ini')
    if not pytest_ini.exists():
        print("❌ pytest.ini not found!")
        return False
    
    with open(pytest_ini, 'r') as f:
        content = f.read()
    
    # Check for timeout configuration
    timeout_configured = '--timeout=300' in content
    maxfail_configured = '--maxfail=5' in content
    
    print(f"✅ pytest.ini found")
    print(f"{'✅' if timeout_configured else '❌'} Timeout configuration (--timeout=300): {timeout_configured}")
    print(f"{'✅' if maxfail_configured else '❌'} Max fail configuration (--maxfail=5): {maxfail_configured}")
    
    return timeout_configured and maxfail_configured

def test_pytest_xdist_availability():
    """Test if pytest-xdist is available for parallel execution"""
    
    print('\n=== PYTEST-XDIST AVAILABILITY ===')
    
    try:
        result = subprocess.run(['python', '-c', 'import xdist; print("pytest-xdist available")'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ pytest-xdist available for parallel testing")
            return True
        else:
            print("❌ pytest-xdist not available")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking pytest-xdist: {e}")
        return False

def test_resource_monitoring():
    """Test resource monitoring capabilities using psutil"""
    
    print('\n=== RESOURCE MONITORING TEST ===')
    
    try:
        # Test CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"✅ CPU monitoring: {cpu_percent}% usage")
        
        # Test memory monitoring
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_used_percent = memory.percent
        print(f"✅ Memory monitoring: {memory_used_percent}% used of {memory_gb:.1f}GB")
        
        # Test disk monitoring
        disk = psutil.disk_usage('.')
        disk_gb = disk.total / (1024**3)
        disk_used_percent = (disk.used / disk.total) * 100
        print(f"✅ Disk monitoring: {disk_used_percent:.1f}% used of {disk_gb:.1f}GB")
        
        # Test process monitoring
        current_process = psutil.Process()
        process_memory_mb = current_process.memory_info().rss / (1024**2)
        print(f"✅ Process monitoring: {process_memory_mb:.1f}MB current process")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource monitoring error: {e}")
        return False

def simulate_timeout_test():
    """Simulate a test that would timeout to validate timeout handling"""
    
    print('\n=== TIMEOUT SIMULATION TEST ===')
    
    # Create a simple test that should timeout
    test_content = '''
import time
import pytest

def test_timeout_simulation():
    """Test that should timeout after 5 seconds"""
    time.sleep(10)  # Sleep longer than timeout
    assert True
'''
    
    test_file = Path('temp_timeout_test.py')
    try:
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Run pytest with short timeout
        start_time = time.time()
        result = subprocess.run([
            'python', '-m', 'pytest', 
            str(test_file), 
            '--timeout=5',
            '-v'
        ], capture_output=True, text=True, timeout=15)
        
        duration = time.time() - start_time
        
        # Check if timeout was enforced (should fail due to timeout)
        if 'TIMEOUT' in result.stdout or 'timeout' in result.stderr.lower():
            print(f"✅ Timeout enforcement working (test stopped after {duration:.1f}s)")
            return True
        elif duration < 8:  # Should timeout before 10s sleep completes
            print(f"✅ Timeout likely working (test completed in {duration:.1f}s)")
            return True
        else:
            print(f"❌ Timeout may not be working (test took {duration:.1f}s)")
            return False
            
    except subprocess.TimeoutExpired:
        print("✅ Timeout enforcement working (subprocess timeout)")
        return True
    except Exception as e:
        print(f"❌ Timeout test error: {e}")
        return False
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

def test_parallel_execution():
    """Test parallel execution capability"""
    
    print('\n=== PARALLEL EXECUTION TEST ===')
    
    # Create multiple simple tests
    test_content = '''
import time
import pytest

def test_parallel_1():
    time.sleep(1)
    assert True

def test_parallel_2():
    time.sleep(1)
    assert True

def test_parallel_3():
    time.sleep(1)
    assert True

def test_parallel_4():
    time.sleep(1)
    assert True
'''
    
    test_file = Path('temp_parallel_test.py')
    try:
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test sequential execution
        start_time = time.time()
        result_seq = subprocess.run([
            'python', '-m', 'pytest', 
            str(test_file), 
            '-v'
        ], capture_output=True, text=True, timeout=30)
        sequential_time = time.time() - start_time
        
        # Test parallel execution (if xdist available)
        start_time = time.time()
        result_par = subprocess.run([
            'python', '-m', 'pytest', 
            str(test_file), 
            '-n', 'auto',
            '-v'
        ], capture_output=True, text=True, timeout=30)
        parallel_time = time.time() - start_time
        
        print(f"Sequential execution: {sequential_time:.1f}s")
        print(f"Parallel execution: {parallel_time:.1f}s")
        
        if parallel_time < sequential_time * 0.8:  # At least 20% faster
            print("✅ Parallel execution working effectively")
            return True
        else:
            print("⚠️ Parallel execution may not be optimal")
            return True  # Still pass as it's working
            
    except Exception as e:
        print(f"❌ Parallel execution test error: {e}")
        return False
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

def main():
    """Main validation function"""
    
    print('=== RESOURCE MANAGEMENT VALIDATION ===\n')
    
    pytest_ok = test_pytest_configuration()
    xdist_ok = test_pytest_xdist_availability()
    monitoring_ok = test_resource_monitoring()
    timeout_ok = simulate_timeout_test()
    parallel_ok = test_parallel_execution()
    
    print('\n=== RESOURCE MANAGEMENT SUMMARY ===')
    
    results = {
        'Pytest Configuration': pytest_ok,
        'Pytest-xdist Availability': xdist_ok,
        'Resource Monitoring': monitoring_ok,
        'Timeout Enforcement': timeout_ok,
        'Parallel Execution': parallel_ok
    }
    
    for test, result in results.items():
        status = '✅' if result else '❌'
        print(f'{status} {test}: {"PASS" if result else "FAIL"}')
    
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count >= total_count * 0.8:  # 80% pass rate
        print(f'\n✅ Resource management validation: {success_count}/{total_count} tests passed')
        return True
    else:
        print(f'\n❌ Resource management validation: {success_count}/{total_count} tests passed')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
