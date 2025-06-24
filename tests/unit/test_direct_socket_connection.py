#!/usr/bin/env python3
"""
Test Direct Socket Connection

Test if the 2-second delay is in socket connections vs HTTP layer.
"""

import socket
import time
import requests

def test_socket_connection_speed():
    """Test raw socket connection speed"""
    print("🔗 Testing Raw Socket Connection Speed...")
    
    ports = [8003, 8004, 8005]  # Skip 8002 since it's not running
    
    for port in ports:
        print(f"\n   Testing port {port}:")
        
        for i in range(3):
            try:
                start_time = time.time()
                
                # Create socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                
                # Connect
                result = sock.connect_ex(('127.0.0.1', port))
                connect_time = time.time() - start_time
                
                if result == 0:
                    print(f"     Socket {i+1}: {connect_time*1000:.1f}ms (SUCCESS)")
                else:
                    print(f"     Socket {i+1}: {connect_time*1000:.1f}ms (FAILED - {result})")
                
                sock.close()
                
            except Exception as e:
                connect_time = time.time() - start_time
                print(f"     Socket {i+1}: {connect_time*1000:.1f}ms (ERROR - {str(e)})")

def test_http_vs_socket():
    """Compare HTTP vs raw socket timing"""
    print("\n🌐 Comparing HTTP vs Socket Timing...")
    
    port = 8003  # Document processing server
    
    # Test socket connection
    print(f"   Raw socket to port {port}:")
    socket_times = []
    for i in range(3):
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', port))
            socket_time = time.time() - start_time
            sock.close()
            
            if result == 0:
                socket_times.append(socket_time)
                print(f"     Socket {i+1}: {socket_time*1000:.1f}ms")
            else:
                print(f"     Socket {i+1}: FAILED")
        except Exception as e:
            print(f"     Socket {i+1}: ERROR ({str(e)})")
    
    # Test HTTP request
    print(f"   HTTP request to port {port}:")
    http_times = []
    for i in range(3):
        try:
            start_time = time.time()
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            http_time = time.time() - start_time
            
            if response.status_code == 200:
                http_times.append(http_time)
                print(f"     HTTP {i+1}: {http_time*1000:.1f}ms")
            else:
                print(f"     HTTP {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"     HTTP {i+1}: ERROR ({str(e)})")
    
    # Compare results
    if socket_times and http_times:
        avg_socket = sum(socket_times) / len(socket_times)
        avg_http = sum(http_times) / len(http_times)
        
        print(f"\n   📊 COMPARISON:")
        print(f"     Average Socket: {avg_socket*1000:.1f}ms")
        print(f"     Average HTTP: {avg_http*1000:.1f}ms")
        print(f"     HTTP Overhead: {(avg_http - avg_socket)*1000:.1f}ms")
        
        if avg_http > avg_socket * 10:  # HTTP is 10x slower than socket
            print(f"     🚨 ISSUE: HTTP layer is causing significant delays!")
        elif avg_socket > 0.1:  # Socket itself is slow
            print(f"     🚨 ISSUE: Socket connection is slow!")
        else:
            print(f"     ✅ Both socket and HTTP are reasonably fast")

def test_localhost_vs_127001():
    """Test localhost vs 127.0.0.1"""
    print("\n🏠 Testing localhost vs 127.0.0.1...")
    
    port = 8003
    
    # Test localhost
    print(f"   Testing localhost:{port}:")
    localhost_times = []
    for i in range(3):
        try:
            start_time = time.time()
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            req_time = time.time() - start_time
            
            if response.status_code == 200:
                localhost_times.append(req_time)
                print(f"     localhost {i+1}: {req_time*1000:.1f}ms")
            else:
                print(f"     localhost {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"     localhost {i+1}: ERROR ({str(e)})")
    
    # Test 127.0.0.1
    print(f"   Testing 127.0.0.1:{port}:")
    ip_times = []
    for i in range(3):
        try:
            start_time = time.time()
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            req_time = time.time() - start_time
            
            if response.status_code == 200:
                ip_times.append(req_time)
                print(f"     127.0.0.1 {i+1}: {req_time*1000:.1f}ms")
            else:
                print(f"     127.0.0.1 {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"     127.0.0.1 {i+1}: ERROR ({str(e)})")
    
    # Compare results
    if localhost_times and ip_times:
        avg_localhost = sum(localhost_times) / len(localhost_times)
        avg_ip = sum(ip_times) / len(ip_times)
        
        print(f"\n   📊 COMPARISON:")
        print(f"     Average localhost: {avg_localhost*1000:.1f}ms")
        print(f"     Average 127.0.0.1: {avg_ip*1000:.1f}ms")
        
        if abs(avg_localhost - avg_ip) > 0.5:  # More than 500ms difference
            print(f"     🚨 ISSUE: Significant difference between localhost and 127.0.0.1!")
            if avg_localhost > avg_ip:
                print(f"     💡 SOLUTION: Use 127.0.0.1 instead of localhost")
            else:
                print(f"     💡 SOLUTION: Use localhost instead of 127.0.0.1")
        else:
            print(f"     ✅ No significant difference between localhost and 127.0.0.1")

def main():
    """Main execution"""
    print("🔍 Direct Socket Connection Analysis")
    print("=" * 50)
    
    test_socket_connection_speed()
    test_http_vs_socket()
    test_localhost_vs_127001()
    
    print("\n" + "=" * 50)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
