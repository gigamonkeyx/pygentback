#!/usr/bin/env python3
"""
Test script to verify document loading endpoints and help debug
the corrupted document issue in DocumentationPageV2.
"""

import requests
import json
import sys
import os

def test_document_loading():
    """Test various document loading endpoints to identify working ones."""
    
    base_url = "http://localhost:8000"
    
    print("Testing Document Loading Endpoints")
    print("=" * 50)
    
    # First, get the list of available files
    try:
        response = requests.get(f"{base_url}/api/files", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                files = data.get('data', {}).get('files', [])
                print(f"✓ Found {len(files)} files available")
                
                # Test loading a few different documents
                test_files = files[:3]  # Test first 3 files
                
                for file_info in test_files:
                    file_id = file_info.get('id', '')
                    file_path = file_info.get('path', '')
                    file_title = file_info.get('title', 'Unknown')
                    
                    print(f"\n--- Testing file: {file_title} ---")
                    print(f"ID: {file_id}")
                    print(f"Path: {file_path}")
                    
                    # Test different endpoint formats
                    endpoints_to_test = [
                        f"/api/files/{requests.utils.quote(file_path, safe='')}",
                        f"/api/files/{requests.utils.quote(file_id, safe='')}",
                        f"/api/files/{requests.utils.quote(file_path.split('/')[-1], safe='')}",
                        f"/api/documentation/files/{requests.utils.quote(file_path, safe='')}"
                    ]
                    
                    for endpoint in endpoints_to_test:
                        try:
                            print(f"  Testing: {endpoint}")
                            response = requests.get(f"{base_url}{endpoint}", timeout=5)
                            print(f"    Status: {response.status_code}")
                            
                            if response.status_code == 200:
                                try:
                                    data = response.json()
                                    if data.get('status') == 'success':
                                        content = data.get('data', {})
                                        has_content = bool(content.get('content') or content.get('raw_content'))
                                        print(f"    ✓ SUCCESS - Has content: {has_content}")
                                        if has_content:
                                            content_preview = (content.get('content') or content.get('raw_content', ''))[:100]
                                            print(f"    Preview: {content_preview}...")
                                            break  # Found working endpoint
                                    else:
                                        print(f"    ✗ API returned error: {data.get('error', 'Unknown error')}")
                                except json.JSONDecodeError:
                                    print(f"    ✗ Invalid JSON response")
                            else:
                                print(f"    ✗ HTTP Error: {response.status_code}")
                                
                        except Exception as e:
                            print(f"    ✗ Request failed: {e}")
                    
                    print()
                    
                return True
            else:
                print(f"✗ API error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Failed to get file list: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error getting file list: {e}")
        return False

def test_backend_direct():
    """Test if backend files exist directly in the filesystem."""
    
    print("\nTesting Backend File System")
    print("=" * 50)
    
    # Common documentation paths
    possible_paths = [
        "docs/",
        "documentation/",
        "src/docs/",
        "data/docs/",
        "storage/docs/",
        "files/docs/",
        "README.md",
        "docs/README.md"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for path in possible_paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            print(f"✓ Found: {full_path}")
            if os.path.isdir(full_path):
                files = os.listdir(full_path)
                print(f"  Contains {len(files)} items: {files[:5]}...")
            else:
                print(f"  File size: {os.path.getsize(full_path)} bytes")
        else:
            print(f"✗ Not found: {full_path}")

def main():
    """Main test function."""
    print("DocumentationPageV2 - Document Loading Debug Test")
    print("=" * 60)
    
    print("\n1. Testing API endpoints...")
    if not test_document_loading():
        print("API endpoint tests failed!")
        
    print("\n2. Testing file system...")
    test_backend_direct()
    
    print("\n" + "=" * 60)
    print("Debug Summary:")
    print("- If API endpoints are working but returning empty content,")
    print("  the issue is likely in the backend file reading logic.")
    print("- If file system shows files exist, check backend permissions.")
    print("- The frontend should now show better error messages and")
    print("  provide retry functionality for failed documents.")
    
    print("\nTo test the UI improvements:")
    print("1. Start backend: python -m uvicorn src.api.main:app --reload")
    print("2. Start frontend: cd ui && npm run dev")
    print("3. Open browser and test document loading")
    print("4. Look for improved error messages and retry button")

if __name__ == "__main__":
    main()
