import requests

print('🔍 Testing PyGent Factory → Ollama connection...')
try:
    # Test backend health with Ollama status
    response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
    if response.status_code == 200:
        data = response.json()
        print('✅ Backend health check successful')
        
        # Check Ollama status in health response
        components = data.get('components', {})
        ollama_status = components.get('ollama', {})
        print(f'Ollama status: {ollama_status.get("status", "unknown")}')
        
        if ollama_status.get('status') == 'healthy':
            print('🎉 Ollama is properly connected to PyGent Factory!')
        else:
            print('⚠️ Ollama connection may have issues')
    else:
        print(f'❌ Backend health check failed: {response.status_code}')
except Exception as e:
    print(f'❌ Connection test failed: {e}')
