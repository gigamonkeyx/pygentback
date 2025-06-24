import requests

print('Quick Ollama test...')
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=3)
    print(f'Ollama direct: {response.status_code}')
except Exception as e:
    print(f'Ollama failed: {e}')

print('Quick backend test...')
try:
    response = requests.get('http://localhost:8000/api/v1/health', timeout=3)
    print(f'Backend: {response.status_code}')
except Exception as e:
    print(f'Backend failed: {e}')
