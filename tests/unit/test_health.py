import requests

try:
    response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
    print(f'Backend health: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Overall status: {data.get("status", "unknown")}')
        print(f'Ollama status: {data.get("components", {}).get("ollama", {}).get("status", "unknown")}')
except Exception as e:
    print(f'Health check error: {e}')
