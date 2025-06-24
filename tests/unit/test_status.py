import requests

workflow_id = 'research_analysis_1749091766_2859'
print(f'Checking status for {workflow_id}...')

try:
    response = requests.get(f'http://localhost:8000/api/v1/workflows/research-analysis/{workflow_id}/status', timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f'Status: {data.get("status", "unknown")}')
        progress = data.get('progress', {})
        print(f'Progress: {progress.get("progress_percentage", 0):.1f}%')
        print(f'Step: {progress.get("current_step", "unknown")}')
        if progress.get('research_papers_found', 0) > 0:
            print(f'Papers found: {progress["research_papers_found"]}')
    else:
        print(f'Status check failed: {response.status_code}')
except Exception as e:
    print(f'Error: {e}')
