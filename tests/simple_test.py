import asyncio
import aiohttp

async def test():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/api/v1/workflows/research-analysis',
            json={'query': 'quantum computing', 'max_papers': 3},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f'SUCCESS: {data.get("workflow_id")}')
            else:
                print(f'FAILED: {response.status}')

asyncio.run(test())
