#!/usr/bin/env python3
"""Simple WebSocket test"""

from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/")
async def root():
    return {"message": "WebSocket test server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
