import asyncio
import websockets
import json
from src.config import CHROME_PORT

class CDPClient:
    def __init__(self):
        self.ws_url = f"ws://localhost:{CHROME_PORT}/devtools/browser"
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(self.ws_url)
        print("Connected to Chrome CDP successfully!")

    async def send_command(self, method, params=None):
        if params is None:
            params = {}
        payload = {
            "id": 1,
            "method": method,
            "params": params
        }
        await self.websocket.send(json.dumps(payload))
        response = await self.websocket.recv()
        return json.loads(response)

    async def enable_headless_renderer(self):
        await self.send_command("HeadlessExperimental.enable")