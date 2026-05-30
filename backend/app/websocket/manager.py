from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.channels: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        self.channels[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        if websocket in self.channels[channel]:
            self.channels[channel].remove(websocket)

    async def broadcast(self, channel: str, message: dict) -> None:
        for websocket in list(self.channels[channel]):
            try:
                await websocket.send_json(message)
            except RuntimeError:
                self.disconnect(websocket, channel)


manager = ConnectionManager()
