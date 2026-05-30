import asyncio
import random

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket.manager import manager

router = APIRouter()


@router.websocket("/ws/stocks")
async def stocks_socket(websocket: WebSocket):
    await manager.connect(websocket, "stocks")
    try:
        while True:
            await websocket.send_json({"type": "stock_tick", "ticker": "AAPL", "price": round(210 + random.random() * 8, 2)})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "stocks")


@router.websocket("/ws/portfolio")
async def portfolio_socket(websocket: WebSocket):
    await manager.connect(websocket, "portfolio")
    try:
        while True:
            await websocket.send_json({"type": "portfolio_update", "value": round(428000 + random.random() * 2500, 2), "daily_pnl": round(random.uniform(-900, 1400), 2)})
            await asyncio.sleep(4)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "portfolio")


@router.websocket("/ws/alerts")
async def alerts_socket(websocket: WebSocket):
    await manager.connect(websocket, "alerts")
    try:
        while True:
            payload = await websocket.receive_json()
            await websocket.send_json({"type": "alert_ack", "payload": payload})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "alerts")
