from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
import os
import asyncio
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Dexter Gliksbot Cerebral API")

# Enable CORS for the Next.js dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _db_path() -> Path:
    env_path = os.getenv("DEXTER_DB_PATH")
    if env_path:
        return Path(env_path)
    # Resolve relative to repo root
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent
    config_path = repo_root / "configs" / "core_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                db_path = cfg.get("database_path", "brain.db")
                path = Path(db_path)
                if not path.is_absolute():
                    path = repo_root / path
                return path
        except Exception:
            pass
    return repo_root / "brain.db"

DB_PATH = _db_path()

class QueryRequest(BaseModel):
    question: str

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/status")
def get_status():
    return {"status": "online", "identity": "Dexter Gliksbot", "user": "Jeffrey Gliksman"}

@app.get("/triples")
def get_triples(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT subject, predicate, object, ts FROM triples ORDER BY ts DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"subject": r[0], "predicate": r[1], "object": r[2], "ts": r[3]} for r in rows]

@app.post("/ask")
async def ask_dexter(req: QueryRequest):
    from skills.graph_query import async_ask_graph
    result = await async_ask_graph(req.question)
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep connection alive, we primarily broadcast OUT
            data = await websocket.receive_text()
            # If dashboard sends something, we could process it here
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Global helper to send data to the dashboard from anywhere in the process
def broadcast_thought(thought_type: str, content: Any):
    """
    Helper to send thoughts/logs to the dashboard.
    NOTE: Must be called within the same event loop or via threadsafe.
    """
    message = json.dumps({"type": thought_type, "content": content})
    asyncio.create_task(manager.broadcast(message))

def start_api_server():
    import uvicorn
    # --- PORT JANITOR ---
    try:
        import psutil
        for conn in psutil.net_connections():
            if conn.laddr.port == 8000:
                print(f"[API] Port 8000 in use by PID {conn.pid}. Clearing...", flush=True)
                try:
                    p = psutil.Process(conn.pid)
                    if p.name().lower() == "python.exe" or "python" in p.name().lower():
                        p.terminate()
                except:
                    pass
    except:
        pass

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

if __name__ == "__main__":
    start_api_server()
