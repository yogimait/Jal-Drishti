from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.api import stream, ws_server
from app.auth import auth_router

# Core Modules
from app.video.video_reader import RawVideoSource
from app.scheduler.frame_scheduler import FrameScheduler
from app.ml.dummy_ml import DummyML
from app.services.video_stream_manager import video_stream_manager

import threading
import asyncio
import os

app = FastAPI(title="Jal-Drishti Backend", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
# app.include_router(stream.router, prefix="/ws", tags=["stream"]) # Keeping old one for now if needed?
app.include_router(ws_server.router, prefix="/ws", tags=["websocket"])

# --- RAW FEED WEBSOCKET ---
@app.websocket("/ws/raw_feed")
async def raw_feed_endpoint(websocket: WebSocket):
    await video_stream_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for any client messages (ignore them or handle control)
            data = await websocket.receive_text()
            # Optional: Heartbeat or keepalive logic
    except WebSocketDisconnect:
        video_stream_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] Error in raw_feed: {e}")
        video_stream_manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    # Capture the main event loop for the WS server to use
    loop = asyncio.get_running_loop()
    ws_server.set_event_loop(loop)
    
    # Initialize Core Pipeline
    # 1. Setup Video Source (Configurable via Env or defaults to backend/dummy.mp4)
    # The RawVideoSource class handles logic, but we can verify here.
    try:
        reader = RawVideoSource() # Uses VIDEO_SOURCE_PATH env or default
    except Exception as e:
         print(f"[Startup] Error initializing Video Source: {e}")
         return

    ml_module = DummyML()
    
    # 2. Define Callbacks
    
    # ML Result Callback (WS Server for ML data)
    def on_ml_result(payload):
        ws_server.broadcast(payload)

    # Raw Frame Callback (Stream Manager)
    # Scheduler runs in a thread, so we need to schedule the async broadcast on the main event loop
    def on_raw_frame(frame, frame_id, timestamp):
        # Fire-and-forget broadcast
        if video_stream_manager.active_connections:
            asyncio.run_coroutine_threadsafe(
                video_stream_manager.broadcast_raw_frame(frame, frame_id, timestamp), 
                loop
            )

    # 3. Initialize & Start Scheduler
    # We pass the raw_callback to decouple raw streaming
    scheduler = FrameScheduler(
        reader, 
        target_fps=15, 
        ml_module=ml_module, 
        result_callback=on_ml_result,
        raw_callback=on_raw_frame
    )
    
    # Run in background thread
    t = threading.Thread(target=scheduler.run, daemon=True)
    t.start()
    print("[Startup] Scheduler thread started.")


@app.get("/")
def read_root():
    return {"message": "Jal-Drishti Backend is running"}

# Entry point for debugging if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
