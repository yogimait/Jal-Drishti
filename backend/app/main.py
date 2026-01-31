from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.api import stream, ws_server

# Core Modules
from app.video.video_reader import RawVideoSource
from app.scheduler.frame_scheduler import FrameScheduler
from app.ml.dummy_ml import DummyML
from app.services.video_stream_manager import video_stream_manager
from app.config_loader import config

import threading
import asyncio
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jal-Drishti Backend", version="1.0.0")

# Global references for graceful shutdown
_scheduler_thread = None
_shutdown_event = threading.Event()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
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
        logger.error(f"[WS] Error in raw_feed: {e}")
        video_stream_manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    # Load configuration
    config.print_summary()
    if not config.validate():
        logger.warning("[Startup] Configuration has validation errors")
    
    # Capture the main event loop for the WS server to use
    loop = asyncio.get_running_loop()
    ws_server.set_event_loop(loop)
    
    # Initialize ML service (ML engine handles device selection internally)
    from app.services.ml_service import MLService

    debug_mode = config.get("ml_service.debug_mode", True)
    ml_service = MLService(debug_mode=debug_mode)
    # Expose the single MLService instance via the FastAPI app state so
    # other routers (e.g. stream) can access it without importing a global.
    app.state.ml_service = ml_service
    
    video_path = config.get("video.file_path", "backend/dummy.mp4")
    if not os.path.exists(video_path):
        # Check relative to root as well
        if os.path.exists("dummy.mp4"):
            video_path = "dummy.mp4"
        else:
            logger.warning(f"[Startup] Warning: {video_path} not found.")
            return
    
    reader = RawVideoSource(video_path)

    # Callback to push to WebSocket
    def on_result(envelope):
        """
        Flatten the payload for the frontend.
        The frontend expects 'state', 'image_data', etc. at the top level.
        """
        if envelope.get("type") == "data" and "payload" in envelope:
            flat_payload = {
                "type": "data",
                **envelope["payload"]
            }
            ws_server.broadcast(flat_payload)
        else:
            ws_server.broadcast(envelope)

    # Scheduler
    # Raw frame callback: schedule the coroutine on the main loop
    def raw_frame_callback(frame, frame_id, timestamp):
        try:
            # Skip if shutting down
            if video_stream_manager.is_shutting_down:
                return
            # Use run_coroutine_threadsafe to schedule async broadcast from scheduler thread
            asyncio.run_coroutine_threadsafe(
                video_stream_manager.broadcast_raw_frame(frame, frame_id, timestamp),
                loop
            )
        except RuntimeError as e:
            # Event loop is closed or being closed; silently ignore during shutdown
            if "closed" in str(e) or "is not running" in str(e):
                pass
            else:
                logger.error(f"[Startup] Error scheduling raw frame broadcast: {e}")
        except Exception as e:
            # Silently ignore exceptions during shutdown
            if not video_stream_manager.is_shutting_down:
                logger.error(f"[Startup] Error scheduling raw frame broadcast: {e}")

    target_fps = config.get("performance.target_fps", 5)
    scheduler = FrameScheduler(reader, target_fps=target_fps, ml_module=ml_service, result_callback=on_result, raw_callback=raw_frame_callback, shutdown_event=_shutdown_event)
    
    # Run in background thread and store reference for shutdown
    global _scheduler_thread
    _scheduler_thread = threading.Thread(target=scheduler.run, daemon=False)  # daemon=False ensures thread joins on shutdown
    _scheduler_thread.start()
    logger.info("[Startup] Scheduler thread started (ML engine will initialize lazily).")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Gracefully shut down the scheduler thread to avoid asyncio errors on exit.
    """
    global _scheduler_thread, _shutdown_event
    
    # Signal the video stream manager to stop sending frames
    video_stream_manager.is_shutting_down = True
    
    # Signal the scheduler to stop immediately
    _shutdown_event.set()
    logger.info("[Shutdown] Signaled scheduler thread to stop.")
    
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.info("[Shutdown] Waiting for scheduler thread to finish...")
        # Give the thread a chance to finish (e.g., end of video loop)
        _scheduler_thread.join(timeout=5.0)
        if _scheduler_thread.is_alive():
            logger.warning("[Shutdown] Scheduler thread did not stop gracefully (timeout). Continuing shutdown.")
        else:
            logger.info("[Shutdown] Scheduler thread stopped cleanly.")



@app.get("/")
def read_root():
    return {"message": "Jal-Drishti Backend is running"}

# Entry point for debugging if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
