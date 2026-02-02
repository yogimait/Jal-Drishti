import base64
import cv2
import numpy as np
import asyncio
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

class VideoStreamManager:
    """
    Manages WebSocket connections and broadcasts raw video frames.
    Strictly responsible for encoding and transport, NOT scheduling.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.is_shutting_down = False
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[VideoStreamManager] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[VideoStreamManager] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast_raw_frame(self, frame: np.ndarray, frame_id: int, timestamp: float):
        """
        Encodes and broadcasts a raw frame to all connected clients.
        
        Args:
            frame: RGB numpy array
            frame_id: Integer frame counter
            timestamp: Time of capture
        """
        if not self.active_connections or self.is_shutting_down:
            return

        try:
            # 1. Encode Frame (Input is BGR from VideoReader, so no conversion needed)
            bgr_frame = frame
            
            # Encode to JPEG
            success, buffer = cv2.imencode(".jpg", bgr_frame)
            if not success:
                print("[VideoStreamManager] Failed to encode frame")
                return
            
            # Base64 Encode
            b64_image = base64.b64encode(buffer).decode("utf-8")
            
            # 2. Construct Payload
            # Schema:
            # {
            #   "type": "RAW_FRAME",
            #   "frame_id": 947,
            #   "timestamp": 1769603962.78,
            #   "image": "<base64-encoded JPEG>",
            #   "resolution": [480, 640]
            # }
            payload = {
                "type": "RAW_FRAME",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image": b64_image,
                "resolution": [frame.shape[0], frame.shape[1]]
            }
            
            # 3. Broadcast to all clients
            # We use a copy of the list to allow safe removal during iteration if needed
            for connection in self.active_connections[:]:
                try:
                    await connection.send_json(payload)
                except Exception as e:
                    # WebSocketDisconnect is a subclass of Exception; handle it specifically
                    error_msg = str(e).lower() if e else ""
                    
                    # Normal disconnects (client-side close) - silent unless verbose
                    if isinstance(e, WebSocketDisconnect) or "connection closed" in error_msg or "receive failed" in error_msg:
                        pass  # Normal client disconnect, just remove quietly
                    # Abnormal send failures - log if not shutting down
                    elif not self.is_shutting_down:
                        print(f"[VideoStreamManager] Send error: {type(e).__name__}: {e}")
                    
                    # Remove from active connections
                    self.disconnect(connection)
                    
        except Exception as e:
            print(f"[VideoStreamManager] Broadcasting error: {e}")

# Global instance
video_stream_manager = VideoStreamManager()
