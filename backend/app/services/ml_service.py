import base64
import numpy as np
import cv2
import sys
import os
import logging

# Ensure backend can import ml-engine core
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
ml_engine_path = os.path.join(root_dir, "ml-engine")
if ml_engine_path not in sys.path:
    sys.path.append(ml_engine_path)


logger = logging.getLogger(__name__)

class MLService:
    """
    PHASE-3 CORE REFACTORED ML SERVICE
    
    Key Changes:
    1. STOPS sending Base64 enhanced images on every frame (huge bandwidth/CPU load)
    2. Only sends detections, confidence, and metadata
    3. Enhanced image sent ONLY on anomaly detection or debug toggle
    4. Reduces frontend processing and network overhead
    """
    
    def __init__(self, debug_mode=True):
        """
        Initialize ML Service.

        Note: Backend must never attempt to detect or use GPU. The actual
        device selection and any `torch` usage lives inside the ml-engine
        (JalDrishtiEngine). The engine is lazily initialized on first use
        to avoid importing torch at backend import time.

        Args:
            debug_mode (bool): If True, send enhanced images for debugging
        """
        self.debug_mode = debug_mode
        self.engine = None
        self.frame_count = 0

    def _ensure_engine(self):
        """Lazily import and initialize the ML engine inside the ML service.

        This keeps all torch imports and device detection inside the
        ml-engine module (core.pipeline)."""
        if self.engine is not None:
            return

        try:
            # Import the engine lazily so backend modules do not import torch
            from core.pipeline import JalDrishtiEngine
            # Let the engine decide device/FP16 internally based on its own config
            self.engine = JalDrishtiEngine()
            logger.info("[ML Service] Engine initialized (lazy)")
        except Exception as e:
            logger.error("[ML Service] CRITICAL ERROR: Engine failed to start: %s", e, exc_info=True)
            self.engine = None

    def run_inference(self, frame: np.ndarray, send_enhanced: bool = False) -> dict:
        """
        Processes a raw BGR frame from the scheduler.
        
        PHASE-3 CORE behavior:
        - Always returns detections and metadata
        - Only includes enhanced frame if send_enhanced=True
        
        Args:
            frame: Input BGR frame
            send_enhanced: Whether to include enhanced frame (send on anomaly or debug)
        
        Returns:
            dict with detections, confidence, state, optionally image_data
        """
        # Ensure engine is available and lazily initialize if necessary.
        if self.engine is None:
            self._ensure_engine()

        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        result_json, enhanced_frame = self.engine.infer(frame)
        
        # PHASE-3 CORE: Base response WITHOUT image (always)
        response = {
            "detections": result_json.get('detections', []),
            "max_confidence": result_json.get('max_confidence', 0.0),
            "state": result_json.get('state', 'NORMAL'),
            "latency_ms": result_json.get('latency_ms', 0.0),
        }
        
        # OPTIONAL: Send enhanced image only on anomaly or debug
        if send_enhanced or self.debug_mode:
            _, buffer = cv2.imencode('.jpg', enhanced_frame)
            b64_image = base64.b64encode(buffer).decode('utf-8')
            response['image_data'] = b64_image
        
        return response

    def process_frame(self, binary_frame: bytes, send_enhanced: bool = False) -> dict:
        """
        Real ML Processing for individual frames (e.g. from WebSocket or HTTP).
        
        Args:
            binary_frame: Raw frame bytes
            send_enhanced: Whether to include enhanced frame
        """
        self.frame_count += 1
        
        try:
            # Decode image
            nparr = np.frombuffer(binary_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return self._error_response("Image decode failed")
            
            # Run Inference (optionally with enhanced frame)
            result = self.run_inference(frame, send_enhanced=send_enhanced or self.debug_mode)
            
            return {
                "status": "success",
                "frame_id": self.frame_count,
                **result
            }

        except Exception as e:
            logger.error("[ML Service] Processing Error: %s", e, exc_info=True)
            return self._error_response(str(e))

    def _error_response(self, msg):
        """Return minimal error response without heavy image data"""
        return {
            "status": "error",
            "message": msg,
            "state": "SAFE_MODE",
            "frame_id": self.frame_count,
            "detections": [],
            "max_confidence": 0.0,
        }

# Do not instantiate MLService at module import time: the application
# should create the service from `app.main` so initialization and
# configuration are explicit and backend does not import torch unless
# the ML engine is required.
