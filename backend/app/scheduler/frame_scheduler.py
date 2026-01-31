import time
import threading
import queue
from app.video.video_reader import RawVideoSource

class FrameScheduler:
    """
    PHASE-3 CORE REFACTORED SCHEDULER
    
    Key Changes:
    1. ML inference moved to BACKGROUND WORKER (non-blocking)
    2. RAW frames emitted immediately (decoupled from ML)
    3. Frame drops only affect ML queue, never raw stream
    4. FPS target is now 8-12 (stable) instead of 15 (unstable)
    """
    
    def __init__(self, video_source: RawVideoSource, target_fps: int = 12, 
                 ml_module=None, result_callback=None, raw_callback=None, shutdown_event=None):
        """
        Initializes the FrameScheduler with background ML worker.

        Args:
            video_source (RawVideoSource): Instance of video source.
            target_fps (int): Target frames per second. Default 12 (stable).
            ml_module (object): Optional ML module with run_inference method.
            result_callback (callable): Optional callback for ML results.
            raw_callback (callable): Optional callback for RAW frames.
            shutdown_event (threading.Event): Signal to stop gracefully.
        """
        self.video_source = video_source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.ml_module = ml_module
        self.result_callback = result_callback
        self.raw_callback = raw_callback
        self.shutdown_event = shutdown_event
        
        # BACKGROUND WORKER SETUP
        self.ml_queue = queue.Queue(maxsize=1)  # Drop frames if ML is busy
        self.ml_worker_thread = None
        self.in_safe_mode = False
        self.safe_mode_reason = None
        self.last_recovery_check = 0.0
        self.recovery_interval = 5.0

        
    def _ml_worker(self):
        """
        BACKGROUND WORKER: Processes ML inference without blocking scheduler.
        If ML is slow, frames are dropped from queue (not from raw stream).
        """
        print("[ML Worker] Background worker started")
        
        while True:
            # Check shutdown
            if self.shutdown_event and self.shutdown_event.is_set():
                print("[ML Worker] Shutdown signal received, exiting")
                break
            
            try:
                # Non-blocking get with timeout
                frame, frame_id, timestamp = self.ml_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                # EXECUTE ML INFERENCE
                ml_start = time.time()
                result = self.ml_module.run_inference(frame)
                ml_duration = time.time() - ml_start
                
                # SEND RESULT BACK
                if self.result_callback:
                    payload = result.copy()
                    payload['frame_id'] = frame_id
                    payload['system'] = {
                        'fps': self.target_fps,
                        'latency_ms': ml_duration * 1000
                    }
                    
                    self.result_callback({
                        "type": "data",
                        "status": "success",
                        "message": "ML result",
                        "payload": payload
                    })
                
                # RECOVERY CHECK
                if self.in_safe_mode and ml_duration < 0.2:
                    self.in_safe_mode = False
                    self.safe_mode_reason = None
                    print("[ML Worker] System recovered from safe mode")
                    
            except Exception as e:
                print(f"[ML Worker] Error processing frame {frame_id}: {e}")
                
                # ENTER SAFE MODE
                if not self.in_safe_mode:
                    self.in_safe_mode = True
                    self.safe_mode_reason = str(e)
                    self.last_recovery_check = time.time()
                    print(f"[Scheduler] ENTERING SAFE MODE: {e}")
                    
                    if self.result_callback:
                        self.result_callback({
                            "type": "system",
                            "status": "safe_mode",
                            "message": str(e),
                            "payload": None
                        })

    def run(self):
        """
        MAIN SCHEDULER LOOP
        
        Contract: 
        1. ALWAYS emit raw frames immediately (never block on ML)
        2. Try to send to ML queue (drop if queue full = ML is busy)
        3. Maintain stable FPS through frame drops, not delays
        """
        print(f"[Scheduler] PHASE-3 CORE: Target FPS={self.target_fps}, Interval={self.frame_interval:.4f}s")
        
        # Start background worker if ML module present
        if self.ml_module:
            self.ml_worker_thread = threading.Thread(target=self._ml_worker, daemon=True)
            self.ml_worker_thread.start()
            print("[Scheduler] Background ML worker thread started (non-blocking)")
        
        start_time = time.time()
        frames_processed = 0
        last_fps_log = start_time
        current_fps = 0.0
        
        try:
            for frame, frame_id, timestamp in self.video_source.read():
                # Check shutdown
                if self.shutdown_event and self.shutdown_event.is_set():
                    print("[Scheduler] Shutdown signal received")
                    break
                
                # Align timing on first frame
                if frame_id == 0:
                    start_time = time.time()
                
                current_time = time.time()
                
                # ========== STEP 1: EMIT RAW FRAME (DECOUPLED) ==========
                if self.raw_callback:
                    self.raw_callback(frame, frame_id, timestamp)
                
                # ========== STEP 2: CHECK IF WE SHOULD PROCESS THIS FRAME ==========
                expected_time = frame_id * self.frame_interval
                actual_elapsed = current_time - start_time
                drift = actual_elapsed - expected_time
                
                # If we are SIGNIFICANTLY behind, drop from ML (but raw continues)
                if drift > self.frame_interval:
                    print(f"[Scheduler] Frame {frame_id}: SKIPPED (Drift={drift:.4f}s) - Raw emitted, ML skipped")
                    continue
                
                # ========== STEP 3: TRY TO SEND TO ML (NON-BLOCKING) ==========
                if self.ml_module:
                    try:
                        # Non-blocking put: if queue full, frame is dropped silently
                        self.ml_queue.put_nowait((frame, frame_id, timestamp))
                        # print(f"[Scheduler] Frame {frame_id}: Queued for ML")
                    except queue.Full:
                        # ML worker is busy, drop this frame from ML (raw already emitted)
                        print(f"[Scheduler] Frame {frame_id}: ML QUEUE FULL - Dropped from processing (raw emitted)")
                
                # ========== STEP 4: MAINTAIN FPS THROUGH SLEEP ==========
                frames_processed += 1
                
                post_time = time.time()
                post_elapsed = post_time - start_time
                next_expected = (frame_id + 1) * self.frame_interval
                
                sleep_duration = next_expected - post_elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                
                # ========== STEP 5: LOG FPS ==========
                now = time.time()
                if now - last_fps_log >= 1.0:
                    current_fps = frames_processed
                    print(f"[Scheduler] FPS={frames_processed}, Drift={drift:.3f}s")
                    frames_processed = 0
                    last_fps_log = now
        
        except Exception as e:
            print(f"[Scheduler] CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if self.shutdown_event:
            self.shutdown_event.set()
        
        if self.ml_worker_thread:
            self.ml_worker_thread.join(timeout=2.0)
            print("[Scheduler] Graceful shutdown complete")


if __name__ == "__main__":
    from app.video.video_reader import RawVideoSource
    
    print("Testing PHASE-3 CORE Scheduler...")
    source = RawVideoSource()
    scheduler = FrameScheduler(source, target_fps=12) 
    scheduler.run()
