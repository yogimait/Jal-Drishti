import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_CONFIG, RECONNECT_CONFIG } from '../constants';

const useRawFeed = () => {
    const [status, setStatus] = useState('DISCONNECTED');
    const [resolution, setResolution] = useState(null);
    const [frameSrc, setFrameSrc] = useState(null);
    const [reconnectAttempt, setReconnectAttempt] = useState(0);

    const wsRef = useRef(null);
    const lastFrameIdRef = useRef(-1);
    const reconnectTimeoutRef = useRef(null);
    const connectRef = useRef(null);

    const getReconnectDelay = useCallback((attempt) => {
        return Math.min(
            RECONNECT_CONFIG.BASE_DELAY_MS * (2 ** attempt),
            RECONNECT_CONFIG.MAX_DELAY_MS
        );
    }, []);

    const connect = useCallback(() => {
        // Construct WS URL - explicitly triggering the raw_feed endpoint
        // Assuming WS_CONFIG.URL is base like "ws://localhost:8000/ws" 
        // We need to parse/modify it to "ws://localhost:8000/ws/raw_feed"

        // Parse the Configured URL
        // WS_CONFIG.URL is typically "ws://localhost:9000/ws/stream"
        // We need "ws://localhost:9000/ws/raw_feed"

        try {
            // Use URL object for robust parsing
            // Note: 'ws' protocol might need 'http' for URL constructor if environment is weird, 
            // but standard browsers handle ws/wss in URL object nowadays.
            // If not, we can treat it as string.

            const rawConfigUrl = WS_CONFIG.URL;
            let finalUrl;

            if (rawConfigUrl.includes('/ws/stream')) {
                finalUrl = rawConfigUrl.replace('/ws/stream', '/ws/raw_feed');
            } else if (rawConfigUrl.endsWith('/ws')) {
                finalUrl = `${rawConfigUrl}/raw_feed`;
            } else {
                // Fallback: URL might be just host
                const wsUrl = rawConfigUrl.endsWith('/') ? rawConfigUrl.slice(0, -1) : rawConfigUrl;
                finalUrl = `${wsUrl}/raw_feed`;
            }

            console.log(`[RawFeed] Connecting to ${finalUrl}`);
            const ws = new WebSocket(finalUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[RawFeed] Connected');
                setStatus('CONNECTED');
                setReconnectAttempt(0);
                lastFrameIdRef.current = -1;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'RAW_FRAME') {
                        // Out-of-order check
                        if (data.frame_id <= lastFrameIdRef.current) {
                            return; // Drop old frame
                        }
                        lastFrameIdRef.current = data.frame_id;

                        if (data.resolution) {
                            setResolution(data.resolution);
                        }

                        // Decode base64 
                        const src = `data:image/jpeg;base64,${data.image}`;
                        setFrameSrc(src);

                        // If we were waiting for signal, update status
                        setStatus('STREAMING');
                    }
                } catch (err) {
                    console.error('[RawFeed] Parse error', err);
                }
            };

            ws.onclose = () => {
                console.log('[RawFeed] Disconnected');
                setStatus('DISCONNECTED');
                setFrameSrc(null);
                
                // Attempt reconnection
                setReconnectAttempt((prev) => {
                    const nextAttempt = prev + 1;
                    const delay = getReconnectDelay(nextAttempt);
                    console.log(`[RawFeed] Reconnecting in ${delay}ms (attempt ${nextAttempt})`);
                    
                    reconnectTimeoutRef.current = setTimeout(() => {
                        if (connectRef.current) {
                            connectRef.current();
                        }
                    }, delay);
                    
                    return nextAttempt;
                });
            };

            ws.onerror = (err) => {
                console.error('[RawFeed] Error', err);
                setStatus('ERROR');
            };

        } catch (e) {
            console.error("[RawFeed] URL Construction Error", e);
            setStatus('ERROR');
        }
    }, [getReconnectDelay]);

    useEffect(() => {
        connectRef.current = connect;
    }, [connect]);

    useEffect(() => {
        connect();
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
        };
    }, [connect]);

    return {
        status,
        frameSrc,
        resolution
    };
}
    export default useRawFeed;
