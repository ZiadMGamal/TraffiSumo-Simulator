import { useCallback, useEffect, useRef, useState } from "react";

export function useWebSocket(url, options = {}) {
  const { reconnectInterval = 3000, maxRetries = 10 } = options;
  const [data, setData] = useState([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const retries = useRef(0);
  const wsRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => {
      setConnected(true);
      setError(null);
      retries.current = 0;
    };
    ws.onclose = () => {
      setConnected(false);
      if (retries.current < maxRetries) {
        retries.current += 1;
        setTimeout(connect, reconnectInterval);
      }
    };
    ws.onerror = () => setError("WebSocket connection failed");
    ws.onmessage = (event) => {
      try {
        setData(JSON.parse(event.data));
      } catch {
        setData([]);
      }
    };
  }, [url, reconnectInterval, maxRetries]);

  useEffect(() => {
    connect();
    return () => {
      retries.current = maxRetries;
      wsRef.current?.close();
    };
  }, [connect, maxRetries]);

  return { data, connected, error, reconnect: connect };
}
