import { useEffect, useRef, useState } from "react";

export interface JointState {
  arm: number[];
  head: number[];
  gripper: number;
  loads: number[] | null;
}

export interface IKState {
  target: number[] | null;
  error: number;
  iterations: number;
  converged: boolean;
}

export interface DetectionState {
  bboxes: [number, number, number, number][];
  centroids: [number, number][];
  label: string;
}

export interface BehaviorState {
  name: string;
  status: string;
}

export interface RobotTelemetry {
  stamp: number;
  joints: JointState;
  ee: number[];
  ik: IKState;
  detection: DetectionState;
  loop_hz: number;
  behavior: BehaviorState;
  torque_enabled: boolean;
}

export function useRobotState(): {
  state: RobotTelemetry | null;
  connected: boolean;
} {
  const [state, setState] = useState<RobotTelemetry | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    function connect() {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${window.location.host}/ws`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (event) => {
        try {
          const data: RobotTelemetry = JSON.parse(event.data);
          setState(data);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        reconnectTimer.current = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { state, connected };
}
