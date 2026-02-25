import { useState } from "react";
import type { BehaviorState } from "../hooks/useRobotState";

const BEHAVIORS = ["detect_and_grasp"];

interface Props {
  behavior: BehaviorState | null;
}

export default function BehaviorControl({ behavior }: Props) {
  const [selected, setSelected] = useState(BEHAVIORS[0]);
  const [loading, setLoading] = useState(false);

  const isRunning = behavior?.status === "running";

  async function startBehavior() {
    setLoading(true);
    try {
      await fetch("/api/behavior/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selected, params: {} }),
      });
    } finally {
      setLoading(false);
    }
  }

  async function stopBehavior() {
    setLoading(true);
    try {
      await fetch("/api/behavior/stop", { method: "POST" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h3>Behavior</h3>
      <div className="behavior-controls">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={isRunning || loading}
        >
          {BEHAVIORS.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        {isRunning ? (
          <button onClick={stopBehavior} disabled={loading} className="btn-stop">
            Stop
          </button>
        ) : (
          <button onClick={startBehavior} disabled={loading} className="btn-start">
            Start
          </button>
        )}
      </div>
      <div className="behavior-status">
        <span className={`status-dot status-${behavior?.status ?? "idle"}`} />
        {behavior?.status === "running" && behavior.name
          ? `Running: ${behavior.name}`
          : behavior?.status === "completed"
            ? `Completed: ${behavior.name}`
            : behavior?.status === "error"
              ? `Error: ${behavior.name}`
              : "Idle"}
      </div>
    </div>
  );
}
