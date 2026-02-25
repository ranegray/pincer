import { useMemo, useState } from "react";
import type { RecordingState } from "../hooks/useRobotState";

interface Props {
  recording: RecordingState | null;
}

function tailPath(path: string | null): string {
  if (!path) return "--";
  const parts = path.split("/");
  return parts.slice(-3).join("/");
}

export default function RecordToggle({ recording }: Props) {
  const [loading, setLoading] = useState(false);

  const enabled = recording?.enabled ?? false;
  const hasError = Boolean(recording?.error);
  const episodeLabel = useMemo(() => {
    if (recording?.active_episode) return tailPath(recording.active_episode);
    if (recording?.last_episode) return tailPath(recording.last_episode);
    return "--";
  }, [recording]);

  async function toggle() {
    setLoading(true);
    try {
      const endpoint = enabled ? "/api/recording/stop" : "/api/recording/start";
      await fetch(endpoint, { method: "POST" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h3>Rerun Record</h3>
      <div className="torque-controls">
        <span className={`status-dot ${enabled ? "status-running" : "status-idle"}`} />
        <span className="torque-label">{enabled ? "Recording" : "Idle"}</span>
        <button
          onClick={toggle}
          disabled={loading}
          className={enabled ? "btn-stop" : "btn-start"}
        >
          {loading ? "..." : enabled ? "Stop" : "Record"}
        </button>
      </div>
      <div className="record-meta">
        <div>Run: {tailPath(recording?.run_dir ?? null)}</div>
        <div>Episode: {episodeLabel}</div>
        <div>Count: {recording?.episode_count ?? 0}</div>
        {hasError ? <div className="warn">Error: {recording?.error}</div> : null}
      </div>
    </div>
  );
}
