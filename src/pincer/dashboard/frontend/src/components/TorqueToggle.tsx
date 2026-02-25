import { useState } from "react";

interface Props {
  enabled: boolean;
}

export default function TorqueToggle({ enabled }: Props) {
  const [loading, setLoading] = useState(false);

  async function toggle() {
    setLoading(true);
    try {
      const endpoint = enabled ? "/api/torque/disable" : "/api/torque/enable";
      await fetch(endpoint, { method: "POST" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h3>Torque</h3>
      <div className="torque-controls">
        <span className={`status-dot ${enabled ? "status-running" : "status-idle"}`} />
        <span className="torque-label">{enabled ? "Enabled" : "Disabled"}</span>
        <button
          onClick={toggle}
          disabled={loading}
          className={enabled ? "btn-stop" : "btn-start"}
        >
          {loading ? "..." : enabled ? "Disable" : "Enable"}
        </button>
      </div>
    </div>
  );
}
