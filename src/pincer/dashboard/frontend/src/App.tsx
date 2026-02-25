import { useRobotState } from "./hooks/useRobotState";
import JointPanel from "./components/JointPanel";
import CameraFeed from "./components/CameraFeed";
import IKStatus from "./components/IKStatus";
import EEPosition from "./components/EEPosition";
import BehaviorControl from "./components/BehaviorControl";
import "./App.css";

export default function App() {
  const { state, connected } = useRobotState();

  return (
    <div className="dashboard">
      <header>
        <h1>Pincer</h1>
        <span className={`conn-badge ${connected ? "conn-ok" : "conn-off"}`}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      </header>
      <div className="grid">
        <div className="col-main">
          <CameraFeed />
          <BehaviorControl behavior={state?.behavior ?? null} />
        </div>
        <div className="col-side">
          <JointPanel
            joints={state?.joints ?? null}
            loopHz={state?.loop_hz ?? 0}
          />
          <EEPosition ee={state?.ee ?? null} />
          <IKStatus ik={state?.ik ?? null} />
        </div>
      </div>
    </div>
  );
}
