import type { JointState } from "../hooks/useRobotState";

const ARM_NAMES = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
];

const HEAD_NAMES = ["head_pan", "head_tilt"];

interface Props {
  joints: JointState | null;
  loopHz: number;
}

export default function JointPanel({ joints, loopHz }: Props) {
  return (
    <div className="panel">
      <h3>
        Joints{" "}
        <span className="hz">{loopHz > 0 ? `${loopHz.toFixed(1)} Hz` : "--"}</span>
      </h3>
      <table>
        <thead>
          <tr>
            <th>Joint</th>
            <th>Position</th>
            <th>Load</th>
          </tr>
        </thead>
        <tbody>
          {ARM_NAMES.map((name, i) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{joints ? joints.arm[i].toFixed(1) + "\u00b0" : "--"}</td>
              <td>{joints?.loads ? joints.loads[i].toFixed(0) : "--"}</td>
            </tr>
          ))}
          {HEAD_NAMES.map((name, i) => (
            <tr key={name} className="head-row">
              <td>{name}</td>
              <td>{joints ? joints.head[i].toFixed(1) + "\u00b0" : "--"}</td>
              <td>--</td>
            </tr>
          ))}
          <tr className="gripper-row">
            <td>gripper</td>
            <td>{joints ? joints.gripper.toFixed(1) + "\u00b0" : "--"}</td>
            <td>--</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
