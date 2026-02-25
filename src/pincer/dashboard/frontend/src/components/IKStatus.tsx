import type { IKState } from "../hooks/useRobotState";

interface Props {
  ik: IKState | null;
}

function fmtVec(v: number[] | null): string {
  if (!v) return "--";
  return `(${v.map((x) => x.toFixed(3)).join(", ")})`;
}

export default function IKStatus({ ik }: Props) {
  return (
    <div className="panel">
      <h3>IK Solver</h3>
      <dl>
        <dt>Target</dt>
        <dd>{fmtVec(ik?.target ?? null)} m</dd>
        <dt>Error</dt>
        <dd>{ik ? ik.error.toFixed(4) + " m" : "--"}</dd>
        <dt>Iterations</dt>
        <dd>{ik ? ik.iterations : "--"}</dd>
        <dt>Converged</dt>
        <dd>
          {ik ? (
            <span className={ik.converged ? "ok" : "warn"}>
              {ik.converged ? "yes" : "no"}
            </span>
          ) : (
            "--"
          )}
        </dd>
      </dl>
    </div>
  );
}
