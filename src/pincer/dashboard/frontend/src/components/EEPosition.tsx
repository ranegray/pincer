interface Props {
  ee: number[] | null;
}

export default function EEPosition({ ee }: Props) {
  return (
    <div className="panel">
      <h3>End Effector</h3>
      <dl>
        <dt>X</dt>
        <dd>{ee ? ee[0].toFixed(3) + " m" : "--"}</dd>
        <dt>Y</dt>
        <dd>{ee ? ee[1].toFixed(3) + " m" : "--"}</dd>
        <dt>Z</dt>
        <dd>{ee ? ee[2].toFixed(3) + " m" : "--"}</dd>
      </dl>
    </div>
  );
}
