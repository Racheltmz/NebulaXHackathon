/** A side elevation of the trainset — a C151-family silhouette with cab noses on the end cars —
 * with exactly one car tinted: the one predicted to have the fault. `cars` is
 * [{ id, intensity (0 to 1), caption }]; the car with the highest intensity is flagged, the rest
 * stay plain. The ranking itself is still carried by the captions under each car. */
const CAR_W = 150;
const CAR_GAP = 10;
const HEIGHT = 150;

const BODY_Y = 30;
const BODY_H = 54;
const WIN_Y = 46;
const WIN_H = 18;
const STRIPE_Y = 76;
const WHEEL_Y = 94;
const RAIL_Y = 101;
const LABEL_Y = 122;
const CAPTION_Y = 139;

// Livery colours are literal: a real train is the same colour in every theme.
const BODY = "#f7f8f9";
const BODY_EDGE = "#c7ced4";
const GLASS = "#2b343b";
const STRIPE = "#e0431d";
const WHEEL = "#3a4247";

/** Car outline — flat, or with a raked nose on the outer end of a cab car. */
function bodyPath(cab) {
  const y = BODY_Y;
  const b = BODY_Y + BODY_H;
  const r = 9;
  const n = 30;
  if (cab === "left") {
    return `M ${n} ${y} H ${CAR_W - r} Q ${CAR_W} ${y} ${CAR_W} ${y + r} V ${b} H 8 Q 0 ${b} 0 ${b - 8} V ${y + 22} Q 2 ${y + 2} ${n} ${y} Z`;
  }
  if (cab === "right") {
    return `M ${r} ${y} H ${CAR_W - n} Q ${CAR_W - 2} ${y + 2} ${CAR_W} ${y + 22} V ${b - 8} Q ${CAR_W} ${b} ${CAR_W - 8} ${b} H 0 V ${y + r} Q 0 ${y} ${r} ${y} Z`;
  }
  return `M ${r} ${y} H ${CAR_W - r} Q ${CAR_W} ${y} ${CAR_W} ${y + r} V ${b} H 0 V ${y + r} Q 0 ${y} ${r} ${y} Z`;
}

function Windows({ cab }) {
  if (cab === "left") {
    return (
      <>
        <path d={`M 12 ${WIN_Y + 4} Q 14 ${WIN_Y} 24 ${WIN_Y} H 36 V ${WIN_Y + WIN_H} H 12 Z`} fill={GLASS} />
        <rect x={46} y={WIN_Y} width={CAR_W - 58} height={WIN_H} rx={4} fill={GLASS} />
      </>
    );
  }
  if (cab === "right") {
    return (
      <>
        <rect x={12} y={WIN_Y} width={CAR_W - 58} height={WIN_H} rx={4} fill={GLASS} />
        <path
          d={`M ${CAR_W - 36} ${WIN_Y} H ${CAR_W - 24} Q ${CAR_W - 14} ${WIN_Y} ${CAR_W - 12} ${WIN_Y + 4} V ${WIN_Y + WIN_H} H ${CAR_W - 36} Z`}
          fill={GLASS}
        />
      </>
    );
  }
  return <rect x={12} y={WIN_Y} width={CAR_W - 24} height={WIN_H} rx={4} fill={GLASS} />;
}

function Car({ car, cab, flagged, neutral }) {
  const stripeX = cab === "left" ? 6 : 0;
  const stripeW = cab ? CAR_W - 6 : CAR_W;
  return (
    <>
      <title>{`Car ${car.id}: ${car.caption}`}</title>
      <path
        d={bodyPath(cab)}
        fill={flagged ? "var(--status-abnormal)" : BODY}
        stroke={flagged ? "var(--status-abnormal)" : BODY_EDGE}
        strokeWidth={1.2}
      />
      <Windows cab={cab} />
      {/* the livery stripe drops to white on the flagged car so it stays visible on red */}
      <rect x={stripeX} y={STRIPE_Y} width={stripeW} height={4} fill={flagged ? "rgba(255,255,255,.85)" : STRIPE} />
      {[28, 44, CAR_W - 44, CAR_W - 28].map((cx) => (
        <circle key={cx} cx={cx} cy={WHEEL_Y} r={5} fill={WHEEL} />
      ))}
      <text
        x={CAR_W / 2}
        y={LABEL_Y}
        textAnchor="middle"
        style={{ fill: "var(--fe-ink)", fontSize: 13, fontWeight: 600 }}
      >
        {car.id}
      </text>
      <text
        x={CAR_W / 2}
        y={CAPTION_Y}
        textAnchor="middle"
        style={{
          fill: flagged ? "var(--status-abnormal)" : "var(--fe-pewter)",
          fontSize: 12,
          fontWeight: flagged ? 600 : 400,
        }}
      >
        {neutral ? "—" : car.caption}
      </text>
    </>
  );
}

/** Turns a `ranked_cars` string ("03|01|05|…") into this component's `cars` prop. Only the
 * top-ranked car is flagged — the rest of the ranking is carried by the captions. */
export function carsFromRanking(rankedCars) {
  return (rankedCars || "")
    .split("|")
    .filter(Boolean)
    .map((id, i) => ({ id, intensity: i === 0 ? 1 : 0, caption: `#${i + 1}` }));
}

export default function TrainDiagram({ cars, flagLabel = "Predicted faulty car" }) {
  if (!cars || cars.length === 0) return <p>No car ranking available.</p>;

  const sorted = [...cars].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
  const width = sorted.length * CAR_W + (sorted.length - 1) * CAR_GAP;
  const strongest = Math.max(...sorted.map((c) => c.intensity));
  const neutral = !(strongest > 0);

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${width} ${HEIGHT}`}
          style={{ width: "100%", minWidth: 640, display: "block" }}
          role="img"
          aria-label={`Train of ${sorted.length} cars with the car predicted to be faulty highlighted`}
        >
          <rect x={0} y={RAIL_Y} width={width} height={2} style={{ fill: "var(--fe-edge)" }} />
          {sorted.map((car, i) => {
            const cab = i === 0 ? "left" : i === sorted.length - 1 ? "right" : null;
            const flagged = !neutral && car.intensity === strongest;
            return (
              <g key={car.id} transform={`translate(${i * (CAR_W + CAR_GAP)} 0)`}>
                {i > 0 && (
                  <rect x={-CAR_GAP} y={BODY_Y + 20} width={CAR_GAP} height={16} fill={BODY_EDGE} />
                )}
                <Car car={car} cab={cab} flagged={flagged} neutral={neutral} />
              </g>
            );
          })}
        </svg>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 12, fontSize: 13, color: "var(--fe-pewter)" }}>
        <span style={{ width: 14, height: 14, borderRadius: 3, background: "var(--status-abnormal)" }} />
        <span>{flagLabel}</span>
      </div>
    </div>
  );
}
