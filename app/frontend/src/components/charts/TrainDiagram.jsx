/** A row of train carriages, each shaded by how strongly it's flagged. `cars` is
 * [{ id, intensity (0 to 1), caption }] — a darker carriage is the one more likely to be faulty. */
const CAR_WIDTH = 112;
const CAR_GAP = 14;
const TOP_INTENSITY_FLOOR = 0.1; // even the least likely car keeps a faint tint so it reads as a carriage

export default function TrainDiagram({ cars, lowLabel = "Less likely faulty", highLabel = "More likely faulty" }) {
  if (!cars || cars.length === 0) return <p>No car ranking available.</p>;

  const sorted = [...cars].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
  const width = sorted.length * CAR_WIDTH + (sorted.length - 1) * CAR_GAP;
  const strongest = Math.max(...sorted.map((c) => c.intensity));

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${width} 108`}
          style={{ width: "100%", minWidth: 640, display: "block" }}
          role="img"
          aria-label={`Train of ${sorted.length} carriages shaded by how likely each is to be faulty`}
        >
          <rect x={0} y={73} width={width} height={2} style={{ fill: "var(--fe-edge)" }} />
          {sorted.map((car, i) => {
            const opacity = TOP_INTENSITY_FLOOR + (1 - TOP_INTENSITY_FLOOR) * car.intensity;
            const isStrongest = car.intensity === strongest && strongest > 0;
            return (
              <g key={car.id} transform={`translate(${i * (CAR_WIDTH + CAR_GAP)} 0)`}>
                <title>{`Car ${car.id}: ${car.caption}`}</title>
                {i > 0 && <rect x={-CAR_GAP} y={38} width={CAR_GAP} height={6} style={{ fill: "var(--fe-pewter)" }} />}
                <rect
                  x={0}
                  y={8}
                  width={CAR_WIDTH}
                  height={52}
                  rx={12}
                  style={{ fill: "var(--status-abnormal)", fillOpacity: opacity, stroke: "var(--fe-edge)" }}
                />
                {[0, 1, 2, 3].map((w) => (
                  <rect
                    key={w}
                    x={11 + w * 24}
                    y={18}
                    width={18}
                    height={16}
                    rx={4}
                    style={{ fill: "#fff", fillOpacity: 0.9 }}
                  />
                ))}
                <text
                  x={CAR_WIDTH / 2}
                  y={52}
                  textAnchor="middle"
                  style={{ fill: opacity > 0.55 ? "#fff" : "var(--fe-ink)", fontSize: 13, fontWeight: 600 }}
                >
                  {car.id}
                </text>
                <circle cx={24} cy={68} r={6} style={{ fill: "var(--fe-ink)" }} />
                <circle cx={88} cy={68} r={6} style={{ fill: "var(--fe-ink)" }} />
                <text
                  x={CAR_WIDTH / 2}
                  y={98}
                  textAnchor="middle"
                  style={{
                    fill: isStrongest ? "var(--fe-ink)" : "var(--fe-pewter)",
                    fontSize: 12,
                    fontWeight: isStrongest ? 600 : 400,
                  }}
                >
                  {car.caption}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 12, fontSize: 13, color: "var(--fe-pewter)" }}>
        <span>{lowLabel}</span>
        <span
          style={{
            width: 120,
            height: 10,
            borderRadius: 5,
            background: "linear-gradient(to right, color-mix(in srgb, var(--status-abnormal) 10%, white), var(--status-abnormal))",
          }}
        />
        <span>{highLabel}</span>
      </div>
    </div>
  );
}
