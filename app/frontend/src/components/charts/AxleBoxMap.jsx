/** Plan view of the trainset's 64 axle boxes — 8 cars × 8 positions, two bogies per car, laid out
 * as the info kit's Figure 2: positions 1, 3, 5, 7 ride the Side I rail (drawn along the top) and
 * 2, 4, 6, 8 the Side II rail (along the bottom).
 *
 * Each box is shaded by how strongly it shakes in the corrugation band, so a rail with corrugation
 * shows up as one side of the train reading darker than the other. Shade is magnitude, so it's one
 * hue light→dark (never a rainbow), and every value is also reachable from the tooltip, the table
 * view in the chart below, and the per-side figures beside the map. */

const CAR_W = 132;
const CAR_GAP = 9;
const BOX_W = 20;
const BOX_H = 15;
const TOP_Y = 30; // Side I boxes sit on this line
const BODY_Y = 56;
const BODY_H = 40;
const BOTTOM_Y = BODY_Y + BODY_H + 11; // Side II boxes
const HEIGHT = BOTTOM_Y + BOX_H + 40;

// Where each position sits along the car: two bogies, two boxes per rail on each.
const POSITION_X = { 1: 0.19, 3: 0.35, 5: 0.65, 7: 0.81, 2: 0.19, 4: 0.35, 6: 0.65, 8: 0.81 };

// The reference palette's blue ramp, steps 100→700: magnitude reads as one hue, light to dark. This
// is sequential (continuous magnitude over a grid), so the lightest step is allowed to recede
// toward the surface — a quiet box should look quiet. Each box also carries its printed position
// number, so a pale box is still locatable, and its value is in the tooltip and the table below.
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const EMPTY = "#eceae4";

// One unusually loud box (a bearing, a rail joint) would otherwise take the whole top of the scale
// and leave every other box pale — hiding the side-wide pattern the map exists to show. So the
// ramp tops out at the 90th percentile and the loudest boxes share the darkest step.
const SCALE_PERCENTILE = 0.9;

function scaleTop(values) {
  const sorted = values.filter((v) => v != null).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * SCALE_PERCENTILE))];
}

function shadeOf(value, top) {
  if (value == null || !(top > 0)) return EMPTY;
  const step = Math.min(RAMP.length - 1, Math.floor((value / top) * RAMP.length));
  return RAMP[step];
}

// A box's shade is only readable against dark ink at the light end of the ramp.
const inkOn = (shade) => (RAMP.indexOf(shade) >= 4 ? "#ffffff" : "#0b0b0b");

export default function AxleBoxMap({ telemetry, selected, onSelect, metricLabel = "Corrugation band vibration" }) {
  const boxes = telemetry.boxes ?? {};
  const cars = [...new Set(Object.values(boxes).map((b) => b.car))].sort((a, b) => a - b);
  const positions = [...new Set(Object.values(boxes).map((b) => b.position))].sort((a, b) => a - b);
  const topPositions = positions.filter((p) => p % 2 === 1);
  const bottomPositions = positions.filter((p) => p % 2 === 0);

  const values = Object.values(boxes).map((b) => b.peak_amplitude ?? null);
  const top = scaleTop(values);
  const width = cars.length * CAR_W + (cars.length - 1) * CAR_GAP;

  const boxAt = (car, position) => boxes[`${car}-${position}`];

  const renderBox = (car, position, y) => {
    const box = boxAt(car, position);
    if (!box) return null;
    const x = POSITION_X[position] * CAR_W - BOX_W / 2;
    const shade = shadeOf(box.peak_amplitude, top);
    const id = `${car}-${position}`;
    const isSelected = id === selected;
    return (
      <g
        key={id}
        className="axle-box"
        transform={`translate(${x} ${y})`}
        role="button"
        tabIndex={0}
        aria-label={`Car ${car}, position ${position}, Side ${box.side}. ${metricLabel} ${box.peak_amplitude ?? "no reading"}`}
        aria-pressed={isSelected}
        onClick={() => onSelect?.(id)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onSelect?.(id);
          }
        }}
      >
        <title>
          {`Car ${car}, position ${position} (Side ${box.side})\n${metricLabel}: ${box.peak_amplitude ?? "n/a"}`}
          {box.peak_frequency ? ` at ${box.peak_frequency} Hz` : ""}
          {box.rms_vibration != null ? `\nOverall vibration RMS: ${box.rms_vibration} m/s²` : ""}
        </title>
        <rect width={BOX_W} height={BOX_H} rx={3} fill={shade} />
        {isSelected && <rect className="axle-box-ring" width={BOX_W} height={BOX_H} rx={3} />}
        <text x={BOX_W / 2} y={BOX_H / 2 + 3.5} textAnchor="middle" style={{ fill: inkOn(shade), fontSize: 9.5 }}>
          {position}
        </text>
      </g>
    );
  };

  return (
    <div className="axle-box-map">
      <div className="axle-box-scroll">
        <svg
          viewBox={`0 0 ${width} ${HEIGHT}`}
          style={{ width: "100%", minWidth: 700, display: "block" }}
          role="group"
          aria-label="Plan view of the train's 64 axle boxes, shaded by corrugation band vibration"
        >
          {/* the two rails the boxes ride on */}
          <rect className="axle-rail" x={0} y={TOP_Y - 7} width={width} height={2.5} />
          <rect className="axle-rail" x={0} y={BOTTOM_Y + BOX_H + 5} width={width} height={2.5} />
          <text className="axle-rail-label" x={0} y={TOP_Y - 13}>
            Side I rail, positions {topPositions.join(", ")}
          </text>
          <text className="axle-rail-label" x={0} y={BOTTOM_Y + BOX_H + 22}>
            Side II rail, positions {bottomPositions.join(", ")}
          </text>

          {cars.map((car, i) => (
            <g key={car} transform={`translate(${i * (CAR_W + CAR_GAP)} 0)`}>
              <rect className="axle-car-body" x={0} y={BODY_Y} width={CAR_W} height={BODY_H} rx={6} />
              <text className="axle-car-label" x={CAR_W / 2} y={BODY_Y + BODY_H / 2 + 4} textAnchor="middle">
                Car {car}
              </text>
              {topPositions.map((p) => renderBox(car, p, TOP_Y))}
              {bottomPositions.map((p) => renderBox(car, p, BOTTOM_Y))}
            </g>
          ))}
        </svg>
      </div>

      <div className="axle-legend">
        <span className="axle-legend-title">{metricLabel}</span>
        <span className="axle-legend-end">0</span>
        {RAMP.map((shade) => (
          <span key={shade} className="axle-legend-swatch" style={{ background: shade }} />
        ))}
        <span className="axle-legend-end">{top ? `≥ ${top.toFixed(3)}` : "n/a"}</span>
        <span className="axle-legend-hint">Select a box to plot it below.</span>
      </div>
    </div>
  );
}
