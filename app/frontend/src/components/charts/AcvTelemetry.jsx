import { useEffect, useMemo, useRef, useState } from "react";

/** Per-car ACV telemetry for one run: temperatures as lines, and the modes/status flags as state
 * strips, on one shared time axis with one crosshair. Reads `telemetry` as built by the backend's
 * ml/acv_telemetry.py — a fixed number of equal time bins per series, null where a bin has no
 * valid reading (so gaps in the recording stay gaps). */

const GUTTER = 116; // left: temperature ticks and the strip labels
const PAD_RIGHT = 16;
const TEMP_TOP = 8;
const PANEL_GAP = 26; // between the temperature panels
const SECTION_GAP = 28; // between the temperatures and the strips
const STRIP_H = 18;
const STRIP_GAP = 8;
const AXIS_H = 30;
const MIN_WIDTH = 360;

// Series colours are the reference palette's categorical slots 1–4 in order, checked with the
// dataviz validate_palette script (adjacent pairs pass; aqua and yellow sit under 3:1 on the light
// surface, which the legend, tooltip and table view relieve).
const TEMP_SERIES = [
  { key: "indoor", label: "Indoor average", color: "#2a78d6" },
  { key: "outdoor", label: "Outdoor", color: "#eb6834" },
  { key: "control_cooling", label: "Control (cooling)", color: "#1baf7a" },
  { key: "control_heating", label: "Control (heating)", color: "#eda100" },
];

// Outdoor gets its own short panel: sharing an axis with the cabin lines stretches it to the outdoor
// range and squashes the gap between indoor temperature and setpoint, which is the part to read.
// Panels share the time axis but each has its own scale — not a dual-axis plot.
const TEMP_PANELS = [
  { key: "cabin", title: "Cabin (°C)", height: 170, signals: ["indoor", "control_cooling", "control_heating"] },
  { key: "outdoor", title: "Outdoor (°C)", height: 80, signals: ["outdoor"] },
];

const STATE_ROWS = [
  { key: "setting_mode", label: "Setting mode" },
  { key: "running_mode", label: "Running mode" },
  { key: "load_halved", label: "Load halved" },
  { key: "info_valid", label: "Info valid" },
];

// The parameter each signal has in the model A/C files. Anything else was matched by the backend
// from a differently named column (model B, model C's outdoor sensor) and is called out below the chart.
const CANONICAL_SOURCE = {
  indoor: "Indoor Average Temperature",
  outdoor: "Outdoor Average Temperature",
  control_cooling: "ACV Control Temperature (Cooling)",
  control_heating: "ACV Control Temperature (Heating)",
  setting_mode: "ACV Setting Mode",
  running_mode: "ACV Running Mode",
  load_halved: "Load Halved",
  info_valid: "ACV Information Valid",
};
const SIGNAL_LABEL = Object.fromEntries([...TEMP_SERIES, ...STATE_ROWS].map((s) => [s.key, s.label]));

// State colours, keyed by the value so a state looks the same in every strip and on every car.
// Default/quiet states share one neutral; cooling levels are one blue ramp, light to dark
// (half → automatic → full); ventilation, emergency and manual take categorical hues that were
// validated all-pairs. "Invalid" is a dark neutral, not red: red sits too close to the orange for
// Emergency Ventilation, and the two can be neighbours in the running-mode strip.
const NEUTRAL = "#c3c2b7";
const STATE_COLORS = {
  "Centralized Control": NEUTRAL,
  Stop: NEUTRAL,
  Stopped: NEUTRAL,
  Normal: NEUTRAL,
  Valid: NEUTRAL,
  "Manual Control": "#4a3aa7",
  "Half Cooling": "#86b6ef",
  "Automatic Cooling": "#3987e5",
  "Full Cooling": "#184f95",
  Ventilation: "#1baf7a",
  "Emergency Ventilation": "#eb6834",
  "Self-Check": "#e87ba4",
  Invalid: "#52514e",
};
const LOAD_ABNORMAL = "#fab219"; // a "Load halved" value other than Normal: reduced capacity
const UNKNOWN_STATE = "#898781";
const QUIET_STATES = new Set(["Normal", "Valid"]); // not worth a label on every stretch

function stateColor(signal, name) {
  if (STATE_COLORS[name]) return STATE_COLORS[name];
  return signal === "load_halved" ? LOAD_ABNORMAL : UNKNOWN_STATE;
}

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// White or ink, whichever has more contrast on the fill — for labels set inside a segment.
function labelColor(fill) {
  const l = luminance(fill);
  return (1.05) / (l + 0.05) > (l + 0.05) / (luminance("#0b0b0b") + 0.05) ? "#ffffff" : "#0b0b0b";
}

function niceScale(min, max) {
  const lo0 = Math.floor(min - 0.5);
  const hi0 = Math.ceil(max + 0.5);
  const range = hi0 - lo0;
  const step = range <= 6 ? 1 : range <= 12 ? 2 : range <= 30 ? 5 : 10;
  const lo = Math.floor(lo0 / step) * step;
  const hi = Math.ceil(hi0 / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(v);
  return { lo, hi, ticks };
}

// Path for one line, broken wherever a bin has no reading. An isolated bin gets a zero-length
// segment so the round cap still draws it as a dot.
function linePath(values, xOf, yOf) {
  let d = "";
  let pen = false;
  values.forEach((v, i) => {
    if (v == null) {
      pen = false;
      return;
    }
    const point = `${xOf(i).toFixed(1)} ${yOf(v).toFixed(1)}`;
    if (pen) {
      d += `L${point}`;
    } else {
      d += `M${point}`;
      if (values[i + 1] == null) d += "h0";
    }
    pen = true;
  });
  return d;
}

// Consecutive bins with the same state, as [firstBin, lastBin, index] runs.
function stateRuns(values) {
  const runs = [];
  values.forEach((v, i) => {
    if (v == null) return;
    const last = runs[runs.length - 1];
    if (last && last[2] === v && last[1] === i - 1) last[1] = i;
    else runs.push([i, i, v]);
  });
  return runs;
}

const TIME_FORMAT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC", // the files' timestamps carry no zone; show them as recorded
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const formatTime = (seconds) => TIME_FORMAT.format(new Date(seconds * 1000));

export default function AcvTelemetry({ telemetry, rankedCars }) {
  const carIds = useMemo(() => {
    const ranked = (rankedCars || "").split("|").filter(Boolean);
    return ranked.length > 0 ? ranked : Object.keys(telemetry.cars);
  }, [rankedCars, telemetry]);

  const hasReadings = (id) => Object.keys(telemetry.cars[id] ?? {}).length > 0;
  const [carId, setCarId] = useState(carIds.find(hasReadings) ?? carIds[0]);
  const [hover, setHover] = useState(null); // bin index under the pointer / keyboard focus
  const [tableOpen, setTableOpen] = useState(false);
  const [width, setWidth] = useState(720);
  const wrapRef = useRef(null);

  const bins = telemetry.t.length;
  const data = telemetry.cars[carId] ?? {};
  const tempSeries = TEMP_SERIES.filter((s) => data[s.key]);
  const stateRows = STATE_ROWS.filter((r) => data[r.key]);
  const hasData = tempSeries.length > 0 || stateRows.length > 0;

  // The measured element only exists for a car that has data, so re-attach when that changes.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const update = () => setWidth(Math.max(el.clientWidth, MIN_WIDTH));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasData]);

  const plotW = width - GUTTER - PAD_RIGHT;
  const xOf = (i) => GUTTER + ((i + 0.5) / bins) * plotW;

  let cursor = TEMP_TOP;
  const panels = [];
  TEMP_PANELS.forEach((panel) => {
    const series = tempSeries.filter((s) => panel.signals.includes(s.key));
    if (series.length === 0) return;
    const values = series.flatMap((s) => data[s.key].filter((v) => v != null));
    const scale = niceScale(Math.min(...values), Math.max(...values));
    const top = cursor;
    panels.push({
      ...panel,
      series,
      scale,
      top,
      yOf: (v) => top + (1 - (v - scale.lo) / (scale.hi - scale.lo)) * panel.height,
    });
    cursor += panel.height + PANEL_GAP;
  });
  const tempBottom = panels.length > 0 ? cursor - PANEL_GAP : TEMP_TOP;

  const stripsTop = panels.length > 0 ? tempBottom + SECTION_GAP : TEMP_TOP;
  const stripsBottom = stripsTop + stateRows.length * (STRIP_H + STRIP_GAP) - (stateRows.length ? STRIP_GAP : 0);
  const plotBottom = stateRows.length > 0 ? stripsBottom : tempBottom;
  const height = plotBottom + AXIS_H;

  const firstT = telemetry.t[0];
  const lastT = telemetry.t[bins - 1];
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => ({
    x: xOf(f * (bins - 1)),
    label: formatTime(firstT + f * (lastT - firstT)),
    anchor: f === 0 ? "start" : f === 1 ? "end" : "middle",
  }));

  const stateName = (signal, index) => (index == null ? null : telemetry.categories[signal][index]);
  const legendStates = [];
  stateRows.forEach(({ key }) => {
    new Set(data[key].filter((v) => v != null)).forEach((index) => {
      const name = stateName(key, index);
      if (!legendStates.some((s) => s.name === name)) legendStates.push({ key, name });
    });
  });

  const binFromPointer = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / rect.width;
    return Math.min(bins - 1, Math.max(0, Math.floor(fraction * bins)));
  };

  const onKeyDown = (event) => {
    if (event.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? bins) - 1));
    else if (event.key === "ArrowRight") setHover((h) => Math.min(bins - 1, (h ?? -1) + 1));
    else if (event.key === "Escape") setHover(null);
    else return;
    event.preventDefault();
  };

  const selectCar = (id) => {
    setCarId(id);
    setHover(null);
  };

  // What to tell the reader about signals this car (or this file) doesn't plot.
  const notRecorded = Object.keys(CANONICAL_SOURCE).filter((s) => !telemetry.sources[s]);
  const noReadings = Object.keys(telemetry.sources).filter((s) => !data[s]);
  const renamed = Object.entries(telemetry.sources).filter(
    ([signal, source]) => data[signal] && source !== CANONICAL_SOURCE[signal],
  );
  const tooltipOnRight = hover !== null && xOf(hover) < width * 0.6;

  return (
    <section className="acv-telemetry">
      <h3 className="acv-telemetry-title">Car telemetry</h3>
      <p className="history-dashboard-note">
        Pick a car to see its temperatures and modes over the recording. Cars are listed from most to
        least likely faulty, so the first tab is the one to check first.
      </p>

      <div className="history-filters acv-car-tabs" role="tablist" aria-label="Car">
        {carIds.map((id, i) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={id === carId}
            className={`${id === carId ? "active" : ""} ${hasReadings(id) ? "" : "no-data"}`}
            title={hasReadings(id) ? undefined : "No readings for this car in this file"}
            onClick={() => selectCar(id)}
          >
            Car {id} (#{i + 1})
          </button>
        ))}
      </div>

      <div className="chart-card">
        {!hasData ? (
          <p>
            Car {carId} has no readings in this file
            {telemetry.sources.running_mode ? " (its columns are empty)" : ""}.
          </p>
        ) : (
          <>
            {tempSeries.length > 0 && (
              <div className="viz-legend">
                <strong>Temperature</strong>
                {tempSeries.map((s) => (
                  <span key={s.key}>
                    <span className="viz-key-line" style={{ background: s.color }} />
                    {s.label}
                  </span>
                ))}
              </div>
            )}

            <div className="viz-wrap" ref={wrapRef}>
              <svg
                width={width}
                height={height}
                role="group"
                aria-label={`Car ${carId} temperatures and modes over time. Use the arrow keys to move between readings.`}
                onPointerLeave={() => setHover(null)}
              >
                {panels.map((panel) => (
                  <g key={panel.key}>
                    <text className="viz-text viz-panel-title" x={0} y={panel.top + 4}>
                      {panel.title}
                    </text>
                    {panel.scale.ticks.map((v) => (
                      <g key={v}>
                        <line
                          className="viz-grid"
                          x1={GUTTER}
                          x2={GUTTER + plotW}
                          y1={panel.yOf(v)}
                          y2={panel.yOf(v)}
                        />
                        <text className="viz-text" x={GUTTER - 10} y={panel.yOf(v) + 4} textAnchor="end">
                          {v}°
                        </text>
                      </g>
                    ))}
                    {panel.series.map((s) => (
                      <path
                        key={s.key}
                        d={linePath(data[s.key], xOf, panel.yOf)}
                        fill="none"
                        stroke={s.color}
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    ))}
                  </g>
                ))}

                {stateRows.map((row, r) => {
                  const y = stripsTop + r * (STRIP_H + STRIP_GAP);
                  return (
                    <g key={row.key}>
                      <text className="viz-text viz-strip-label" x={GUTTER - 10} y={y + STRIP_H / 2 + 4} textAnchor="end">
                        {row.label}
                      </text>
                      {stateRuns(data[row.key]).map(([from, to, index]) => {
                        const name = stateName(row.key, index);
                        const fill = stateColor(row.key, name);
                        const left = GUTTER + (from / bins) * plotW;
                        const w = ((to - from + 1) / bins) * plotW;
                        const inset = w > 6 ? 1 : 0; // the 2px surface gap between neighbours
                        const fits = w >= name.length * 5.9 + 14;
                        return (
                          <g key={`${from}-${index}`}>
                            <rect
                              x={left + inset}
                              y={y}
                              width={Math.max(w - inset * 2, 2)}
                              height={STRIP_H}
                              rx={w > 6 ? 3 : 0}
                              fill={fill}
                            />
                            {fits && !QUIET_STATES.has(name) && (
                              <text
                                className="viz-segment-label"
                                x={left + w / 2}
                                y={y + STRIP_H / 2 + 4}
                                textAnchor="middle"
                                fill={labelColor(fill)}
                              >
                                {name}
                              </text>
                            )}
                          </g>
                        );
                      })}
                    </g>
                  );
                })}

                <line className="viz-axis" x1={GUTTER} x2={GUTTER + plotW} y1={plotBottom + 8} y2={plotBottom + 8} />
                {xTicks.map((tick) => (
                  <text key={tick.x} className="viz-text" x={tick.x} y={plotBottom + 24} textAnchor={tick.anchor}>
                    {tick.label}
                  </text>
                ))}

                {hover !== null && (
                  <g pointerEvents="none">
                    <line className="viz-crosshair" x1={xOf(hover)} x2={xOf(hover)} y1={TEMP_TOP} y2={plotBottom} />
                    {panels.flatMap((panel) =>
                      panel.series.map((s) =>
                        data[s.key][hover] == null ? null : (
                          <circle
                            key={s.key}
                            className="viz-dot"
                            cx={xOf(hover)}
                            cy={panel.yOf(data[s.key][hover])}
                            r="4"
                            fill={s.color}
                          />
                        ),
                      ),
                    )}
                  </g>
                )}

                <rect
                  className="viz-hit"
                  x={GUTTER}
                  y={0}
                  width={plotW}
                  height={plotBottom + 8}
                  fill="transparent"
                  tabIndex={0}
                  onPointerMove={(event) => setHover(binFromPointer(event))}
                  onFocus={() => setHover((h) => h ?? Math.floor(bins / 2))}
                  onBlur={() => setHover(null)}
                  onKeyDown={onKeyDown}
                />
              </svg>

              {hover !== null && (
                <div
                  className="viz-tooltip"
                  style={
                    tooltipOnRight
                      ? { left: xOf(hover) + 14, top: TEMP_TOP }
                      : { left: xOf(hover) - 14, top: TEMP_TOP, transform: "translateX(-100%)" }
                  }
                >
                  <div className="viz-tooltip-time">{formatTime(telemetry.t[hover])}</div>
                  {tempSeries.map((s) => (
                    <div className="viz-tooltip-row" key={s.key}>
                      <span className="viz-key-line" style={{ background: s.color }} />
                      <strong>{data[s.key][hover] == null ? "—" : `${data[s.key][hover]} °C`}</strong>
                      <span>{s.label}</span>
                    </div>
                  ))}
                  {stateRows.map((row) => {
                    const name = stateName(row.key, data[row.key][hover]);
                    return (
                      <div className="viz-tooltip-row" key={row.key}>
                        <span
                          className={`viz-key-swatch ${name ? "" : "empty"}`}
                          style={name ? { background: stateColor(row.key, name) } : undefined}
                        />
                        <strong>{name ?? "—"}</strong>
                        <span>{row.label}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {stateRows.length > 0 && (
              <div className="viz-legend">
                <strong>Modes and status</strong>
                {legendStates.map(({ key, name }) => (
                  <span key={name}>
                    <span className="viz-key-swatch" style={{ background: stateColor(key, name) }} />
                    {name}
                  </span>
                ))}
                <span>
                  <span className="viz-key-swatch empty" />
                  No data
                </span>
              </div>
            )}

            <p className="chart-caption">
              Each point is the average (temperatures) or most common value (modes) over roughly{" "}
              {Math.max(1, Math.round((lastT - firstT) / bins / 60))} minutes. A temperature of 0 is
              treated as no reading. Hover the chart, or focus it and use the arrow keys, to read exact values.
            </p>
            {renamed.length > 0 && (
              <p className="chart-caption">
                Read from differently named columns in this file:{" "}
                {renamed.map(([signal, source]) => `${SIGNAL_LABEL[signal]} ← ${source}`).join("; ")}.
              </p>
            )}
            {notRecorded.length > 0 && (
              <p className="chart-caption">
                Not recorded in this file: {notRecorded.map((s) => SIGNAL_LABEL[s]).join(", ")}.
              </p>
            )}
            {noReadings.length > 0 && (
              <p className="chart-caption">
                No valid readings for car {carId}: {noReadings.map((s) => SIGNAL_LABEL[s]).join(", ")}.
              </p>
            )}

            <details className="viz-table-view" onToggle={(event) => setTableOpen(event.currentTarget.open)}>
              <summary>View as table</summary>
              {tableOpen && (
                <div className="viz-table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        {[...tempSeries, ...stateRows].map((s) => (
                          <th key={s.key}>{s.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {telemetry.t.map((t, i) => (
                        <tr key={t}>
                          <td>{formatTime(t)}</td>
                          {tempSeries.map((s) => (
                            <td key={s.key}>{data[s.key][i] ?? "—"}</td>
                          ))}
                          {stateRows.map((row) => (
                            <td key={row.key}>{stateName(row.key, data[row.key][i]) ?? "—"}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </details>
          </>
        )}
      </div>
    </section>
  );
}
