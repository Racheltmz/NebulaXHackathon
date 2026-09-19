import { useEffect, useRef, useState } from "react";

/** Line panels and state strips on one shared axis, with one crosshair and tooltip across all of
 * them and a table view. Shared by the ACV and Door run dashboards, which only differ in what
 * they feed it (see AcvTelemetry.jsx, DoorTelemetry.jsx).
 *
 * `times`   seconds since the epoch for each point (the x axis is by point, so uneven spacing —
 *           e.g. Door's skipped idle time — doesn't leave empty stretches).
 * `panels`  [{ key, title, height, unit?, series: [{ key, label, color, values, format? }] }] —
 *           a line panel per scale, so nothing is ever plotted on two axes. `values` has one
 *           entry per point, null where there is no reading.
 * `strips`  [{ key, label, hint?, names, values, colorOf(name), quiet?, showLabels?, compact? }] —
 *           `values` are indexes into `names`, null for no data. A compact strip is an on/off flag:
 *           the tooltip lists the flags that are on in one line, not a row each.
 * `breaks`  point indexes to draw a faint vertical line before (e.g. the start of each cycle).
 * `bands`   [{ from, to, color, opacity? }] — point ranges to tint faintly behind every panel and
 *           strip, to draw the eye to the stretches that matter. */

const GUTTER = 116; // left: axis ticks and the strip labels
const PAD_RIGHT = 16;
const TOP = 8;
const TITLE_H = 22; // the line above each panel that carries its title
const PANEL_GAP = 12;
const SECTION_GAP = 26; // between the line panels and the strips
const AXIS_H = 30;
const MIN_WIDTH = 360;

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
  return 1.05 / (l + 0.05) > (l + 0.05) / (luminance("#0b0b0b") + 0.05) ? "#ffffff" : "#0b0b0b";
}

// The panel spans the data with a little air, and ticks fall on round numbers inside it — so the
// line fills the panel instead of sitting in the middle of a scale rounded out to the next step.
function niceScale(min, max, height) {
  const range = Math.max(max - min, 1e-9);
  const target = Math.max(2, Math.floor(height / 34));
  const raw = range / (target + 2);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw);
  const pad = range * 0.05;
  const lo = min - pad;
  const hi = max + pad;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step / 1000; v += step) ticks.push(Math.round(v * 1000) / 1000);
  return { lo, hi, ticks };
}

// Path for one line, broken wherever a point has no reading. An isolated point gets a zero-length
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

// Consecutive points with the same state, as [first, last, index] runs.
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

// Series keys for a panel with more than one line, right-aligned on its title line.
function keyLayout(series, right) {
  let x = right;
  return [...series].reverse().map((s) => {
    const textWidth = s.label.length * 6.3;
    const item = { s, textX: x, keyStart: x - textWidth - 22, keyEnd: x - textWidth - 8 };
    x = item.keyStart - 18;
    return item;
  });
}

export default function TelemetryChart({
  ariaLabel,
  times,
  formatTick,
  formatTooltipTime,
  panels,
  strips,
  breaks = [],
  bands = [],
  stripHeight = 18,
  stripGap = 8,
  stripsLegendTitle = "States",
  footer,
}) {
  const [hover, setHover] = useState(null); // point index under the pointer / keyboard focus
  const [tableOpen, setTableOpen] = useState(false);
  const [width, setWidth] = useState(720);
  const wrapRef = useRef(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const update = () => setWidth(Math.max(el.clientWidth, MIN_WIDTH));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const count = times.length;
  const plotW = width - GUTTER - PAD_RIGHT;
  const xOf = (i) => GUTTER + ((i + 0.5) / count) * plotW;

  let cursor = TOP;
  const laidPanels = [];
  panels.forEach((panel) => {
    const series = panel.series.filter((s) => s.values.some((v) => v != null));
    if (series.length === 0) return;
    const values = series.flatMap((s) => s.values.filter((v) => v != null));
    const scale = niceScale(Math.min(...values), Math.max(...values), panel.height);
    const top = cursor + TITLE_H;
    laidPanels.push({
      ...panel,
      series,
      scale,
      top,
      yOf: (v) => top + (1 - (v - scale.lo) / (scale.hi - scale.lo)) * panel.height,
    });
    cursor = top + panel.height + PANEL_GAP;
  });
  const panelsBottom = laidPanels.length > 0 ? cursor - PANEL_GAP : TOP;

  const laidStrips = strips.filter((s) => s.values.some((v) => v != null));
  const stripsTop = laidPanels.length > 0 ? panelsBottom + SECTION_GAP : TOP;
  const stripsBottom = stripsTop + laidStrips.length * (stripHeight + stripGap) - (laidStrips.length ? stripGap : 0);
  const plotBottom = laidStrips.length > 0 ? stripsBottom : panelsBottom;
  const height = plotBottom + AXIS_H;

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => ({
    x: xOf(f * (count - 1)),
    label: formatTick(times[Math.round(f * (count - 1))]),
    anchor: f === 0 ? "start" : f === 1 ? "end" : "middle",
  }));

  const nameAt = (strip, i) => (strip.values[i] == null ? null : strip.names[strip.values[i]]);
  const legendStates = [];
  laidStrips.forEach((strip) => {
    new Set(strip.values.filter((v) => v != null)).forEach((index) => {
      const name = strip.names[index];
      if (!legendStates.some((s) => s.name === name)) legendStates.push({ name, color: strip.colorOf(name) });
    });
  });

  const pointFromPointer = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / rect.width;
    return Math.min(count - 1, Math.max(0, Math.floor(fraction * count)));
  };

  const onKeyDown = (event) => {
    if (event.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? count) - 1));
    else if (event.key === "ArrowRight") setHover((h) => Math.min(count - 1, (h ?? -1) + 1));
    else if (event.key === "Escape") setHover(null);
    else return;
    event.preventDefault();
  };

  const tooltipOnRight = hover !== null && xOf(hover) < width * 0.6;
  const rowStrips = laidStrips.filter((s) => !s.compact);
  const flagStrips = laidStrips.filter((s) => s.compact);
  const activeFlags = hover === null ? [] : flagStrips.filter((s) => nameAt(s, hover) === "On").map((s) => s.label);
  const tableSeries = laidPanels.flatMap((p) => p.series);

  return (
    <div className="telemetry-chart">
      <div className="viz-wrap" ref={wrapRef}>
        <svg
          width={width}
          height={height}
          role="group"
          aria-label={`${ariaLabel}. Use the arrow keys to move between readings.`}
          onPointerLeave={() => setHover(null)}
        >
          {bands.map((band) => (
            <rect
              key={`${band.from}-${band.to}`}
              className="viz-band"
              x={GUTTER + (band.from / count) * plotW}
              y={TOP}
              width={((band.to - band.from + 1) / count) * plotW}
              height={plotBottom - TOP}
              fill={band.color}
              opacity={band.opacity ?? 0.08}
            />
          ))}

          {breaks.map((b) => (
            <line
              key={b}
              className="viz-break"
              x1={GUTTER + (b / count) * plotW}
              x2={GUTTER + (b / count) * plotW}
              y1={TOP}
              y2={plotBottom}
            />
          ))}

          {laidPanels.map((panel) => (
            <g key={panel.key}>
              <text className="viz-text viz-panel-title" x={GUTTER} y={panel.top - 8}>
                {panel.title}
              </text>
              {panel.series.length > 1 &&
                keyLayout(panel.series, GUTTER + plotW).map(({ s, textX, keyStart, keyEnd }) => (
                  <g key={s.key}>
                    <line
                      x1={keyStart}
                      x2={keyEnd}
                      y1={panel.top - 12}
                      y2={panel.top - 12}
                      stroke={s.color}
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                    <text className="viz-text viz-key-text" x={textX} y={panel.top - 8} textAnchor="end">
                      {s.label}
                    </text>
                  </g>
                ))}
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
                    {v}
                    {panel.unit ?? ""}
                  </text>
                </g>
              ))}
              {panel.series.map((s) => (
                <path
                  key={s.key}
                  d={linePath(s.values, xOf, panel.yOf)}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}
            </g>
          ))}

          {laidStrips.map((strip, r) => {
            const y = stripsTop + r * (stripHeight + stripGap);
            return (
              <g key={strip.key}>
                <text
                  className="viz-text viz-strip-label"
                  x={GUTTER - 10}
                  y={y + stripHeight / 2 + 4}
                  textAnchor="end"
                >
                  {strip.hint && <title>{strip.hint}</title>}
                  {strip.label}
                </text>
                {stateRuns(strip.values).map(([from, to, index]) => {
                  const name = strip.names[index];
                  const fill = strip.colorOf(name);
                  const left = GUTTER + (from / count) * plotW;
                  const w = ((to - from + 1) / count) * plotW;
                  const inset = w > 6 ? 1 : 0; // the 2px surface gap between neighbours
                  const fits = w >= name.length * 5.9 + 14;
                  const labelled = strip.showLabels !== false && !strip.quiet?.has(name) && fits;
                  return (
                    <g key={`${from}-${index}`}>
                      <rect
                        x={left + inset}
                        y={y}
                        width={Math.max(w - inset * 2, 1.5)}
                        height={stripHeight}
                        rx={w > 6 ? 3 : 0}
                        fill={fill}
                      />
                      {labelled && (
                        <text
                          className="viz-segment-label"
                          x={left + w / 2}
                          y={y + stripHeight / 2 + 4}
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
              <line className="viz-crosshair" x1={xOf(hover)} x2={xOf(hover)} y1={TOP} y2={plotBottom} />
              {laidPanels.flatMap((panel) =>
                panel.series.map((s) =>
                  s.values[hover] == null ? null : (
                    <circle
                      key={s.key}
                      className="viz-dot"
                      cx={xOf(hover)}
                      cy={panel.yOf(s.values[hover])}
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
            onPointerMove={(event) => setHover(pointFromPointer(event))}
            onFocus={() => setHover((h) => h ?? Math.floor(count / 2))}
            onBlur={() => setHover(null)}
            onKeyDown={onKeyDown}
          />
        </svg>

        {hover !== null && (
          <div
            className="viz-tooltip"
            style={
              tooltipOnRight
                ? { left: xOf(hover) + 14, top: TOP }
                : { left: xOf(hover) - 14, top: TOP, transform: "translateX(-100%)" }
            }
          >
            <div className="viz-tooltip-time">{formatTooltipTime(times[hover])}</div>
            {tableSeries.map((s) => (
              <div className="viz-tooltip-row" key={s.key}>
                <span className="viz-key-line" style={{ background: s.color }} />
                <strong>{s.values[hover] == null ? "—" : (s.format ?? String)(s.values[hover])}</strong>
                <span>{s.label}</span>
              </div>
            ))}
            {rowStrips.map((strip) => {
              const name = nameAt(strip, hover);
              return (
                <div className="viz-tooltip-row" key={strip.key}>
                  <span
                    className={`viz-key-swatch ${name ? "" : "empty"}`}
                    style={name ? { background: strip.colorOf(name) } : undefined}
                  />
                  <strong>{name ?? "—"}</strong>
                  <span>{strip.label}</span>
                </div>
              );
            })}
            {flagStrips.length > 0 && (
              <div className="viz-tooltip-flags">
                <span>On:</span> {activeFlags.length > 0 ? activeFlags.join(", ") : "none"}
              </div>
            )}
          </div>
        )}
      </div>

      {laidStrips.length > 0 && (
        <div className="viz-legend">
          <strong>{stripsLegendTitle}</strong>
          {legendStates.map(({ name, color }) => (
            <span key={name}>
              <span className="viz-key-swatch" style={{ background: color }} />
              {name}
            </span>
          ))}
          <span>
            <span className="viz-key-swatch empty" />
            No data
          </span>
        </div>
      )}

      {footer}

      <details className="viz-table-view" onToggle={(event) => setTableOpen(event.currentTarget.open)}>
        <summary>View as table</summary>
        {tableOpen && (
          <div className="viz-table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  {[...tableSeries, ...laidStrips].map((s) => (
                    <th key={s.key}>{s.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {times.map((t, i) => (
                  <tr key={t}>
                    <td>{formatTooltipTime(t)}</td>
                    {tableSeries.map((s) => (
                      <td key={s.key}>{s.values[i] ?? "—"}</td>
                    ))}
                    {laidStrips.map((strip) => (
                      <td key={strip.key}>{nameAt(strip, i) ?? "—"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </details>
    </div>
  );
}
