import { useMemo, useState } from "react";

import TelemetryChart from "./TelemetryChart";

/** Per-car ACV telemetry for one run: temperatures as lines, and the modes/status flags as state
 * strips, for one car at a time. Reads `telemetry` as built by the backend's ml/acv_telemetry.py —
 * a fixed number of equal time bins per series, null where a bin has no valid reading (so gaps in
 * the recording stay gaps). Drawing and interaction are in TelemetryChart. */

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

  const data = telemetry.cars[carId] ?? {};
  const bins = telemetry.t.length;

  const panels = TEMP_PANELS.map((panel) => ({
    key: panel.key,
    title: panel.title,
    height: panel.height,
    unit: "°",
    series: TEMP_SERIES.filter((s) => panel.signals.includes(s.key) && data[s.key]).map((s) => ({
      ...s,
      values: data[s.key],
      format: (v) => `${v} °C`,
    })),
  }));
  const strips = STATE_ROWS.filter((r) => data[r.key]).map((r) => ({
    ...r,
    names: telemetry.categories[r.key],
    values: data[r.key],
    colorOf: (name) => stateColor(r.key, name),
    quiet: QUIET_STATES,
  }));

  // What to tell the reader about signals this car (or this file) doesn't plot.
  const notRecorded = Object.keys(CANONICAL_SOURCE).filter((s) => !telemetry.sources[s]);
  const noReadings = Object.keys(telemetry.sources).filter((s) => !data[s]);
  const renamed = Object.entries(telemetry.sources).filter(
    ([signal, source]) => data[signal] && source !== CANONICAL_SOURCE[signal],
  );
  const minutesPerPoint = Math.max(1, Math.round((telemetry.t[bins - 1] - telemetry.t[0]) / bins / 60));

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
            onClick={() => setCarId(id)}
          >
            Car {id} (#{i + 1})
          </button>
        ))}
      </div>

      <div className="chart-card">
        {!hasReadings(carId) ? (
          <p>
            Car {carId} has no readings in this file
            {telemetry.sources.running_mode ? " (its columns are empty)" : ""}.
          </p>
        ) : (
          <TelemetryChart
            key={carId}
            ariaLabel={`Car ${carId} temperatures and modes over time`}
            times={telemetry.t}
            formatTick={formatTime}
            formatTooltipTime={formatTime}
            panels={panels}
            strips={strips}
            stripsLegendTitle="Modes and status"
            footer={
              <>
                <p className="chart-caption">
                  Each point is the average (temperatures) or most common value (modes) over roughly{" "}
                  {minutesPerPoint} minutes. A temperature of 0 is treated as no reading. Hover the chart, or
                  focus it and use the arrow keys, to read exact values.
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
              </>
            }
          />
        )}
      </div>
    </section>
  );
}
