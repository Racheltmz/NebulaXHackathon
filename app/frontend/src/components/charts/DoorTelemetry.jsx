import { useMemo } from "react";

import { parseTimestamp } from "./DoorTimeline";
import TelemetryChart from "./TelemetryChart";

/** The Door stream's readings for one run: the analog signals as line panels, the detected
 * segments and the on/off flags as strips, on one shared axis. Reads `telemetry` as built by the
 * backend's ml/door_telemetry.py. Drawing and interaction are in TelemetryChart.
 *
 * The axis is by reading, not by clock time: the provided streams are cycles stitched together,
 * with the time between them missing, so a clock axis would be mostly empty. `breaks` mark where
 * the recording jumps ahead, so each stretch between two lines is one door cycle. */

// Each signal has its own panel, so colour isn't what identifies it (the panel title does) — one
// accent for all, and the second slot only where two lines share a panel.
const BLUE = "#2a78d6";
const ORANGE = "#eb6834";

const PANELS = [
  { key: "current", title: "Motor current (mA)", height: 100, series: [{ key: "current", label: "Motor current", color: BLUE, format: (v) => `${v} mA` }] },
  { key: "voltage", title: "Motor voltage (10 mV units)", height: 72, series: [{ key: "voltage", label: "Motor voltage", color: BLUE, format: (v) => `${v} (×10 mV)` }] },
  { key: "back_emf", title: "Motor back-EMF", height: 72, series: [{ key: "back_emf", label: "Back-EMF", color: BLUE }] },
  { key: "position", title: "Door leaf position", height: 72, series: [{ key: "position", label: "Door leaf position", color: BLUE }] },
  {
    key: "cycle_time",
    title: "Measured cycle time (0.1 s units)",
    height: 60,
    series: [
      { key: "opening_time", label: "Opening time", color: BLUE, format: (v) => `${v} (×0.1 s)` },
      { key: "closing_time", label: "Closing time", color: ORANGE, format: (v) => `${v} (×0.1 s)` },
    ],
  },
];

// From the dataset's own parameter list (Door Data Headers): what the four switches measure.
const SWITCH_HINT = "changes from released to actuated while the door closes, and back while it opens";
const FLAGS = [
  { key: "close_command", label: "Close command", hint: "A value of 1 triggers the door-closing action" },
  { key: "open_command", label: "Open command", hint: "A value of 1 triggers the door-opening action" },
  { key: "opening", label: "Door is opening" },
  { key: "closing", label: "Door is closing" },
  { key: "dcsr", label: "DCSR", hint: `Door Close Switch Right — ${SWITCH_HINT}` },
  { key: "dcsl", label: "DCSL", hint: `Door Close Switch Left — ${SWITCH_HINT}` },
  { key: "dlsr", label: "DLSR", hint: `Door Locked Switch Right — ${SWITCH_HINT}` },
  { key: "dlsl", label: "DLSL", hint: `Door Locked Switch Left — ${SWITCH_HINT}` },
  { key: "door_opened", label: "Door opened" },
  { key: "door_locked", label: "Door locked" },
];

// "Off" recedes and "On" carries the accent, so what is active stands out.
const FLAG_COLORS = { Off: "#e1e0d9", On: BLUE };
// The app's own status colours, as the segment timeline uses.
const SEGMENT_NAMES = ["Normal", "Abnormal resistance"];
const SEGMENT_COLORS = { Normal: "#16a34a", "Abnormal resistance": "#dc2626" };

const DATE_FORMAT = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "numeric", month: "short" });
const CLOCK_FORMAT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC", // the stream's timestamps carry no zone; show them as recorded
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});
const formatTick = (seconds) => CLOCK_FORMAT.format(new Date(seconds * 1000));
const formatTooltipTime = (seconds) => {
  const date = new Date(Math.round(seconds * 1000));
  const ms = String(date.getUTCMilliseconds()).padStart(3, "0");
  return `${DATE_FORMAT.format(date)}, ${CLOCK_FORMAT.format(date)}.${ms}`;
};

export default function DoorTelemetry({ telemetry, rows }) {
  // Which detected segment, if any, each point falls in.
  const segmentValues = useMemo(() => {
    const segments = (rows ?? []).map((r) => ({
      start: parseTimestamp(r.start_time),
      end: parseTimestamp(r.end_time),
      index: Math.max(SEGMENT_NAMES.indexOf(r.label), 0),
    }));
    return telemetry.times.map((t) => {
      const ms = t * 1000;
      const hit = segments.find((s) => ms >= s.start - 1 && ms <= s.end + 1);
      return hit ? hit.index : null;
    });
  }, [telemetry, rows]);

  // A faint tint behind every panel for each detected segment, red a little stronger than green so
  // an abnormal stretch stands out from the normal ones around it.
  const bands = useMemo(() => {
    const out = [];
    let start = 0;
    let current = null;
    const close = (end) => {
      if (current !== null) {
        out.push({
          from: start,
          to: end,
          color: SEGMENT_COLORS[SEGMENT_NAMES[current]],
          opacity: SEGMENT_NAMES[current] === "Abnormal resistance" ? 0.12 : 0.06,
        });
      }
    };
    segmentValues.forEach((value, i) => {
      if (value !== current) {
        close(i - 1);
        start = i;
        current = value;
      }
    });
    close(segmentValues.length - 1);
    return out;
  }, [segmentValues]);

  const panels = PANELS.map((panel) => ({
    ...panel,
    series: panel.series.filter((s) => telemetry.series[s.key]).map((s) => ({ ...s, values: telemetry.series[s.key] })),
  }));
  const strips = [
    {
      key: "segments",
      label: "Segments",
      names: SEGMENT_NAMES,
      values: segmentValues,
      colorOf: (name) => SEGMENT_COLORS[name],
    },
    ...FLAGS.filter((f) => telemetry.flags[f.key]).map((f) => ({
      ...f,
      names: ["Off", "On"],
      values: telemetry.flags[f.key],
      colorOf: (name) => FLAG_COLORS[name],
      showLabels: false,
      compact: true,
    })),
  ];

  const cycles = telemetry.breaks.length + 1;
  const missing = [...PANELS.flatMap((p) => p.series), ...FLAGS].filter(
    (s) => !telemetry.series[s.key] && !telemetry.flags[s.key],
  );

  return (
    <TelemetryChart
      ariaLabel="Door motor signals, segments and switch states over the recording"
      times={telemetry.times}
      formatTick={formatTick}
      formatTooltipTime={formatTooltipTime}
      panels={panels}
      strips={strips}
      breaks={telemetry.breaks}
      bands={bands}
      stripHeight={14}
      stripGap={6}
      stripsLegendTitle="Segments and switches"
      footer={
        <>
          <p className="chart-caption">
            Each point averages about {telemetry.rows_per_point} readings
            {telemetry.sample_ms ? ` (${telemetry.rows_per_point * telemetry.sample_ms} ms)` : ""}; a flag shows as
            on if it was on at any point within it. An opening or closing time of 0 is treated as no reading.
            Hover the chart, or focus it and use the arrow keys, to read exact values. The faint tint behind
            the chart marks each detected segment: green for Normal, red for Abnormal resistance.
          </p>
          {telemetry.breaks.length > 0 && (
            <p className="chart-caption">
              The recording jumps ahead in time between door cycles, so the axis skips that time. The faint
              vertical lines mark each jump, splitting the stream into {cycles} stretches, one door cycle each.
            </p>
          )}
          {missing.length > 0 && (
            <p className="chart-caption">Not recorded in this file: {missing.map((s) => s.label).join(", ")}.</p>
          )}
        </>
      }
    />
  );
}
