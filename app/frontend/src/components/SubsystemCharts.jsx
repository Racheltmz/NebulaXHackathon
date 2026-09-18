import { useEffect, useState } from "react";

import apiClient from "../lib/apiClient";
import BarChart from "./charts/BarChart";
import DoorTimeline from "./charts/DoorTimeline";
import TrainDiagram, { carsFromRanking } from "./charts/TrainDiagram";
import { InfoIcon } from "./icons/NavIcons";
import StatTile from "./StatTile";

/** Per-subsystem dashboard charts. Each takes `job` ({ rows, summary }) — a single run on the
 * run dashboard, or (with `aggregate`) the latest prediction for every file of that subsystem
 * on the History page. */

// The "Understanding ..." card at the top of each dashboard: how to read that subsystem's output.
const UNDERSTANDING = {
  door: {
    title: "Understanding door segments",
    points: [
      "Each segment is one door open or close cycle found in a continuous stream of door controller data, with its start and end time.",
      "Normal means the door moved freely. Abnormal resistance means the motor met extra resistance, for example from a foreign object in the slide rail, a jammed rubber strip or a deformed door leaf.",
      "Repeated abnormal cycles can lead to door jamming and motor overload, so that door should be inspected.",
    ],
  },
  acv: {
    title: "Understanding car rankings",
    points: [
      "Each file is one air conditioning case in which exactly one of the cars has a refrigerant leak.",
      "Cars are ranked from most to least likely to be the leaking car, so the first car listed is the best guess.",
      "Inspect the top ranked car first, then move down the ranking if it is clear.",
    ],
  },
  rail_corrugation: {
    title: "Understanding corrugation classes",
    points: [
      "Each file is a 1 second recording of axle box vibration and shock, classified as Normal, Side I or Side II.",
      "Normal means both rails are healthy. Side I or Side II means corrugation (wave like wear) on that side's rail while the other side is normal.",
      "A Side I or Side II result means that side of the track should be inspected.",
    ],
  },
  shm: {
    title: "Understanding cumulative damage",
    points: [
      "This value comes from Miner's linear cumulative damage rule applied to rstress cycles from the uploaded signal.",
      "It ranges from 0 (no accumulated fatigue damage) up to 1.0 (the point at which fatigue failure is expected).",
      "The higher the value, the more of the component's fatigue life has already been used up, and the component should be inspected.",
    ],
  },
};

function UnderstandingCard({ title, points }) {
  return (
    <div className="chart-card">
      <h3 className="title-with-icon">
        <span className="icon-tile">
          <InfoIcon />
        </span>
        {title}
      </h3>
      <ul>
        {points.map((point) => (
          <li key={point}>{point}</li>
        ))}
      </ul>
    </div>
  );
}

function DoorChart({ job }) {
  const normal = job.rows.filter((r) => r.label !== "Abnormal resistance").length;
  const abnormal = job.rows.length - normal;
  return (
    <>
      <UnderstandingCard {...UNDERSTANDING.door} />
      <div className="stat-tiles">
        <StatTile label="Segments detected" value={job.rows.length} />
        <StatTile label="Normal" value={normal} />
        <StatTile label="Abnormal resistance" value={abnormal} />
      </div>
      <div className="chart-card">
        <h3>Segment timeline</h3>
        <DoorTimeline rows={job.rows} />
      </div>
    </>
  );
}

function AcvChart({ job, aggregate }) {
  if (aggregate) {
    // No chart here on purpose. Each file is its own train, and a car identifier is only
    // meaningful within the file it came from — car 03 on one trainset has nothing to do with
    // car 03 on another — so counts or rankings pooled across files would be misleading. The
    // ranking per file is in the table below, and the train diagram belongs to a single
    // prediction (Predict page and run dashboard).
    return (
      <>
        <UnderstandingCard {...UNDERSTANDING.acv} />
        <div className="stat-tiles">
          <StatTile label="Files Analysed" value={job.rows.length} />
        </div>
        <p className="history-dashboard-note">
          Each file is a separate trainset, so car rankings aren&apos;t comparable between files and
          aren&apos;t summarised here. See each file&apos;s ranking in the table below, or open a run
          for its train diagram.
        </p>
      </>
    );
  }

  const byFile = {};
  job.rows.forEach((r) => {
    byFile[r.file_id] = r.ranked_cars || "";
  });

  return (
    <>
      <UnderstandingCard {...UNDERSTANDING.acv} />
      <div className="stat-tiles">
        <StatTile label="Files Analysed" value={job.rows.length} />
        <StatTile
          label="Most likely faulty (first file)"
          value={carsFromRanking(Object.values(byFile)[0])[0]?.id ?? "—"}
        />
      </div>
      {Object.entries(byFile).map(([fileId, ranked]) => (
        <div className="chart-card" key={fileId}>
          <h3>{fileId} — cars ranked most → least likely faulty</h3>
          <TrainDiagram cars={carsFromRanking(ranked)} />
        </div>
      ))}
    </>
  );
}

function RailChart({ job }) {
  const counts = { Normal: 0, "Side I": 0, "Side II": 0 };
  job.rows.forEach((r) => {
    if (counts[r.label] !== undefined) counts[r.label] += 1;
  });

  return (
    <>
      <UnderstandingCard {...UNDERSTANDING.rail_corrugation} />
      <div className="stat-tiles">
        {Object.entries(counts).map(([label, value]) => (
          <StatTile key={label} label={label} value={value} />
        ))}
      </div>
      <div className="chart-card">
        <h3>Class distribution</h3>
        <BarChart
          data={Object.entries(counts).map(([label, value]) => ({
            label,
            value,
            color: label === "Normal" ? "var(--status-normal)" : "var(--status-abnormal)",
          }))}
        />
      </div>
    </>
  );
}

// Fixed-width bins over the damage scale — 0 up to the failure threshold, or the largest value if
// that's higher. Data-driven rules (sqrt, Sturges, Freedman–Diaconis) all give only 3–6 bins at
// this app's typical n of ~10–20 records; a fixed width keeps the resolution constant and puts
// the same damage level in the same place on every run.
const HISTOGRAM_BIN_WIDTH = 0.1;

function buildHistogram(values, highlightValues, scaleMax) {
  if (values.length === 0) return [];
  const top = Math.max(scaleMax, ...values);
  const binCount = Math.ceil(top / HISTOGRAM_BIN_WIDTH - 1e-9);
  const binIndex = (v) => Math.min(Math.max(Math.floor(v / HISTOGRAM_BIN_WIDTH + 1e-9), 0), binCount - 1);

  const counts = new Array(binCount).fill(0);
  values.forEach((v) => {
    counts[binIndex(v)] += 1;
  });
  const highlighted = new Set(highlightValues.map(binIndex));

  return counts.map((count, i) => ({
    label: (i * HISTOGRAM_BIN_WIDTH).toFixed(2),
    value: count,
    color: highlighted.has(i) ? "var(--fe-cobalt)" : "var(--fe-edge)",
  }));
}

// Damage is the fraction of fatigue life used: 1.0 is failure (the histogram's scale tops out
// there), and severity bands are fixed cut-offs below it. Same rules as backend ml/severity.py —
// change them together. Order matters: it's the count plot's x-axis order.
const SHM_SCALE_MAX = 1.0;
const SHM_ELEVATED_THRESHOLD = 0.5;
const SHM_CRITICAL_THRESHOLD = 0.8;
const SEVERITY_LEVELS = [
  { level: "normal", label: "Normal", color: "var(--status-normal)", badge: "normal" },
  { level: "elevated", label: "Elevated", color: "var(--db-warning)", badge: "elevated" },
  { level: "critical", label: "Critical", color: "var(--status-abnormal)", badge: "abnormal" },
];

function shmSeverity(value) {
  if (value >= SHM_CRITICAL_THRESHOLD) return SEVERITY_LEVELS[2];
  if (value >= SHM_ELEVATED_THRESHOLD) return SEVERITY_LEVELS[1];
  return SEVERITY_LEVELS[0];
}

function ShmChart({ job, aggregate }) {
  const [fetchedDistribution, setFetchedDistribution] = useState(null);

  useEffect(() => {
    if (aggregate) return; // the rows already are the whole population
    apiClient
      .get("/api/subsystems/shm/damage-distribution")
      .then((res) => setFetchedDistribution(res.data.values))
      .catch(() => setFetchedDistribution([]));
  }, [aggregate]);

  const runValues = job.rows.map((r) => r.value).filter((v) => v !== null && v !== undefined);
  const distribution = aggregate ? runValues : fetchedDistribution;
  const meanValue = runValues.length > 0 ? runValues.reduce((a, b) => a + b, 0) / runValues.length : null;
  const maxValue = runValues.length > 0 ? Math.max(...runValues) : null;
  const headlineValue = aggregate ? meanValue : job.summary.mean ?? job.rows[0]?.value ?? null;
  const headlineSeverity = headlineValue !== null ? shmSeverity(headlineValue) : null;
  const histogram = distribution ? buildHistogram(distribution, runValues, SHM_SCALE_MAX) : [];
  // Every level is listed even at zero, so the plot always shows the full scale.
  const severityCounts = SEVERITY_LEVELS.map(({ level, label, color }) => ({
    label,
    color,
    value: runValues.filter((v) => shmSeverity(v).level === level).length,
  }));

  const sorted = [...job.rows].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

  return (
    <>
      <UnderstandingCard {...UNDERSTANDING.shm} />

      <div className="shm-hero-row">
        <div className="chart-card shm-hero-card">
          <div className="shm-hero-value">{headlineValue !== null ? headlineValue.toFixed(4) : "—"}</div>
          <div className="shm-hero-label">
            {aggregate ? "Mean Predicted Cumulative Damage" : "Predicted Cumulative Damage"}
          </div>
          {!aggregate && headlineSeverity && (
            <span className={`badge shm-hero-badge ${headlineSeverity.badge}`}>
              Severity: {headlineSeverity.label}
            </span>
          )}
        </div>

        <div className="chart-card shm-hero-card">
          <div className="shm-hero-value">{maxValue !== null ? maxValue.toFixed(4) : "—"}</div>
          <div className="shm-hero-label">Max Predicted Cumulative Damage</div>
        </div>

        <div className="chart-card shm-hero-card">
          <div className="shm-hero-value">{job.rows.length}</div>
          <div className="shm-hero-label">Files Analysed</div>
        </div>
      </div>

      <div className="shm-chart-row">
        <div className="chart-card">
          <h3 className="chart-card-header">
            {aggregate ? "Distribution of predicted damage" : "Where this value sits vs other records"}
          </h3>
          {distribution === null ? (
            <p>Loading…</p>
          ) : histogram.length > 0 ? (
            <BarChart data={histogram} tight />
          ) : (
            <p>Not enough historical data yet to compare against.</p>
          )}
          <p className="chart-caption">
            {aggregate
              ? "Each bar counts files at that damage level (labelled by its lower bound)."
              : "Each bar counts records at that damage level (labelled by its lower bound); blue bars contain this run's records. So we know how serious it is."}{" "}
          </p>
        </div>

        <div className="chart-card">
          <h3 className="chart-card-header">Records by severity</h3>
          <BarChart data={severityCounts} />
          <p className="chart-caption">
            Normal: below {SHM_ELEVATED_THRESHOLD}. Elevated: {SHM_ELEVATED_THRESHOLD} to below {SHM_CRITICAL_THRESHOLD}.
            Critical: {SHM_CRITICAL_THRESHOLD} and above.
          </p>
        </div>
      </div>

      <div className="chart-card">
        <h3 className="chart-card-header">Predicted cumulative damage per file (sorted, coloured by severity)</h3>
        <BarChart
          data={sorted.map((r) => ({
            label: r.file_id,
            value: r.value,
            color: shmSeverity(r.value).color,
          }))}
          valueFormatter={(v) => v.toFixed(2)}
        />
      </div>
    </>
  );
}

export const CHARTS = {
  door: DoorChart,
  acv: AcvChart,
  rail_corrugation: RailChart,
  shm: ShmChart,
};
