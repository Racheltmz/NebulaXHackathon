import { useEffect, useState } from "react";

import apiClient from "../lib/apiClient";
import BarChart from "./charts/BarChart";
import DoorTelemetry from "./charts/DoorTelemetry";
import RailTelemetry from "./charts/RailTelemetry";
import DoorTimeline from "./charts/DoorTimeline";
import AcvTelemetry from "./charts/AcvTelemetry";
import TrainDiagram, { carsFromRanking } from "./charts/TrainDiagram";
import { InfoIcon } from "./icons/NavIcons";
import { StatusBadge } from "./ResultsTable";
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
      "Each segment is rated Normal or Abnormal resistance from the motor and door position signals recorded during that cycle.",
      "Judge the door by how many segments are abnormal compared with the total detected, not by any single segment. One abnormal cycle among hundreds is probably noise.",
      "A growing share of abnormal cycles suggests a real developing fault rather than noise.",
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
      "Each file gets one result covering both rails, so a file is never rated for one rail only.",
      "To interpret a result, look at the axle boxes, the averaged spectrum, the vibration and shock of a single box, and the rotating speed. Compare the two rails rather than trusting any one high reading, and check the speed first, because a faster train makes every box vibrate harder.",
    ],
  },
  shm: {
    title: "Understanding cumulative damage",
    points: [
      "This value comes from Miner's linear cumulative damage rule applied to stress cycles from the uploaded signal.",
      "It ranges from 0 (no accumulated fatigue damage) up to 1.0 (the point at which fatigue failure is expected).",
      "The higher the value, the more of the component's fatigue life has already been used up, and the component should be inspected.",
    ],
  },
};

function UnderstandingCard({ title, points, children }) {
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
      {children}
    </div>
  );
}

// Background caveats and what each status means, for the subsystems with no chart to put beside
// them — the counterpart of ACV's car model disclaimer, in the same layout. Only SHM is graded
// for severity, so these two explain their status labels instead.
const SUBSYSTEM_NOTES = {
  door: {
    notes: [
      {
        term: "One continuous stream",
        detail:
          "Each file is a continuous recording of many door cycles back to back, so segments are found first and then classified.",
      },
      {
        term: "Doors differ",
        detail:
          "Motor readings that are normal on one door can be abnormal on another, so slight resistance faults are easy to miss with a fixed threshold.",
      },
      {
        term: "Open or close is not predicted",
        detail: "Each segment is only rated Normal or Abnormal resistance, not labelled as opening or closing.",
      },
    ],
    statuses: [
      { term: "Normal", detail: "The door moved freely through the cycle." },
      {
        term: "Abnormal resistance",
        detail:
          "The motor met extra resistance, for example from a foreign object in the slide rail, a jammed rubber strip or a deformed door leaf. Repeated abnormal cycles can lead to door jamming and motor overload, so inspect that door.",
      },
    ],
  },
  rail_corrugation: {
    notes: [
      {
        term: "Two rails, one recording",
        detail:
          "Side I and Side II are judged independently from the same recording: axle box positions 1, 3, 5 and 7 sit on the Side I rail, and 2, 4, 6 and 8 on the Side II rail.",
      },
      {
        term: "One side at a time",
        detail: "A file with corrugation on one side is reported as that side only, and the other side is normal.",
      },
      {
        term: "Faults are rare",
        detail:
          "Corrugation is a small minority of the training files, so Side I and Side II results rest on far fewer examples than Normal.",
      },
    ],
    statuses: [
      { term: "Normal", detail: "Both rails are healthy." },
      {
        term: "Side I",
        detail:
          "Corrugation (wave like wear) on the Side I rail, while the Side II rail is normal. Inspect that side of the track.",
      },
      {
        term: "Side II",
        detail:
          "Corrugation (wave like wear) on the Side II rail, while the Side I rail is normal. Inspect that side of the track.",
      },
    ],
  },
};

function NotesList({ notes }) {
  return (
    <div>
      <h4>Things to know</h4>
      <dl className="car-model-list">
        {notes.map(({ term, detail }) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{detail}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function StatusList({ statuses }) {
  return (
    <dl className="car-model-list">
      {statuses.map(({ term, detail }) => (
        <div key={term}>
          {/* the same pill the History table and dashboards use, so a status maps by sight */}
          <dt className="status-term">
            <StatusBadge label={term} />
          </dt>
          <dd>{detail}</dd>
        </div>
      ))}
    </dl>
  );
}

function SubsystemNotes({ notes, statuses }) {
  return (
    <div className="car-model-disclaimer">
      <NotesList notes={notes} />
      <div>
        <h4>What each status means</h4>
        <StatusList statuses={statuses} />
      </div>
    </div>
  );
}

function DoorChart({ job, aggregate }) {
  const normal = job.rows.filter((r) => r.label !== "Abnormal resistance").length;
  const abnormal = job.rows.length - normal;
  const fileName = job.input_files?.[0]?.filename;
  // The backend rebuilds a run's signals from its saved upload the first time it is opened, so a
  // run has none only if its file wasn't saved; that one keeps the plain timeline.
  const telemetry = fileName ? job.summary?.telemetry?.[fileName] : undefined;
  return (
    <>
      <UnderstandingCard {...UNDERSTANDING.door}>
        <SubsystemNotes {...SUBSYSTEM_NOTES.door} />
      </UnderstandingCard>
      {aggregate ? (
        // No merged view: each file is a separate door stream, and stretching segments from
        // different doors onto one time axis says nothing about any of them.
        <p className="history-dashboard-note">
          Each file is a separate door stream, so segments from different files aren&apos;t shown together.
          Open a run&apos;s dashboard from the table below to see its signals and segments.
        </p>
      ) : (
        <>
          <div className="stat-tiles stat-tiles-fill">
            <StatTile label="Total Segments Detected" value={job.rows.length} />
            <StatTile label="Normal Segments" value={normal} />
            <StatTile label="Abnormal Resistance Segments" value={abnormal} />
          </div>
          <div className="chart-card">
            <h3>
              {telemetry
                ? `${fileName} door signals`
                : fileName
                  ? `${fileName} segment timeline`
                  : "Segment timeline"}
            </h3>
            {telemetry ? (
              <DoorTelemetry telemetry={telemetry} rows={job.rows} />
            ) : (
              <>
                <DoorTimeline rows={job.rows} />
                <p className="chart-caption">
                  This run&apos;s signals aren&apos;t available (its file wasn&apos;t saved), so only the segments are
                  shown. Upload the file again to see them.
                </p>
              </>
            )}
          </div>
        </>
      )}
    </>
  );
}

// The three car models in the provided ACV data (train type, from each file's "Car model"
// column — not to be confused with the 8 cars of a train). Order is the count plot's x-axis order.
const CAR_MODELS = [
  { id: "A", description: "8 parameters per car, sampled every 30 seconds, including an outdoor average temperature." },
  {
    id: "B",
    description:
      "63 parameters per car, sampled every 10 seconds. A far richer set, including refrigeration pressures and compressor faults.",
  },
  {
    id: "C",
    description: "The same 8 parameters as model A every 30 seconds, but outdoor temperature is a raw sensor reading rather than an average.",
  },
];
const UNKNOWN_CAR_MODEL = "Unknown";

// Files per car model. Known models are always listed, even at zero, so the plot shows the full
// set; a model outside A–C (or a run recorded before car model was saved) still gets its own bar.
function countCarModels(carModels) {
  const counts = new Map(CAR_MODELS.map(({ id }) => [id, 0]));
  Object.values(carModels ?? {}).forEach((model) => {
    const id = model ?? UNKNOWN_CAR_MODEL;
    counts.set(id, (counts.get(id) ?? 0) + 1);
  });
  if (counts.get(UNKNOWN_CAR_MODEL) === 0) counts.delete(UNKNOWN_CAR_MODEL);
  return [...counts].map(([id, value]) => ({
    label: id === UNKNOWN_CAR_MODEL ? id : `Model ${id}`,
    value,
    color: value > 0 ? "var(--fe-cobalt)" : "var(--fe-edge)",
  }));
}

function AcvUnderstanding({ carModels, fileCount }) {
  return (
    <UnderstandingCard {...UNDERSTANDING.acv}>
      <div className="car-model-disclaimer">
        <div>
          <h4>Disclaimer: car models differ</h4>
          <p>
            The data comes from three car models, and they don&apos;t record the same things. A ranking
            is only comparable with other files of the same model, and models B and C have far fewer
            labelled cases to learn from than model A.
          </p>
          <dl className="car-model-list">
            {CAR_MODELS.map(({ id, description }) => (
              <div key={id}>
                <dt>Model {id}</dt>
                <dd>{description}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div>
          <h4>
            Files by car model{" "}
            <span className="heading-note">
              ({fileCount} {fileCount === 1 ? "File" : "Files"} Analysed)
            </span>
          </h4>
          <BarChart data={countCarModels(carModels)} />
          <p className="chart-caption">Each bar counts the files analysed for that car model.</p>
        </div>
      </div>
    </UnderstandingCard>
  );
}

// Runs from before telemetry was recorded have none saved; a run whose file had no usable Time
// column has a null entry.
function AcvTelemetrySection({ telemetry, fileId, ranked }) {
  const forFile = telemetry?.[fileId];
  if (forFile) return <AcvTelemetry telemetry={forFile} rankedCars={ranked} />;
  return (
    <div className="chart-card">
      <h3>Car telemetry</h3>
      <p>
        {telemetry === undefined
          ? "Telemetry wasn't saved for this run. Upload the file again to see it."
          : "Telemetry couldn't be built from this file."}
      </p>
    </div>
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
        <AcvUnderstanding carModels={job.summary?.car_models} fileCount={job.rows.length} />
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
      <AcvUnderstanding carModels={job.summary?.car_models} fileCount={job.rows.length} />
      {Object.entries(byFile).map(([fileId, ranked]) => (
        <div key={fileId}>
          <div className="chart-card">
            <h3>
              {fileId} — cars ranked most → least likely faulty
              <span className="highlight-pill">Most likely faulty: Car {carsFromRanking(ranked)[0]?.id ?? "—"}</span>
            </h3>
            <TrainDiagram cars={carsFromRanking(ranked)} />
          </div>
          <AcvTelemetrySection telemetry={job.summary?.telemetry} fileId={fileId} ranked={ranked} />
        </div>
      ))}
    </>
  );
}

function RailChart({ job, aggregate }) {
  const counts = { Normal: 0, "Side I": 0, "Side II": 0 };
  job.rows.forEach((r) => {
    if (counts[r.label] !== undefined) counts[r.label] += 1;
  });

  const fileName = job.input_files?.[0]?.filename;
  // The backend rebuilds a run's axle-box signals from its saved upload the first time it is
  // opened, so a run has none only if its file wasn't saved.
  const telemetry = fileName ? job.summary?.telemetry?.[fileName] : undefined;

  return (
    <>
      <UnderstandingCard {...UNDERSTANDING.rail_corrugation}>
        {aggregate ? (
          // The status meanings sit beside the class distribution below, so the notes get the full width
          <div className="car-model-disclaimer car-model-disclaimer-single">
            <NotesList notes={SUBSYSTEM_NOTES.rail_corrugation.notes} />
          </div>
        ) : (
          <SubsystemNotes {...SUBSYSTEM_NOTES.rail_corrugation} />
        )}
      </UnderstandingCard>
      {aggregate ? (
        // Unlike ACV and Door, every rail file is the same measurement of a different track
        // section, so counting them together is meaningful.
        <div className="rail-summary-row">
          <div className="chart-card shm-hero-card rail-records-card">
            <div className="shm-hero-value">{job.rows.length}</div>
            <div className="shm-hero-label">Number of Records</div>
          </div>
          <div className="chart-card rail-class-card">
            <h3 className="chart-card-header">Class distribution</h3>
            <BarChart
              data={Object.entries(counts).map(([label, value]) => ({
                label,
                value,
                color: label === "Normal" ? "var(--status-normal)" : "var(--status-abnormal)",
              }))}
            />
            <p className="chart-caption">
              Each bar counts files with that result. Open a run&apos;s dashboard from the table below to
              see its 64 axle boxes.
            </p>
          </div>
          <div className="chart-card rail-status-card">
            <h3 className="chart-card-header">What each status means</h3>
            <StatusList statuses={SUBSYSTEM_NOTES.rail_corrugation.statuses} />
          </div>
        </div>
      ) : telemetry ? (
        <RailTelemetry telemetry={telemetry} />
      ) : (
        <div className="chart-card">
          <h3>Axle-box signals</h3>
          <p>
            This run&apos;s signals aren&apos;t available (its file wasn&apos;t saved). Upload the file again to
            see its 64 axle boxes.
          </p>
        </div>
      )}
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
