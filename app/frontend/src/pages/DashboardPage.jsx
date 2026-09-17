import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import BarChart from "../components/charts/BarChart";
import DoorTimeline from "../components/charts/DoorTimeline";
import ResultsTable from "../components/ResultsTable";
import StatTile from "../components/StatTile";
import apiClient from "../lib/apiClient";
import { downloadJobCsv } from "../lib/downloadJob";

const LABELS = {
  door: "Door",
  acv: "ACV",
  rail_corrugation: "Rail Corrugation",
  shm: "SHM",
};

function DoorChart({ job }) {
  const normal = job.rows.filter((r) => r.label !== "Abnormal resistance").length;
  const abnormal = job.rows.length - normal;
  return (
    <>
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

function AcvChart({ job }) {
  const byFile = {};
  job.rows.forEach((r) => {
    byFile[r.file_id] = (r.ranked_cars || "").split("|").filter(Boolean);
  });

  return (
    <>
      <div className="stat-tiles">
        <StatTile label="Files analysed" value={job.rows.length} />
        <StatTile label="Most likely faulty (first file)" value={Object.values(byFile)[0]?.[0] ?? "—"} />
      </div>
      {Object.entries(byFile).map(([fileId, cars]) => (
        <div className="chart-card" key={fileId}>
          <h3>{fileId} — cars ranked most → least likely faulty</h3>
          <BarChart
            data={cars.map((car, i) => ({ label: car, value: cars.length - i }))}
            barColor="var(--fe-cobalt)"
          />
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

function ShmChart({ job }) {
  const values = job.rows.map((r) => r.value ?? 0);
  const sorted = [...job.rows].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

  return (
    <>
      <div className="stat-tiles">
        <StatTile label="Files" value={job.rows.length} />
        <StatTile label="Mean damage" value={(job.summary.mean ?? 0).toFixed?.(4) ?? job.summary.mean} />
        <StatTile label="Max damage" value={(job.summary.max ?? 0).toFixed?.(4) ?? job.summary.max} />
      </div>
      <div className="chart-card">
        <h3>Predicted cumulative damage per file (sorted, failure threshold = 1.0)</h3>
        <BarChart
          data={sorted.map((r) => ({
            label: r.file_id,
            value: r.value,
            color: r.value >= 1 ? "var(--status-abnormal)" : "var(--fe-cobalt)",
          }))}
          valueFormatter={(v) => v.toFixed(2)}
        />
      </div>
      {values.length === 0 && <p>No files to chart.</p>}
    </>
  );
}

const CHARTS = {
  door: DoorChart,
  acv: AcvChart,
  rail_corrugation: RailChart,
  shm: ShmChart,
};

export default function DashboardPage() {
  const { jobId } = useParams();
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    apiClient
      .get(`/api/jobs/${jobId}`)
      .then((res) => setJob(res.data))
      .catch((err) => setError(err.response?.data?.detail || err.message));
  }, [jobId]);

  if (error) return <div className="page-container error-banner">{error}</div>;
  if (!job) return <div className="page-loading">Loading…</div>;

  const ChartComponent = CHARTS[job.subsystem];

  return (
    <div className="page-container">
      <div className="dashboard-header">
        <div>
          <h1>{LABELS[job.subsystem] ?? job.subsystem} run</h1>
          <div className="dashboard-meta">Run on {new Date(job.created_at).toLocaleString()}</div>
        </div>
        <button className="btn-secondary" type="button" onClick={() => downloadJobCsv(job.id, job.subsystem)}>
          Download predictions.csv
        </button>
      </div>

      {ChartComponent && <ChartComponent job={job} />}

      <div className="chart-card">
        <h3>All predictions</h3>
        <ResultsTable subsystem={job.subsystem} rows={job.rows} />
      </div>
    </div>
  );
}
