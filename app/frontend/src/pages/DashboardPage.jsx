import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import ResultsTable from "../components/ResultsTable";
import { CHARTS } from "../components/SubsystemCharts";
import apiClient from "../lib/apiClient";

const LABELS = {
  door: "Door",
  acv: "ACV",
  rail_corrugation: "Rail Corrugation",
  shm: "SHM",
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

  if (error) {
    return (
      <div className="page-container">
        <Link className="back-link" to="/history">
          ← Back to History
        </Link>
        <div className="error-banner">{error}</div>
      </div>
    );
  }
  if (!job) return <div className="page-loading">Loading…</div>;

  const ChartComponent = CHARTS[job.subsystem];

  return (
    <div className="page-container dashboard-page-wide">
      <Link className="back-link" to={`/history?subsystem=${job.subsystem}`}>
        ← Back to {LABELS[job.subsystem] ?? job.subsystem} history
      </Link>
      <div className="dashboard-header">
        <div>
          <h1>{LABELS[job.subsystem] ?? job.subsystem} run</h1>
          <div className="dashboard-meta">Ran on {new Date(job.created_at).toLocaleString()}</div>
        </div>
      </div>

      {ChartComponent && <ChartComponent job={job} />}

      <div className="chart-card">
        <h3>All predictions</h3>
        <ResultsTable subsystem={job.subsystem} rows={job.rows} />
      </div>
    </div>
  );
}
