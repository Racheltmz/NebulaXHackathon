import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import apiClient from "../lib/apiClient";
import { downloadJobCsv } from "../lib/downloadJob";

const SUBSYSTEM_LABELS = {
  door: "Door",
  acv: "ACV",
  rail_corrugation: "Rail Corrugation",
  shm: "SHM",
};

export default function HistoryPage() {
  const [jobs, setJobs] = useState([]);
  const [filter, setFilter] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const params = filter ? { subsystem: filter } : {};
    apiClient
      .get("/api/history", { params })
      .then((res) => setJobs(res.data))
      .catch((err) => setError(err.response?.data?.detail || err.message));
  }, [filter]);

  return (
    <div className="page-container history-page">
      <h1>History</h1>
      <p style={{ color: "var(--db-muted)", fontSize: 14 }}>Every run, newest first.</p>

      <div className="history-filters">
        <button className={!filter ? "active" : ""} type="button" onClick={() => setFilter(null)}>
          All
        </button>
        {Object.entries(SUBSYSTEM_LABELS).map(([key, label]) => (
          <button
            key={key}
            className={filter === key ? "active" : ""}
            type="button"
            onClick={() => setFilter(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="history-table-card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Subsystem</th>
              <th>Date</th>
              <th>Status</th>
              <th>Download</th>
              <th>Dashboard</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td>{SUBSYSTEM_LABELS[job.subsystem] ?? job.subsystem}</td>
                <td>{new Date(job.created_at).toLocaleString()}</td>
                <td>
                  <span className={`badge ${job.status === "done" ? "normal" : "abnormal"}`}>{job.status}</span>
                </td>
                <td>
                  <button className="link-button" type="button" onClick={() => downloadJobCsv(job.id, job.subsystem)}>
                    Download
                  </button>
                </td>
                <td>
                  <Link className="link-button" to={`/jobs/${job.id}`}>
                    View →
                  </Link>
                </td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr>
                <td colSpan={5} style={{ textAlign: "center", color: "var(--db-muted)" }}>
                  No runs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
