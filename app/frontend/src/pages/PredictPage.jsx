import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import FileDropzone from "../components/FileDropzone";
import FormatPanel from "../components/FormatPanel";
import SubsystemSelector from "../components/SubsystemSelector";
import apiClient from "../lib/apiClient";

/** One-line result for a finished run — a run is one file, so this is its single prediction
 * (or, for Door, a count of the segments found in that stream). */
function summarizeRun(subsystem, rows) {
  if (subsystem === "door") {
    const abnormal = rows.filter((r) => r.label === "Abnormal resistance").length;
    return `${rows.length} segment(s), ${abnormal} abnormal`;
  }
  const row = rows[0];
  if (!row) return "—";
  if (subsystem === "acv") return row.ranked_cars;
  if (subsystem === "shm") return row.value?.toFixed?.(4) ?? String(row.value);
  return row.label;
}

export default function PredictPage() {
  const [subsystems, setSubsystems] = useState([]);
  const [selectedKey, setSelectedKey] = useState(null);
  const [files, setFiles] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [runs, setRuns] = useState([]); // one entry per uploaded file: { name, status, jobId, rows, error }
  const [subsystemsError, setSubsystemsError] = useState(null);

  useEffect(() => {
    apiClient
      .get("/api/subsystems")
      .then((res) => {
        setSubsystems(res.data);
        setSelectedKey(res.data[0]?.key ?? null);
      })
      .catch((err) => {
        setSubsystemsError(
          err.response?.data?.detail ||
            `Couldn't reach the backend at ${apiClient.defaults.baseURL} — is it running?`
        );
      });
  }, []);

  const selected = subsystems.find((s) => s.key === selectedKey);

  const handleSelect = (key) => {
    setSelectedKey(key);
    setFiles([]);
    setRuns([]);
  };

  const updateRun = (index, patch) =>
    setRuns((prev) => prev.map((run, i) => (i === index ? { ...run, ...patch } : run)));

  // One request per file so each file is its own run in History (and a bad file doesn't sink the
  // rest). Sequential rather than parallel to keep the backend's memory use flat on big batches.
  const handleSubmit = async () => {
    if (!selected || files.length === 0) return;
    const batch = files;
    setSubmitting(true);
    setRuns(batch.map((f) => ({ name: f.name, status: "pending" })));
    setFiles([]);

    for (let i = 0; i < batch.length; i++) {
      updateRun(i, { status: "running" });
      const formData = new FormData();
      formData.append("file", batch[i]);
      try {
        const res = await apiClient.post(`/api/predict/${selected.key}`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        updateRun(i, { status: "done", jobId: res.data.job_id, rows: res.data.rows });
      } catch (err) {
        updateRun(i, { status: "failed", error: err.response?.data?.detail || err.message });
      }
    }
    setSubmitting(false);
  };

  return (
    <div className="form-section">
      <div className="page-container">
        {subsystemsError && <div className="error-banner">{subsystemsError}</div>}
        <SubsystemSelector subsystems={subsystems} selectedKey={selectedKey} onSelect={handleSelect} />
        <FormatPanel subsystem={selected} />

        {selected && (
          <FileDropzone accept={selected.accepted_extensions} files={files} onFilesChange={setFiles} />
        )}

        <button
          className="btn-primary"
          type="button"
          onClick={handleSubmit}
          disabled={!selected || files.length === 0 || submitting}
        >
          {submitting ? "Running predictions…" : files.length > 1 ? `Run ${files.length} predictions` : "Run prediction"}
        </button>

        {runs.length > 0 && (
          <div className="result-panel">
            <table className="data-table">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Status</th>
                  <th>Result</th>
                  <th>Dashboard</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run, i) => (
                  <tr key={i}>
                    <td>{run.name}</td>
                    <td>
                      <span className={`badge ${run.status === "done" ? "normal" : run.status === "failed" ? "abnormal" : ""}`}>
                        {run.status}
                      </span>
                    </td>
                    <td>{run.status === "done" ? summarizeRun(selected.key, run.rows) : run.error ?? "—"}</td>
                    <td>
                      {run.jobId ? (
                        <Link className="link-button" to={`/jobs/${run.jobId}`}>
                          View →
                        </Link>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="preview-truncated-note">
              Each file is saved as its own run. Download predictions from the History page.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
