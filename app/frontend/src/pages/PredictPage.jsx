import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import FileDropzone from "../components/FileDropzone";
import FormatPanel from "../components/FormatPanel";
import ResultsTable from "../components/ResultsTable";
import SubsystemSelector from "../components/SubsystemSelector";
import apiClient from "../lib/apiClient";
import { downloadJobCsv } from "../lib/downloadJob";

export default function PredictPage() {
  const [subsystems, setSubsystems] = useState([]);
  const [selectedKey, setSelectedKey] = useState(null);
  const [files, setFiles] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
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
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selected || files.length === 0) return;
    setSubmitting(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    try {
      const res = await apiClient.post(`/api/predict/${selected.key}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setSubmitting(false);
    }
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

        {error && <div className="error-banner">{error}</div>}

        <button
          className="btn-primary"
          type="button"
          onClick={handleSubmit}
          disabled={!selected || files.length === 0 || submitting}
        >
          {submitting ? "Running prediction…" : "Run prediction"}
        </button>

        {result && (
          <div className="result-panel">
            <div className="result-actions">
              <button className="btn-secondary" type="button" onClick={() => downloadJobCsv(result.job_id, selected.key)}>
                Download predictions.csv
              </button>
              <Link className="btn-lime" to={`/jobs/${result.job_id}`}>
                View dashboard
              </Link>
            </div>
            <ResultsTable subsystem={selected.key} rows={result.rows} />
          </div>
        )}
      </div>
    </div>
  );
}
