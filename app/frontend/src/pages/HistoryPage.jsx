import { useEffect, useMemo, useState } from "react";

import Modal from "../components/Modal";
import PreviewTable from "../components/PreviewTable";
import ResultsTable, { StatusBadge } from "../components/ResultsTable";
import { CHARTS } from "../components/SubsystemCharts";
import apiClient from "../lib/apiClient";
import { downloadInputFile, downloadSubsystemCsv } from "../lib/downloadJob";

const SUBSYSTEM_LABELS = {
  door: "Door",
  acv: "ACV",
  rail_corrugation: "Rail Corrugation",
  shm: "SHM",
};

const SEVERITY_LABELS = { normal: "Normal", elevated: "Elevated", critical: "Critical" };
const SEVERITY_BADGE = { normal: "normal", elevated: "elevated", critical: "abnormal" };
const SEVERITY_RANK = { normal: 0, elevated: 1, critical: 2 };

function SeverityCell({ job }) {
  if (!job.severity) {
    return (
      <span style={{ color: "var(--db-muted)" }} title="This subsystem's output can't be graded for severity">
        —
      </span>
    );
  }
  return (
    <span className={`badge ${SEVERITY_BADGE[job.severity.level]}`} title={job.severity.detail}>
      {SEVERITY_LABELS[job.severity.level] ?? job.severity.level}
    </span>
  );
}

// Sort keys for the sortable columns. Predictions is an action button, so they have none.
const SORT_KEYS = {
  subsystem: (job) => SUBSYSTEM_LABELS[job.subsystem] ?? job.subsystem,
  date: (job) => new Date(job.created_at).getTime(),
  status: (job) => job.status,
  severity: (job) => SEVERITY_RANK[job.severity?.level] ?? -1, // ungradable runs (ACV) sort first
  files: (job) => job.input_files[0]?.filename ?? "",
};

// Per-prediction table (a subsystem tab): one row per file/segment. The uploaded filename already
// has its own column, so ACV/Rail/SHM only add their single output; Door adds its segment breakdown.
// Door's operation (Open/Close) and n_rows aren't produced by the model or stored yet.
const PREDICTION_COLUMNS = {
  door: [
    {
      key: "range",
      header: "Date range",
      sortValue: (r) => r.start_time ?? "",
      render: (r) => `${r.start_time} → ${r.end_time}`,
    },
    { key: "label", header: "Status", sortValue: (r) => r.label ?? "", render: (r) => <StatusBadge label={r.label} /> },
  ],
  acv: [
    {
      key: "cars",
      header: "Ranked cars (most → least likely)",
      sortValue: (r) => r.ranked_cars ?? "",
      render: (r) => r.ranked_cars,
    },
  ],
  rail_corrugation: [
    { key: "label", header: "Prediction", sortValue: (r) => r.label ?? "", render: (r) => <StatusBadge label={r.label} /> },
  ],
  shm: [
    {
      key: "value",
      header: "Predicted cumulative damage",
      sortValue: (r) => r.value ?? -1,
      render: (r) => r.value?.toFixed?.(4) ?? r.value,
    },
  ],
};

const ROW_SORT_KEYS = {
  date: (r) => new Date(r.created_at).getTime(),
  severity: (r) => SEVERITY_RANK[r.severity?.level] ?? -1,
  files: (r) => r.input_file_name ?? "",
};

function compareValues(a, b) {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

function SortableTh({ column, sort, onSort, children }) {
  const active = sort.column === column;
  return (
    <th aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}>
      <button className={`sort-button ${active ? "active" : ""}`} type="button" onClick={() => onSort(column)}>
        {children}
        <span className="sort-arrow" aria-hidden="true">
          {active ? (sort.direction === "asc" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );
}

function InputFilesCell({ job, onPreview }) {
  if (job.input_files.length === 0) return <span style={{ color: "var(--db-muted)" }}>—</span>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {job.input_files.map((f, index) =>
        f.storage_path ? (
          <div key={index} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span>{f.filename}</span>
            <button
              className="link-button"
              type="button"
              onClick={() => onPreview({ type: "input", jobId: job.id, index, filename: f.filename })}
            >
              Preview
            </button>
          </div>
        ) : (
          <span key={index} style={{ color: "var(--db-muted)" }} title="Not saved to storage for this run">
            {f.filename} (unavailable)
          </span>
        )
      )}
    </div>
  );
}

function RowFileCell({ row, onPreview }) {
  if (!row.input_file_name) return <span style={{ color: "var(--db-muted)" }}>—</span>;
  if (!row.input_file_available) {
    return (
      <span style={{ color: "var(--db-muted)" }} title="Not saved to storage for this run">
        {row.input_file_name} (unavailable)
      </span>
    );
  }
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span>{row.input_file_name}</span>
      <button
        className="link-button"
        type="button"
        onClick={() =>
          onPreview({
            type: "input",
            jobId: row.job_id,
            index: row.input_file_index,
            filename: row.input_file_name,
          })
        }
      >
        Preview
      </button>
    </div>
  );
}

export default function HistoryPage() {
  const [jobs, setJobs] = useState([]);
  const [rows, setRows] = useState(null); // per-prediction rows for the selected subsystem
  const [filter, setFilter] = useState(null);
  const [error, setError] = useState(null);
  const [dashboard, setDashboard] = useState(null); // { runs, rows } for the selected subsystem
  const [dashboardError, setDashboardError] = useState(null);
  const [sort, setSort] = useState({ column: "date", direction: "desc" });
  const [downloadKey, setDownloadKey] = useState("door");
  const [downloadError, setDownloadError] = useState(null);

  const [preview, setPreview] = useState(null); // { type: "input" | "predictions", jobId, ... }
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);

  useEffect(() => {
    setError(null);
    if (filter) return;

    let stale = false;
    apiClient
      .get("/api/history")
      .then((res) => !stale && setJobs(res.data))
      .catch((err) => !stale && setError(err.response?.data?.detail || err.message));
    return () => {
      stale = true;
    };
  }, [filter]);

  useEffect(() => {
    setRows(null);
    if (!filter) return;

    let stale = false;
    apiClient
      .get(`/api/history/${filter}/rows`)
      .then((res) => !stale && setRows(res.data))
      .catch((err) => !stale && setError(err.response?.data?.detail || err.message));
    return () => {
      stale = true;
    };
  }, [filter]);

  useEffect(() => {
    setDashboard(null);
    setDashboardError(null);
    if (!filter) return;

    let stale = false;
    apiClient
      .get("/api/history/dashboard", { params: { subsystem: filter } })
      .then((res) => !stale && setDashboard(res.data))
      .catch((err) => !stale && setDashboardError(err.response?.data?.detail || err.message));
    return () => {
      stale = true;
    };
  }, [filter]);

  useEffect(() => {
    if (!preview) return;
    setPreviewData(null);
    setPreviewError(null);
    setPreviewLoading(true);

    const request =
      preview.type === "input"
        ? apiClient.get(`/api/jobs/${preview.jobId}/input-files/${preview.index}/preview`)
        : apiClient.get(`/api/jobs/${preview.jobId}`);

    request
      .then((res) => setPreviewData(res.data))
      .catch((err) => setPreviewError(err.response?.data?.detail || err.message))
      .finally(() => setPreviewLoading(false));
  }, [preview]);

  const sortedJobs = useMemo(() => {
    const keyFn = SORT_KEYS[sort.column];
    const sign = sort.direction === "asc" ? 1 : -1;
    return [...jobs].sort((a, b) => sign * compareValues(keyFn(a), keyFn(b)));
  }, [jobs, sort]);

  const predictionColumns = filter ? PREDICTION_COLUMNS[filter] : [];

  const sortedRows = useMemo(() => {
    if (!rows) return [];
    const keyFn =
      ROW_SORT_KEYS[sort.column] ?? predictionColumns.find((c) => `pred:${c.key}` === sort.column)?.sortValue;
    if (!keyFn) return rows;
    const sign = sort.direction === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => sign * compareValues(keyFn(a), keyFn(b)));
  }, [rows, sort, predictionColumns]);

  const selectFilter = (key) => {
    setFilter(key);
    setSort({ column: "date", direction: "desc" });
  };

  const handleSort = (column) =>
    setSort((prev) =>
      prev.column === column
        ? { column, direction: prev.direction === "asc" ? "desc" : "asc" }
        : { column, direction: "asc" }
    );

  const Chart = filter ? CHARTS[filter] : null;

  const closePreview = () => setPreview(null);

  const handleDownload = () => {
    setDownloadError(null);
    downloadSubsystemCsv(downloadKey).catch((err) => setDownloadError(err.message));
  };

  return (
    <div className="page-container dashboard-page-wide history-page">
      <h1>History</h1>
      <p style={{ color: "var(--db-muted)", fontSize: 14 }}>Every run: click a column header to sort.</p>

      <div className="download-panel">
        <div>
          <strong>Download predictions</strong>
          <p>One CSV per subsystem covering all its runs, in the submission format (goes into predictions.zip).</p>
        </div>
        <select value={downloadKey} onChange={(e) => setDownloadKey(e.target.value)} aria-label="Subsystem to download">
          {Object.entries(SUBSYSTEM_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
        <button className="btn-secondary" type="button" onClick={handleDownload}>
          Download {downloadKey}_predictions.csv
        </button>
      </div>
      {downloadError && <div className="error-banner">{downloadError}</div>}

      <div className="history-filters">
        <button className={!filter ? "active" : ""} type="button" onClick={() => selectFilter(null)}>
          All
        </button>
        {Object.entries(SUBSYSTEM_LABELS).map(([key, label]) => (
          <button
            key={key}
            className={filter === key ? "active" : ""}
            type="button"
            onClick={() => selectFilter(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {filter ? (
        <section className="history-dashboard">
          <h2>{SUBSYSTEM_LABELS[filter]} dashboard</h2>
          {dashboard && (
            <p className="history-dashboard-note">
              {dashboard.rows.length} {filter === "door" ? "segment(s)" : "file(s)"} across {dashboard.runs} run(s)
              {filter === "door" ? "" : ": the latest prediction for each file"}.
            </p>
          )}
          {dashboardError && <div className="error-banner">{dashboardError}</div>}
          {!dashboard && !dashboardError && <p>Loading…</p>}
          {dashboard &&
            (dashboard.rows.length > 0 ? (
              <Chart job={{ rows: dashboard.rows, summary: {} }} aggregate />
            ) : (
              <p>No {SUBSYSTEM_LABELS[filter]} runs yet.</p>
            ))}
        </section>
      ) : (
        <p className="history-dashboard-note">Select a subsystem above to see its dashboard.</p>
      )}

      {filter ? (
        <div className="history-table-card">
          <table className="data-table">
            <thead>
              <tr>
                <SortableTh column="date" sort={sort} onSort={handleSort}>
                  Run date
                </SortableTh>
                <SortableTh column="severity" sort={sort} onSort={handleSort}>
                  Severity
                </SortableTh>
                <SortableTh column="files" sort={sort} onSort={handleSort}>
                  Uploaded File
                </SortableTh>
                {predictionColumns.map((c) => (
                  <SortableTh key={c.key} column={`pred:${c.key}`} sort={sort} onSort={handleSort}>
                    {c.header}
                  </SortableTh>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row, i) => (
                <tr key={`${row.job_id}-${i}`}>
                  <td>{new Date(row.created_at).toLocaleString()}</td>
                  <td>
                    <SeverityCell job={row} />
                  </td>
                  <td>
                    <RowFileCell row={row} onPreview={setPreview} />
                  </td>
                  {predictionColumns.map((c) => (
                    <td key={c.key}>{c.render(row)}</td>
                  ))}
                </tr>
              ))}
              {rows !== null && rows.length === 0 && (
                <tr>
                  <td colSpan={3 + predictionColumns.length} style={{ textAlign: "center", color: "var(--db-muted)" }}>
                    No {SUBSYSTEM_LABELS[filter]} runs yet.
                  </td>
                </tr>
              )}
              {rows === null && !error && (
                <tr>
                  <td colSpan={3 + predictionColumns.length} style={{ textAlign: "center", color: "var(--db-muted)" }}>
                    Loading…
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
      <div className="history-table-card">
        <table className="data-table">
          <thead>
            <tr>
              <SortableTh column="subsystem" sort={sort} onSort={handleSort}>
                Subsystem
              </SortableTh>
              <SortableTh column="date" sort={sort} onSort={handleSort}>
                Date
              </SortableTh>
              <SortableTh column="status" sort={sort} onSort={handleSort}>
                Status
              </SortableTh>
              <SortableTh column="severity" sort={sort} onSort={handleSort}>
                Severity
              </SortableTh>
              <SortableTh column="files" sort={sort} onSort={handleSort}>
                Uploaded Files
              </SortableTh>
              <th>Predictions</th>
            </tr>
          </thead>
          <tbody>
            {sortedJobs.map((job) => (
              <tr key={job.id}>
                <td>{SUBSYSTEM_LABELS[job.subsystem] ?? job.subsystem}</td>
                <td>{new Date(job.created_at).toLocaleString()}</td>
                <td>
                  <span className={`badge ${job.status === "done" ? "normal" : "abnormal"}`}>{job.status}</span>
                </td>
                <td>
                  <SeverityCell job={job} />
                </td>
                <td>
                  <InputFilesCell job={job} onPreview={setPreview} />
                </td>
                <td>
                  <div style={{ display: "flex", gap: 12 }}>
                    <button
                      className="link-button"
                      type="button"
                      onClick={() => setPreview({ type: "predictions", jobId: job.id, subsystem: job.subsystem })}
                    >
                      Preview
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", color: "var(--db-muted)" }}>
                  No runs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      )}

      {preview && (
        <Modal
          title={preview.type === "input" ? preview.filename : "Predictions preview"}
          onClose={closePreview}
        >
          {previewLoading && <p>Loading preview…</p>}
          {previewError && <div className="error-banner">{previewError}</div>}
          {previewData && preview.type === "input" && (
            <PreviewTable columns={previewData.columns} rows={previewData.rows} truncated={previewData.truncated} />
          )}
          {previewData && preview.type === "predictions" && (
            <ResultsTable subsystem={previewData.subsystem} rows={previewData.rows} />
          )}
          {previewData && preview.type === "input" && (
            <div style={{ marginTop: 16 }}>
              <button
                className="btn-secondary"
                type="button"
                onClick={() => downloadInputFile(preview.jobId, preview.index, preview.filename)}
              >
                Download
              </button>
            </div>
          )}
        </Modal>
      )}
    </div>
  );
}
