function StatusBadge({ label }) {
  const cls = label === "Abnormal resistance" ? "abnormal" : label === "Side I" ? "side1" : label === "Side II" ? "side2" : "normal";
  return <span className={`badge ${cls}`}>{label}</span>;
}

export default function ResultsTable({ subsystem, rows }) {
  if (!rows || rows.length === 0) return <p>No rows to show.</p>;

  if (subsystem === "door") {
    return (
      <table className="data-table">
        <thead>
          <tr>
            <th>Start</th>
            <th>End</th>
            <th>Prediction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.start_time}</td>
              <td>{r.end_time}</td>
              <td>
                <StatusBadge label={r.label} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  if (subsystem === "acv") {
    return (
      <table className="data-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Ranked cars (most → least likely)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.file_id}</td>
              <td>{r.ranked_cars}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  if (subsystem === "rail_corrugation") {
    return (
      <table className="data-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Prediction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.file_id}</td>
              <td>
                <StatusBadge label={r.label} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  // shm
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>File</th>
          <th>Predicted cumulative damage</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td>{r.file_id}</td>
            <td>{r.value?.toFixed?.(4) ?? r.value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
