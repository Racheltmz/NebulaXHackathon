export default function PreviewTable({ columns, rows, truncated }) {
  if (!rows || rows.length === 0) return <p>No rows to show.</p>;

  return (
    <>
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((c, i) => (
              <th key={i}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell === null || cell === undefined ? "" : String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {truncated && <p className="preview-truncated-note">Showing the first {rows.length} rows only.</p>}
    </>
  );
}
