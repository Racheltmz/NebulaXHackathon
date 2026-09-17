/** Minimal dependency-free bar chart — v1 dashboards are intentionally simple
 * (docs/DESIGN.md Section 7.2). `data` is [{ label, value }]. */
export default function BarChart({ data, valueFormatter = (v) => v, barColor }) {
  if (!data || data.length === 0) return <p>No data to chart.</p>;

  const max = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className="bar-chart">
      {data.map((d) => (
        <div className="bar-chart-col" key={d.label}>
          <span className="bar-chart-value">{valueFormatter(d.value)}</span>
          <div
            className="bar-chart-bar"
            style={{
              height: `${Math.max((d.value / max) * 100, 2)}%`,
              background: d.color || barColor,
            }}
          />
          <span className="bar-chart-label">{d.label}</span>
        </div>
      ))}
    </div>
  );
}
