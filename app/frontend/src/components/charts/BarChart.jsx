/** Minimal dependency-free bar chart — v1 dashboards are intentionally simple
 * (docs/DESIGN.md Section 7.2). `data` is [{ label, value }].
 *
 * `tight` renders it as a histogram: bars touch with no gaps, empty bins draw as a flat baseline
 * (not a stub bar), and since there can be many bins only every Nth x-label / non-zero count is
 * printed so they don't collide. */
export default function BarChart({ data, valueFormatter = (v) => v, barColor, tight = false }) {
  if (!data || data.length === 0) return <p>No data to chart.</p>;

  const max = Math.max(...data.map((d) => d.value), 1);
  const labelStep = tight ? Math.ceil(data.length / 10) : 1;
  // Hidden text is a non-breaking space, not "": an empty span collapses to zero height, which
  // would sit its bar lower than the bars whose label is showing and break the shared baseline.
  const BLANK = "\u00a0";

  return (
    <div className={`bar-chart ${tight ? "tight" : ""}`}>
      {data.map((d, i) => (
        <div className="bar-chart-col" key={`${d.label}-${i}`}>
          <span className="bar-chart-value">{tight && d.value === 0 ? BLANK : valueFormatter(d.value)}</span>
          <div
            className="bar-chart-bar"
            style={{
              height: tight && d.value === 0 ? "2px" : `${Math.max((d.value / max) * 100, 2)}%`,
              background: d.color || barColor,
              ...(tight && d.value === 0 ? { borderRadius: 0 } : {}),
            }}
          />
          <span className="bar-chart-label">{i % labelStep === 0 ? d.label : BLANK}</span>
        </div>
      ))}
    </div>
  );
}
