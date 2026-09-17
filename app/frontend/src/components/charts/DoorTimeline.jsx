/** Renders detected door-cycle segments as colored spans along a single horizontal timeline.
 * Accepts either the dataset's native Year-Month-Date-Hour-Minute-Second-Millisecond format
 * (hyphen-separated, not zero-padded, e.g. "2023-7-5-0-0-3-760") or an ISO-parseable timestamp,
 * matching what the Door subsystem's output schema itself accepts (see docs/DESIGN.md Section 6).
 */
function parseTimestamp(value) {
  const parts = String(value).split("-");
  if (parts.length === 7) {
    const [year, month, day, hour, minute, second, ms] = parts.map(Number);
    return Date.UTC(year, month - 1, day, hour, minute, second, ms);
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

export default function DoorTimeline({ rows }) {
  if (!rows || rows.length === 0) return <p>No segments detected.</p>;

  const spans = rows.map((r) => ({
    start: parseTimestamp(r.start_time),
    end: parseTimestamp(r.end_time),
    label: r.label,
  }));

  const min = Math.min(...spans.map((s) => s.start));
  const max = Math.max(...spans.map((s) => s.end));
  const range = Math.max(max - min, 1);

  return (
    <div>
      <div className="timeline">
        {spans.map((s, i) => (
          <div
            key={i}
            className={`timeline-segment ${s.label === "Abnormal resistance" ? "abnormal" : "normal"}`}
            style={{
              left: `${((s.start - min) / range) * 100}%`,
              width: `${Math.max(((s.end - s.start) / range) * 100, 0.3)}%`,
            }}
            title={`${s.label}: ${new Date(s.start).toISOString()} → ${new Date(s.end).toISOString()}`}
          />
        ))}
      </div>
      <div className="legend">
        <span>
          <span className="legend-dot" style={{ background: "var(--status-normal)" }} />
          Normal
        </span>
        <span>
          <span className="legend-dot" style={{ background: "var(--status-abnormal)" }} />
          Abnormal resistance
        </span>
      </div>
    </div>
  );
}
