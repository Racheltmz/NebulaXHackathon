export default function FormatPanel({ subsystem }) {
  if (!subsystem) return null;

  // schema_hint is `\n`-separated (see backend/ml/subsystems.py) — a lead-in sentence followed
  // by one item per line. Subsystems with no `\n` at all just render as a single paragraph.
  const [intro, ...items] = subsystem.schema_hint.split("\n");

  return (
    <div className="format-panel">
      <h3>
        Required format for {subsystem.label}
        {subsystem.accepted_extensions.map((ext) => (
          <span className="accepted-ext" key={ext}>
            {ext}
          </span>
        ))}
      </h3>
      <p>{intro}</p>
      {items.length > 0 && (
        <ul>
          {items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
