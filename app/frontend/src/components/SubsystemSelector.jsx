import { SUBSYSTEM_ICONS } from "./icons/SubsystemIcons";

export default function SubsystemSelector({ subsystems, selectedKey, onSelect }) {
  return (
    <div className="subsystem-grid">
      {subsystems.map((s) => {
        const icon = SUBSYSTEM_ICONS[s.key];
        const Icon = icon?.Icon;
        return (
          <button
            key={s.key}
            type="button"
            className={`subsystem-card ${s.key === selectedKey ? "active" : ""}`}
            onClick={() => onSelect(s.key)}
          >
            {Icon && (
              <span className="subsystem-card-icon" style={{ background: icon.tint, color: icon.color }}>
                <Icon />
              </span>
            )}
            <span className="subsystem-card-label">{s.label}</span>
          </button>
        );
      })}
    </div>
  );
}
