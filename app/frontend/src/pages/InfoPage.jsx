import { SUBSYSTEM_ICONS } from "../components/icons/SubsystemIcons";

const USAGE_STEPS = [
  "Select the subsystem you want to run predictions for.",
  "Upload (or drag & drop) the test file(s) in the format shown below.",
  "Get predictions back on screen, download the CSV, or open the dashboard.",
];

const SUBSYSTEM_INFO = [
  {
    key: "door",
    label: "Door",
    description:
      "Finds every door open and close cycle in a continuous sensor stream and flags any cycle with abnormal motor resistance.",
  },
  {
    key: "acv",
    label: "ACV",
    description:
      "Identifies which train car has a refrigerant leak from cabin temperature and control mode telemetry.",
  },
  {
    key: "rail_corrugation",
    label: "Rail Corrugation",
    description: "Classifies axle box vibration and shock readings as normal or one of two corrugation types.",
  },
  {
    key: "shm",
    label: "SHM",
    description: "Estimates cumulative fatigue damage on the vehicle structure from dynamic stress readings.",
  },
];

export default function InfoPage() {
  return (
    <div className="page-container info-page">
      <div className="predict-header">
        <h1>Run a prediction</h1>
        <p>
          Pick a subsystem, upload the matching test file(s), and get predictions back, with a
          downloadable CSV and a visual dashboard for the run.
        </p>
      </div>

      <div className="usage-steps">
        {USAGE_STEPS.map((text, i) => (
          <div className="usage-step" key={i}>
            <div className="step-number">{i + 1}</div>
            <p>{text}</p>
          </div>
        ))}
      </div>

      <h2 className="subsystem-info-heading">Subsystems</h2>
      <div className="subsystem-info-grid">
        {SUBSYSTEM_INFO.map((s) => {
          const icon = SUBSYSTEM_ICONS[s.key];
          const Icon = icon?.Icon;
          return (
            <div className="subsystem-info-card" key={s.key}>
              {Icon && (
                <span className="subsystem-info-card-icon" style={{ background: icon.tint, color: icon.color }}>
                  <Icon />
                </span>
              )}
              <h3>{s.label}</h3>
              <p>{s.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
