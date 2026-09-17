const USAGE_STEPS = [
  "Select the subsystem you want to run predictions for.",
  "Upload (or drag & drop) the test file(s) in the format shown below.",
  "Get predictions back on screen, download the CSV, or open the dashboard.",
];

export default function InfoPage() {
  return (
    <div className="page-container">
      <div className="predict-header">
        <h1>Run a prediction</h1>
        <p>
          Pick a subsystem, upload the matching test file(s), and get predictions back — with a
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
    </div>
  );
}
