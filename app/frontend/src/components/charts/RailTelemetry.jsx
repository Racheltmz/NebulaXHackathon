import { useMemo, useState } from "react";

import AxleBoxMap from "./AxleBoxMap";
import InterpretNote from "./InterpretNote";
import TelemetryChart from "./TelemetryChart";

/** Axle-box signals for one Rail Corrugation run: a plan view of all 64 boxes, then — for the box
 * you pick — how hard it shakes through the second, and the spectrum that the Side I / Side II call
 * actually rests on. Reads `telemetry` as built by the backend's ml/rail_telemetry.py.
 *
 * Time and frequency are separate charts on purpose: they are different domains, and putting them
 * on one axis would invent a relationship that isn't there. Drawing is in TelemetryChart. */

const VIBRATION = "#2a78d6"; // palette slot 1
const SHOCK = "#eb6834"; // slot 2
const SIDE_I = "#2a78d6";
const SIDE_II = "#eb6834";
const SELECTED = "#4a3aa7"; // slot 7 — validated against both side colours

const formatSeconds = (s) => `${s.toFixed(2)} s`;
const formatHz = (hz) => `${Math.round(hz)} Hz`;

// Reading guides, one per plot. Each claim below was checked against the 272 labelled training files:
// - speed drives the vibration level hard (correlation 0.83; the level nearly triples from under
//   20 km/h to over 55 km/h);
// - shock barely separates the three classes (side ratio 0.91 Normal, 0.93 Side I, 0.86 Side II), so it
//   is described as context and not as evidence;
// - the spectrum peak does NOT move with speed (correlation 0.00, median 54 to 61 Hz in every speed
//   band), so no text here may say that it does.
const HOW_TO_READ = {
  map: [
    { term: "Darker box:", text: "that box shakes harder in the band where corrugation shows up." },
    {
      term: "Dark boxes along one row:",
      text: "point to that rail. One dark box on its own can come from a rail joint, weld or bearing, so it is a reason to look closer, not a verdict.",
    },
    {
      term: "Rotating speed:",
      text: "a faster train makes every box vibrate harder, corrugation or not. In the training files the vibration level nearly triples from under 20 km/h to over 55 km/h, so compare files recorded at similar speeds.",
    },
  ],
  spectrum: [
    {
      term: "Taller line:",
      text: "more vibration at that frequency. A taller peak in the shaded band on one rail suggests corrugation on that rail.",
    },
    {
      term: "Two lines that overlap:",
      text: "the rails look alike, as in a Normal file. Both lines tall together usually means a fast train or noisy track, not a fault.",
    },
    {
      term: "Where the peak sits:",
      text: "in the training files it stays near 55 to 60 Hz at every speed, so judge its height rather than its position.",
    },
  ],
  box: [
    {
      term: "Vibration:",
      text: "a higher level means the box is shaken harder. Corrugation is a repeating wave along the rail, so it tends to keep the level raised for the whole second, while a short tall spike points to a single bump.",
    },
    {
      term: "Shock:",
      text: "a high value means harder knocks between wheel and rail.",
    },
  ],
  boxSpectrum: [
    {
      term: "Tall peak:",
      text: "this box shakes strongly at corrugation frequencies. One box alone cannot say which rail is faulty, so compare it with the side averages above.",
    },
  ],
};

export default function RailTelemetry({ telemetry }) {
  const boxIds = useMemo(() => Object.keys(telemetry.boxes ?? {}), [telemetry]);

  // Open on the box with the strongest corrugation-band reading — the one worth looking at first.
  const [selected, setSelected] = useState(() => {
    let best = null;
    let bestValue = -Infinity;
    Object.entries(telemetry.boxes ?? {}).forEach(([id, box]) => {
      if ((box.peak_amplitude ?? -Infinity) > bestValue) {
        bestValue = box.peak_amplitude ?? -Infinity;
        best = id;
      }
    });
    return best;
  });

  const box = telemetry.boxes?.[selected];
  const sides = telemetry.sides ?? {};
  const bins = box?.vibration?.length ?? telemetry.time_bins ?? 0;
  const duration = telemetry.duration_s ?? 1;

  // Bin centres in seconds, for the envelope chart's axis.
  const times = useMemo(
    () => Array.from({ length: bins }, (_, i) => ((i + 0.5) / bins) * duration),
    [bins, duration],
  );
  const freqs = telemetry.spectrum_freqs ?? [];

  const [bandLow, bandHigh] = telemetry.corrugation_band ?? [];
  // The corrugation band, tinted behind the spectrum so the comparison area is obvious.
  const bands = useMemo(() => {
    if (bandLow == null || !freqs.length) return [];
    const from = freqs.findIndex((f) => f >= bandLow);
    let to = freqs.findIndex((f) => f > bandHigh);
    to = to === -1 ? freqs.length - 1 : to - 1;
    return from === -1 || to < from ? [] : [{ from, to, color: "#4a3aa7", opacity: 0.07 }];
  }, [freqs, bandLow, bandHigh]);

  // How the two rails compare in the corrugation band, said plainly. Corrugation raises the peak on
  // the affected rail, so a clear lead on one side is the thing to notice — but it is evidence, not
  // a verdict: healthy files vary too, and the prediction is made elsewhere.
  const comparison = (() => {
    const a = sides.I?.peak_amplitude;
    const b = sides.II?.peak_amplitude;
    if (!a || !b) return null;
    const lead = a >= b ? a / b : b / a;
    if (lead < 1.05) return "both rails peak within 5% of each other";
    return `Side ${a >= b ? "I" : "II"} peaks ${lead.toFixed(2)}× higher than Side ${a >= b ? "II" : "I"}`;
  })();

  if (boxIds.length === 0) return <p>This file has no recognisable axle box columns.</p>;

  return (
    <section className="rail-telemetry">
      <div className="chart-card">
        <h3>Axle boxes: 8 cars × 8 positions</h3>
        <p className="history-dashboard-note">
          Each square is one axle box, shaded by how hard it shakes in the{" "}
          {bandLow != null ? `${bandLow} to ${bandHigh} Hz` : "corrugation"} band, where corrugation shows up. A
          darker square means stronger shaking at that box. The top row is the Side I rail and the bottom row is
          the Side II rail. The result covers a whole rail side across all 8 cars, not a single car or box, so
          read the two rows together rather than picking out one dark square.
        </p>
        {telemetry.speed_kmh != null && (
          <p className="history-dashboard-note">
            Train speed for this recording: about {telemetry.speed_kmh} km/h, from the rotating speed sensor.
          </p>
        )}
        <AxleBoxMap telemetry={telemetry} selected={selected} onSelect={setSelected} />
        <InterpretNote items={HOW_TO_READ.map} />
      </div>

      <div className="chart-card">
        <h3>
          Side I vs Side II: averaged spectrum
          {comparison && <span className="heading-note"> ({comparison})</span>}
        </h3>
        <p className="history-dashboard-note">
          The 32 boxes on each rail, averaged, with the vibration split by frequency. This is the main evidence
          for the Side I or Side II call. Corrugation shows up as a taller peak in the shaded band on the
          affected side.
        </p>
        <TelemetryChart
          ariaLabel="Averaged vibration spectrum for the Side I and Side II rails"
          times={freqs}
          formatTick={formatHz}
          formatTooltipTime={formatHz}
          bands={bands}
          panels={[
            {
              key: "spectrum",
              title: "Vibration amplitude (m/s²)",
              height: 180,
              series: [
                sides.I && { key: "sideI", label: "Side I (mean)", color: SIDE_I, values: sides.I.spectrum },
                sides.II && { key: "sideII", label: "Side II (mean)", color: SIDE_II, values: sides.II.spectrum },
              ].filter(Boolean),
            },
          ]}
          strips={[]}
          footer={
            <>
              <InterpretNote items={HOW_TO_READ.spectrum} />
              <p className="chart-caption">
                Each point is the strongest amplitude in that frequency bin. The shaded band is{" "}
                {bandLow != null ? `${bandLow} to ${bandHigh} Hz` : "the corrugation band"}, where the two sides
                are compared.
              </p>
            </>
          }
        />
      </div>

      {box && (
        <div className="chart-card">
          <h3>
            Car {box.car}, position {box.position} : through the recording
            <span className="heading-note"> (Side {box.side} rail)</span>
          </h3>
          <p className="history-dashboard-note">
            This is the box selected on the map. It opens on the loudest box in this file, and you can select any
            other box on the map to plot it instead. Vibration is the steady shaking and shock is the sharp
            knocks, each shown as an average level over every{" "}
            {Math.round((duration / Math.max(bins, 1)) * 1000)} ms.
          </p>
          <TelemetryChart
            key={selected}
            ariaLabel={`Vibration and shock amplitude for car ${box.car} position ${box.position}`}
            times={times}
            formatTick={formatSeconds}
            formatTooltipTime={formatSeconds}
            panels={[
              {
                key: "vibration",
                title: "Vibration amplitude (m/s²)",
                height: 110,
                series: box.vibration
                  ? [{ key: "vibration", label: "Vibration", color: VIBRATION, values: box.vibration }]
                  : [],
              },
              {
                key: "shock",
                title: "Shock amplitude (m/s²)",
                height: 110,
                series: box.shock ? [{ key: "shock", label: "Shock", color: SHOCK, values: box.shock }] : [],
              },
            ]}
            strips={[]}
            footer={
              <>
                <InterpretNote items={HOW_TO_READ.box} />
                <p className="chart-caption">
                  Each point is the RMS over about {Math.round((duration / Math.max(bins, 1)) * 1000)} ms, the
                  envelope of the signal and not the raw {telemetry.samples?.toLocaleString?.() ?? ""} samples,
                  which oscillate far too fast to draw. Overall RMS: {box.rms_vibration ?? "n/a"} m/s² vibration,{" "}
                  {box.rms_shock ?? "n/a"} m/s² shock.
                </p>
              </>
            }
          />

          {box.spectrum && (
            <>
              <h4 className="rail-subhead">This box&apos;s own spectrum</h4>
              <TelemetryChart
                key={`${selected}-spectrum`}
                ariaLabel={`Vibration spectrum for car ${box.car} position ${box.position}`}
                times={freqs}
                formatTick={formatHz}
                formatTooltipTime={formatHz}
                bands={bands}
                panels={[
                  {
                    key: "box-spectrum",
                    title: "Vibration amplitude (m/s²)",
                    height: 120,
                    series: [
                      {
                        key: "box",
                        label: `Car ${box.car} pos ${box.position}`,
                        color: SELECTED,
                        values: box.spectrum,
                      },
                    ],
                  },
                ]}
                strips={[]}
                footer={
                  <>
                    <InterpretNote items={HOW_TO_READ.boxSpectrum} />
                    <p className="chart-caption">
                      On its own scale, so a loud box does not have to be read against the side averages above.
                      Its strongest reading in the shaded band is {box.peak_amplitude ?? "n/a"} m/s²
                      {box.peak_frequency ? ` at ${box.peak_frequency} Hz` : ""}.
                    </p>
                  </>
                }
              />
            </>
          )}
        </div>
      )}
    </section>
  );
}
