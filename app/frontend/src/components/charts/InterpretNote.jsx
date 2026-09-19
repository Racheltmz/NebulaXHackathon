/** The "how to read this plot" part of a chart card: a clear pill labelling it, then a few short
 * points. `items` is [{ term?, text }]. Kept visually distinct from the technical captions under a
 * chart (what a point is, how it was averaged) so the reader can find the interpretation quickly. */
export default function InterpretNote({ items, label = "How to interpret" }) {
  return (
    <div className="interpret-note">
      <span className="interpret-pill">{label}</span>
      <ul>
        {items.map(({ term, text }) => (
          <li key={text}>
            {term && <strong>{term} </strong>}
            {text}
          </li>
        ))}
      </ul>
    </div>
  );
}
