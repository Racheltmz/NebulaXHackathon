import { useRef, useState } from "react";

export default function FileDropzone({ accept, files, onFilesChange }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const addFiles = (fileList) => {
    const incoming = Array.from(fileList);
    const existingNames = new Set(files.map((f) => f.name));
    const merged = [...files, ...incoming.filter((f) => !existingNames.has(f.name))];
    onFilesChange(merged);
  };

  const removeFile = (name) => {
    onFilesChange(files.filter((f) => f.name !== name));
  };

  return (
    <div>
      <div
        className={`dropzone ${dragOver ? "dragover" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          addFiles(e.dataTransfer.files);
        }}
      >
        <p>
          Drag &amp; drop your file(s) here, or click to browse
          <br />
          <small>Accepted: {accept.join(", ")}</small>
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={accept.join(",")}
          onChange={(e) => addFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <ul className="file-list">
          {files.map((f) => (
            <li key={f.name}>
              <span>{f.name}</span>
              <button type="button" onClick={() => removeFile(f.name)}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
