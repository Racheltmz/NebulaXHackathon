import apiClient from "./apiClient";

function saveBlob(data, filename) {
  const url = window.URL.createObjectURL(new Blob([data]));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

/** The combined predictions CSV for one subsystem (every finished run, PS3 submission schema).
 * Fetched as a blob through the axios client and saved via a throwaway object URL. */
export async function downloadSubsystemCsv(subsystem) {
  try {
    const response = await apiClient.get("/api/history/download", {
      params: { subsystem },
      responseType: "blob",
    });
    saveBlob(response.data, `${subsystem}_predictions.csv`);
  } catch (err) {
    // With responseType "blob" an error body arrives as a Blob too — unwrap the FastAPI detail.
    const body = err.response?.data instanceof Blob ? await err.response.data.text() : null;
    let detail = null;
    try {
      detail = JSON.parse(body)?.detail;
    } catch {
      // not JSON — fall through to the generic message
    }
    throw new Error(detail || err.message);
  }
}

export async function downloadInputFile(jobId, index, filename) {
  const response = await apiClient.get(`/api/jobs/${jobId}/input-files/${index}/download`, {
    responseType: "blob",
  });
  saveBlob(response.data, filename);
}
