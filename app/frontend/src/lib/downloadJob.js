import apiClient from "./apiClient";

/** The download endpoint requires the Supabase bearer token, so a plain <a href> won't carry
 * auth — fetch it as a blob through the authenticated axios client instead and trigger a
 * save via a throwaway object URL. */
export async function downloadJobCsv(jobId, subsystem) {
  const response = await apiClient.get(`/api/jobs/${jobId}/download`, { responseType: "blob" });
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${subsystem}_predictions.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}
