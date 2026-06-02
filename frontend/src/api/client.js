const API_BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

export const api = {
  getHealth: () => request("/api/system/health"),
  getSystemInfo: () => request("/api/system/info"),
  getAnalyticsSummary: (limit = 100) =>
    request(`/api/analytics/summary?limit=${limit}`),
  getLeaderboard: () => request("/api/analytics/leaderboard"),
  getTrainingHistory: () => request("/api/analytics/training-history"),
  getTrainingStatus: () => request("/api/training/status"),
  getAlgorithms: () => request("/api/training/algorithms"),
  startTraining: (config) =>
    request("/api/training/start", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  stopTraining: () => request("/api/training/stop", { method: "POST" }),
  getTimeSeries: (intersectionId, hours = 24) =>
    request(`/api/analytics/timeseries/${intersectionId}?hours=${hours}`),
};
