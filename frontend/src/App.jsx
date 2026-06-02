import React from "react";
import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { DashboardPage } from "./pages/DashboardPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { TrainingPage } from "./pages/TrainingPage";
import { SystemPage } from "./pages/SystemPage";
import { ModelsPage } from "./pages/ModelsPage";
import { useWebSocket } from "./hooks/useWebSocket";

const WS_URL =
  import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/traffic";

export default function App() {
  const { data, connected } = useWebSocket(WS_URL);

  return (
    <Layout connected={connected}>
      <Routes>
        <Route path="/" element={<DashboardPage data={data} />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/system" element={<SystemPage />} />
      </Routes>
    </Layout>
  );
}
