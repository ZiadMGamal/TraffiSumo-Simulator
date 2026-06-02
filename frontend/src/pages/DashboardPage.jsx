import React from "react";
import { MetricsCards } from "../components/MetricsCards";
import { TrafficCharts } from "../components/TrafficCharts";

export function DashboardPage({ data = [] }) {
  return (
    <div className="p-6">
      <header className="mb-6">
        <h2 className="text-xl font-bold text-white">Live Traffic Dashboard</h2>
        <p className="text-sm text-slate-500">
          Real-time multi-agent intersection monitoring
        </p>
      </header>
      <MetricsCards metrics={data} />
      <TrafficCharts data={data} />
    </div>
  );
}
