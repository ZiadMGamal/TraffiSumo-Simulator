import React from "react";
import { api } from "../api/client";
import { useApi } from "../hooks/useApi";

export function SystemPage() {
  const health = useApi(() => api.getHealth(), []);
  const info = useApi(() => api.getSystemInfo(), []);

  return (
    <div className="p-6 space-y-6">
      <header>
        <h2 className="text-xl font-bold text-white">System Status</h2>
        <p className="text-sm text-slate-500">Infrastructure and configuration</p>
      </header>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <InfoCard title="Health" loading={health.loading}>
          {health.data && (
            <dl className="space-y-2 text-sm">
              <Row label="Status" value={health.data.status} />
              <Row label="Version" value={health.data.version} />
              <Row
                label="SUMO"
                value={health.data.sumo_available ? "Available" : "Not detected"}
              />
              <Row
                label="Database"
                value={
                  health.data.database_connected ? "Connected" : "Disconnected"
                }
              />
            </dl>
          )}
        </InfoCard>
        <InfoCard title="Configuration" loading={info.loading}>
          {info.data && (
            <dl className="space-y-2 text-sm">
              <Row label="Project" value={info.data.project} />
              <Row
                label="Algorithms"
                value={info.data.algorithms?.join(", ")}
              />
              <Row
                label="State Dim"
                value={info.data.config?.state_dim}
              />
              <Row
                label="SUMO Config"
                value={info.data.config?.sumo_config}
              />
            </dl>
          )}
        </InfoCard>
      </div>
    </div>
  );
}

function InfoCard({ title, children, loading }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <h3 className="text-sm font-medium text-slate-300 mb-4">{title}</h3>
      {loading ? <div className="animate-pulse h-24 bg-slate-800 rounded" /> : children}
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-slate-200 font-mono text-xs">{value}</dd>
    </div>
  );
}
