import React from "react";
import { api } from "../api/client";
import { useApi } from "../hooks/useApi";
import { LeaderboardTable } from "../components/LeaderboardTable";

export function AnalyticsPage() {
  const summary = useApi(() => api.getAnalyticsSummary(), []);
  const leaderboard = useApi(() => api.getLeaderboard(), []);
  const history = useApi(() => api.getTrainingHistory(), []);

  return (
    <div className="p-6 space-y-6">
      <header>
        <h2 className="text-xl font-bold text-white">Analytics</h2>
        <p className="text-sm text-slate-500">Historical performance metrics</p>
      </header>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Avg Queue"
          value={summary.data?.avg_queue ?? "—"}
          loading={summary.loading}
        />
        <StatCard
          label="Avg Wait"
          value={summary.data?.avg_wait ?? "—"}
          loading={summary.loading}
        />
        <StatCard
          label="Throughput"
          value={summary.data?.total_throughput ?? "—"}
          loading={summary.loading}
        />
        <StatCard
          label="Active Nodes"
          value={summary.data?.active_intersections ?? "—"}
          loading={summary.loading}
        />
      </div>
      <section>
        <h3 className="text-sm font-medium text-slate-300 mb-3">
          Intersection Leaderboard
        </h3>
        <LeaderboardTable
          data={leaderboard.data}
          loading={leaderboard.loading}
        />
      </section>
      <section>
        <h3 className="text-sm font-medium text-slate-300 mb-3">Training History</h3>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          {history.loading ? (
            <div className="animate-pulse h-20" />
          ) : !history.data?.length ? (
            <p className="text-slate-500 text-sm">No training runs recorded</p>
          ) : (
            <ul className="space-y-2">
              {history.data.map((run) => (
                <li
                  key={run.id}
                  className="flex justify-between text-sm border-b border-slate-800 pb-2"
                >
                  <span className="text-cyan-400">{run.algorithm}</span>
                  <span className="text-slate-400">{run.episodes} episodes</span>
                  <span
                    className={
                      run.status === "completed"
                        ? "text-emerald-400"
                        : "text-amber-400"
                    }
                  >
                    {run.status}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}

function StatCard({ label, value, loading }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
      {loading ? (
        <div className="h-8 mt-1 bg-slate-800 rounded animate-pulse" />
      ) : (
        <p className="text-2xl font-semibold text-white mt-1">{value}</p>
      )}
    </div>
  );
}
