import React from "react";
import { api } from "../api/client";
import { useApi } from "../hooks/useApi";
import { TrainingPanel } from "../components/TrainingPanel";

export function TrainingPage() {
  const status = useApi(() => api.getTrainingStatus(), []);
  const algorithms = useApi(() => api.getAlgorithms(), []);

  return (
    <div className="p-6 space-y-6">
      <header>
        <h2 className="text-xl font-bold text-white">Training Center</h2>
        <p className="text-sm text-slate-500">
          Configure and launch multi-agent reinforcement learning
        </p>
      </header>
      <TrainingPanel status={status.data} onRefresh={status.refetch} />
      <section>
        <h3 className="text-sm font-medium text-slate-300 mb-3">
          Available Algorithms
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {(algorithms.data || []).map((algo) => (
            <div
              key={algo.name}
              className="bg-slate-900 border border-slate-800 rounded-lg p-4"
            >
              <p className="font-medium text-cyan-400 uppercase text-sm">
                {algo.name}
              </p>
              <p className="text-xs text-slate-500 mt-1">{algo.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
