import React from "react";

export function MetricsCards({ metrics }) {
  if (!metrics?.length) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="bg-slate-900 border border-slate-800 rounded-lg p-4 animate-pulse h-28"
          />
        ))}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {metrics.map((node) => (
        <div
          key={node.id}
          className="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-cyan-500/30 transition-colors"
        >
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">
              {node.id}
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">
              Phase {node.phase ?? 0}
            </span>
          </div>
          <div className="text-2xl font-semibold text-white mb-1">
            {node.queue}
            <span className="text-xs text-slate-500 font-normal ml-1">queued</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs text-slate-400 border-t border-slate-800 pt-2">
            <span>
              Wait:{" "}
              <strong className="text-amber-500">{node.wait?.toFixed(1)}s</strong>
            </span>
            <span>
              Reward:{" "}
              <strong
                className={
                  node.reward < -20 ? "text-rose-500" : "text-emerald-500"
                }
              >
                {node.reward?.toFixed(1)}
              </strong>
            </span>
            <span>
              Pressure:{" "}
              <strong className="text-cyan-400">
                {node.pressure?.toFixed(1) ?? "—"}
              </strong>
            </span>
            <span>
              Throughput:{" "}
              <strong className="text-violet-400">{node.throughput ?? 0}</strong>
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
