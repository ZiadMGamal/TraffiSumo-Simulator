import React from "react";
import { api } from "../api/client";
import { useApi } from "../hooks/useApi";

export function ModelsPage() {
  const models = useApi(async () => {
    const res = await fetch("/api/models/");
    return res.json();
  }, []);

  return (
    <div className="p-6 space-y-6">
      <header>
        <h2 className="text-xl font-bold text-white">Model Registry</h2>
        <p className="text-sm text-slate-500">Trained policies and checkpoints</p>
      </header>
      {models.loading ? (
        <div className="animate-pulse h-40 bg-slate-900 rounded-xl" />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Model Directory</p>
              <p className="text-sm text-cyan-400 font-mono mt-1">
                {models.data?.model_dir}
              </p>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <p className="text-xs text-slate-500">Artifacts</p>
              <p className="text-2xl font-semibold text-white mt-1">
                {models.data?.model_count ?? 0} models /{" "}
                {models.data?.checkpoint_count ?? 0} checkpoints
              </p>
            </div>
          </div>
          <section>
            <h3 className="text-sm font-medium text-slate-300 mb-3">Policies</h3>
            <div className="space-y-2">
              {(models.data?.models || []).map((m) => (
                <div
                  key={m.name}
                  className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex justify-between text-sm"
                >
                  <span className="text-white">{m.name}</span>
                  <span className="text-slate-500">{m.type}</span>
                </div>
              ))}
              {!models.data?.models?.length && (
                <p className="text-slate-500 text-sm">No models trained yet</p>
              )}
            </div>
          </section>
          <section>
            <h3 className="text-sm font-medium text-slate-300 mb-3">Checkpoints</h3>
            <div className="space-y-2">
              {(models.data?.checkpoints || []).map((c) => (
                <div
                  key={c.name}
                  className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-sm"
                >
                  <span className="text-cyan-400">{c.name}</span>
                  {c.episode != null && (
                    <span className="text-slate-500 ml-3">ep {c.episode}</span>
                  )}
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
