import React, { useState } from "react";
import { Play, Square } from "lucide-react";
import { api } from "../api/client";

export function TrainingPanel({ status, onRefresh }) {
  const [algorithm, setAlgorithm] = useState("dqn");
  const [episodes, setEpisodes] = useState(100);
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    setLoading(true);
    try {
      await api.startTraining({
        algorithm,
        total_episodes: episodes,
        batch_size: 64,
        learning_rate: 0.0001,
        gamma: 0.99,
      });
      onRefresh?.();
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    await api.stopTraining();
    onRefresh?.();
  };

  const isRunning = status?.status === "running";

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Training Control</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <label className="block">
          <span className="text-xs text-slate-400 uppercase tracking-wider">
            Algorithm
          </span>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            disabled={isRunning}
            className="mt-1 w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm"
          >
            <option value="dqn">DQN</option>
            <option value="rainbow">Rainbow DQN</option>
            <option value="ppo">PPO</option>
            <option value="qmix">QMIX</option>
            <option value="maddpg">MADDPG</option>
          </select>
        </label>
        <label className="block">
          <span className="text-xs text-slate-400 uppercase tracking-wider">
            Episodes
          </span>
          <input
            type="number"
            value={episodes}
            onChange={(e) => setEpisodes(Number(e.target.value))}
            disabled={isRunning}
            className="mt-1 w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm"
          />
        </label>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={handleStart}
          disabled={isRunning || loading}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg text-sm font-medium"
        >
          <Play size={16} />
          Start Training
        </button>
        <button
          onClick={handleStop}
          disabled={!isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg text-sm font-medium"
        >
          <Square size={16} />
          Stop
        </button>
        <span className="ml-auto text-sm text-slate-400">
          Status:{" "}
          <span
            className={
              isRunning ? "text-emerald-400" : "text-slate-300"
            }
          >
            {status?.status ?? "idle"}
          </span>
        </span>
      </div>
    </div>
  );
}
