import React from "react";

export function LeaderboardTable({ data, loading }) {
  if (loading) {
    return <div className="animate-pulse h-48 bg-slate-900 rounded-xl" />;
  }
  if (!data?.length) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-500">
        No leaderboard data available
      </div>
    );
  }
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800 text-slate-400 text-left">
            <th className="p-4 font-medium">Rank</th>
            <th className="p-4 font-medium">Intersection</th>
            <th className="p-4 font-medium">Avg Reward</th>
            <th className="p-4 font-medium">Avg Queue</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr
              key={row.intersection_id}
              className="border-b border-slate-800/50 hover:bg-slate-800/30"
            >
              <td className="p-4 text-slate-500">#{idx + 1}</td>
              <td className="p-4 font-medium text-cyan-400">
                {row.intersection_id}
              </td>
              <td className="p-4 text-emerald-400">{row.avg_reward}</td>
              <td className="p-4 text-amber-400">{row.avg_queue}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
