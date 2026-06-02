import React from "react";
import { NavLink } from "react-router-dom";
import { Activity, BarChart3, Cpu, Database, Home, Settings } from "lucide-react";

const navItems = [
  { to: "/", icon: Home, label: "Dashboard" },
  { to: "/analytics", icon: BarChart3, label: "Analytics" },
  { to: "/training", icon: Cpu, label: "Training" },
  { to: "/models", icon: Database, label: "Models" },
  { to: "/system", icon: Settings, label: "System" },
];

export function Layout({ children, connected }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex">
      <aside className="w-56 border-r border-slate-800 bg-slate-900/50 flex flex-col">
        <div className="p-5 border-b border-slate-800">
          <h1 className="text-sm font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
            Traffic MARL
          </h1>
          <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-widest">
            Cooperative Control
          </p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-cyan-500/10 text-cyan-400"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs">
            <Activity
              size={14}
              className={connected ? "text-emerald-500" : "text-rose-500"}
            />
            <span className="text-slate-400">
              {connected ? "Live Sync" : "Offline"}
            </span>
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
