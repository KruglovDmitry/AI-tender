import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useAnalysis } from "./AnalysisProvider";
import { AnimatedOutlet } from "./AnimatedOutlet";
import { BackgroundDecor } from "./BackgroundDecor";
import { SettingsDialog } from "./SettingsPanel";

const EASE = "duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]";

const tabClass = ({ isActive }: { isActive: boolean }) =>
  isActive
    ? `rounded-md border border-blue-800 bg-blue-800 px-4 py-2 text-sm font-medium text-white transition-all ${EASE}`
    : `rounded-md border border-gray-300 bg-white/80 px-4 py-2 text-sm font-medium text-[#3b3e47] backdrop-blur-sm transition-all ${EASE} hover:border-gray-400 hover:bg-white`;

export function Layout() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const location = useLocation();
  const { running, progress, report } = useAnalysis();
  const onAnalysisPage = location.pathname === "/";

  return (
    <div className="relative flex min-h-screen flex-col">
      <BackgroundDecor />

      <header className={`panel-header transition-colors ${EASE}`}>
        <div className="flex items-center gap-3 px-4 py-3 sm:px-6">
          <h1 className="ml-8 shrink-0 text-lg font-semibold tracking-[-0.01em] text-[#12161e] sm:ml-12">
            AI Tender
          </h1>

          <nav className="ml-auto flex items-center gap-2">
            <NavLink to="/" end className={tabClass}>
              <span className="inline-flex items-center gap-2">
                Анализ тендера
                {running && (
                  <span
                    className="inline-flex min-w-[2.25rem] items-center justify-center rounded-full bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
                    title="Анализ выполняется"
                  >
                    {Math.round(progress * 100)}%
                  </span>
                )}
                {!running && report && !onAnalysisPage && (
                  <span
                    className="size-2 rounded-full bg-emerald-400"
                    title="Результат анализа готов"
                  />
                )}
              </span>
            </NavLink>
            <NavLink to="/references" className={tabClass}>
              Эталоны
            </NavLink>
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className={`inline-flex size-9 items-center justify-center rounded-md border border-gray-300 bg-white/80 text-[#3b3e47] backdrop-blur-sm transition-all ${EASE} hover:border-gray-400 hover:bg-white`}
              aria-label="Настройки"
              title="Настройки"
            >
              <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path
                  d="M12 15a3 3 0 100-6 3 3 0 000 6z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </nav>
        </div>
      </header>

      <main className="flex min-h-0 min-w-0 flex-1 flex-col px-5 py-6 sm:px-8 lg:px-10">
        <AnimatedOutlet />
      </main>

      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
