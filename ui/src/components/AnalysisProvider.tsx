import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { pollJob, startAnalyze } from "../api";
import type { AnalysisReport } from "../types";

interface AnalysisContextValue {
  running: boolean;
  progress: number;
  status: string;
  error: string | null;
  report: AnalysisReport | null;
  runAnalyze: (form: FormData) => void;
  clearReport: () => void;
  clearError: () => void;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const runAnalyze = useCallback((form: FormData) => {
    setError(null);
    setReport(null);
    setRunning(true);
    setProgress(0);
    setStatus("Запуск…");

    void (async () => {
      try {
        const job = await startAnalyze(form);
        const done = await pollJob(job.id, (jobUpdate) => {
          setProgress(jobUpdate.progress);
          setStatus(jobUpdate.message);
        });
        const result = done.result as { report?: AnalysisReport } | undefined;
        if (result?.report) {
          setReport(result.report);
        } else {
          setError("Отчёт не получен от сервера");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setRunning(false);
      }
    })();
  }, []);

  const clearReport = useCallback(() => {
    setReport(null);
    setError(null);
    setProgress(0);
    setStatus("");
  }, []);

  const clearError = useCallback(() => setError(null), []);

  const value = useMemo(
    () => ({
      running,
      progress,
      status,
      error,
      report,
      runAnalyze,
      clearReport,
      clearError,
    }),
    [running, progress, status, error, report, runAnalyze, clearReport, clearError],
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) {
    throw new Error("useAnalysis must be used within AnalysisProvider");
  }
  return ctx;
}
