import { useRef, useState } from "react";
import { pollJob, startAnalyze } from "../api";
import { ReportView } from "../components/ReportView";
import { useSettings } from "../components/SettingsPanel";
import {
  alertErrorClass,
  btnActionClass,
  mutedTextClass,
  pageActionBarClass,
  progressBarClass,
  progressTrackClass,
} from "../lib/styles";
import type { AnalysisReport } from "../types";

export function AnalysisPage() {
  const { settings } = useSettings();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const runAnalyze = async (uploadFiles: File[]) => {
    setError(null);
    setReport(null);
    setRunning(true);
    setProgress(0);
    setStatus("Запуск…");

    const form = new FormData();
    form.append("llm_provider", settings.llmProvider);
    form.append("ocr_enabled", String(settings.ocrEnabled));
    form.append("max_reqs_per_scope_item", String(settings.maxReqs));
    form.append("tender_source", settings.tenderSource);
    form.append("tender_folder", settings.tenderPath);
    form.append("assets_path", settings.assetsPath);
    if (settings.tenderSource === "upload") {
      for (const file of uploadFiles) {
        form.append("files", file);
      }
    }

    try {
      const job = await startAnalyze(form);
      const done = await pollJob(job.id, (j) => {
        setProgress(j.progress);
        setStatus(j.message);
      });
      const result = done.result as { report?: AnalysisReport } | undefined;
      if (result?.report) {
        setReport(result.report);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const onStart = () => {
    if (running) return;
    setError(null);

    if (settings.tenderSource === "folder") {
      if (!settings.tenderPath.trim()) {
        setError("Укажите папку с документами тендера в настройках.");
        return;
      }
      void runAnalyze([]);
      return;
    }

    fileInputRef.current?.click();
  };

  const onFilesPicked = (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    if (!picked.length) return;
    void runAnalyze(picked);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={onFilesPicked}
      />

      {!report && (
        <div className={pageActionBarClass}>
          <button type="button" disabled={running} onClick={onStart} className={btnActionClass}>
            {running ? "Анализ…" : "Начать"}
          </button>

          {running && (
            <div className="grid w-full max-w-md gap-2">
              <p className={`text-center ${mutedTextClass}`}>{status}</p>
              <div className={progressTrackClass}>
                <div
                  className={progressBarClass}
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
            </div>
          )}

          {error && <p className={`max-w-md text-center ${alertErrorClass}`}>{error}</p>}
        </div>
      )}

      {report && (
        <div className="min-h-0 overflow-y-auto">
          <ReportView report={report} />
        </div>
      )}
    </div>
  );
}
