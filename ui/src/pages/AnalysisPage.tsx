import { useRef, useState } from "react";
import { useAnalysis } from "../components/AnalysisProvider";
import { ProgressTicker } from "../components/ProgressTicker";
import { ReportView } from "../components/ReportView";
import { useSettings } from "../components/SettingsPanel";
import {
  alertErrorClass,
  btnActionClass,
  btnSecondaryClass,
  pageActionBarClass,
} from "../lib/styles";

export function AnalysisPage() {
  const { config, settings } = useSettings();
  const { running, progress, status, error, report, runAnalyze, clearReport, clearError } =
    useAnalysis();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const buildForm = (uploadFiles: File[]) => {
    const form = new FormData();
    form.append("llm_provider", config?.llm_provider ?? "local");
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
    return form;
  };

  const onStart = () => {
    if (running) return;
    clearError();
    setValidationError(null);

    if (settings.tenderSource === "folder") {
      if (!settings.tenderPath.trim()) {
        setValidationError("Укажите папку с документами тендера в настройках.");
        return;
      }
      runAnalyze(buildForm([]));
      return;
    }

    fileInputRef.current?.click();
  };

  const onFilesPicked = (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    if (!picked.length) return;
    setValidationError(null);
    runAnalyze(buildForm(picked));
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

          {running && <ProgressTicker message={status} progress={progress} />}

          {(error || validationError) && (
            <p className={`max-w-md text-center ${alertErrorClass}`}>
              {validationError || error}
            </p>
          )}
        </div>
      )}

      {report && (
        <div className="flex min-h-0 flex-1 flex-col gap-4">
          <div className={pageActionBarClass}>
            <button
              type="button"
              disabled={running}
              onClick={clearReport}
              className={btnSecondaryClass}
            >
              Новый анализ
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            <ReportView report={report} />
          </div>
        </div>
      )}
    </div>
  );
}
