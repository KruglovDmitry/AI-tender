import { useEffect, useRef, useState } from "react";
import { useAnalysis } from "../components/AnalysisProvider";
import { ProgressTicker } from "../components/ProgressTicker";
import { ReportView } from "../components/ReportView";
import { useSettings } from "../components/SettingsPanel";
import {
  alertErrorClass,
  btnActionClass,
  btnSecondaryClass,
  inputClass,
  labelClass,
  mutedTextClass,
  pageActionBarClass,
} from "../lib/styles";

const EASE = "transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]";

function UrlPromptDialog({
  open,
  onClose,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (url: string) => void;
}) {
  const [url, setUrl] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setUrl("");
    const t = window.setTimeout(() => inputRef.current?.focus(), 50);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  const submit = () => {
    const value = url.trim();
    if (!value) return;
    onSubmit(value);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <button
        type="button"
        className={`absolute inset-0 bg-black/25 backdrop-blur-[2px] ${EASE}`}
        onClick={onClose}
        aria-label="Закрыть"
      />
      <div className="page-transition relative z-10 w-full max-w-md rounded-lg border border-[#e3e4e8] bg-white p-6 shadow-xl">
        <h2 className="mb-4 text-lg font-semibold tracking-[-0.01em] text-[#12161e]">
          Ссылка на тендер
        </h2>
        <label className={labelClass} htmlFor="tender-url-prompt">
          URL страницы закупки
        </label>
        <input
          ref={inputRef}
          id="tender-url-prompt"
          className={inputClass}
          type="url"
          placeholder="https://zakupki.gov.ru/..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
        />
        <p className={`mt-2 ${mutedTextClass}`}>
          Страница откроется в headless-браузере; VL найдёт и скачает документы.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button type="button" className={btnSecondaryClass} onClick={onClose}>
            Отмена
          </button>
          <button
            type="button"
            className={btnActionClass}
            disabled={!url.trim()}
            onClick={submit}
          >
            Продолжить
          </button>
        </div>
      </div>
    </div>
  );
}

export function AnalysisPage() {
  const { settings } = useSettings();
  const { running, progress, status, error, report, runAnalyze, clearReport, clearError } =
    useAnalysis();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [urlPromptOpen, setUrlPromptOpen] = useState(false);

  useEffect(() => {
    const input = folderInputRef.current;
    if (!input) return;
    input.setAttribute("webkitdirectory", "");
    input.setAttribute("directory", "");
  }, []);

  const buildForm = (opts: {
    source: "upload" | "folder" | "url";
    files?: File[];
    url?: string;
  }) => {
    const form = new FormData();
    form.append("vl_enabled", String(settings.vlEnabled));
    form.append("max_reqs_per_scope_item", String(settings.maxReqs));
    form.append("tender_source", opts.source);
    form.append("tender_folder", "");
    form.append("tender_url", opts.url || "");
    form.append("assets_path", settings.assetsPath);
    if (opts.source === "upload" && opts.files) {
      for (const file of opts.files) {
        const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
        form.append("files", file, relative || file.name);
      }
    }
    return form;
  };

  const onStart = () => {
    if (running) return;
    clearError();
    setValidationError(null);

    if (settings.tenderSource === "folder") {
      folderInputRef.current?.click();
      return;
    }

    if (settings.tenderSource === "url") {
      setUrlPromptOpen(true);
      return;
    }

    fileInputRef.current?.click();
  };

  const onFilesPicked = (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    if (!picked.length) return;
    setValidationError(null);
    runAnalyze(buildForm({ source: "upload", files: picked }));
  };

  const onFolderPicked = (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    if (!picked.length) {
      setValidationError("Папка пуста или файлы не выбраны.");
      return;
    }
    setValidationError(null);
    // Браузер отдаёт файлы папки — отправляем как upload с относительными путями.
    runAnalyze(buildForm({ source: "upload", files: picked }));
  };

  const onUrlSubmit = (url: string) => {
    setUrlPromptOpen(false);
    setValidationError(null);
    runAnalyze(buildForm({ source: "url", url }));
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
      <input
        ref={folderInputRef}
        type="file"
        className="hidden"
        multiple
        onChange={onFolderPicked}
      />

      <UrlPromptDialog
        open={urlPromptOpen}
        onClose={() => setUrlPromptOpen(false)}
        onSubmit={onUrlSubmit}
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
