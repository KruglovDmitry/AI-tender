import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { fetchConfig, fetchOcrStatus } from "../api";
import {
  alertWarningClass,
  btnGroupClass,
  btnGroupItemActiveClass,
  btnGroupItemClass,
  inputClass,
  labelClass,
  mutedTextClass,
  selectClass,
} from "../lib/styles";
import type { AppConfig } from "../types";

export interface SettingsState {
  llmProvider: string;
  ocrEnabled: boolean;
  maxReqs: number;
  assetsPath: string;
  tenderPath: string;
  tenderSource: "upload" | "folder";
}

interface SettingsContextValue {
  config: AppConfig | null;
  settings: SettingsState;
  setSettings: React.Dispatch<React.SetStateAction<SettingsState>>;
  ocrOk: boolean;
  ocrHint: string;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [settings, setSettings] = useState<SettingsState>({
    llmProvider: "deepseek",
    ocrEnabled: true,
    maxReqs: 10,
    assetsPath: "",
    tenderPath: "",
    tenderSource: "upload",
  });
  const [ocrOk, setOcrOk] = useState(true);
  const [ocrHint, setOcrHint] = useState("");

  useEffect(() => {
    void (async () => {
      const [cfg, ocr] = await Promise.all([fetchConfig(), fetchOcrStatus()]);
      setConfig(cfg);
      setSettings((prev) => ({
        ...prev,
        llmProvider: cfg.llm_provider,
        ocrEnabled: cfg.ocr_enabled,
        maxReqs: cfg.max_reqs_per_scope_item,
        assetsPath: cfg.default_assets_path,
        tenderPath: cfg.default_tender_path,
      }));
      setOcrOk(ocr.ok);
      setOcrHint(ocr.hint);
    })();
  }, []);

  const value = useMemo(
    () => ({ config, settings, setSettings, ocrOk, ocrHint }),
    [config, settings, ocrOk, ocrHint],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("SettingsProvider required");
  return ctx;
}

export function SettingsPanel() {
  const { config, settings, setSettings, ocrOk, ocrHint } = useSettings();

  if (!config) {
    return <p className={mutedTextClass}>Загрузка настроек…</p>;
  }

  return (
    <div className="grid gap-4">
        <div>
          <label className={labelClass} htmlFor="llm-provider">
            LLM-провайдер
          </label>
          <select
            id="llm-provider"
            className={selectClass}
            value={settings.llmProvider}
            onChange={(e) => setSettings((s) => ({ ...s, llmProvider: e.target.value }))}
          >
            <option value="deepseek">DeepSeek</option>
            <option value="openai">Local</option>
          </select>
        </div>

        <div>
          <span className={labelClass}>Файлы тендера</span>
          <div className={`${btnGroupClass} w-full`} role="group">
            <button
              type="button"
              className={
                settings.tenderSource === "upload"
                  ? `${btnGroupItemActiveClass} flex-1`
                  : `${btnGroupItemClass} flex-1`
              }
              onClick={() => setSettings((s) => ({ ...s, tenderSource: "upload" }))}
            >
              Загрузить
            </button>
            <button
              type="button"
              className={
                settings.tenderSource === "folder"
                  ? `${btnGroupItemActiveClass} flex-1`
                  : `${btnGroupItemClass} flex-1`
              }
              onClick={() => setSettings((s) => ({ ...s, tenderSource: "folder" }))}
            >
              Папка
            </button>
          </div>
        </div>

        {settings.tenderSource === "folder" && (
          <div>
            <label className={labelClass} htmlFor="tender-path">
              Папка с документами тендера
            </label>
            <input
              id="tender-path"
              className={inputClass}
              value={settings.tenderPath}
              onChange={(e) => setSettings((s) => ({ ...s, tenderPath: e.target.value }))}
            />
          </div>
        )}

        <div className="flex items-center gap-2">
          <input
            id="ocr-enabled"
            type="checkbox"
            checked={settings.ocrEnabled}
            onChange={(e) => setSettings((s) => ({ ...s, ocrEnabled: e.target.checked }))}
            className="size-4 rounded border-gray-300 text-blue-800 focus:ring-blue-800/30"
          />
          <label htmlFor="ocr-enabled" className="text-sm text-gray-700">
            OCR для сканов PDF
          </label>
        </div>

        {!ocrOk && settings.ocrEnabled && <p className={alertWarningClass}>{ocrHint}</p>}

        <div>
          <label className={labelClass} htmlFor="max-reqs">
            Макс. требований на позицию: {settings.maxReqs}
          </label>
          <input
            id="max-reqs"
            type="range"
            min={1}
            max={20}
            value={settings.maxReqs}
            onChange={(e) =>
              setSettings((s) => ({ ...s, maxReqs: Number(e.target.value) }))
            }
          />
        </div>

        {config.running_in_docker && (
          <p className="text-xs leading-relaxed text-gray-500">
            Docker: sources → /data/tender, assets → /data/assets
          </p>
        )}
    </div>
  );
}

const EASE = "transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]";

export function SettingsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <button
        type="button"
        className={`absolute inset-0 bg-black/25 backdrop-blur-[2px] ${EASE}`}
        onClick={onClose}
        aria-label="Закрыть настройки"
      />
      <div className="page-transition relative z-10 w-full max-w-md rounded-lg border border-[#e3e4e8] bg-white p-6 shadow-xl">
        <div className="mb-5 flex items-center justify-between gap-3">
          <h2 className="text-lg font-semibold tracking-[-0.01em] text-[#12161e]">Настройки</h2>
          <button
            type="button"
            onClick={onClose}
            className={`inline-flex size-8 items-center justify-center rounded-md text-[#7c7f88] hover:bg-gray-100 hover:text-[#12161e] ${EASE}`}
            aria-label="Закрыть"
          >
            ✕
          </button>
        </div>
        <SettingsPanel />
      </div>
    </div>
  );
}
