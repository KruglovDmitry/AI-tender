import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { fetchConfig, fetchVlStatus } from "../api";
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
  vlEnabled: boolean;
  maxReqs: number;
  assetsPath: string;
  tenderPath: string;
  tenderSource: "upload" | "folder";
  reportDownloadFormat: "md" | "json";
}

interface SettingsContextValue {
  config: AppConfig | null;
  settings: SettingsState;
  setSettings: React.Dispatch<React.SetStateAction<SettingsState>>;
  vlOk: boolean;
  vlHint: string;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [settings, setSettings] = useState<SettingsState>({
    vlEnabled: true,
    maxReqs: 10,
    assetsPath: "",
    tenderPath: "",
    tenderSource: "upload",
    reportDownloadFormat: "md",
  });
  const [vlOk, setVlOk] = useState(true);
  const [vlHint, setVlHint] = useState("");

  useEffect(() => {
    void (async () => {
      const [cfg, vl] = await Promise.all([fetchConfig(), fetchVlStatus()]);
      setConfig(cfg);
      setSettings((prev) => ({
        ...prev,
        vlEnabled: cfg.vl_enabled,
        maxReqs: cfg.max_reqs_per_scope_item,
        assetsPath: cfg.default_assets_path,
        tenderPath: cfg.default_tender_path,
      }));
      setVlOk(vl.ok);
      setVlHint(vl.hint);
    })();
  }, []);

  const value = useMemo(
    () => ({ config, settings, setSettings, vlOk, vlHint }),
    [config, settings, vlOk, vlHint],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("SettingsProvider required");
  return ctx;
}

export function SettingsPanel() {
  const { config, settings, setSettings, vlOk, vlHint } = useSettings();

  if (!config) {
    return <p className={mutedTextClass}>Загрузка настроек…</p>;
  }

  return (
    <div className="grid gap-4">
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

        <div
          className="flex items-center gap-2"
          title={`Сканы без текста всегда идут через VL (${config.qwen_vl_model}). Включите переключатель, чтобы VL использовался и для текстовых PDF.`}
        >
          <input
            id="vl-enabled"
            type="checkbox"
            checked={settings.vlEnabled}
            onChange={(e) => setSettings((s) => ({ ...s, vlEnabled: e.target.checked }))}
            className="size-4 rounded border-gray-300 text-blue-800 focus:ring-blue-800/30"
          />
          <label htmlFor="vl-enabled" className="text-sm text-gray-700">
            VL-модель для извлечения
          </label>
        </div>

        {!vlOk && <p className={alertWarningClass}>{vlHint}</p>}

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

        <div>
          <label className={labelClass} htmlFor="report-format">
            Формат скачиваемого отчёта
          </label>
          <select
            id="report-format"
            className={selectClass}
            value={settings.reportDownloadFormat}
            onChange={(e) =>
              setSettings((s) => ({
                ...s,
                reportDownloadFormat: e.target.value as "md" | "json",
              }))
            }
          >
            <option value="md">Markdown (.md)</option>
            <option value="json">JSON (.json)</option>
          </select>
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
