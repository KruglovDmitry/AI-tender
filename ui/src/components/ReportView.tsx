import type { AnalysisReport } from "../types";
import { STATUS_LABELS } from "../types";
import { downloadReport } from "../api";
import { useSettings } from "./SettingsPanel";
import {
  btnActionClass,
  formGridClass,
  itemClass,
  mutedTextClass,
  reportAccordionClass,
  reportAccordionSummaryClass,
  sectionClass,
} from "../lib/styles";

function formatElapsed(seconds?: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} с`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes} мин ${Math.round(rest)} с`;
}

function statusColor(status: string): string {
  if (status === "matched") return "text-green-700";
  if (status === "partial") return "text-amber-700";
  return "text-red-700";
}

export function ReportView({ report }: { report: AnalysisReport }) {
  const { settings } = useSettings();
  const cacheNote = report.index_reused
    ? "VL-каталог из кэша"
    : "VL-каталог загружен";
  const scope = (report.query_selection?.scope as Record<string, unknown>) || {};
  const docSel = (report.query_selection?.doc_selection as Record<string, unknown>) || {};
  const matches = report.position_matches || [];

  const handleDownload = async () => {
    const format = settings.reportDownloadFormat;
    const blob = await downloadReport(report, format);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = format === "md" ? "ai-tender-report.md" : "ai-tender-report.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className={sectionClass}>
      <div className={formGridClass}>
        {report.verdict && (
          <p className="whitespace-pre-wrap rounded-md border border-blue-200 bg-blue-50 px-4 py-3 text-sm leading-relaxed text-blue-950">
            {report.verdict}
          </p>
        )}

        <p className={`text-sm ${mutedTextClass}`}>
          Готово за {formatElapsed(report.elapsed_seconds)}. {cacheNote}.
        </p>

        <div className="mt-8 grid gap-6">
          <details className={reportAccordionClass} open>
            <summary className={reportAccordionSummaryClass}>Предмет закупки</summary>
            <div className="mt-4 grid gap-4">
              <p className="text-xs text-gray-500">
                confidence={String(scope.overall_confidence ?? "—")} · needs_more_docs=
                {String(scope.needs_more_docs ?? false)}
              </p>
              {scope.summary ? (
                <p className="text-sm text-gray-800">
                  <span className="font-medium">Титул:</span> {String(scope.summary)}
                </p>
              ) : null}

              {matches.length > 0 ? (
                matches.map((match, index) => {
                  const qtyPart =
                    match.qty != null ? ` — ${match.qty} ${match.unit || ""}`.trimEnd() : "";
                  const label = STATUS_LABELS[match.status] || match.status;
                  return (
                    <div key={`${match.scope_name}-${index}`} className={itemClass}>
                      <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
                        <span className="text-sm font-medium text-gray-900">
                          {index + 1}. {match.scope_name}
                          {qtyPart}
                        </span>
                        <span className={`text-xs font-medium ${statusColor(match.status)}`}>
                          {label}
                        </span>
                      </div>
                      <details>
                        <summary className="cursor-pointer text-xs text-gray-500 hover:text-gray-700">
                          Детали
                        </summary>
                        <div className="mt-3 grid gap-2 text-sm text-gray-700">
                          <p className="text-xs text-gray-500">
                            Статус: {label} · conf={match.confidence.toFixed(2)}
                          </p>
                          {match.requirements?.length ? (
                            <>
                              <p className="font-medium text-gray-900">Требования:</p>
                              <ul className="list-inside list-disc space-y-0.5 text-gray-600">
                                {match.requirements.map((req, i) => (
                                  <li key={i}>{req.text}</li>
                                ))}
                              </ul>
                            </>
                          ) : (
                            <p className={mutedTextClass}>Требования не найдены.</p>
                          )}
                          {match.required_product ? (
                            <p>
                              <span className="font-medium">Требуется:</span>{" "}
                              {match.required_product}
                            </p>
                          ) : (
                            <p className={mutedTextClass}>
                              Требуется: конкретное обозначение в тендере не указано
                            </p>
                          )}
                          {match.status === "none" || !match.product_name ? (
                            <p>
                              <span className="font-medium">Подобрано:</span> нет подходящего
                              варианта
                            </p>
                          ) : (
                            <p>
                              <span className="font-medium">Подобрано:</span> {match.product_name}
                            </p>
                          )}
                          {match.explanation && (
                            <p className="text-gray-600">{match.explanation}</p>
                          )}
                          {match.asset_hits && match.asset_hits.length > 0 && (
                            <details>
                              <summary className="cursor-pointer text-xs text-gray-500">
                                Фрагменты каталога ({match.asset_hits.length})
                              </summary>
                              <div className="mt-2 grid gap-2">
                                {match.asset_hits.slice(0, 8).map((hit, hi) => (
                                  <div key={hi} className="rounded-md bg-gray-50 p-3">
                                    <p className="text-xs text-gray-500">
                                      {hit.file} · {hit.location}
                                      {hit.score != null ? ` · score=${hit.score.toFixed(3)}` : ""}
                                    </p>
                                    <p className="mt-1 text-sm text-gray-700">
                                      {hit.quote.slice(0, 800)}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </details>
                          )}
                        </div>
                      </details>
                    </div>
                  );
                })
              ) : (
                <p className={mutedTextClass}>Перечень позиций не извлечён.</p>
              )}
            </div>
          </details>

          {Boolean(docSel.selected) && (
            <details className={reportAccordionClass}>
              <summary className={reportAccordionSummaryClass}>
                Выбор файлов тендера (
                {Array.isArray(docSel.loaded) ? docSel.loaded.length : 0} загружено)
              </summary>
              <p className={`mt-3 ${mutedTextClass}`}>Режим: {String(docSel.mode || "—")}</p>
            </details>
          )}

          {report.warnings?.length > 0 && (
            <details className={reportAccordionClass}>
              <summary className={reportAccordionSummaryClass}>
                Предупреждения ({report.warnings.length})
              </summary>
              <ul className="mt-3 list-inside list-disc space-y-1 text-sm text-amber-800">
                {report.warnings.map((warning, i) => (
                  <li key={i}>{warning}</li>
                ))}
              </ul>
            </details>
          )}
        </div>

        <div className="mt-10 pt-2">
          <button type="button" onClick={() => void handleDownload()} className={btnActionClass}>
            Скачать отчёт
          </button>
        </div>
      </div>
    </section>
  );
}
