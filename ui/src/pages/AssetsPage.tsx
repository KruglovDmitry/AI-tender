import { useCallback, useEffect, useRef, useState } from "react";
import { deleteAsset, fetchAssets, fetchProducts, pollJob, uploadAssets } from "../api";
import { useSettings } from "../components/SettingsPanel";
import {
  alertErrorClass,
  alertWarningClass,
  btnActionClass,
  btnOutlineDangerClass,
  itemClass,
  mutedTextClass,
  pageActionBarClass,
  progressBarClass,
  progressTrackClass,
} from "../lib/styles";
import type { AssetFile, ProductDocumentIndex } from "../types";

function fileName(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path;
}

export function AssetsPage() {
  const { settings } = useSettings();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<AssetFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [jobMessage, setJobMessage] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [productsCache, setProductsCache] = useState<Record<string, ProductDocumentIndex>>({});
  const [loadingProducts, setLoadingProducts] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchAssets(settings.assetsPath);
      setFiles(data.files.filter((f) => f.indexed));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [settings.assetsPath]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const loadProducts = async (path: string) => {
    if (productsCache[path]) return productsCache[path];
    setLoadingProducts(path);
    try {
      const data = await fetchProducts(path, settings.assetsPath);
      setProductsCache((prev) => ({ ...prev, [path]: data }));
      return data;
    } catch {
      return null;
    } finally {
      setLoadingProducts((current) => (current === path ? null : current));
    }
  };

  const toggleExpand = (file: AssetFile) => {
    if (expanded === file.path) {
      setExpanded(null);
      return;
    }
    setExpanded(file.path);
    void loadProducts(file.path);
  };

  const onAddClick = () => {
    if (uploading) return;
    setError(null);
    fileInputRef.current?.click();
  };

  const onFilesPicked = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    if (!picked.length) return;

    setUploading(true);
    setError(null);
    setJobMessage("Загрузка…");
    setJobProgress(0);
    try {
      const job = await uploadAssets(picked, settings.assetsPath);
      await pollJob(job.id, (j) => {
        setJobProgress(j.progress);
        setJobMessage(j.message);
      });
      setProductsCache({});
      setExpanded(null);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      setJobMessage("");
      setJobProgress(0);
    }
  };

  const onDelete = async (path: string) => {
    if (!confirm(`Удалить ${path}?`)) return;
    setError(null);
    try {
      await deleteAsset(path, settings.assetsPath);
      if (expanded === path) setExpanded(null);
      setProductsCache((prev) => {
        const next = { ...prev };
        delete next[path];
        return next;
      });
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        className="hidden"
        onChange={(e) => void onFilesPicked(e)}
      />

      <div className={pageActionBarClass}>
        <button
          type="button"
          disabled={uploading}
          onClick={onAddClick}
          className={btnActionClass}
        >
          {uploading ? "Индексация…" : "Добавить"}
        </button>

        {uploading && (
          <div className="grid w-full max-w-md gap-2">
            <p className={`text-center ${mutedTextClass}`}>{jobMessage}</p>
            <div className={progressTrackClass}>
              <div
                className={progressBarClass}
                style={{ width: `${Math.round(jobProgress * 100)}%` }}
              />
            </div>
          </div>
        )}

        {error && <p className={`max-w-md text-center ${alertErrorClass}`}>{error}</p>}
      </div>

      <div className="mx-auto w-full max-w-3xl flex-1 pb-8">
        {loading ? (
          <p className={`text-center ${mutedTextClass}`}>Загрузка…</p>
        ) : (
          <ul className="grid gap-4">
            {files.map((file) => {
              const products = productsCache[file.path];
              const isOpen = expanded === file.path;
              return (
                <li key={file.path} className={itemClass}>
                  <div className="flex items-start gap-3">
                    <button
                      type="button"
                      onClick={() => toggleExpand(file)}
                      className="min-w-0 flex-1 text-left"
                    >
                      <p className="font-medium text-gray-900">{fileName(file.path)}</p>
                      <p className="mt-0.5 text-xs text-gray-500">{file.path}</p>
                      <p className={`mt-1 ${mutedTextClass}`}>
                        {[
                          file.catalog_name,
                          file.doc_kind,
                          `${file.product_count} продуктов`,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </p>
                    </button>
                    <button
                      type="button"
                      onClick={() => void onDelete(file.path)}
                      className={btnOutlineDangerClass}
                      aria-label={`Удалить ${file.path}`}
                    >
                      ✕
                    </button>
                  </div>

                  {isOpen && (
                    <div className="mt-4 border-t border-gray-100 pt-4">
                      {loadingProducts === file.path && !products && (
                        <p className={mutedTextClass}>Загрузка продуктов…</p>
                      )}
                      {products && (
                        <div className="grid gap-4">
                          {products.warnings.map((w, i) => (
                            <p key={i} className={alertWarningClass}>
                              {w}
                            </p>
                          ))}
                          {products.product_pages.length > 0 && (
                            <p className={`text-xs ${mutedTextClass}`}>
                              Страницы: {products.product_pages.join(", ")}
                            </p>
                          )}
                          <div className="grid gap-3">
                            {products.products.map((product) => (
                              <article
                                key={product.id}
                                className="rounded-md border border-gray-100 bg-gray-50/80 p-3"
                              >
                                <h3 className="font-medium text-gray-900">
                                  {product.model || product.id}
                                </h3>
                                <div className="mt-1 flex flex-wrap gap-1.5">
                                  {product.manufacturer && (
                                    <span className="rounded-full bg-white px-2.5 py-0.5 text-xs text-gray-600">
                                      {product.manufacturer}
                                    </span>
                                  )}
                                  {product.category && (
                                    <span className="rounded-full bg-white px-2.5 py-0.5 text-xs text-gray-600">
                                      {product.category}
                                    </span>
                                  )}
                                </div>
                                {product.canonical_desc && (
                                  <p className="mt-2 text-sm text-gray-700">
                                    {product.canonical_desc}
                                  </p>
                                )}
                                {product.characteristics.length > 0 && (
                                  <ul className="mt-2 list-inside list-disc space-y-0.5 text-sm text-gray-600">
                                    {product.characteristics.map((item, i) => (
                                      <li key={i}>{item}</li>
                                    ))}
                                  </ul>
                                )}
                              </article>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
