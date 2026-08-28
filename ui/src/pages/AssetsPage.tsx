import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteAsset,
  fetchAssets,
  fetchProducts,
  pollJob,
  reindexAsset,
  uploadAssets,
} from "../api";
import { useSettings } from "../components/SettingsPanel";
import {
  alertErrorClass,
  assetItemClass,
  btnActionClass,
  btnIconNeutralClass,
  btnOutlineDangerClass,
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
  const [reindexingPath, setReindexingPath] = useState<string | null>(null);
  const [reindexMessage, setReindexMessage] = useState("");
  const [reindexProgress, setReindexProgress] = useState(0);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [productsCache, setProductsCache] = useState<Record<string, ProductDocumentIndex>>({});
  const [loadingProducts, setLoadingProducts] = useState<string | null>(null);

  const reload = useCallback(async (options?: { silent?: boolean }) => {
    if (!options?.silent) setLoading(true);
    try {
      const data = await fetchAssets(settings.assetsPath);
      setFiles(data.files);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (!options?.silent) setLoading(false);
    }
  }, [settings.assetsPath]);

  const refreshAsset = useCallback(
    async (path: string) => {
      const data = await fetchAssets(settings.assetsPath);
      const updated = data.files.find((file) => file.path === path);
      if (!updated) return;
      setFiles((prev) => prev.map((file) => (file.path === path ? updated : file)));
    },
    [settings.assetsPath],
  );

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
    if (uploading || reindexingPath !== null) return;
    setError(null);
    fileInputRef.current?.click();
  };

  const onFilesPicked = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    if (!picked.length) return;

    setUploading(true);
    setError(null);
    setUploadMessage("Загрузка…");
    setUploadProgress(0);
    try {
      const job = await uploadAssets(picked, settings.assetsPath);
      await pollJob(job.id, (j) => {
        setUploadProgress(j.progress);
        setUploadMessage(j.message);
      });
      setProductsCache({});
      setExpanded(null);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      setUploadMessage("");
      setUploadProgress(0);
    }
  };

  const onReindex = async (path: string) => {
    if (reindexingPath !== null) return;

    setReindexingPath(path);
    setError(null);
    setReindexMessage("Повторная индексация…");
    setReindexProgress(0);
    try {
      const job = await reindexAsset(path, settings.assetsPath);
      await pollJob(job.id, (j) => {
        setReindexProgress(j.progress);
        setReindexMessage(j.message);
      });
      setProductsCache((prev) => {
        const next = { ...prev };
        delete next[path];
        return next;
      });
      await refreshAsset(path);
      if (expanded === path) {
        void loadProducts(path);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReindexingPath(null);
      setReindexMessage("");
      setReindexProgress(0);
    }
  };

  const onDelete = async (path: string) => {
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

      <div className={`${pageActionBarClass} mb-14`}>
        <button
          type="button"
          disabled={uploading || reindexingPath !== null}
          onClick={onAddClick}
          className={btnActionClass}
        >
          {uploading ? "Индексация…" : "Добавить"}
        </button>

        {uploading && (
          <div className="grid w-full max-w-md gap-2">
            <p className={`text-center ${mutedTextClass}`}>{uploadMessage}</p>
            <div className={progressTrackClass}>
              <div
                className={progressBarClass}
                style={{ width: `${Math.round(uploadProgress * 100)}%` }}
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
          <ul className="grid list-none gap-4 p-0">
            {files.map((file) => {
              const products = productsCache[file.path];
              const isOpen = expanded === file.path;
              const isReindexing = reindexingPath === file.path;
              return (
                <li key={file.path}>
                  <div className={assetItemClass}>
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
                          isReindexing
                            ? reindexMessage || "Повторная индексация…"
                            : file.catalog_name,
                          !isReindexing && file.doc_kind,
                          !isReindexing &&
                            (file.indexed
                              ? `${file.product_count} продуктов`
                              : "не проиндексирован"),
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </p>
                    </button>
                    <button
                      type="button"
                      disabled={reindexingPath !== null}
                      onClick={() => void onReindex(file.path)}
                      className={`${btnIconNeutralClass} size-9 shrink-0 p-0`}
                      aria-label={`Повторная индексация ${fileName(file.path)}`}
                      title="Повторная индексация"
                    >
                      <svg
                        className={`size-4 ${isReindexing ? "animate-spin" : ""}`}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.25"
                        aria-hidden
                      >
                        <path
                          d="M1 4v6h6M23 20v-6h-6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4-4.64 4.36A9 9 0 0 1 3.51 15"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                    <button
                      type="button"
                      disabled={isReindexing}
                      onClick={() => void onDelete(file.path)}
                      className={`${btnOutlineDangerClass} inline-flex size-9 shrink-0 items-center justify-center p-0`}
                      aria-label={`Удалить ${file.path}`}
                    >
                      <svg
                        className="size-4"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        aria-hidden
                      >
                        <path
                          d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path d="M10 11v6M14 11v6" strokeLinecap="round" />
                      </svg>
                    </button>
                  </div>

                  {isReindexing && (
                    <div className="mt-3 grid gap-2">
                      <div className={progressTrackClass}>
                        <div
                          className={progressBarClass}
                          style={{ width: `${Math.round(reindexProgress * 100)}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {isOpen && (
                    <div className="mt-4 border-t border-gray-400 pt-4">
                      {loadingProducts === file.path && !products && (
                        <p className={mutedTextClass}>Загрузка продуктов…</p>
                      )}
                      {products && (
                        <div className="grid gap-4">
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
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
