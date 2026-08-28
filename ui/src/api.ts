import type {
  AnalysisReport,
  AppConfig,
  AssetFile,
  Job,
  ProductDocumentIndex,
} from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    let message = text || response.statusText;
    try {
      const body = JSON.parse(text) as { detail?: string | { msg?: string }[] };
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail) && body.detail[0]?.msg) {
        message = body.detail.map((item) => item.msg).filter(Boolean).join("; ");
      }
    } catch {
      /* keep raw text */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function fetchConfig(): Promise<AppConfig> {
  return request<AppConfig>("/api/config");
}

export async function fetchOcrStatus(): Promise<{ ok: boolean; hint: string }> {
  return request("/api/ocr-status");
}

export async function fetchAssets(assetsPath?: string): Promise<{
  assets_path: string;
  files: AssetFile[];
}> {
  const query = assetsPath ? `?assets_path=${encodeURIComponent(assetsPath)}` : "";
  return request(`/api/assets${query}`);
}

export async function fetchProducts(
  path: string,
  assetsPath?: string,
): Promise<ProductDocumentIndex> {
  const params = new URLSearchParams({ path });
  if (assetsPath) params.set("assets_path", assetsPath);
  return request(`/api/assets/products?${params}`);
}

export async function deleteAsset(path: string, assetsPath?: string): Promise<void> {
  const query = assetsPath ? `?assets_path=${encodeURIComponent(assetsPath)}` : "";
  await request(`/api/assets/delete${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

export async function uploadAssets(files: File[], assetsPath?: string): Promise<Job> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  if (assetsPath) form.append("assets_path", assetsPath);
  return request<Job>("/api/assets/upload", { method: "POST", body: form });
}

export async function reindexAsset(path: string, assetsPath?: string): Promise<Job> {
  const query = assetsPath ? `?assets_path=${encodeURIComponent(assetsPath)}` : "";
  return request<Job>(`/api/assets/reindex${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

export async function startAnalyze(form: FormData): Promise<Job> {
  return request<Job>("/api/analyze", { method: "POST", body: form });
}

export async function fetchJob(jobId: string): Promise<Job> {
  return request<Job>(`/api/jobs/${jobId}`);
}

export async function pollJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  intervalMs = 800,
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const job = await fetchJob(jobId);
        onUpdate(job);
        if (job.status === "done") {
          resolve(job);
          return;
        }
        if (job.status === "failed") {
          reject(new Error(job.error || "Задача завершилась с ошибкой"));
          return;
        }
        setTimeout(tick, intervalMs);
      } catch (error) {
        reject(error);
      }
    };
    void tick();
  });
}

export async function downloadReport(
  report: AnalysisReport,
  format: "md" | "json",
): Promise<Blob> {
  const response = await fetch(`/api/reports/download?format=${format}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(report),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.blob();
}
