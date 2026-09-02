export interface AppConfig {
  llm_provider: string;
  llm_model: string;
  vl_enabled: boolean;
  qwen_vl_model: string;
  max_reqs_per_scope_item: number;
  embedding_model: string;
  default_tender_path: string;
  default_assets_path: string;
  running_in_docker: boolean;
}

export interface AssetFile {
  path: string;
  product_count: number;
  indexed: boolean;
  doc_kind: string | null;
  catalog_name: string;
}

export interface Job {
  id: string;
  kind: string;
  status: "pending" | "running" | "done" | "failed";
  progress: number;
  message: string;
  result?: Record<string, unknown>;
  error?: string | null;
}

export interface Product {
  id: string;
  model: string;
  manufacturer: string;
  category: string;
  canonical_desc: string;
  raw_chunk: string;
  characteristics: string[];
  standards: string[];
}

export interface ProductDocumentIndex {
  source_file: string;
  doc_kind: string;
  catalog_name: string;
  products: Product[];
  product_pages: number[];
  embedding_model: string;
  warnings: string[];
}

export interface AssetHit {
  file: string;
  location: string;
  quote: string;
  score?: number | null;
}

export interface Requirement {
  text: string;
}

export interface PositionMatch {
  scope_name: string;
  qty?: number | null;
  unit?: string;
  status: "matched" | "partial" | "none";
  confidence: number;
  requirements: Requirement[];
  required_product?: string;
  product_name?: string;
  explanation?: string;
  asset_hits?: AssetHit[];
}

export interface AnalysisReport {
  tender_path: string;
  assets_path: string;
  embedding_model: string;
  llm_model: string;
  summary: string;
  verdict: string;
  position_matches: PositionMatch[];
  warnings: string[];
  indexed_files: string[];
  index_reused: boolean;
  elapsed_seconds?: number | null;
  query_selection: Record<string, unknown>;
}

export const STATUS_LABELS: Record<string, string> = {
  matched: "Есть вариант",
  partial: "Частично",
  none: "Нет варианта",
};
