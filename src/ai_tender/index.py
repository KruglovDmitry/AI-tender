"""LlamaIndex индекс эталонов (с дисковым кэшем) и гибридный поиск."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from .loaders import load_documents, split_documents


def _resolve_device(device: str | None) -> str | None:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def configure_embeddings(model_name: str, device: str | None = None) -> HuggingFaceEmbedding:
    # batch 8 на CPU очень медленный для ~1000 чанков; 32 заметно быстрее.
    kwargs: dict = {"model_name": model_name, "embed_batch_size": 32}
    resolved = _resolve_device(device)
    if resolved:
        kwargs["device"] = resolved
    embed_model = HuggingFaceEmbedding(**kwargs)
    Settings.embed_model = embed_model
    return embed_model


def folder_fingerprint(folder: Path) -> str:
    parts: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {
            ".pdf",
            ".docx",
            ".xlsx",
            ".xls",
            ".txt",
            ".md",
            ".csv",
            ".zip",
            ".rar",
        }:
            continue
        relative = path.relative_to(folder).as_posix()
        stat = path.stat()
        parts.append(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def cache_key(folder: Path, embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    raw = (
        f"{folder.resolve()}|{folder_fingerprint(folder)}|{embedding_model}|"
        f"{chunk_size}|{chunk_overlap}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_index_from_folder(
    folder: Path,
    corpus: str,
    chunk_size: int,
    chunk_overlap: int,
    technical_only: bool = False,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[VectorStoreIndex, list[BaseNode], list[str]]:
    documents, warnings = load_documents(
        folder,
        corpus=corpus,
        technical_only=technical_only,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    if not documents:
        details = "; ".join(warnings) if warnings else "файлы не найдены или пусты"
        raise ValueError(f"Не удалось извлечь документы из {folder}. {details}")
    nodes = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not nodes:
        raise ValueError(f"После разбиения на чанки документов нет: {folder}")
    index = VectorStoreIndex(nodes, show_progress=True)
    return index, nodes, warnings


def load_or_build_assets_index(
    assets_path: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str | None = None,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[VectorStoreIndex, list[BaseNode], list[str], bool]:
    configure_embeddings(embedding_model, device)
    key = cache_key(assets_path, embedding_model, chunk_size, chunk_overlap)
    entry_dir = cache_dir / "llama_assets" / key
    meta_path = entry_dir / "meta.json"

    if meta_path.exists() and (entry_dir / "docstore.json").exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("embedding_model") == embedding_model
            and meta.get("chunk_size") == chunk_size
            and meta.get("chunk_overlap") == chunk_overlap
            and meta.get("folder_fingerprint") == folder_fingerprint(assets_path)
        ):
            storage = StorageContext.from_defaults(persist_dir=str(entry_dir))
            index = load_index_from_storage(storage)
            nodes = list(index.docstore.docs.values())
            return index, nodes, list(meta.get("warnings", [])), True

    index, nodes, warnings = build_index_from_folder(
        assets_path,
        corpus="assets",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    entry_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(entry_dir))
    meta_path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "folder_fingerprint": folder_fingerprint(assets_path),
                "assets_path": str(assets_path.resolve()),
                "node_count": len(nodes),
                "indexed_files": sorted(
                    {
                        str(node.metadata.get("file_path") or node.metadata.get("file_name"))
                        for node in nodes
                        if node.metadata
                    }
                ),
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index, nodes, warnings, False


def load_tender_nodes(
    tender_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[list[BaseNode], list[str]]:
    """Тендер только как источник запросов — векторный индекс не нужен."""
    documents, warnings = load_documents(
        tender_path,
        corpus="tender",
        technical_only=True,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    if not documents:
        details = "; ".join(warnings) if warnings else "файлы не найдены или пусты"
        raise ValueError(f"Не удалось извлечь документы из {tender_path}. {details}")
    nodes = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not nodes:
        raise ValueError(f"После разбиения на чанки документов нет: {tender_path}")
    return nodes, warnings


def _rrf_fuse(
    result_lists: list[list[NodeWithScore]],
    top_k: int,
    k: int = 60,
) -> list[NodeWithScore]:
    """Reciprocal Rank Fusion без LLM."""
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            node_id = hit.node.node_id
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)
            current = nodes.get(node_id)
            if current is None or (hit.score or 0) > (current.score or 0):
                nodes[node_id] = hit
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    output: list[NodeWithScore] = []
    for node_id, rrf_score in fused[:top_k]:
        hit = nodes[node_id]
        output.append(NodeWithScore(node=hit.node, score=rrf_score))
    return output


def build_bm25_retriever(index: VectorStoreIndex, top_k: int) -> BM25Retriever:
    try:
        return BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=top_k,
            language="russian",
        )
    except Exception:
        return BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=top_k,
        )


def hybrid_retrieve(
    index: VectorStoreIndex,
    query: str,
    top_k: int,
    bm25_retriever: BM25Retriever | None = None,
) -> list[NodeWithScore]:
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25 = bm25_retriever or build_bm25_retriever(index, top_k)
    vector_hits = vector_retriever.retrieve(query)
    bm25_hits = bm25.retrieve(query)
    return _rrf_fuse([vector_hits, bm25_hits], top_k=top_k)


def select_query_nodes(nodes: list[BaseNode], limit: int) -> list[BaseNode]:
    """Равномерно выбираем чанки как поисковые запросы."""
    if not nodes:
        return []
    substantive = [
        node
        for node in nodes
        if len(" ".join(node.get_content(metadata_mode="none").split())) >= 40
    ]
    pool = substantive or list(nodes)
    if len(pool) <= limit:
        return list(pool)
    step = len(pool) / limit
    return [pool[int(index * step)] for index in range(limit)]


# Обратная совместимость для тестов/импортов.
select_asset_query_nodes = select_query_nodes


def node_to_evidence(node: BaseNode, score: float | None = None):
    from .models import Evidence

    meta = node.metadata or {}
    text = node.get_content(metadata_mode="none")
    quote = " ".join(text.split())
    if len(quote) > 500:
        quote = quote[:499] + "…"
    return Evidence(
        file=str(meta.get("file_path") or meta.get("file_name") or "unknown"),
        location=str(meta.get("location") or "фрагмент"),
        quote=quote,
        score=None if score is None else round(float(score), 4),
    )


def indexed_file_paths(nodes: list[BaseNode]) -> list[str]:
    files = {
        str(node.metadata.get("file_path") or node.metadata.get("file_name"))
        for node in nodes
        if node.metadata
    }
    return sorted(path for path in files if path and path != "None")


def retrieve_candidates(
    query_nodes: list[BaseNode],
    search_index: VectorStoreIndex,
    top_k: int,
    min_score: float = 0.0,
) -> list[tuple[BaseNode, list[NodeWithScore]]]:
    """Для каждого запроса (обычно чанк тендера) ищем хиты в индексе эталонов."""
    del min_score  # RRF-score мал; отсечение по абсолютному порогу вводит в заблуждение
    bm25 = build_bm25_retriever(search_index, top_k)
    vector_retriever = search_index.as_retriever(similarity_top_k=top_k)
    results: list[tuple[BaseNode, list[NodeWithScore]]] = []
    for query_node in query_nodes:
        query = query_node.get_content(metadata_mode="none")
        query = " ".join(query.split())[:800]
        vector_hits = vector_retriever.retrieve(query)
        bm25_hits = bm25.retrieve(query)
        hits = _rrf_fuse([vector_hits, bm25_hits], top_k=top_k)
        if hits:
            results.append((query_node, hits[:top_k]))
    return results
