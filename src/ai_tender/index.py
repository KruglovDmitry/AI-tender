"""LlamaIndex индекс эталонов (с дисковым кэшем) и гибридный поиск по тендеру."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from .loaders import load_documents, split_documents


def configure_embeddings(model_name: str, device: str | None = None) -> HuggingFaceEmbedding:
    kwargs = {"model_name": model_name, "embed_batch_size": 8}
    if device:
        kwargs["device"] = device
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
) -> tuple[VectorStoreIndex, list[BaseNode], list[str]]:
    documents, warnings = load_documents(folder, corpus=corpus, technical_only=technical_only)
    if not documents:
        details = "; ".join(warnings) if warnings else "файлы не найдены или пусты"
        raise ValueError(f"Не удалось извлечь документы из {folder}. {details}")
    nodes = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not nodes:
        raise ValueError(f"После разбиения на чанки документов нет: {folder}")
    index = VectorStoreIndex(nodes)
    return index, nodes, warnings


def load_or_build_assets_index(
    assets_path: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str | None = None,
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
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index, nodes, warnings, False


def build_tender_index(
    tender_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[VectorStoreIndex, list[BaseNode], list[str]]:
    return build_index_from_folder(
        tender_path,
        corpus="tender",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        technical_only=True,
    )


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


def hybrid_retrieve(index: VectorStoreIndex, query: str, top_k: int) -> list[NodeWithScore]:
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    try:
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=top_k,
            language="russian",
        )
    except Exception:
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=top_k,
        )
    vector_hits = vector_retriever.retrieve(query)
    bm25_hits = bm25_retriever.retrieve(query)
    return _rrf_fuse([vector_hits, bm25_hits], top_k=top_k)


def select_asset_query_nodes(nodes: list[BaseNode], limit: int) -> list[BaseNode]:
    """Берём равномерно распределённые чанки эталона как поисковые запросы."""
    if not nodes:
        return []
    if len(nodes) <= limit:
        return list(nodes)
    step = len(nodes) / limit
    selected = []
    for index in range(limit):
        selected.append(nodes[int(index * step)])
    return selected


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


def retrieve_candidates(
    asset_nodes: list[BaseNode],
    tender_index: VectorStoreIndex,
    top_k: int,
    min_score: float,
) -> list[tuple[BaseNode, list[NodeWithScore]]]:
    results: list[tuple[BaseNode, list[NodeWithScore]]] = []
    for asset_node in asset_nodes:
        query = asset_node.get_content(metadata_mode="none")
        # короткий запрос лучше для BM25 по моделям/артикулам
        query = " ".join(query.split())[:800]
        hits = hybrid_retrieve(tender_index, query, top_k=top_k)
        filtered = [hit for hit in hits if (hit.score or 0) >= min_score or hit.score is None]
        if not filtered:
            filtered = hits[:top_k]
        if filtered:
            results.append((asset_node, filtered))
    return results
