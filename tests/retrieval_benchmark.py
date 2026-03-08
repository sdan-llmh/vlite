from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings, semantic_search_usearch
from sentence_transformers.util import semantic_search

from bench import load_rust_module
from retrieval_eval import compute_retrieval_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_quora_retrieval_dataset(max_rows: int = 800, max_queries: int = 100) -> dict:
    """
    Build a lightweight retrieval benchmark from the public sentence-transformers Quora duplicates dataset.

    We use unique `sentence2` entries as corpus documents and duplicate (`label == 1`) pairs as
    query -> relevant document mappings.
    """
    rows = load_dataset(
        "sentence-transformers/quora-duplicates",
        "pair-class",
        split=f"train[:{max_rows}]",
    )

    corpus: list[str] = []
    corpus_map: dict[str, int] = {}
    for row in rows:
        document = row["sentence2"].strip()
        if document and document not in corpus_map:
            corpus_map[document] = len(corpus)
            corpus.append(document)

    queries: list[str] = []
    relevant_docs: dict[int, set[int]] = {}
    for row in rows:
        if row["label"] != 1:
            continue
        query = row["sentence1"].strip()
        document = row["sentence2"].strip()
        if not query or document not in corpus_map:
            continue
        query_id = len(queries)
        queries.append(query)
        relevant_docs[query_id] = {corpus_map[document]}
        if len(queries) >= max_queries:
            break

    return {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs,
    }


def benchmark_vlite(dataset: dict, top_k: int = 10) -> dict:
    rust_module = load_rust_module()
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    relevant_docs = dataset["relevant_docs"]

    with tempfile.TemporaryDirectory() as tmpdir:
        db = rust_module.open_local(tmpdir)
        document_ids = [f"doc-{index}" for index in range(len(corpus))]

        ingest_start = time.perf_counter()
        db.add_many(corpus, None, document_ids)
        ingest_seconds = time.perf_counter() - ingest_start

        rankings: list[list[int]] = []
        search_times = []
        for query in queries:
            query_start = time.perf_counter()
            hits = db.search(query, top_k, None)
            search_times.append(time.perf_counter() - query_start)
            rankings.append([int(hit.document_id.split("-")[1]) for hit in hits])

        metrics = compute_retrieval_metrics(
            rankings,
            relevant_docs,
            accuracy_at_k=[1, 10],
            precision_recall_at_k=[10],
            mrr_at_k=[10],
            ndcg_at_k=[10],
            map_at_k=[10],
        )

        return {
            "system": "vlite_rust_hashed_exact",
            "corpus_size": len(corpus),
            "query_size": len(queries),
            "ingest_seconds": ingest_seconds,
            "avg_query_seconds": statistics.mean(search_times),
            "p95_query_seconds": sorted(search_times)[max(0, int(len(search_times) * 0.95) - 1)],
            **metrics,
        }


def benchmark_sentence_transformers(dataset: dict, top_k: int = 10) -> tuple[dict, dict]:
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    relevant_docs = dataset["relevant_docs"]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    corpus_encode_start = time.perf_counter()
    corpus_embeddings = model.encode(
        corpus,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    corpus_encode_seconds = time.perf_counter() - corpus_encode_start

    query_encode_start = time.perf_counter()
    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    query_encode_seconds = time.perf_counter() - query_encode_start

    exact_search_start = time.perf_counter()
    exact_hits = semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)
    exact_search_seconds = time.perf_counter() - exact_search_start
    exact_rankings = [[hit["corpus_id"] for hit in hits] for hits in exact_hits]
    exact_metrics = compute_retrieval_metrics(
        exact_rankings,
        relevant_docs,
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
    )

    quantized_corpus = quantize_embeddings(corpus_embeddings, precision="int8")
    usearch_hits, usearch_search_seconds = semantic_search_usearch(
        query_embeddings,
        corpus_embeddings=quantized_corpus,
        corpus_precision="int8",
        top_k=top_k,
        calibration_embeddings=corpus_embeddings,
        rescore=True,
        rescore_multiplier=4,
        exact=False,
    )
    usearch_rankings = [[hit["corpus_id"] for hit in hits] for hits in usearch_hits]
    usearch_metrics = compute_retrieval_metrics(
        usearch_rankings,
        relevant_docs,
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
    )

    exact_result = {
        "system": "sentence_transformers_exact_semantic_search",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "corpus_encode_seconds": corpus_encode_seconds,
        "query_encode_seconds": query_encode_seconds,
        "search_seconds": exact_search_seconds,
        **exact_metrics,
    }
    usearch_result = {
        "system": "sentence_transformers_usearch_int8_rescored",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "corpus_encode_seconds": corpus_encode_seconds,
        "query_encode_seconds": query_encode_seconds,
        "search_seconds": usearch_search_seconds,
        **usearch_metrics,
    }
    return exact_result, usearch_result


def run_retrieval_benchmark(max_rows: int = 800, max_queries: int = 100, top_k: int = 10) -> list[dict]:
    dataset = build_quora_retrieval_dataset(max_rows=max_rows, max_queries=max_queries)
    vlite_result = benchmark_vlite(dataset, top_k=top_k)
    st_exact_result, st_usearch_result = benchmark_sentence_transformers(dataset, top_k=top_k)
    return [vlite_result, st_exact_result, st_usearch_result]


if __name__ == "__main__":
    results = run_retrieval_benchmark()
    print(json.dumps(results, indent=2))
