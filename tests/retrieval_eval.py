from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def cosine_rankings(
    query_embeddings: Sequence[Sequence[float]],
    corpus_embeddings: Sequence[Sequence[float]],
    top_k: int,
) -> list[list[int]]:
    query = np.asarray(query_embeddings, dtype=np.float32)
    corpus = np.asarray(corpus_embeddings, dtype=np.float32)

    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    scores = query @ corpus.T
    order = np.argsort(-scores, axis=1)[:, :top_k]
    return order.tolist()


def _dcg(relevances: list[int]) -> float:
    score = 0.0
    for index, relevance in enumerate(relevances, start=1):
        if relevance:
            score += relevance / math.log2(index + 1)
    return score


def compute_retrieval_metrics(
    rankings: Sequence[Sequence[int]],
    relevant_docs: dict[int, set[int]],
    *,
    accuracy_at_k: Sequence[int] = (1, 3, 10),
    precision_recall_at_k: Sequence[int] = (1, 3, 10),
    mrr_at_k: Sequence[int] = (10,),
    ndcg_at_k: Sequence[int] = (10,),
    map_at_k: Sequence[int] = (10,),
) -> dict[str, float]:
    num_queries = len(rankings)
    metrics: dict[str, float] = {}

    for k in accuracy_at_k:
        values = []
        for query_id, ranking in enumerate(rankings):
            hits = set(ranking[:k])
            values.append(1.0 if hits & relevant_docs[query_id] else 0.0)
        metrics[f"accuracy@{k}"] = sum(values) / num_queries

    for k in precision_recall_at_k:
        precisions = []
        recalls = []
        for query_id, ranking in enumerate(rankings):
            relevant = relevant_docs[query_id]
            top_k_hits = ranking[:k]
            true_positives = sum(1 for doc_id in top_k_hits if doc_id in relevant)
            precisions.append(true_positives / k)
            recalls.append(true_positives / len(relevant) if relevant else 0.0)
        metrics[f"precision@{k}"] = sum(precisions) / num_queries
        metrics[f"recall@{k}"] = sum(recalls) / num_queries

    for k in mrr_at_k:
        values = []
        for query_id, ranking in enumerate(rankings):
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(ranking[:k], start=1):
                if doc_id in relevant_docs[query_id]:
                    reciprocal_rank = 1.0 / rank
                    break
            values.append(reciprocal_rank)
        metrics[f"mrr@{k}"] = sum(values) / num_queries

    for k in ndcg_at_k:
        values = []
        for query_id, ranking in enumerate(rankings):
            relevant = relevant_docs[query_id]
            retrieved_relevances = [1 if doc_id in relevant else 0 for doc_id in ranking[:k]]
            ideal_relevances = [1] * min(len(relevant), k)
            ideal = _dcg(ideal_relevances)
            values.append(_dcg(retrieved_relevances) / ideal if ideal else 0.0)
        metrics[f"ndcg@{k}"] = sum(values) / num_queries

    for k in map_at_k:
        values = []
        for query_id, ranking in enumerate(rankings):
            relevant = relevant_docs[query_id]
            hit_count = 0
            precision_sum = 0.0
            for rank, doc_id in enumerate(ranking[:k], start=1):
                if doc_id in relevant:
                    hit_count += 1
                    precision_sum += hit_count / rank
            normalizer = min(len(relevant), k)
            values.append(precision_sum / normalizer if normalizer else 0.0)
        metrics[f"map@{k}"] = sum(values) / num_queries

    return metrics
