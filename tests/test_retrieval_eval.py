from __future__ import annotations

import numpy as np

from retrieval_eval import compute_retrieval_metrics, cosine_rankings


def test_cosine_rankings_matches_bruteforce_topk():
    rng = np.random.default_rng(seed=42)
    corpus = rng.normal(size=(100, 32))
    queries = rng.normal(size=(10, 32))

    rankings = cosine_rankings(queries, corpus, top_k=5)

    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    query_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    scores = query_norm @ corpus_norm.T
    expected = np.argsort(-scores, axis=1)[:, :5].tolist()

    assert rankings == expected


def test_information_retrieval_metrics_match_sentence_transformers_expected_values():
    queries = {
        0: "What is a pokemon?",
        1: "What is a vegetable?",
        2: "What is a fruit?",
        3: "What is a vehicle?",
        4: "What is a car?",
    }
    corpus = {
        0: "A pokemon is a fictional creature",
        1: "A vegetable is a plant",
        2: "A fruit is a plant",
        3: "A vehicle is a machine",
        4: "A car is a vehicle",
    }
    relevant_docs = {0: {0}, 1: {1}, 2: {2}, 3: {3, 4}, 4: {4}}

    one_hot_encodings = {
        "pokemon": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "car": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "vehicle": np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "fruit": np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "vegetable": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }

    def encode(text: str) -> np.ndarray:
        vector = np.zeros(5, dtype=np.float32)
        for keyword, value in one_hot_encodings.items():
            if keyword in text:
                vector += value
        return vector

    query_embeddings = np.stack([encode(queries[idx].lower()) for idx in sorted(queries)])
    corpus_embeddings = np.stack([encode(corpus[idx].lower()) for idx in sorted(corpus)])
    rankings = cosine_rankings(query_embeddings, corpus_embeddings, top_k=5)

    metrics = compute_retrieval_metrics(
        rankings,
        relevant_docs,
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )

    expected = {
        "accuracy@1": 1.0,
        "accuracy@3": 1.0,
        "precision@1": 1.0,
        "precision@3": 0.4,
        "recall@1": 0.9,
        "recall@3": 1.0,
        "ndcg@3": 1.0,
        "mrr@3": 1.0,
        "map@5": 1.0,
    }
    assert metrics == expected
