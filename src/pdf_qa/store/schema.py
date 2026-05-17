"""Index mapping definition for the chunk store."""

from __future__ import annotations


def index_body(embedding_dim: int) -> dict:
    """Return the request body for `PUT /<index>` that creates the chunks index.

    - `text` is BM25-indexed for lexical / keyword search.
    - `embedding` is a knn_vector field using HNSW + cosine for k-NN.
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "doc_id":      {"type": "keyword"},
                "doc_name":    {"type": "keyword"},
                "page":        {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "kind":        {"type": "keyword"},
                "text":        {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }


def pages_index_body() -> dict:
    """Return the request body for `PUT /<index>` that creates the pages index.

    One record per (doc_id, page). Holds the raw extracted page text, used by
    keyword_search for accurate literal/regex counting WITHOUT chunk-overlap
    duplication. No embeddings — purely lexical.
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "doc_id":   {"type": "keyword"},
                "doc_name": {"type": "keyword"},
                "page":     {"type": "integer"},
                "text":     {"type": "text"},
            }
        },
    }
