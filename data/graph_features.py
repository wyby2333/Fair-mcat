import hashlib
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp


GRAPH_FEATURES_SCHEMA_VERSION = 1


def adjacency_fingerprint(adjacency: sp.csr_matrix) -> str:
    """Return a stable fingerprint for the exact CSR graph representation."""
    adjacency = adjacency.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(adjacency.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(adjacency.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(adjacency.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(adjacency.data, dtype=np.float32).tobytes())
    return digest.hexdigest()


def compute_graph_features(graph, adjacency: sp.csr_matrix) -> Dict[str, Any]:
    """计算模型与基线所需的图特征。"""
    import networkx as nx

    num_nodes = int(adjacency.shape[0])
    if graph.number_of_nodes() != num_nodes:
        raise ValueError("NetworkX 图与邻接矩阵的节点数不一致。")

    clustering = nx.clustering(graph)
    degree_centrality = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph)

    # 排序结果用于构造 Degree 与 PageRank 基线。
    degree_order = sorted(
        degree_centrality,
        key=degree_centrality.get,
        reverse=True,
    )
    pagerank_order = sorted(pagerank, key=pagerank.get, reverse=True)

    return {
        "schema_version": GRAPH_FEATURES_SCHEMA_VERSION,
        "adjacency_fingerprint": adjacency_fingerprint(adjacency),
        "clustering_coefficients": np.asarray(
            [clustering[node] for node in range(num_nodes)],
            dtype=np.float32,
        ),
        "degree_centrality": np.asarray(
            [degree_centrality[node] for node in range(num_nodes)],
            dtype=np.float32,
        ),
        "degree_order": np.asarray(degree_order, dtype=np.int64),
        "pagerank_scores": np.asarray(
            [pagerank[node] for node in range(num_nodes)],
            dtype=np.float32,
        ),
        "pagerank_order": np.asarray(pagerank_order, dtype=np.int64),
        "metadata": {
            "schema_version": GRAPH_FEATURES_SCHEMA_VERSION,
            "networkx_version": nx.__version__,
            "clustering_call": "networkx.clustering(graph)",
            "degree_call": "networkx.degree_centrality(graph)",
            "pagerank_call": "networkx.pagerank(graph)",
        },
    }


def validate_graph_features(
        graph_features: Dict[str, Any],
        adjacency: sp.csr_matrix,
) -> None:
    if not isinstance(graph_features, dict):
        raise TypeError("图特征缓存必须是字典。")
    num_nodes = int(adjacency.shape[0])
    if int(graph_features.get("schema_version", -1)) != GRAPH_FEATURES_SCHEMA_VERSION:
        raise ValueError("图特征缓存版本不受支持。")
    if graph_features.get("adjacency_fingerprint") != adjacency_fingerprint(adjacency):
        raise ValueError("图特征缓存与当前邻接矩阵不匹配。")

    expected_lengths = {
        "clustering_coefficients": num_nodes,
        "degree_centrality": num_nodes,
        "degree_order": num_nodes,
        "pagerank_scores": num_nodes,
        "pagerank_order": num_nodes,
    }
    for key, expected_length in expected_lengths.items():
        if key not in graph_features:
            raise ValueError(f"图特征缓存缺少字段: {key}")
        if np.asarray(graph_features[key]).shape != (expected_length,):
            raise ValueError(f"图特征缓存字段 {key} 的形状不正确。")
