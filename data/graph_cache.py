import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp

try:
    from .graph_features import (
        GRAPH_FEATURES_SCHEMA_VERSION,
        compute_graph_features,
        validate_graph_features,
    )
except ImportError:
    from graph_features import (
        GRAPH_FEATURES_SCHEMA_VERSION,
        compute_graph_features,
        validate_graph_features,
    )


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_communities(communities: List[List[int]], num_nodes: int) -> None:
    if not communities:
        raise ValueError("社区划分为空。")
    seen = set()
    for community_id, nodes in enumerate(communities):
        if not nodes:
            raise ValueError(f"社区 {community_id} 为空。")
        for node in nodes:
            node = int(node)
            if node < 0 or node >= num_nodes:
                raise ValueError(f"社区 {community_id} 包含越界节点 {node}。")
            if node in seen:
                raise ValueError(f"节点 {node} 同时出现在多个社区中。")
            seen.add(node)
    expected = set(range(num_nodes))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"社区划分未完整覆盖图节点；缺失 {missing[:10]}，额外 {extra[:10]}。"
        )


def detect_best_louvain(
        graph: nx.Graph,
        runs: int = 20,
        resolution: float = 1.0,
        random_seed: int = 42
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    if runs < 1:
        raise ValueError("louvain_runs 必须大于或等于 1。")
    if resolution <= 0:
        raise ValueError("resolution 必须大于 0。")

    try:
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community.quality import modularity
    except ImportError as exc:
        raise RuntimeError("当前 NetworkX 版本不支持 Louvain。") from exc

    best_communities = None
    best_modularity = -float("inf")
    best_seed = None
    run_results = []
    print(f"[社区] 次数={runs} | 分辨率={resolution}")
    for run_index in range(runs):
        run_seed = int(random_seed + run_index)
        communities = [
            sorted(int(node) for node in community)
            for community in louvain_communities(
                graph,
                weight=None,
                resolution=resolution,
                seed=run_seed
            )
        ]
        validate_communities(communities, graph.number_of_nodes())
        q_value = float(modularity(
            graph,
            communities,
            weight=None,
            resolution=resolution
        ))
        result = {
            "run": run_index + 1,
            "seed": run_seed,
            "modularity": q_value,
            "num_communities": len(communities)
        }
        run_results.append(result)
        if q_value > best_modularity:
            best_modularity = q_value
            best_communities = communities
            best_seed = run_seed

    best_communities = sorted(best_communities, key=lambda nodes: nodes[0])
    validate_communities(best_communities, graph.number_of_nodes())
    community_dict = {
        community_id: nodes
        for community_id, nodes in enumerate(best_communities)
    }
    metadata = {
        "algorithm": "louvain",
        "runs": int(runs),
        "resolution": float(resolution),
        "weight": None,
        "base_random_seed": int(random_seed),
        "best_seed": int(best_seed),
        "best_modularity": float(best_modularity),
        "run_results": run_results
    }
    print(
        f"[社区] Q={best_modularity:.8f} | "
        f"社区={len(community_dict)} | seed={best_seed}"
    )
    return community_dict, metadata


def prepare_graph(
        edges_file: Path,
        cache_file: Path,
        louvain_runs: int,
        resolution: float,
        random_seed: int,
        directed: bool = False,
        expected_nodes: int = None,
        expected_edges: int = None,
        force: bool = False
) -> Dict[str, Any]:
    """Preprocess a graph once and cache its CSR matrix and fixed partition."""
    edges_file = Path(edges_file).resolve()
    cache_file = Path(cache_file).resolve()
    source_hash = file_sha256(edges_file)
    signature = {
        "source_sha256": source_hash,
        "directed_input": bool(directed),
        "make_unweighted": True,
        "make_undirected": True,
        "no_self_loops": True,
        "select_lcc": False,
        "louvain_runs": int(louvain_runs),
        "resolution": float(resolution),
        "random_seed": int(random_seed)
    }

    if cache_file.exists() and not force:
        with cache_file.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("cache_signature") == signature:
            try:
                validate_graph_features(
                    cached.get("graph_features"), cached["adj"]
                )
            except (KeyError, TypeError, ValueError):
                pass
            else:
                print(f"[缓存] 读取 {cache_file}")
                return cached
        print("[缓存] 重建")

    create_using = nx.DiGraph if directed else nx.Graph
    graph_raw = nx.read_edgelist(
        edges_file,
        nodetype=int,
        create_using=create_using()
    )
    original_node_ids = np.asarray(list(graph_raw.nodes()), dtype=np.int64)
    mapping = {
        original_id: internal_id
        for internal_id, original_id in enumerate(original_node_ids.tolist())
    }
    graph_raw = nx.relabel_nodes(graph_raw, mapping, copy=True)
    adjacency_raw = sp.csr_matrix(
        nx.to_scipy_sparse_array(graph_raw, format="csr", dtype=np.float32)
    )
    adjacency = adjacency_raw.maximum(adjacency_raw.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.0)
    adjacency = adjacency.astype(np.float32)
    adjacency.sort_indices()
    del graph_raw, adjacency_raw

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("清洗后的邻接矩阵不是方阵。")
    if adjacency.diagonal().any():
        raise ValueError("清洗后的邻接矩阵仍包含自环。")
    if (adjacency != adjacency.T).nnz != 0:
        raise ValueError("清洗后的邻接矩阵不是对称矩阵。")
    if adjacency.nnz and not np.all(adjacency.data == 1):
        raise ValueError("清洗后的邻接矩阵仍包含非 1 边权。")

    num_nodes = int(adjacency.shape[0])
    num_edges = int(adjacency.nnz // 2)
    if expected_nodes is not None and num_nodes != int(expected_nodes):
        raise ValueError(f"节点数不符：期望 {expected_nodes}，实际 {num_nodes}。")
    if expected_edges is not None and num_edges != int(expected_edges):
        raise ValueError(f"边数不符：期望 {expected_edges}，实际 {num_edges}。")

    graph_clean = nx.from_scipy_sparse_array(adjacency)
    community_dict, community_metadata = detect_best_louvain(
        graph_clean,
        runs=louvain_runs,
        resolution=resolution,
        random_seed=random_seed
    )
    graph_features = compute_graph_features(graph_clean, adjacency)
    graph_data = {
        "adj": adjacency,
        "degrees": np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float32),
        "communities": community_dict,
        "community_detection": community_metadata,
        "graph_features": graph_features,
        "graph_features_schema_version": GRAPH_FEATURES_SCHEMA_VERSION,
        "original_node_ids": original_node_ids,
        "cache_signature": signature,
        "num_nodes": num_nodes,
        "num_edges": num_edges
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(graph_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_file)
    print(f"[缓存] 保存 {cache_file}")
    return graph_data
