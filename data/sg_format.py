import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

try:
    from .config import DIFFUSION_FILE_SUFFIX
except ImportError:
    from config import DIFFUSION_FILE_SUFFIX


SG_FORMAT_VERSION = 2


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            f"读取或写入 {DIFFUSION_FILE_SUFFIX} v2 需要 h5py。"
            "请在项目环境中安装 h5py。"
        ) from exc
    return h5py


def is_sg_v2(path: str) -> bool:
    h5py = _h5py()
    path = str(path)
    return os.path.isfile(path) and h5py.is_hdf5(path)


def _encode_communities(communities: Dict[int, List[int]]):
    ordered = [communities[key] for key in sorted(communities, key=int)]
    offsets = [0]
    nodes = []
    for community in ordered:
        nodes.extend(int(node) for node in community)
        offsets.append(len(nodes))
    return np.asarray(offsets, dtype=np.int64), np.asarray(nodes, dtype=np.int64)


def _decode_communities(offsets: np.ndarray, nodes: np.ndarray) -> List[List[int]]:
    return [
        nodes[offsets[index]:offsets[index + 1]].astype(np.int64).tolist()
        for index in range(len(offsets) - 1)
    ]


class SGWriterV2:
    """Chunked, resumable writer for the single-file HDF5 .IM v2 format."""

    def __init__(
            self,
            output_path: str,
            adjacency: sp.csr_matrix,
            communities: Dict[int, List[int]],
            original_node_ids: np.ndarray,
            metadata: Dict[str, Any],
            seed_size: int,
            target_samples: int,
            target_dtype: str = "float32",
            compression: Optional[str] = "lzf",
            resume: bool = False,
            overwrite: bool = False,
            graph_features: Dict[str, Any] = None
    ):
        h5py = _h5py()
        self.output_path = Path(output_path).resolve()
        self.partial_path = Path(str(self.output_path) + ".partial")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.target_samples = int(target_samples)
        self.seed_size = int(seed_size)
        self.num_nodes = int(adjacency.shape[0])
        self.target_dtype = np.dtype(target_dtype)

        if overwrite:
            for candidate in (self.output_path, self.partial_path):
                if candidate.exists():
                    candidate.unlink()

        if resume and self.output_path.exists() and not self.partial_path.exists():
            self.output_path.replace(self.partial_path)

        if resume and self.partial_path.exists():
            self.handle = h5py.File(self.partial_path, "r+")
            if int(self.handle.attrs["sg_format_version"]) != SG_FORMAT_VERSION:
                raise ValueError(
                    f"不能续写非 {DIFFUSION_FILE_SUFFIX} v2 文件。"
                )
            if int(self.handle.attrs["seed_size"]) != self.seed_size:
                raise ValueError("续写文件的 seed_size 与当前参数不一致。")
            if int(self.handle.attrs["num_nodes"]) != self.num_nodes:
                raise ValueError("续写文件的节点数与当前图不一致。")
            self._ensure_graph_features(graph_features)
            self._resize(self.target_samples)
            return

        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(
                f"输出已存在: {self.output_path}。请使用 --resume 或 --overwrite。"
            )

        self.handle = h5py.File(self.partial_path, "w")
        self.handle.attrs["sg_format_version"] = SG_FORMAT_VERSION
        self.handle.attrs["completed_samples"] = 0
        self.handle.attrs["target_samples"] = self.target_samples
        self.handle.attrs["seed_size"] = self.seed_size
        self.handle.attrs["num_nodes"] = self.num_nodes
        self.handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

        graph_group = self.handle.create_group("graph")
        adjacency = adjacency.astype(np.float32).tocsr()
        graph_group.create_dataset("indptr", data=adjacency.indptr.astype(np.int64))
        graph_group.create_dataset("indices", data=adjacency.indices.astype(np.int64))
        graph_group.create_dataset("data", data=adjacency.data.astype(np.float32))
        graph_group.create_dataset(
            "original_node_ids",
            data=np.asarray(original_node_ids, dtype=np.int64)
        )
        self._ensure_graph_features(graph_features)

        offsets, community_nodes = _encode_communities(communities)
        community_group = self.handle.create_group("communities")
        community_group.create_dataset("offsets", data=offsets)
        community_group.create_dataset("nodes", data=community_nodes)

        sample_group = self.handle.create_group("samples")
        sample_group.create_dataset(
            "seed_indices",
            shape=(self.target_samples, self.seed_size),
            maxshape=(None, self.seed_size),
            dtype=np.int32,
            chunks=(min(256, max(1, self.target_samples)), self.seed_size),
            compression=compression
        )
        sample_group.create_dataset(
            "influence_probs",
            shape=(self.target_samples, self.num_nodes),
            maxshape=(None, self.num_nodes),
            dtype=self.target_dtype,
            chunks=(1, self.num_nodes),
            compression=compression
        )
        self.handle.flush()

    def _ensure_graph_features(
            self,
            graph_features: Dict[str, Any],
    ) -> None:
        if graph_features is None:
            raise ValueError("缺少 graph_features。")
        graph_group = self.handle["graph"]
        if "features" in graph_group:
            return
        feature_group = graph_group.create_group("features")
        feature_group.attrs["schema_version"] = int(
            graph_features["schema_version"]
        )
        feature_group.attrs["adjacency_fingerprint"] = str(
            graph_features["adjacency_fingerprint"]
        )
        feature_group.attrs["metadata_json"] = json.dumps(
            graph_features.get("metadata", {}), ensure_ascii=False
        )
        feature_group.create_dataset(
            "clustering_coefficients",
            data=np.asarray(
                graph_features["clustering_coefficients"], dtype=np.float32
            ),
        )
        feature_group.create_dataset(
            "degree_centrality",
            data=np.asarray(graph_features["degree_centrality"], dtype=np.float32),
        )
        feature_group.create_dataset(
            "degree_order",
            data=np.asarray(graph_features["degree_order"], dtype=np.int64),
        )
        feature_group.create_dataset(
            "pagerank_scores",
            data=np.asarray(graph_features["pagerank_scores"], dtype=np.float32),
        )
        feature_group.create_dataset(
            "pagerank_order",
            data=np.asarray(graph_features["pagerank_order"], dtype=np.int64),
        )

    @property
    def completed_samples(self) -> int:
        return int(self.handle.attrs["completed_samples"])

    def _resize(self, target_samples: int) -> None:
        current_size = int(self.handle["samples/seed_indices"].shape[0])
        if target_samples < self.completed_samples:
            raise ValueError("target_samples 不能小于已完成样本数。")
        if target_samples != current_size:
            self.handle["samples/seed_indices"].resize(target_samples, axis=0)
            self.handle["samples/influence_probs"].resize(target_samples, axis=0)
        self.target_samples = int(target_samples)
        self.handle.attrs["target_samples"] = self.target_samples
        self.handle.flush()

    def write_batch(
            self,
            start_index: int,
            seed_indices: np.ndarray,
            influence_probs: np.ndarray
    ) -> None:
        start_index = int(start_index)
        count = int(seed_indices.shape[0])
        end_index = start_index + count
        if start_index != self.completed_samples:
            raise ValueError(
                f"写入起点 {start_index} 与 completed_samples={self.completed_samples} 不一致。"
            )
        if end_index > self.target_samples:
            raise ValueError("写入样本超过 target_samples。")
        if seed_indices.shape != (count, self.seed_size):
            raise ValueError("seed_indices 形状不正确。")
        if influence_probs.shape != (count, self.num_nodes):
            raise ValueError("influence_probs 形状不正确。")
        self.handle["samples/seed_indices"][start_index:end_index] = seed_indices
        self.handle["samples/influence_probs"][start_index:end_index] = influence_probs
        self.handle.attrs["completed_samples"] = end_index
        self.handle.flush()

    def finalize(self) -> None:
        if self.completed_samples != self.target_samples:
            raise RuntimeError(
                f"仅完成 {self.completed_samples}/{self.target_samples} 条样本，不能完成文件。"
            )
        self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        self.partial_path.replace(self.output_path)

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            try:
                self.handle.close()
            finally:
                self.handle = None


def load_sg_graph(path: str) -> Dict[str, Any]:
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("sg_format_version", -1)) != SG_FORMAT_VERSION:
            raise ValueError(
                f"不支持的 {DIFFUSION_FILE_SUFFIX} 格式版本。"
            )
        num_nodes = int(handle.attrs["num_nodes"])
        graph_group = handle["graph"]
        adjacency = sp.csr_matrix(
            (
                graph_group["data"][:],
                graph_group["indices"][:],
                graph_group["indptr"][:]
            ),
            shape=(num_nodes, num_nodes)
        )
        communities = _decode_communities(
            handle["communities/offsets"][:],
            handle["communities/nodes"][:]
        )
        metadata = json.loads(handle.attrs["metadata_json"])
        if "features" not in graph_group:
            raise ValueError("数据文件缺少 graph/features。")
        feature_group = graph_group["features"]
        graph_features = {
            "schema_version": int(feature_group.attrs["schema_version"]),
            "adjacency_fingerprint": str(
                feature_group.attrs["adjacency_fingerprint"]
            ),
            "clustering_coefficients": feature_group[
                "clustering_coefficients"
            ][:],
            "degree_centrality": feature_group["degree_centrality"][:],
            "degree_order": feature_group["degree_order"][:],
            "pagerank_scores": feature_group["pagerank_scores"][:],
            "pagerank_order": feature_group["pagerank_order"][:],
            "metadata": json.loads(
                feature_group.attrs.get("metadata_json", "{}")
            ),
        }
        return {
            "adj": adjacency,
            "communities": communities,
            "community_detection": metadata.get("community_detection"),
            "metadata": metadata,
            "original_node_ids": graph_group["original_node_ids"][:],
            "num_samples": int(handle.attrs["completed_samples"]),
            "seed_size": int(handle.attrs["seed_size"]),
            "num_nodes": num_nodes,
            "graph_features": graph_features,
        }


class SGDatasetV2(Dataset):
    """Lazy HDF5 dataset. Each DataLoader process opens its own file handle."""

    def __init__(self, path: str):
        h5py = _h5py()
        self.path = str(Path(path).resolve())
        with h5py.File(self.path, "r") as handle:
            self.length = int(handle.attrs["completed_samples"])
            self.seed_size = int(handle.attrs["seed_size"])
            self.num_nodes = int(handle.attrs["num_nodes"])
        self._handle = None

    def _file(self):
        if self._handle is None:
            self._handle = _h5py().File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        handle = self._file()
        seeds = np.asarray(handle["samples/seed_indices"][index], dtype=np.int64)
        targets = np.asarray(handle["samples/influence_probs"][index], dtype=np.float32)
        return torch.from_numpy(seeds), torch.from_numpy(targets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        if getattr(self, "_handle", None) is not None:
            try:
                self._handle.close()
            except Exception:
                pass
