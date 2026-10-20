"""Generate compact, resumable Fair-MCAT diffusion datasets (.IM v2)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

try:
    from .config import DIFFUSION_FILE_SUFFIX, load_dataset_config, resolve_project_path
    from .graph_cache import prepare_graph
    from .sg_format import SGWriterV2
except ImportError:
    from config import DIFFUSION_FILE_SUFFIX, load_dataset_config, resolve_project_path
    from graph_cache import prepare_graph
    from sg_format import SGWriterV2


DEFAULTS = {
    "diffusion_model": "IC",
    "seed_num": 10,
    "num_samples": 5000,
    "mc_runs": 50,
    "louvain_runs": 20,
    "resolution": 1.0,
    "random_seed": 42,
    "write_batch_size": 64,
    "target_dtype": "float32",
    "compression": "lzf"
}


def run_ic_once(
        adjacency: sp.csr_matrix,
        seeds: Sequence[int],
        rng: np.random.Generator,
        activation_probability: float
) -> np.ndarray:
    """Run one Independent Cascade simulation on an unweighted CSR graph."""
    num_nodes = adjacency.shape[0]
    activated = np.zeros(num_nodes, dtype=np.bool_)
    seed_array = np.asarray(seeds, dtype=np.int64)
    activated[seed_array] = True
    frontier = seed_array

    while frontier.size:
        newly_activated = []
        for node in frontier:
            start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
            neighbors = adjacency.indices[start:end]
            inactive_neighbors = neighbors[~activated[neighbors]]
            if inactive_neighbors.size == 0:
                continue
            successful = inactive_neighbors[
                rng.random(inactive_neighbors.size) < activation_probability
            ]
            if successful.size:
                newly_activated.extend(successful.tolist())
                activated[successful] = True
        if newly_activated:
            frontier = np.unique(np.asarray(newly_activated, dtype=np.int64))
        else:
            frontier = np.empty(0, dtype=np.int64)
    return activated


def run_lt_once(
        adjacency: sp.csr_matrix,
        seeds: Sequence[int],
        rng: np.random.Generator,
        degrees: np.ndarray
) -> np.ndarray:
    """Run one Linear Threshold simulation on an unweighted CSR graph."""
    num_nodes = adjacency.shape[0]
    activated = np.zeros(num_nodes, dtype=np.bool_)
    activated[np.asarray(seeds, dtype=np.int64)] = True
    thresholds = rng.random(num_nodes)

    while True:
        newly_activated = []
        for node in np.flatnonzero(~activated):
            start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
            neighbors = adjacency.indices[start:end]
            if neighbors.size == 0:
                continue
            active_fraction = activated[neighbors].sum() / degrees[node]
            if active_fraction >= thresholds[node]:
                newly_activated.append(int(node))
        if not newly_activated:
            break
        activated[np.asarray(newly_activated, dtype=np.int64)] = True
    return activated


def _sample_uniform(
        rng: np.random.Generator,
        num_nodes: int,
        seed_size: int
) -> np.ndarray:
    return np.sort(rng.choice(num_nodes, size=seed_size, replace=False)).astype(np.int32)


def sample_seeds_diversified(
        rng: np.random.Generator,
        num_nodes: int,
        seed_size: int,
        degrees: np.ndarray,
        communities: Dict[int, List[int]],
        sampling_mix: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
) -> np.ndarray:
    """Sample uniform, degree-biased, concentrated, or community-balanced seeds."""
    if seed_size < 1 or seed_size > num_nodes:
        raise ValueError(f"seed_size 必须位于 1~{num_nodes}。")
    if len(sampling_mix) != 4 or not np.isclose(sum(sampling_mix), 1.0):
        raise ValueError("sampling_mix 必须包含4个且总和为1的概率。")

    draw = rng.random()
    uniform_limit = sampling_mix[0]
    degree_limit = uniform_limit + sampling_mix[1]
    concentrated_limit = degree_limit + sampling_mix[2]

    if draw < uniform_limit:
        return _sample_uniform(rng, num_nodes, seed_size)

    if draw < degree_limit:
        total_degree = float(degrees.sum())
        if total_degree <= 0 or np.count_nonzero(degrees) < seed_size:
            return _sample_uniform(rng, num_nodes, seed_size)
        probabilities = degrees.astype(np.float64) / total_degree
        return np.sort(
            rng.choice(num_nodes, size=seed_size, replace=False, p=probabilities)
        ).astype(np.int32)

    community_ids = np.asarray(sorted(communities), dtype=np.int64)
    if draw < concentrated_limit:
        community_id = int(rng.choice(community_ids))
        nodes = np.asarray(communities[community_id], dtype=np.int64)
        if nodes.size < seed_size:
            return _sample_uniform(rng, num_nodes, seed_size)
        return np.sort(rng.choice(nodes, size=seed_size, replace=False)).astype(np.int32)

    # Community-balanced: take at most one seed from each selected community,
    # then fill any remaining positions uniformly without replacement.
    selected = []
    selected_count = min(seed_size, community_ids.size)
    selected_communities = rng.choice(
        community_ids,
        size=selected_count,
        replace=False
    )
    for community_id in selected_communities:
        nodes = np.asarray(communities[int(community_id)], dtype=np.int64)
        selected.append(int(rng.choice(nodes)))

    selected = list(dict.fromkeys(selected))
    if len(selected) < seed_size:
        available = np.setdiff1d(
            np.arange(num_nodes, dtype=np.int64),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False
        )
        selected.extend(
            rng.choice(
                available,
                size=seed_size - len(selected),
                replace=False
            ).astype(np.int64).tolist()
        )
    return np.sort(np.asarray(selected, dtype=np.int32))


def generate_one_sample(
        sample_index: int,
        adjacency: sp.csr_matrix,
        diffusion_model: str,
        seed_size: int,
        mc_runs: int,
        activation_probability: float,
        degrees: np.ndarray,
        communities: Dict[int, List[int]],
        base_random_seed: int,
        sampling_mix: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one deterministic sample independent of worker scheduling."""
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_random_seed), int(sample_index)])
    )
    seeds = sample_seeds_diversified(
        rng,
        adjacency.shape[0],
        seed_size,
        degrees,
        communities,
        sampling_mix=sampling_mix
    )
    influence_sum = np.zeros(adjacency.shape[0], dtype=np.float32)
    for _ in range(mc_runs):
        if diffusion_model == "IC":
            influence_sum += run_ic_once(
                adjacency,
                seeds,
                rng,
                activation_probability
            )
        elif diffusion_model == "LT":
            influence_sum += run_lt_once(adjacency, seeds, rng, degrees)
        else:
            raise ValueError(f"不支持的扩散模型: {diffusion_model}")
    return seeds, influence_sum / float(mc_runs)


def generate_to_sg_v2(
        writer: SGWriterV2,
        parallel: Parallel,
        adjacency: sp.csr_matrix,
        diffusion_model: str,
        seed_size: int,
        mc_runs: int,
        activation_probability: float,
        degrees: np.ndarray,
        communities: Dict[int, List[int]],
        base_random_seed: int,
        sampling_mix: Tuple[float, float, float, float],
        write_batch_size: int,
        target_dtype: str
) -> None:
    """Generate samples in bounded chunks and write each completed chunk immediately."""
    if mc_runs < 1:
        raise ValueError("mc_runs 必须大于或等于 1。")
    if write_batch_size < 1:
        raise ValueError("write_batch_size 必须大于或等于 1。")
    start_index = writer.completed_samples
    if start_index:
        print(f"[继续] {start_index}/{writer.target_samples}")

    while start_index < writer.target_samples:
        end_index = min(start_index + write_batch_size, writer.target_samples)
        results = parallel(
            delayed(generate_one_sample)(
                sample_index,
                adjacency,
                diffusion_model,
                seed_size,
                mc_runs,
                activation_probability,
                degrees,
                communities,
                base_random_seed,
                sampling_mix
            )
            for sample_index in range(start_index, end_index)
        )
        seed_batch = np.stack([result[0] for result in results]).astype(np.int32)
        probability_batch = np.stack([result[1] for result in results]).astype(target_dtype)
        writer.write_batch(start_index, seed_batch, probability_batch)
        start_index = end_index
        print(
            f"[生成] {start_index}/{writer.target_samples} "
            f"({100.0 * start_index / writer.target_samples:.1f}%)"
        )


def _worker_ready(worker_index: int) -> int:
    """Start joblib workers before opening the HDF5 writer on Windows."""
    return worker_index


def _validate_resume_metadata(output_path: Path, metadata: Dict) -> None:
    """Reject appends that would mix incompatible generation settings."""
    partial_path = Path(str(output_path) + ".partial")
    candidate = partial_path if partial_path.exists() else output_path
    if not candidate.exists():
        return
    try:
        import h5py
        with h5py.File(candidate, "r") as handle:
            existing = json.loads(handle.attrs["metadata_json"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"无法验证待续写文件的 {DIFFUSION_FILE_SUFFIX} v2 元数据: {candidate}"
        ) from exc

    immutable_keys = (
        "dataset",
        "source_sha256",
        "diffusion_model",
        "ic_prob",
        "mc_runs",
        "seed_size",
        "random_seed",
        "sampling_mix",
        "target_dtype",
        "num_nodes",
        "num_edges"
    )
    mismatches = [
        key for key in immutable_keys
        if existing.get(key) != metadata.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"续写参数与原 {DIFFUSION_FILE_SUFFIX} 不一致，拒绝混写。差异字段: "
            + ", ".join(mismatches)
        )


def _value(value, config: Dict, key: str, fallback=None):
    if value is not None:
        return value
    if key in config:
        return config[key]
    return DEFAULTS[key] if fallback is None else fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"生成可分块、可恢复的 Fair-MCAT "
            f"{DIFFUSION_FILE_SUFFIX} v2 数据文件"
        )
    )
    parser.add_argument("--dataset", type=str, help="configs/datasets.json 中的数据集名称")
    parser.add_argument("--config", type=str, help="自定义数据集配置 JSON")
    parser.add_argument("--edges", type=str, help="覆盖配置中的 .edges 输入路径")
    parser.add_argument(
        "--output",
        type=str,
        help=f"覆盖默认 {DIFFUSION_FILE_SUFFIX} 输出路径"
    )
    parser.add_argument("--graph_cache", type=str, help="覆盖默认图缓存路径")
    parser.add_argument("--diffusion", choices=["IC", "LT"], help="扩散模型")
    parser.add_argument("--seed_num", type=int, help="种子节点数量")
    parser.add_argument("--num_samples", type=int, help="目标样本总数")
    parser.add_argument("--mc_runs", type=int, help="每条样本的 Monte Carlo 次数")
    parser.add_argument("--ic_prob", type=float, help="IC 单边激活概率")
    parser.add_argument("--n_jobs", type=int, help="并行 worker 数，-1 表示全部核心")
    parser.add_argument("--write_batch_size", type=int, help="每次生成并写入的样本数")
    parser.add_argument("--target_dtype", choices=["float16", "float32"], help="标签存储精度")
    parser.add_argument("--compression", choices=["none", "lzf", "gzip"], help="HDF5 压缩方式")
    parser.add_argument("--louvain_runs", type=int, help="Louvain 独立运行次数")
    parser.add_argument("--resolution", type=float, help="Louvain resolution")
    parser.add_argument("--random_seed", type=int, help="全流程基准随机种子")
    parser.add_argument(
        "--sampling_mix",
        type=float,
        nargs=4,
        metavar=("RANDOM", "DEGREE", "CONCENTRATED", "BALANCED"),
        help="四类种子采样概率，必须总和为1"
    )
    parser.add_argument("--directed", action="store_true", help="按有向边读取，随后仍标准化为无向图")
    parser.add_argument("--force_graph_cache", action="store_true", help="强制重建图缓存")
    parser.add_argument("--prepare_only", action="store_true", help="只构建图缓存，不生成样本")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"从 .partial 或已有 {DIFFUSION_FILE_SUFFIX} 继续追加"
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    return parser


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")
    if not args.dataset and not args.edges:
        raise ValueError("必须提供 --dataset，或者至少提供 --edges。")

    if args.dataset:
        dataset_name, config = load_dataset_config(args.dataset, args.config)
    else:
        dataset_name, config = "custom", {}

    configured_edges = config.get("edges")
    edges_path = resolve_project_path(args.edges or configured_edges)
    if not edges_path.exists():
        raise FileNotFoundError(f"边文件不存在: {edges_path}")

    diffusion_model = _value(args.diffusion, config, "diffusion_model")
    seed_size = int(_value(args.seed_num, config, "seed_num", DEFAULTS["seed_num"]))
    num_samples = int(_value(args.num_samples, config, "num_samples"))
    mc_runs = int(_value(args.mc_runs, config, "mc_runs"))
    configured_ic_prob = (
        args.ic_prob if args.ic_prob is not None else config.get("ic_prob")
    )
    if configured_ic_prob is None:
        raise ValueError(
            "缺少 ic_prob：请在数据集配置中设置，或通过 --ic_prob 显式提供。"
        )
    activation_probability = float(configured_ic_prob)
    louvain_runs = int(_value(args.louvain_runs, config, "louvain_runs"))
    resolution = float(_value(args.resolution, config, "resolution"))
    random_seed = int(_value(args.random_seed, config, "random_seed"))
    write_batch_size = int(_value(args.write_batch_size, config, "write_batch_size"))
    target_dtype = str(_value(args.target_dtype, config, "target_dtype"))
    compression_name = args.compression or config.get("compression", DEFAULTS["compression"])
    compression = None if compression_name == "none" else compression_name
    sampling_mix = tuple(args.sampling_mix or config.get(
        "sampling_mix",
        (0.4, 0.2, 0.2, 0.2)
    ))
    available_cores = os.cpu_count() or 1
    suggested_jobs = int(config.get("suggested_n_jobs", available_cores))
    n_jobs = args.n_jobs if args.n_jobs is not None else min(suggested_jobs, available_cores)

    if not 0.0 <= activation_probability <= 1.0:
        raise ValueError("ic_prob 必须位于 0~1。")
    if num_samples < 1:
        raise ValueError("num_samples 必须大于或等于 1。")
    if not np.isclose(sum(sampling_mix), 1.0) or any(p < 0 for p in sampling_mix):
        raise ValueError("sampling_mix 的值必须非负且总和为1。")

    file_stem = config.get("file_stem", edges_path.stem)
    output_path = resolve_project_path(
        args.output
        or f"data/{file_stem}_mean_{diffusion_model}{seed_size}{DIFFUSION_FILE_SUFFIX}"
    )
    cache_path = resolve_project_path(
        args.graph_cache or f"data/cache/{file_stem}_graph_v2.pkl"
    )

    # Expected counts apply only to the configured source file, not an override.
    expected_nodes = config.get("expected_nodes") if args.edges is None else None
    expected_edges = config.get("expected_edges") if args.edges is None else None
    print("Fair-mcat")
    print(
        f"[数据生成] {dataset_name} | {diffusion_model} | k={seed_size} | "
        f"样本={num_samples} | MC={mc_runs} | 进程={n_jobs}"
    )
    print(f"输入: {edges_path}")
    print(f"输出: {output_path}")

    graph = prepare_graph(
        edges_file=edges_path,
        cache_file=cache_path,
        louvain_runs=louvain_runs,
        resolution=resolution,
        random_seed=random_seed,
        directed=args.directed,
        expected_nodes=expected_nodes,
        expected_edges=expected_edges,
        force=args.force_graph_cache
    )
    if args.prepare_only:
        print("[完成] 图缓存")
        return

    metadata = {
        "dataset": dataset_name,
        "source_edges": str(edges_path),
        "source_sha256": graph["cache_signature"]["source_sha256"],
        "diffusion_model": diffusion_model,
        "ic_prob": activation_probability,
        "mc_runs": mc_runs,
        "seed_size": seed_size,
        "num_samples": num_samples,
        "random_seed": random_seed,
        "sampling_mix": list(sampling_mix),
        "target_dtype": target_dtype,
        "community_detection": graph["community_detection"],
        "num_nodes": graph["num_nodes"],
        "num_edges": graph["num_edges"]
    }
    if args.resume:
        _validate_resume_metadata(output_path, metadata)
    worker_count = available_cores if n_jobs == -1 else max(1, n_jobs)
    writer = None
    # Starting workers before opening HDF5 prevents inherited file handles from
    # locking the final atomic rename on Windows.
    with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
        if worker_count > 1:
            parallel(delayed(_worker_ready)(index) for index in range(worker_count))
        writer = SGWriterV2(
            output_path=str(output_path),
            adjacency=graph["adj"],
            communities=graph["communities"],
            original_node_ids=graph["original_node_ids"],
            metadata=metadata,
            seed_size=seed_size,
            target_samples=num_samples,
            target_dtype=target_dtype,
            compression=compression,
            resume=args.resume,
            overwrite=args.overwrite,
            graph_features=graph["graph_features"]
        )
        # target_samples may grow during a learning-curve append.
        writer.handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        writer.handle.flush()
        try:
            generate_to_sg_v2(
                writer=writer,
                parallel=parallel,
                adjacency=graph["adj"],
                diffusion_model=diffusion_model,
                seed_size=seed_size,
                mc_runs=mc_runs,
                activation_probability=activation_probability,
                degrees=graph["degrees"],
                communities=graph["communities"],
                base_random_seed=random_seed,
                sampling_mix=sampling_mix,
                write_batch_size=write_batch_size,
                target_dtype=target_dtype
            )
            writer.finalize()
        except BaseException:
            writer.close()
            print(f"[中断] {output_path}.partial")
            raise

    print(f"[完成] {output_path}")


if __name__ == "__main__":
    main()
