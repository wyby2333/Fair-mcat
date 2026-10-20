"""End-to-end Fair-MCAT experiment orchestration."""

import time

import numpy as np
import scipy.sparse as sp
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from data.communities import load_saved_communities
from data.config import DIFFUSION_FILE_SUFFIX, resolve_project_path
from data.graph_features import validate_graph_features
from data.sg_format import SGDatasetV2, is_sg_v2, load_sg_graph
from main.baselines import select_degree_seeds, select_pagerank_seeds
from main.evaluation import (
    run_adaptive_paired_fairness_simulation,
    run_paired_fairness_simulation,
)
from main.model.MCATjl import MCATjl
from main.reporting import mark_best, save_json_result
from main.runtime import (
    autocast_context,
    forward_sample_batch,
    normalize_adj,
    scipy_to_torch_sparse,
    unpack_batch,
)
from main.search import (
    local_search_gradient,
    local_search_restarts_batched,
    score_candidate_sets,
)
from main.training import evaluate_predictor, train_model


def _synchronize_device(device):
    """Make wall-clock phase boundaries accurate for asynchronous CUDA work."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _timed_call(function, device):
    _synchronize_device(device)
    started_at = time.perf_counter()
    result = function()
    _synchronize_device(device)
    return result, time.perf_counter() - started_at


def _dataset_spreads(dataset):
    """Read existing targets only; never mutate or regenerate diffusion data."""
    spreads = np.empty(len(dataset), dtype=np.float64)
    for index in range(len(dataset)):
        sample = dataset[index]
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            target = sample[1]
        else:
            target = sample[:, 1]
        spreads[index] = float(target.sum().item())
    base_dataset = dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    if getattr(base_dataset, '_handle', None) is not None:
        base_dataset._handle.close()
        base_dataset._handle = None
    return spreads


def _tail_sampler(train_set, args, random_seed, num_nodes):
    spreads = _dataset_spreads(train_set)
    tail_threshold = float(np.quantile(spreads, args.tail_quantile))
    extreme_threshold = float(np.quantile(spreads, args.extreme_quantile))
    args.tail_spread_threshold = tail_threshold / num_nodes
    args.extreme_spread_threshold = extreme_threshold / num_nodes
    sample_weights = np.ones(len(spreads), dtype=np.float64)
    sample_weights[spreads >= tail_threshold] = args.tail_weight
    sample_weights[spreads >= extreme_threshold] = args.extreme_weight
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(random_seed),
    )
    print(
        "尾部采样: "
        f"q{args.tail_quantile:.2f}={tail_threshold:.1f} (×{args.tail_weight:g}), "
        f"q{args.extreme_quantile:.2f}={extreme_threshold:.1f} "
        f"(×{args.extreme_weight:g})"
    )
    return sampler


def _rank_hybrid(primary, secondary, primary_count, seed_num):
    hybrid = []
    for node in list(primary)[:primary_count]:
        if int(node) not in hybrid:
            hybrid.append(int(node))
    for ranking in (secondary, primary):
        for node in ranking:
            node = int(node)
            if node not in hybrid:
                hybrid.append(node)
            if len(hybrid) == seed_num:
                return hybrid
    raise ValueError("无法构造完整的 warm-start 种子集合。")


def _warm_start_seed_sets(baseline_seed_sets, seed_num, num_restarts):
    degree = list(map(int, baseline_seed_sets['degree']))
    pagerank = list(map(int, baseline_seed_sets['pagerank']))
    candidates = [degree, pagerank]
    for pagerank_ratio in (0.25, 0.5, 0.75):
        candidates.append(_rank_hybrid(
            pagerank,
            degree,
            int(round(seed_num * pagerank_ratio)),
            seed_num,
        ))
    unique = []
    seen = set()
    for seeds in candidates:
        key = tuple(sorted(seeds))
        if key not in seen:
            unique.append(seeds)
            seen.add(key)
    return unique[:num_restarts]


def _deduplicate_candidates(candidate_entries):
    unique = []
    seen = set()
    for label, seeds in candidate_entries:
        normalized = [int(node) for node in seeds]
        key = tuple(sorted(normalized))
        if key not in seen:
            unique.append((label, normalized))
            seen.add(key)
    return unique


def _adaptive_candidate_mc_rerank(
        adj_matrix,
        candidate_entries,
        initial_runs,
        confirm_runs,
        finalist_count,
        diffusion_model,
        communities,
        ic_prob,
        spread_floor_ratio,
        random_seed,
):
    """Select from a small pool with staged, paired diffusion worlds."""
    entries = [
        {
            'label': label,
            'seed_set': list(map(int, seeds)),
        }
        for label, seeds in candidate_entries
    ]
    result = run_adaptive_paired_fairness_simulation(
        adj_matrix,
        entries,
        initial_runs,
        confirm_runs,
        finalist_count,
        diffusion_model,
        communities,
        ic_prob,
        random_seed,
        spread_floor_ratio,
    )
    print("\n候选结果")
    finalist_indices = set(result['finalist_indices'])
    for index, record in enumerate(result['records']):
        metrics = record['metrics']
        label = record['label'] + (' *' if index in finalist_indices else '')
        print(
            f"{label} | 次数 {metrics['runs']} | "
            f"Spread {metrics['spread']:.2f} | VCS {metrics['vcs']:.6f} | "
            f"WCR {metrics['wcr']:.6f} | "
            f"最低10% {metrics['worst_decile_rate']:.6f}"
        )
    print("* 入围")
    return result


def run_experiment(args, dataset_config, random_seed, device):
    RANDOM_SEED = random_seed
    # 数据加载
    diffusion_description = args.diffusion_model
    if args.diffusion_model == 'IC':
        diffusion_description += f" (p={args.ic_prob})"
    print(
        f"数据集: {args.dataset} | 扩散: {diffusion_description} | "
        f"k: {args.seed_num} | 设备: {device}"
    )
    
    file_stem = dataset_config['file_stem']
    data_file = str(resolve_project_path(
        args.data_file
        or (
            f'data/{file_stem}_mean_{args.diffusion_model}{args.seed_num}'
            f'{DIFFUSION_FILE_SUFFIX}'
        )
    ))
    print(f"[数据] {data_file}")
    
    if not is_sg_v2(data_file):
        raise ValueError(
            f"仅支持 {DIFFUSION_FILE_SUFFIX} v2 数据: {data_file}"
        )
    graph = load_sg_graph(data_file)
    adj = graph["adj"]
    sample_dataset = SGDatasetV2(data_file)
    print(
        f"图: {adj.shape} | 样本: {len(sample_dataset)} | "
        f"k: {sample_dataset.seed_size}"
    )
    if sample_dataset.seed_size != args.seed_num:
        raise ValueError(
            f"seed_num={args.seed_num} 与数据中的 "
            f"seed_size={sample_dataset.seed_size} 不一致。"
        )

    num_nodes = adj.shape[0]
    communities = load_saved_communities(graph, num_nodes)
    graph_features = graph.get("graph_features")
    validate_graph_features(graph_features, adj)
    print(f"社区: {len(communities)}")
    community_metadata = graph.get('community_detection')
    if community_metadata:
        print(
            "社区划分: "
            f"{community_metadata.get('algorithm')} | "
            f"Q={community_metadata.get('best_modularity')}"
        )

    adj_raw = adj.maximum(adj.T).astype(np.float32).tocsr()
    adj_norm = normalize_adj(
        adj_raw + sp.eye(adj_raw.shape[0], dtype=np.float32, format='csr')
    )
    adj_single = scipy_to_torch_sparse(adj_norm, device)
    print(f"稀疏边数: {adj_single._nnz()}")
    
    # 数据集划分
    total_size = len(sample_dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size
    if min(train_size, val_size, test_size) < 1:
        raise ValueError(
            "训练、验证、测试集都必须至少包含1条样本；请增加 num_samples。"
        )
    print(f"样本: {total_size} | 训练: {train_size} | 验证: {val_size} | 测试: {test_size}")
    
    train_val_set, test_set = random_split(
        sample_dataset, [train_size + val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_set, val_set = random_split(
        train_val_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    loader_options = {
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': args.num_workers > 0,
    }
    train_sampler = _tail_sampler(
        train_set, args, RANDOM_SEED, num_nodes
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=train_size >= args.batch_size,
        **loader_options
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options
    )
    
    # 模型初始化
    method_label = "Fair-MCAT"
    print("模型: MCATjl")
    
    _synchronize_device(device)
    model_init_started_at = time.perf_counter()
    forward_model = MCATjl(nfeat=1, nhid=64, nclass=1, dropout=0.2, nheads=4, alpha=0.2,
                           raw_adj=adj_raw,
                           degree_values=np.asarray(adj_raw.sum(axis=1)).reshape(-1),
                           clustering_coefficients=graph_features[
                               'clustering_coefficients'
                           ],
                           pagerank_scores=graph_features['pagerank_scores'],
                           global_pool_hidden=args.global_pool_hidden,
                           edge_chunk_size=args.edge_chunk_size,
                           graph_batch_backend=args.graph_batch_backend,
                           output_bias_init=args.output_bias_init).to(device)
    _synchronize_device(device)
    model_init_seconds = time.perf_counter() - model_init_started_at
    if getattr(forward_model, 'supports_batched_graph', False):
        forward_model.cache_graph(adj_single)
        print(
            f"批量: micro={args.model_micro_batch_size or args.batch_size} | "
            f"边分块={args.edge_chunk_size or '关闭'}"
        )
    checkpoint_file = resolve_project_path(
        args.checkpoint
        or (
            f'results/best_model_{args.dataset}_{args.diffusion_model}_'
            f'k{args.seed_num}_MCATjl.pth'
        )
    )
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    def run_training():
        optimizer = Adam(forward_model.parameters(), lr=args.lr)
        return train_model(
            forward_model,
            optimizer,
            train_loader,
            val_loader,
            adj_single,
            num_nodes,
            device,
            args,
            checkpoint_file,
        )

    training_result, optimization_seconds = _timed_call(
        run_training, device,
    )
    best_epoch, best_val_loss, best_val_spearman = training_result
    offline_training_seconds = (
        model_init_seconds + optimization_seconds
    )
    print(f"训练耗时: {offline_training_seconds:.6f}s")

    test_metrics = evaluate_predictor(
        forward_model,
        test_loader,
        len(test_set),
        adj_single,
        num_nodes,
        device,
        args,
        checkpoint_file,
    )
    
    # 搜索前冻结模型参数并建立推理缓存。
    for parameter in forward_model.parameters():
        parameter.requires_grad_(False)
    if hasattr(forward_model, 'prepare_inference_cache'):
        forward_model.prepare_inference_cache()

    # 在线时间从候选构造开始计算。
    _synchronize_device(device)
    online_started_at = time.perf_counter()

    degree_started_at = time.perf_counter()
    degree_seeds = select_degree_seeds(graph_features, args.seed_num)
    degree_selection_seconds = time.perf_counter() - degree_started_at
    pagerank_started_at = time.perf_counter()
    pagerank_seeds = select_pagerank_seeds(graph_features, args.seed_num)
    pagerank_selection_seconds = time.perf_counter() - pagerank_started_at
    baseline_seed_sets = {
        'degree': degree_seeds,
        'pagerank': pagerank_seeds,
    }
    warm_start_sets = _warm_start_seed_sets(
        baseline_seed_sets, args.seed_num, args.num_restarts
    )

    # 局部搜索
    print("\n[3/4] 搜索")
    print(
        f"节点: {num_nodes} | k: {args.seed_num} | "
        f"重启: {args.num_restarts} | 步数: {args.search_steps} | "
        f"模式: {args.search_execution}"
    )
    print(
        f"初始集: {len(warm_start_sets)} | 候选仿真: "
        f"{args.candidate_mc_initial_runs} → "
        f"{args.candidate_mc_confirm_runs} 次 | "
        f"入围={args.candidate_mc_finalist_count} | "
        f"Spread 下限: {args.search_spread_floor_ratio:.1%}"
    )
    
    seed_num = args.seed_num
    
    use_batched_restarts = (
        args.search_execution == 'batched'
        and getattr(forward_model, 'supports_batched_graph', False)
    )
    if args.search_execution == 'batched' and not use_batched_restarts:
        print("[提示] 改用顺序搜索。")
    
    if use_batched_restarts:
        restart_results = local_search_restarts_batched(
            forward_model,
            adj_single,
            num_nodes,
            seed_num,
            args.search_steps,
            args.patience_steps,
            device,
            args.num_restarts,
            fairness_strength=args.fairness_strength,
            communities=communities,
            micro_batch_size=args.model_micro_batch_size,
            random_seed=RANDOM_SEED,
            initial_seed_sets=warm_start_sets,
        )
    else:
        restart_results = []
        for restart in range(args.num_restarts):
            restart_results.append(local_search_gradient(
                forward_model, adj_single, num_nodes, seed_num,
                args.search_steps, args.patience_steps, device,
                fairness_strength=args.fairness_strength,
                communities=communities,
                initial_seeds=(
                    warm_start_sets[restart]
                    if restart < len(warm_start_sets) else None
                ),
            ))

    highest_restart_score = -float('inf')
    for restart, (seeds, score, steps_done) in enumerate(restart_results):
        improved = score > highest_restart_score
        if improved:
            highest_restart_score = score
        print(f"重启 {restart + 1}/{args.num_restarts} | "
              f"步数: {steps_done} | 得分: {score:.2f}"
              f"{' [最佳]' if improved else ''}")

    candidate_entries = [
        (f"重启-{index + 1}", seeds)
        for index, (seeds, _, _) in enumerate(restart_results)
    ]
    candidate_entries.extend([
        ('Degree', baseline_seed_sets['degree']),
        ('PageRank', baseline_seed_sets['pagerank']),
    ])
    candidate_entries.extend([
        (f'混合-{index + 1}', seeds)
        for index, seeds in enumerate(warm_start_sets[2:])
    ])
    candidate_entries = _deduplicate_candidates(candidate_entries)
    candidate_metrics = score_candidate_sets(
        forward_model,
        adj_single,
        [seeds for _, seeds in candidate_entries],
        num_nodes,
        communities,
        device,
        micro_batch_size=args.model_micro_batch_size,
    )
    highest_candidate_score = max(item['score'] for item in candidate_metrics)
    adaptive_selection = _adaptive_candidate_mc_rerank(
        adj,
        candidate_entries,
        args.candidate_mc_initial_runs,
        args.candidate_mc_confirm_runs,
        args.candidate_mc_finalist_count,
        args.diffusion_model,
        communities,
        args.ic_prob,
        args.search_spread_floor_ratio,
        RANDOM_SEED + 161803,
    )
    selected_index = adaptive_selection["selected_index"]
    max_spread_index = adaptive_selection["max_spread_index"]
    selected_label, selected_seeds = candidate_entries[selected_index]
    max_label, max_seeds = candidate_entries[max_spread_index]
    max_spread_seed_set = np.asarray(max_seeds, dtype=np.int64)
    fair_seed_set = np.asarray(selected_seeds, dtype=np.int64)
    max_search_label = max_label
    fair_search_label = selected_label

    print(f"搜索完成 | 最高预测 Spread: {highest_candidate_score:.2f}")
    print(f"候选来源: MaxSpread={max_search_label} | Fair={fair_search_label}")

    adj_full = adj
    seeds_degree = baseline_seed_sets['degree']
    seeds_pr = baseline_seed_sets['pagerank']

    spread_floor = adaptive_selection["spread_floor"]
    selection_source = "adaptive_paired_rerank"

    # 固定最终种子集并结束在线计时。
    _synchronize_device(device)
    online_selection_seconds = time.perf_counter() - online_started_at
    print(
        "在线耗时: "
        f"{online_selection_seconds:.6f}s"
    )

    # 独立配对评估
    evaluation_seed = int(getattr(args, 'evaluation_seed', 314159))
    evaluation_entries = [
        {
            'label': 'Degree Baseline',
            'seed_set': list(map(int, seeds_degree)),
            'role': 'degree',
        },
        {
            'label': 'PageRank Baseline',
            'seed_set': list(map(int, seeds_pr)),
            'role': 'pagerank',
        },
        {
            'label': 'Fair-MCAT-MaxSpread',
            'seed_set': max_spread_seed_set.tolist(),
            'role': 'max_spread',
        },
        {
            'label': 'Fair-MCAT-Final',
            'seed_set': fair_seed_set.tolist(),
            'role': 'fair',
        },
    ]
    print()
    print(
        "[4/4] 评估 | "
        f"{args.diffusion_model} p={args.ic_prob:g} | "
        f"{args.num_simulations} 次 | seed={evaluation_seed}"
    )
    paired_final_evaluation = run_paired_fairness_simulation(
        adj_full,
        evaluation_entries,
        args.num_simulations,
        args.diffusion_model,
        communities,
        args.ic_prob,
        evaluation_seed,
    )
    evaluation_by_role = {
        record['role']: record for record in paired_final_evaluation
    }

    def reported_metrics(role):
        metrics = evaluation_by_role[role]['metrics']
        return (
            metrics['spread'],
            metrics['vcs'],
            metrics['wcr'],
            metrics['worst_decile_rate'],
        )

    deg_spread, deg_vcs, deg_wcr, deg_worst10 = reported_metrics('degree')
    pr_spread, pr_vcs, pr_wcr, pr_worst10 = reported_metrics('pagerank')
    max_spread, max_vcs, max_wcr, max_worst10 = reported_metrics(
        'max_spread'
    )
    fair_spread, fair_vcs, fair_wcr, fair_worst10 = reported_metrics('fair')
    fair_solution_found = bool(fair_vcs < min(deg_vcs, pr_vcs))
    fair_method_name = (
        "Fair-MCAT-Fair"
        if fair_solution_found
        else "Fair-MCAT-Pareto"
    )

    # 汇总最优指标。
    best_spread = max(deg_spread, pr_spread, max_spread, fair_spread)
    best_vcs = min(deg_vcs, pr_vcs, max_vcs, fair_vcs)

    offline_display = (
        f"{offline_training_seconds:.6f}"
        if offline_training_seconds is not None else "—"
    )
    print(f"\n结果 ({args.num_simulations} 次配对仿真平均)")
    print("方法 | 训练(s) | 筛选(s) | Spread↑ | VCS↓")
    print(
        f"Degree | — | {degree_selection_seconds:.6f} | "
        f"{mark_best(deg_spread, best_spread)} | {mark_best(deg_vcs, best_vcs)}"
    )
    print(
        f"PageRank | — | {pagerank_selection_seconds:.6f} | "
        f"{mark_best(pr_spread, best_spread)} | {mark_best(pr_vcs, best_vcs)}"
    )
    print(
        f"Fair-MCAT-MaxSpread | {offline_display} | "
        f"{online_selection_seconds:.6f} | "
        f"{mark_best(max_spread, best_spread)} | {mark_best(max_vcs, best_vcs)}"
    )
    print(
        f"{method_label} | {offline_display} | "
        f"{online_selection_seconds:.6f} | "
        f"{mark_best(fair_spread, best_spread)} | {mark_best(fair_vcs, best_vcs)}"
    )
    print("★ 最优")

    results = {
        'method': method_label,
        'metrics': {
            'spread_unit': 'activated_nodes',
            'vcs_unit': 'variance_of_community_activation_ratio',
        },
        'timing': {
            'offline_training_seconds': (
                float(offline_training_seconds)
                if offline_training_seconds is not None else None
            ),
            'online_selection_seconds': float(online_selection_seconds),
            'degree_selection_seconds': float(degree_selection_seconds),
            'pagerank_selection_seconds': float(
                pagerank_selection_seconds
            ),
            'graph_preparation_included': False,
            'checkpoint_and_graph_loading_included': False,
            'final_evaluation_included': False,
        },
        'evaluation_protocol': {
            'diffusion_model': args.diffusion_model,
            'ic_probability': float(args.ic_prob),
            'mc_runs': int(args.num_simulations),
            'evaluation_seed': evaluation_seed,
            'paired_scenarios_across_methods': True,
            'independent_from_search_random_stream': True,
            'used_for_selection': False,
            'time_included_in_online_selection': False,
            'scenario_protocol': (
                'shared_live_edge_graphs'
                if args.diffusion_model == 'IC'
                else 'shared_lt_thresholds'
            ),
        },
        'model_config': {
            'positive_mapping': 'softplus',
            'structural_features': ['degree', 'clustering_coefficient'],
            'attention_edge_bias': True,
            'global_seed_pooling': True,
            'model_type': 'MCATjl',
        },
        'fairness_config': {
            'algorithm': 'community_gradient_search',
            'fairness_strength': args.fairness_strength,
            'search_spread_floor_ratio': args.search_spread_floor_ratio,
            'selection_objectives': ['maximize_spread', 'minimize_vcs'],
            'other_fairness_metrics_used_for_selection': False,
        },
        'config': {
            'dataset': args.dataset, 'diffusion_model': args.diffusion_model,
            'ic_prob': args.ic_prob, 'seed_num': args.seed_num, 'lr': args.lr,
            'output_bias_init': args.output_bias_init,
            'global_pool_hidden': args.global_pool_hidden,
            'spread_weight': args.spread_weight,
            'spread_beta': args.spread_beta,
            'ranking_weight': args.ranking_weight,
            'ranking_tie_threshold': args.ranking_margin,
            'ranking_temperature': args.ranking_temperature,
            'lr_patience': args.lr_patience,
            'lr_factor': args.lr_factor,
            'min_lr': args.min_lr,
            'tail_quantile': args.tail_quantile,
            'extreme_quantile': args.extreme_quantile,
            'tail_weight': args.tail_weight,
            'extreme_weight': args.extreme_weight,
            'candidate_mc_initial_runs': int(getattr(
                args, 'candidate_mc_initial_runs', 20
            )),
            'candidate_mc_confirm_runs': int(getattr(
                args, 'candidate_mc_confirm_runs', 100
            )),
            'candidate_mc_finalist_count': int(getattr(
                args, 'candidate_mc_finalist_count', 2
            )),
            'final_evaluation_runs': int(args.num_simulations),
            'final_evaluation_seed': evaluation_seed,
        },
        'training': {
            'offline_training_seconds': float(offline_training_seconds),
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
            'best_val_spearman': best_val_spearman,
            'test_mse': test_metrics['mse'],
            'predictor_diagnostics': test_metrics,
        },
        'search': {
            'online_selection_seconds': float(online_selection_seconds),
            'warm_start_count': len(warm_start_sets),
            'candidate_count': len(candidate_entries),
            'highest_predicted_spread': float(highest_candidate_score),
            'initial_candidates': [
                {
                    'label': label,
                    'seed_set': list(map(int, seeds)),
                    'predicted_metrics': predicted,
                }
                for (label, seeds), predicted in zip(
                    candidate_entries, candidate_metrics
                )
            ],
            'adaptive_paired_selection': adaptive_selection,
        },
        'paired_final_evaluation': {
            'runs': int(args.num_simulations),
            'random_seed': evaluation_seed,
            'used_for_selection': False,
            'candidates': paired_final_evaluation,
        },
        'final_results': {
            'seed_set': fair_seed_set.tolist(),
            'spread': float(fair_spread),
            'spread_ratio': float(fair_spread / num_nodes),
            'vcs': float(fair_vcs),
            'wcr': float(fair_wcr),
            'worst_decile_rate': float(fair_worst10),
        },
        'max_spread_results': {
            'search_label': max_search_label,
            'seed_set': max_spread_seed_set.tolist(),
            'spread': float(max_spread),
            'spread_ratio': float(max_spread / num_nodes),
            'vcs': float(max_vcs),
            'wcr': float(max_wcr),
            'worst_decile_rate': float(max_worst10),
        },
        'fair_results': {
            'fair_solution_found': fair_solution_found,
            'reported_method': fair_method_name,
            'search_label': fair_search_label,
            'selection_source': selection_source,
            'spread_floor': float(spread_floor),
            'seed_set': fair_seed_set.tolist(),
            'spread': float(fair_spread),
            'spread_ratio': float(fair_spread / num_nodes),
            'vcs': float(fair_vcs),
            'wcr': float(fair_wcr),
            'worst_decile_rate': float(fair_worst10),
        },
        'baselines': {
            'degree': {
                'online_selection_seconds': float(
                    degree_selection_seconds
                ),
                'spread': float(deg_spread),
                'spread_ratio': float(deg_spread / num_nodes),
                'vcs': float(deg_vcs),
                'wcr': float(deg_wcr),
                'worst_decile_rate': float(deg_worst10),
            },
            'pagerank': {
                'online_selection_seconds': float(
                    pagerank_selection_seconds
                ),
                'spread': float(pr_spread),
                'spread_ratio': float(pr_spread / num_nodes),
                'vcs': float(pr_vcs),
                'wcr': float(pr_wcr),
                'worst_decile_rate': float(pr_worst10),
            }
        },
    }
    
    result_file = resolve_project_path(
        args.result_file
        or (
            f'results/experiment_results_{args.dataset}_{args.diffusion_model}_'
            f'seed{args.seed_num}_zh.json'
        )
    )
    save_json_result(result_file, results)
    
    print(f"[保存] {result_file}")
    print("完成")
    return results


