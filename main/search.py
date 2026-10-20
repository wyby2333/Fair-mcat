"""Fair-mcat 梯度公平搜索。"""

import math
import random

import torch

from main.runtime import forward_sample_batch


def _seed_tensor(seeds, num_nodes, device, requires_grad=False):
    tensor = torch.zeros(num_nodes, device=device)
    tensor[list(seeds)] = 1.0
    tensor.requires_grad_(requires_grad)
    return tensor


def _seed_tensor_batch(seed_sets, num_nodes, device, requires_grad=False):
    batch = torch.zeros(
        (len(seed_sets), num_nodes), dtype=torch.float32, device=device
    )
    for index, seeds in enumerate(seed_sets):
        batch[index, list(seeds)] = 1.0
    batch.requires_grad_(requires_grad)
    return batch


def _community_context(communities, num_nodes, device):
    nodes = [
        torch.as_tensor(group, dtype=torch.long, device=device)
        for group in communities
    ]
    community_id = torch.empty(num_nodes, dtype=torch.long, device=device)
    for index, group in enumerate(nodes):
        community_id[group] = index
    sizes = torch.bincount(
        community_id, minlength=len(nodes)
    ).to(dtype=torch.float32)
    return nodes, community_id, sizes


def _community_rates(prediction, community_id, community_sizes):
    sums = torch.zeros_like(community_sizes)
    sums.scatter_add_(0, community_id, prediction)
    return sums / community_sizes.clamp_min(1.0)


def _community_rates_batch(prediction, community_id, community_sizes):
    batch_size = prediction.size(0)
    indices = community_id.unsqueeze(0).expand(batch_size, -1)
    sums = torch.zeros(
        (batch_size, community_sizes.numel()),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    sums.scatter_add_(1, indices, prediction)
    return sums / community_sizes.clamp_min(1.0).unsqueeze(0)


def _gradient_state(
        model, adjacency, seeds, num_nodes, device,
        community_id, community_sizes,
):
    seed_tensor = _seed_tensor(
        seeds, num_nodes, device, requires_grad=True
    )
    prediction = model(seed_tensor.unsqueeze(-1), adjacency).squeeze(-1)
    score = prediction.sum()
    gradient, = torch.autograd.grad(score, seed_tensor)
    rates = _community_rates(
        prediction.detach(), community_id, community_sizes
    )
    return float(score.detach().item()), gradient.detach(), rates


def _predict_scores_batch(
        model, adjacency, seed_sets, num_nodes, device, micro_batch_size,
):
    seed_batch = _seed_tensor_batch(seed_sets, num_nodes, device)
    with torch.no_grad():
        predictions = forward_sample_batch(
            model, seed_batch, adjacency, micro_batch_size
        )
    return predictions.sum(dim=1), predictions


def _gradient_states_batch(
        model, adjacency, seed_sets, num_nodes, device,
        community_id, community_sizes, micro_batch_size,
):
    seed_batch = _seed_tensor_batch(
        seed_sets, num_nodes, device, requires_grad=True
    )
    predictions = forward_sample_batch(
        model, seed_batch, adjacency, micro_batch_size
    )
    scores = predictions.sum(dim=1)
    gradient, = torch.autograd.grad(scores.sum(), seed_batch)
    rates = _community_rates_batch(
        predictions.detach(), community_id, community_sizes
    )
    return scores.detach(), gradient.detach(), rates


def _random_unselected_node(num_nodes, seeds, rng):
    while True:
        node = rng.randrange(num_nodes)
        if node not in seeds:
            return node


def _resolve_initial_seeds(initial_seeds, num_nodes, seed_num, rng):
    if initial_seeds is None:
        return set(rng.sample(range(num_nodes), seed_num))
    seeds = {int(node) for node in initial_seeds}
    if len(seeds) != seed_num:
        raise ValueError("初始种子数与 seed_num 不一致。")
    if any(node < 0 or node >= num_nodes for node in seeds):
        raise ValueError("初始种子包含越界节点。")
    return seeds


def _select_swap(
        seeds, membership, gradient, rates, community_nodes,
        num_nodes, fairness_strength, rng,
):
    weakest = int(torch.argmin(rates).item())
    strongest = int(torch.argmax(rates).item())

    if rng.random() < fairness_strength:
        candidates = community_nodes[weakest]
        candidates = candidates[~membership[candidates]]
        if candidates.numel():
            values = gradient.index_select(0, candidates)
            in_node = int(candidates[torch.argmax(values)].item())
        else:
            in_node = _random_unselected_node(num_nodes, seeds, rng)
    else:
        in_node = _random_unselected_node(num_nodes, seeds, rng)

    if rng.random() < fairness_strength:
        candidates = community_nodes[strongest]
        candidates = candidates[membership[candidates]]
        if candidates.numel():
            values = gradient.index_select(0, candidates)
            out_node = int(candidates[torch.argmin(values)].item())
        else:
            out_node = rng.choice(tuple(seeds))
    else:
        out_node = rng.choice(tuple(seeds))
    return out_node, in_node


def local_search_gradient(
        model, adjacency, num_nodes, seed_num, steps, patience, device,
        fairness_strength, communities, initial_seeds=None,
):
    """执行一次梯度公平局部搜索。"""
    rng = random
    seeds = _resolve_initial_seeds(
        initial_seeds, num_nodes, seed_num, rng
    )
    membership = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    membership[list(seeds)] = True
    community_nodes, community_id, community_sizes = _community_context(
        communities, num_nodes, device
    )
    best_score, gradient, rates = _gradient_state(
        model, adjacency, seeds, num_nodes, device,
        community_id, community_sizes
    )
    best_seeds = seeds.copy()
    no_improve = 0
    steps_done = 0

    for step in range(steps):
        steps_done = step + 1
        out_node, in_node = _select_swap(
            seeds, membership, gradient, rates, community_nodes,
            num_nodes, fairness_strength, rng
        )
        proposal = seeds.copy()
        proposal.remove(out_node)
        proposal.add(in_node)
        new_score, _ = _predict_scores_batch(
            model, adjacency, [proposal], num_nodes, device, 1
        )
        new_score = float(new_score[0].item())

        if new_score > best_score:
            seeds = proposal
            membership[out_node] = False
            membership[in_node] = True
            best_score, gradient, rates = _gradient_state(
                model, adjacency, seeds, num_nodes, device,
                community_id, community_sizes
            )
            best_seeds = seeds.copy()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    return list(best_seeds), best_score, steps_done


def local_search_restarts_batched(
        model, adjacency, num_nodes, seed_num, steps, patience, device,
        num_restarts, fairness_strength, communities, micro_batch_size=0,
        random_seed=42, initial_seed_sets=None,
):
    """批量执行多次梯度公平局部搜索。"""
    if num_restarts < 1:
        raise ValueError("num_restarts 必须大于或等于1。")

    rngs = [
        random.Random(int(random_seed) + 104729 * index)
        for index in range(num_restarts)
    ]
    initial_seed_sets = list(initial_seed_sets or [])
    if len(initial_seed_sets) > num_restarts:
        raise ValueError("初始种子集合数量不能超过 num_restarts。")
    seed_sets = [
        _resolve_initial_seeds(
            initial_seed_sets[index] if index < len(initial_seed_sets) else None,
            num_nodes,
            seed_num,
            rngs[index],
        )
        for index in range(num_restarts)
    ]
    best_seed_sets = [seeds.copy() for seeds in seed_sets]
    memberships = []
    for seeds in seed_sets:
        membership = torch.zeros(
            num_nodes, dtype=torch.bool, device=device
        )
        membership[list(seeds)] = True
        memberships.append(membership)

    community_nodes, community_id, community_sizes = _community_context(
        communities, num_nodes, device
    )
    scores, gradients, rates = _gradient_states_batch(
        model, adjacency, seed_sets, num_nodes, device,
        community_id, community_sizes, micro_batch_size
    )
    best_scores = [float(value) for value in scores.cpu().tolist()]
    no_improve = [0] * num_restarts
    steps_done = [0] * num_restarts
    active = [True] * num_restarts

    for step in range(steps):
        active_indices = [
            index for index, enabled in enumerate(active) if enabled
        ]
        if not active_indices:
            break

        proposals = []
        swaps = []
        for index in active_indices:
            out_node, in_node = _select_swap(
                seed_sets[index], memberships[index], gradients[index],
                rates[index], community_nodes, num_nodes,
                fairness_strength, rngs[index]
            )
            proposal = seed_sets[index].copy()
            proposal.remove(out_node)
            proposal.add(in_node)
            proposals.append(proposal)
            swaps.append((out_node, in_node))

        proposal_scores, _ = _predict_scores_batch(
            model, adjacency, proposals, num_nodes, device, micro_batch_size
        )
        accepted_indices = []
        accepted_proposals = []
        for batch_index, index in enumerate(active_indices):
            steps_done[index] = step + 1
            proposal_score = float(proposal_scores[batch_index].item())
            if proposal_score > best_scores[index]:
                seed_sets[index] = proposals[batch_index]
                best_seed_sets[index] = proposals[batch_index].copy()
                best_scores[index] = proposal_score
                no_improve[index] = 0
                out_node, in_node = swaps[batch_index]
                memberships[index][out_node] = False
                memberships[index][in_node] = True
                accepted_indices.append(index)
                accepted_proposals.append(proposals[batch_index])
            else:
                no_improve[index] += 1
            if no_improve[index] >= patience:
                active[index] = False

        if accepted_indices:
            accepted_scores, accepted_gradients, accepted_rates = (
                _gradient_states_batch(
                    model, adjacency, accepted_proposals, num_nodes, device,
                    community_id, community_sizes, micro_batch_size
                )
            )
            for batch_index, index in enumerate(accepted_indices):
                best_scores[index] = float(
                    accepted_scores[batch_index].item()
                )
                gradients[index] = accepted_gradients[batch_index]
                rates[index] = accepted_rates[batch_index]

    return [
        (list(best_seed_sets[index]), best_scores[index], steps_done[index])
        for index in range(num_restarts)
    ]


def score_candidate_sets(
        model, adjacency, seed_sets, num_nodes, communities, device,
        micro_batch_size=0,
):
    """计算候选种子集的预测指标。"""
    if not seed_sets:
        return []
    _, community_id, community_sizes = _community_context(
        communities, num_nodes, device
    )
    scores, predictions = _predict_scores_batch(
        model, adjacency, seed_sets, num_nodes, device, micro_batch_size
    )
    rates = _community_rates_batch(
        predictions, community_id, community_sizes
    )
    vcs = rates.var(dim=1, unbiased=False)
    bottom_count = max(1, int(math.ceil(rates.size(1) * 0.1)))
    worst_decile = torch.topk(
        rates, k=bottom_count, dim=1, largest=False
    ).values.mean(dim=1)
    return [
        {
            "score": float(scores[index].item()),
            "vcs": float(vcs[index].item()),
            "worst_decile_rate": float(worst_decile[index].item()),
        }
        for index in range(len(seed_sets))
    ]
