"""Monte Carlo spread and community-fairness evaluation."""

import math
import numpy as np

from main.diffusion import run_ic_once, run_lt_once


def run_fairness_simulation(
        adj_matrix,
        seeds,
        num_simulations,
        diffusion_model,
        communities,
        ic_prob,
        method_name="",
        return_details=False,
        verbose=True,
):
    """Evaluate one seed set using the existing Fair-MCAT metric protocol."""
    if verbose:
        print(f"[仿真] {method_name} | {num_simulations} 次")
    num_nodes = adj_matrix.shape[0]
    total_infections = []
    community_ratios = {i: [] for i in range(len(communities))}

    in_degrees = None
    if diffusion_model == "LT":
        in_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        in_degrees[in_degrees == 0] = 1.0

    for simulation in range(num_simulations):
        if diffusion_model == "IC":
            activated = run_ic_once(
                adj_matrix,
                seeds,
                ic_prob,
                rng=np.random,
            )
        elif diffusion_model == "LT":
            activated = run_lt_once(
                adj_matrix,
                seeds,
                in_degrees=in_degrees,
                rng=np.random,
            )
        else:
            activated = np.zeros(num_nodes, dtype=bool)
            activated[seeds] = True

        total_infected = np.sum(activated)
        total_infections.append(total_infected)
        community_ratios_this = {}
        for community_index, community in enumerate(communities):
            ratio = (
                np.sum(activated[list(community)])
                / len(community)
            )
            community_ratios[community_index].append(ratio)
            community_ratios_this[community_index] = ratio

        community_values = list(community_ratios_this.values())
        community_variance = np.var(community_values)
        progress_interval = max(1, num_simulations // 10)
        if verbose and (
            simulation + 1 == 1
            or simulation + 1 == num_simulations
            or (simulation + 1) % progress_interval == 0
        ):
            print(
                f"[{method_name}] "
                f"{simulation + 1:3d}/{num_simulations} | "
                f"Spread: {total_infected:5d} "
                f"({total_infected / num_nodes:.4f}) | "
                f"VCS: {community_variance:.6f} | "
                f"WCR: {min(community_values):.4f}"
            )

    mean_spread = float(np.mean(total_infections))
    mean_community_ratios = [
        np.mean(community_ratios[index])
        for index in range(len(communities))
    ]
    summary = (
        mean_spread,
        np.var(mean_community_ratios),
        np.min(mean_community_ratios),
    )
    if not return_details:
        return summary
    bottom_count = max(1, int(math.ceil(len(mean_community_ratios) * 0.1)))
    worst_decile_rate = float(np.mean(np.sort(mean_community_ratios)[:bottom_count]))
    return summary + (worst_decile_rate,)


evaluate_seed_set = run_fairness_simulation


def _run_ic_live_edges(adjacency, seeds, live_edges):
    """Evaluate seeds on one pre-sampled IC live-edge graph."""
    activated = np.zeros(adjacency.shape[0], dtype=bool)
    seed_array = np.asarray(seeds, dtype=np.int64)
    activated[seed_array] = True
    active_nodes = list(map(int, seed_array))
    while active_nodes:
        new_active = []
        for node in active_nodes:
            start = adjacency.indptr[node]
            end = adjacency.indptr[node + 1]
            neighbors = adjacency.indices[start:end]
            edge_mask = live_edges[start:end]
            for neighbor in neighbors[edge_mask]:
                neighbor = int(neighbor)
                if not activated[neighbor]:
                    activated[neighbor] = True
                    new_active.append(neighbor)
        active_nodes = new_active
    return activated


def _run_lt_thresholds(adjacency, seeds, in_degrees, thresholds):
    """Evaluate seeds with one shared LT threshold vector."""
    num_nodes = adjacency.shape[0]
    activated = np.zeros(num_nodes, dtype=bool)
    seed_array = np.asarray(seeds, dtype=np.int64)
    activated[seed_array] = True
    active_nodes = list(map(int, seed_array))
    while active_nodes:
        new_active = []
        for node in np.where(~activated)[0]:
            neighbors = adjacency.indices[
                adjacency.indptr[node]:adjacency.indptr[node + 1]
            ]
            active_count = np.sum(activated[neighbors])
            if active_count / in_degrees[node] >= thresholds[node]:
                activated[node] = True
                new_active.append(int(node))
        active_nodes = new_active
    return activated


def run_paired_fairness_simulation(
        adj_matrix,
        candidate_entries,
        num_simulations,
        diffusion_model,
        communities,
        ic_prob,
        random_seed,
        verbose=True,
):
    """Evaluate every candidate on the same diffusion worlds.

    IC worlds pre-sample one live/dead value per sparse edge. LT worlds
    pre-sample one threshold per node. This makes comparisons deterministic
    and independent of candidate evaluation order.
    """
    adjacency = adj_matrix.tocsr()
    candidate_entries = list(candidate_entries)
    if not candidate_entries:
        raise ValueError("paired evaluation requires at least one candidate")
    if int(num_simulations) < 1:
        raise ValueError("num_simulations must be positive")

    num_nodes = adjacency.shape[0]
    community_arrays = [
        np.asarray(community, dtype=np.int64) for community in communities
    ]
    if any(nodes.size == 0 for nodes in community_arrays):
        raise ValueError("communities must not contain empty groups")
    rng = np.random.default_rng(int(random_seed))
    candidate_count = len(candidate_entries)
    spread_samples = np.empty(
        (candidate_count, int(num_simulations)), dtype=np.float64
    )
    community_sums = np.zeros(
        (candidate_count, len(community_arrays)), dtype=np.float64
    )

    in_degrees = None
    if diffusion_model == "LT":
        in_degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        in_degrees[in_degrees == 0] = 1.0
    elif diffusion_model != "IC":
        raise ValueError(f"unsupported paired diffusion model: {diffusion_model}")

    progress_interval = max(1, int(num_simulations) // 10)
    if verbose:
        print(
            "\n[配对评估] "
            f"候选={candidate_count} | 场景={num_simulations} | "
            f"seed={int(random_seed)}"
        )
    for simulation in range(int(num_simulations)):
        if diffusion_model == "IC":
            live_edges = rng.random(adjacency.nnz) < float(ic_prob)
            thresholds = None
        else:
            live_edges = None
            thresholds = rng.random(num_nodes)

        for candidate_index, entry in enumerate(candidate_entries):
            seeds = entry['seed_set']
            if diffusion_model == "IC":
                activated = _run_ic_live_edges(
                    adjacency, seeds, live_edges
                )
            else:
                activated = _run_lt_thresholds(
                    adjacency, seeds, in_degrees, thresholds
                )
            spread_samples[candidate_index, simulation] = np.sum(activated)
            for community_index, nodes in enumerate(community_arrays):
                community_sums[candidate_index, community_index] += (
                    np.sum(activated[nodes]) / nodes.size
                )

        if verbose and (
            simulation + 1 == 1
            or simulation + 1 == int(num_simulations)
            or (simulation + 1) % progress_interval == 0
        ):
            print(
                f"场景 {simulation + 1:4d}/{int(num_simulations)}"
            )

    records = []
    for candidate_index, entry in enumerate(candidate_entries):
        samples = spread_samples[candidate_index]
        mean_community_ratios = (
            community_sums[candidate_index] / int(num_simulations)
        )
        bottom_count = max(
            1, int(math.ceil(len(mean_community_ratios) * 0.1))
        )
        metrics = {
            'spread': float(np.mean(samples)),
            'spread_se': float(
                np.std(samples, ddof=1) / math.sqrt(len(samples))
                if len(samples) > 1 else 0.0
            ),
            'vcs': float(np.var(mean_community_ratios)),
            'wcr': float(np.min(mean_community_ratios)),
            'worst_decile_rate': float(np.mean(
                np.sort(mean_community_ratios)[:bottom_count]
            )),
        }
        record = dict(entry)
        record['seed_set'] = list(map(int, entry['seed_set']))
        record['metrics'] = metrics
        records.append(record)
    return records


def _adaptive_finalist_indices(records, finalist_count, spread_floor_ratio):
    """Keep a spread leader and the strongest plausible fair alternatives."""
    finalist_count = min(int(finalist_count), len(records))
    best_index = max(
        range(len(records)),
        key=lambda index: records[index]['metrics']['spread'],
    )
    best = records[best_index]['metrics']
    best_lower = best['spread'] - 1.645 * best['spread_se']
    plausible = [
        index for index, record in enumerate(records)
        if (
            record['metrics']['spread']
            + 1.645 * record['metrics']['spread_se']
            >= spread_floor_ratio * best_lower
        )
    ]
    fairness_order = sorted(
        plausible,
        key=lambda index: (
            records[index]['metrics']['vcs'],
            -records[index]['metrics']['spread'],
        ),
    )
    spread_order = sorted(
        range(len(records)),
        key=lambda index: (
            -records[index]['metrics']['spread'],
            records[index]['metrics']['vcs'],
        ),
    )
    selected = []
    for index in [best_index, *fairness_order, *spread_order]:
        if index not in selected:
            selected.append(index)
        if len(selected) == finalist_count:
            break
    return selected


def run_adaptive_paired_fairness_simulation(
        adj_matrix,
        candidate_entries,
        initial_runs,
        confirm_runs,
        finalist_count,
        diffusion_model,
        communities,
        ic_prob,
        random_seed,
        spread_floor_ratio,
        verbose=True,
):
    """Screen all candidates on shared worlds, then confirm a short list.

    Statistics from the screening stage are retained. Finalists therefore
    receive ``confirm_runs`` simulations in total, rather than restarting the
    Monte Carlo evaluation after screening.
    """
    adjacency = adj_matrix.tocsr()
    candidate_entries = list(candidate_entries)
    initial_runs = int(initial_runs)
    confirm_runs = int(confirm_runs)
    finalist_count = int(finalist_count)
    if not candidate_entries:
        raise ValueError("adaptive paired evaluation requires candidates")
    if not 1 <= initial_runs <= confirm_runs:
        raise ValueError("require 1 <= initial_runs <= confirm_runs")
    if finalist_count < 1:
        raise ValueError("finalist_count must be positive")
    if not 0.0 < spread_floor_ratio <= 1.0:
        raise ValueError("spread_floor_ratio must be in (0, 1]")

    community_arrays = [
        np.asarray(community, dtype=np.int64) for community in communities
    ]
    if any(nodes.size == 0 for nodes in community_arrays):
        raise ValueError("communities must not contain empty groups")

    num_nodes = adjacency.shape[0]
    in_degrees = None
    if diffusion_model == "LT":
        in_degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        in_degrees[in_degrees == 0] = 1.0
    elif diffusion_model != "IC":
        raise ValueError(
            f"unsupported paired diffusion model: {diffusion_model}"
        )

    states = [
        {
            'runs': 0,
            'spread_sum': 0.0,
            'spread_square_sum': 0.0,
            'community_sum': np.zeros(
                len(community_arrays), dtype=np.float64
            ),
        }
        for _ in candidate_entries
    ]
    rng = np.random.default_rng(int(random_seed))

    def advance(candidate_indices, run_count, stage_name):
        progress_interval = max(1, int(run_count) // 10)
        if verbose:
            print(
                f"\n{stage_name} | 候选={len(candidate_indices)} | "
                f"场景={run_count}"
            )
        for simulation in range(int(run_count)):
            if diffusion_model == "IC":
                live_edges = rng.random(adjacency.nnz) < float(ic_prob)
                thresholds = None
            else:
                live_edges = None
                thresholds = rng.random(num_nodes)

            for candidate_index in candidate_indices:
                seeds = candidate_entries[candidate_index]['seed_set']
                if diffusion_model == "IC":
                    activated = _run_ic_live_edges(
                        adjacency, seeds, live_edges
                    )
                else:
                    activated = _run_lt_thresholds(
                        adjacency, seeds, in_degrees, thresholds
                    )
                spread = float(np.sum(activated))
                state = states[candidate_index]
                state['runs'] += 1
                state['spread_sum'] += spread
                state['spread_square_sum'] += spread * spread
                for community_index, nodes in enumerate(community_arrays):
                    state['community_sum'][community_index] += (
                        np.sum(activated[nodes]) / nodes.size
                    )

            if verbose and (
                simulation + 1 == int(run_count)
                or (simulation + 1) % progress_interval == 0
            ):
                print(
                    f"{stage_name} {simulation + 1:4d}/{int(run_count)}"
                )

    def records_from_states():
        records = []
        for candidate_index, entry in enumerate(candidate_entries):
            state = states[candidate_index]
            runs = state['runs']
            spread = state['spread_sum'] / runs
            spread_variance = (
                max(
                    0.0,
                    (
                        state['spread_square_sum']
                        - state['spread_sum'] ** 2 / runs
                    ) / (runs - 1),
                )
                if runs > 1 else 0.0
            )
            community_means = state['community_sum'] / runs
            bottom_count = max(
                1, int(math.ceil(community_means.size * 0.1))
            )
            record = dict(entry)
            record['seed_set'] = list(map(int, entry['seed_set']))
            record['metrics'] = {
                'runs': int(runs),
                'spread': float(spread),
                'spread_se': float(math.sqrt(spread_variance / runs)),
                'vcs': float(np.var(community_means)),
                'wcr': float(np.min(community_means)),
                'worst_decile_rate': float(np.mean(
                    np.sort(community_means)[:bottom_count]
                )),
            }
            records.append(record)
        return records

    all_indices = list(range(len(candidate_entries)))
    advance(all_indices, initial_runs, "候选初筛")
    screening_records = records_from_states()
    finalist_indices = _adaptive_finalist_indices(
        screening_records,
        finalist_count,
        spread_floor_ratio,
    )
    if verbose:
        labels = [
            candidate_entries[index]['label']
            for index in finalist_indices
        ]
        print(f"入围: {', '.join(labels)}")

    remaining_runs = confirm_runs - initial_runs
    if remaining_runs:
        stage_name = f"候选确认 ({confirm_runs})"
        advance(finalist_indices, remaining_runs, stage_name)
    final_records = records_from_states()
    max_spread_index = max(
        finalist_indices,
        key=lambda index: final_records[index]['metrics']['spread'],
    )
    highest_spread = final_records[max_spread_index]['metrics']['spread']
    spread_floor = spread_floor_ratio * highest_spread
    eligible_indices = [
        index for index in finalist_indices
        if final_records[index]['metrics']['spread'] >= spread_floor
    ]
    selected_index = min(
        eligible_indices,
        key=lambda index: (
            final_records[index]['metrics']['vcs'],
            -final_records[index]['metrics']['spread'],
        ),
    )
    return {
        'initial_runs': initial_runs,
        'confirm_runs': confirm_runs,
        'finalist_count': len(finalist_indices),
        'random_seed': int(random_seed),
        'screening_records': screening_records,
        'records': final_records,
        'finalist_indices': finalist_indices,
        'max_spread_index': int(max_spread_index),
        'selected_index': int(selected_index),
        'highest_spread': float(highest_spread),
        'spread_floor': float(spread_floor),
    }


def pareto_front_records(records):
    """Return Spread-up/VCS-down nondominated records."""
    records = list(records)
    front = []
    for index, record in enumerate(records):
        spread = float(record['metrics']['spread'])
        vcs = float(record['metrics']['vcs'])
        dominated = False
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            other_spread = float(other['metrics']['spread'])
            other_vcs = float(other['metrics']['vcs'])
            if (
                other_spread >= spread
                and other_vcs <= vcs
                and (other_spread > spread or other_vcs < vcs)
            ):
                dominated = True
                break
        if not dominated:
            front.append(record)
    return sorted(
        front,
        key=lambda item: (
            -float(item['metrics']['spread']),
            float(item['metrics']['vcs']),
        ),
    )


def select_pareto_knee(records):
    """Select the normalized Pareto knee without a spread floor."""
    front = pareto_front_records(records)
    if not front:
        raise ValueError("cannot select a Pareto knee from an empty pool")
    spreads = np.asarray(
        [record['metrics']['spread'] for record in front], dtype=np.float64
    )
    vcs_values = np.asarray(
        [record['metrics']['vcs'] for record in front], dtype=np.float64
    )
    spread_range = float(np.max(spreads) - np.min(spreads))
    vcs_range = float(np.max(vcs_values) - np.min(vcs_values))
    spread_scores = (
        (spreads - np.min(spreads)) / spread_range
        if spread_range > 0 else np.ones_like(spreads)
    )
    fairness_scores = (
        (np.max(vcs_values) - vcs_values) / vcs_range
        if vcs_range > 0 else np.ones_like(vcs_values)
    )
    knee_scores = (
        spread_scores + fairness_scores - 1.0
    ) / math.sqrt(2.0)

    annotated = []
    for index, record in enumerate(front):
        item = dict(record)
        item['selection'] = {
            'normalized_spread': float(spread_scores[index]),
            'normalized_fairness': float(fairness_scores[index]),
            'knee_score': float(knee_scores[index]),
        }
        annotated.append(item)
    selected_index = max(
        range(len(annotated)),
        key=lambda index: (
            float(knee_scores[index]),
            float(fairness_scores[index]),
            float(spreads[index]),
        ),
    )
    return annotated[selected_index], annotated
