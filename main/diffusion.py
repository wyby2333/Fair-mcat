"""Single-run diffusion kernels used by final experiment evaluation."""

import numpy as np


def run_ic_once(adjacency, seeds, activation_probability, rng=np.random):
    """Run one Independent Cascade simulation without changing RNG order."""
    activated = np.zeros(adjacency.shape[0], dtype=bool)
    activated[seeds] = True
    active_nodes = list(seeds)

    while active_nodes:
        new_active = []
        for node in active_nodes:
            neighbors = adjacency.indices[
                adjacency.indptr[node]:adjacency.indptr[node + 1]
            ]
            for neighbor in neighbors:
                if (
                    not activated[neighbor]
                    and rng.random() < activation_probability
                ):
                    activated[neighbor] = True
                    new_active.append(neighbor)
        active_nodes = new_active
    return activated


def run_lt_once(adjacency, seeds, in_degrees=None, rng=np.random):
    """Run one Linear Threshold simulation without changing RNG order."""
    num_nodes = adjacency.shape[0]
    if in_degrees is None:
        in_degrees = np.array(adjacency.sum(axis=1)).flatten()
        in_degrees[in_degrees == 0] = 1.0

    activated = np.zeros(num_nodes, dtype=bool)
    activated[seeds] = True
    active_nodes = list(seeds)
    thresholds = rng.uniform(0, 1, num_nodes)

    while active_nodes:
        new_active = []
        inactive_nodes = np.where(~activated)[0]
        for node in inactive_nodes:
            neighbors = adjacency.indices[
                adjacency.indptr[node]:adjacency.indptr[node + 1]
            ]
            active_count = np.sum(activated[neighbors])
            if active_count / in_degrees[node] >= thresholds[node]:
                activated[node] = True
                new_active.append(node)
        active_nodes = new_active
    return activated
