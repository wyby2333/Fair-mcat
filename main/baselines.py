"""Seed-set baselines used by Fair-MCAT experiments."""


def select_degree_seeds(graph_features, seed_size):
    return graph_features["degree_order"][:seed_size].tolist()


def select_pagerank_seeds(graph_features, seed_size):
    return graph_features["pagerank_order"][:seed_size].tolist()
