"""Community-partition loading and validation."""

from data.config import DIFFUSION_FILE_SUFFIX


def load_saved_communities(graph_data, num_nodes):
    """读取并校验数据生成阶段保存的唯一社区划分。"""
    if 'communities' not in graph_data or graph_data['communities'] is None:
        raise ValueError(
            f"当前 {DIFFUSION_FILE_SUFFIX} 文件没有保存 communities，"
            "无法保证训练、搜索与 VCS 使用同一划分。"
            "请使用 data/generate.py 重新生成。"
        )

    raw_communities = graph_data['communities']
    if not isinstance(raw_communities, list):
        raise TypeError("communities 必须是列表。")
    communities = [
        sorted(int(node) for node in nodes)
        for nodes in raw_communities
    ]

    if not communities:
        raise ValueError("保存的 communities 为空。")

    seen = set()
    for community_id, nodes in enumerate(communities):
        if not nodes:
            raise ValueError(f"保存的社区 {community_id} 为空。")
        for node in nodes:
            if node < 0 or node >= num_nodes:
                raise ValueError(
                    f"保存的社区 {community_id} 包含越界节点 {node}，"
                    f"合法范围为 0~{num_nodes - 1}。"
                )
            if node in seen:
                raise ValueError(f"保存的节点 {node} 同时出现在多个社区中。")
            seen.add(node)

    expected = set(range(num_nodes))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            "保存的社区划分未完整覆盖邻接矩阵节点："
            f"缺失节点 {missing[:10]}，额外节点 {extra[:10]}。"
        )
    return communities

