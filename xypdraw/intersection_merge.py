"""交差点の集約（スケルトンのブロブ化解消）。

直接隣接する分岐画素は graph_build.build_skeleton_graph の時点で
既に1ノードにまとまっている。ここで残る仕事は「短い画素チェーンで
隔てられた分岐点同士」の統合で、Union-Find（DSU）による辺の縮約
（edge contraction）として実装する。

XYPWriter (xypwriter/intersection_merge.py) から移植した汎用ロジック。
"""
from __future__ import annotations

from .types import SkeletonEdge, SkeletonGraph, SkeletonNode


class _DSU:
    def __init__(self, ids: list[int]):
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_close_junctions(graph: SkeletonGraph, radius_px: float) -> SkeletonGraph:
    """radius_px 以下の長さのエッジで結ばれた分岐点(junction)同士を1ノードに統合する。

    端点(endpoint)は統合対象にしない（本来別々のストローク終端を誤って
    交点に吸着させないための安全策）。
    """
    if radius_px <= 0 or not graph.nodes:
        return graph

    node_ids = list(graph.nodes.keys())
    dsu = _DSU(node_ids)

    def is_junction(node_id: int) -> bool:
        return graph.nodes[node_id].role == "junction"

    for edge in graph.edges.values():
        if edge.node_a == edge.node_b:
            continue  # 自己ループは統合対象外
        if not (is_junction(edge.node_a) and is_junction(edge.node_b)):
            continue
        if edge.length_px <= radius_px:
            dsu.union(edge.node_a, edge.node_b)

    return _rebuild_graph_from_dsu(graph, dsu)


def _rebuild_graph_from_dsu(graph: SkeletonGraph, dsu: _DSU) -> SkeletonGraph:
    groups: dict[int, list[int]] = {}
    for node_id in graph.nodes:
        root = dsu.find(node_id)
        groups.setdefault(root, []).append(node_id)

    new_nodes: dict[int, SkeletonNode] = {}
    old_to_new: dict[int, int] = {}
    next_id = 1
    for root, members in groups.items():
        pixels: list[tuple[int, int]] = []
        for m in members:
            pixels.extend(graph.nodes[m].pixels)
        weighted_y = sum(p[0] for p in pixels) / len(pixels)
        weighted_x = sum(p[1] for p in pixels) / len(pixels)
        roles = {graph.nodes[m].role for m in members}
        role = "junction" if "junction" in roles else next(iter(roles))
        new_nodes[next_id] = SkeletonNode(
            id=next_id, pixels=pixels, position=(weighted_y, weighted_x), role=role
        )
        for m in members:
            old_to_new[m] = next_id
        next_id += 1

    new_edges: dict[int, SkeletonEdge] = {}
    next_edge_id = 1
    for edge in graph.edges.values():
        new_a = old_to_new[edge.node_a]
        new_b = old_to_new[edge.node_b]
        # 統合により内部エッジ化したもの(new_a == new_b)も、実際の画素列(輪の一部等)を
        # 表しているため破棄せず自己ループとして保持する。path_extraction側は
        # 自己ループを正しく扱える(_pick_start_nodeはnode_a/node_bを別々にカウントするため
        # degreeへの寄与が+2になる)。破棄すると輪(結び)の一部が描画されなくなるバグになる。
        new_edges[next_edge_id] = SkeletonEdge(
            id=next_edge_id,
            node_a=new_a,
            node_b=new_b,
            pixels=edge.pixels,
            length_px=edge.length_px,
        )
        next_edge_id += 1

    return SkeletonGraph(nodes=new_nodes, edges=new_edges, image_shape=graph.image_shape)


def check_merge_safety(before: SkeletonGraph, after: SkeletonGraph) -> list[str]:
    """統合前後で連結成分数や異常な高次数ノードが生じていないか確認する。"""
    warnings: list[str] = []

    def component_count(g: SkeletonGraph) -> int:
        if not g.nodes:
            return 0
        parent = {n: n for n in g.nodes}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for e in g.edges.values():
            union(e.node_a, e.node_b)
        return len({find(n) for n in g.nodes})

    n_before = component_count(before)
    n_after = component_count(after)
    if n_before != n_after:
        warnings.append(
            f"交差点集約前後で連結成分数が変化しました ({n_before} -> {n_after})。"
        )

    for node_id in after.nodes:
        degree = after.degree(node_id)
        if degree > 6:
            warnings.append(f"ノード{node_id}の次数が異常に高いです (degree={degree})。過剰統合の疑いがあります。")

    return warnings
