"""スケルトン画像 → グラフ構造への変換（交差点集約・パス抽出の基盤）。

ノード＝端点(neighbor_count==1)・分岐点(neighbor_count>=3)・孤立点(neighbor_count==0)。
エッジ＝ノード間を結ぶ neighbor_count==2 の画素チェーン（両端ノード込みの画素列）。

分岐/端点を一切持たない純粋な閉ループ（塗りつぶし図形の輪郭など）は、
ループ上の任意の1画素を人工ノード(role="synthetic")として立て、自己ループエッジにする。

XYPWriter (xypwriter/graph_build.py) から移植した汎用ロジック。
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label

from .skeleton_stage import get_neighbors, neighbor_count
from .types import NodeRole, SkeletonEdge, SkeletonGraph, SkeletonNode


def _polyline_pixel_length(pixels: list[tuple[int, int]]) -> float:
    if len(pixels) < 2:
        return 0.0
    arr = np.array(pixels, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def _trace_chain(skel: np.ndarray, start_node_pixel: tuple[int, int], first_chain_pixel: tuple[int, int]):
    """start_node_pixel から first_chain_pixel 方向へ、次のノード画素に到達するまで辿る。

    戻り値: (chain_pixels, terminal_pixel) terminal_pixel は次に到達したノード画素
    （到達できなかった場合は None）。chain_pixels は start_node_pixel を含まず
    terminal_pixel を含む画素列。
    """
    chain = [first_chain_pixel]
    prev, cur = start_node_pixel, first_chain_pixel
    return chain, prev, cur


def build_skeleton_graph(skel: np.ndarray) -> SkeletonGraph:
    if not np.any(skel):
        return SkeletonGraph(nodes={}, edges={}, image_shape=skel.shape)

    nb = neighbor_count(skel)
    node_mask = skel & (nb != 2)

    nodes: dict[int, SkeletonNode] = {}
    pixel_to_node: dict[tuple[int, int], int] = {}
    next_node_id = 1

    if np.any(node_mask):
        labeled, n = label(node_mask, connectivity=2, return_num=True)
        for comp_id in range(1, n + 1):
            ys, xs = np.where(labeled == comp_id)
            pixels = list(zip(ys.tolist(), xs.tolist()))
            role: NodeRole = "junction" if any(nb[y, x] >= 3 for y, x in pixels) else "endpoint"
            centroid = (float(np.mean(ys)), float(np.mean(xs)))
            nodes[next_node_id] = SkeletonNode(id=next_node_id, pixels=pixels, position=centroid, role=role)
            for p in pixels:
                pixel_to_node[p] = next_node_id
            next_node_id += 1

    edges: dict[int, SkeletonEdge] = {}
    next_edge_id = 1
    used_chain_pixels: set[tuple[int, int]] = set()
    used_direct_pairs: set[frozenset] = set()

    for node_id, node in list(nodes.items()):
        for p in node.pixels:
            for nbr in get_neighbors(skel, p):
                if nbr in pixel_to_node:
                    other_id = pixel_to_node[nbr]
                    if other_id == node_id:
                        continue
                    pair_key = frozenset({(node_id, p), (other_id, nbr)})
                    if pair_key in used_direct_pairs:
                        continue
                    used_direct_pairs.add(pair_key)
                    edges[next_edge_id] = SkeletonEdge(
                        id=next_edge_id,
                        node_a=node_id,
                        node_b=other_id,
                        pixels=[p, nbr],
                        length_px=_polyline_pixel_length([p, nbr]),
                    )
                    next_edge_id += 1
                elif nbr not in used_chain_pixels:
                    chain, prev, cur = _trace_chain(skel, p, nbr)
                    used_chain_pixels.add(cur)
                    other_id = None
                    while cur not in pixel_to_node:
                        nxts = [x for x in get_neighbors(skel, cur) if x != prev]
                        if not nxts:
                            break
                        nxt = nxts[0]
                        chain.append(nxt)
                        used_chain_pixels.add(nxt)
                        prev, cur = cur, nxt
                    if cur in pixel_to_node:
                        other_id = pixel_to_node[cur]
                        edges[next_edge_id] = SkeletonEdge(
                            id=next_edge_id,
                            node_a=node_id,
                            node_b=other_id,
                            pixels=[p, *chain],
                            length_px=_polyline_pixel_length([p, *chain]),
                        )
                        next_edge_id += 1

    # 分岐/端点を一切持たない純粋な閉ループ成分を検出し、人工ノードを立てる
    labeled_full, n_full = label(skel, connectivity=2, return_num=True)
    for comp_id in range(1, n_full + 1):
        ys, xs = np.where(labeled_full == comp_id)
        comp_pixels = set(zip(ys.tolist(), xs.tolist()))
        if comp_pixels & pixel_to_node.keys():
            continue  # 既にノードを持つ成分（この成分にはjunction/endpointが存在する）
        start = next(iter(comp_pixels))
        node_id = next_node_id
        next_node_id += 1
        nodes[node_id] = SkeletonNode(
            id=node_id, pixels=[start], position=(float(start[0]), float(start[1])), role="synthetic"
        )
        pixel_to_node[start] = node_id
        neighbors = get_neighbors(skel, start)
        if not neighbors:
            continue  # 孤立単画素（度数0）、エッジなし
        chain = [start]
        prev, cur = start, neighbors[0]
        chain.append(cur)
        while cur != start:
            nxts = [x for x in get_neighbors(skel, cur) if x != prev]
            if not nxts:
                break
            nxt = nxts[0]
            chain.append(nxt)
            prev, cur = cur, nxt
        edges[next_edge_id] = SkeletonEdge(
            id=next_edge_id,
            node_a=node_id,
            node_b=node_id,
            pixels=chain,
            length_px=_polyline_pixel_length(chain),
        )
        next_edge_id += 1

    return SkeletonGraph(nodes=nodes, edges=edges, image_shape=skel.shape)


def edge_points(edge: SkeletonEdge) -> np.ndarray:
    """エッジの画素列を (N, 2) float配列 (row, col) として返す。"""
    return np.array(edge.pixels, dtype=float)
