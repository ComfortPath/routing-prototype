from __future__ import annotations

import heapq
import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import geopandas as gpd

import networkx as nx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

WeightHook = Callable[[str, str, dict[str, Any], float], float]


# ---------------------------------------------------------------------------
# Edge selection
# ---------------------------------------------------------------------------

def _canonical_edge(edges: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """
    Return the attributes of the canonical (forward) edge.
    """
    return edges.get(0) or next(iter(edges.values()))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return x


def _minmax_norm(x: float, xmin: float, xmax: float) -> float:
    if xmax <= xmin:
        return 0.0
    return min(max((x - xmin) / (xmax - xmin), 0.0), 1.0)


def _compute_temp_normalization_stats(
    graph: nx.MultiDiGraph,
    *,
    length_attr: str,
    temp_attr: str,
) -> dict[str, float]:
    """
    Scan canonical edges once and collect min/max values for normalization.
    """
    lengths: list[float] = []
    temps: list[float] = []

    for u, v, key, attrs in graph.edges(keys=True, data=True):
        if key != 0:
            continue

        length = _safe_float(attrs.get(length_attr))
        if length is not None and length > 0:
            lengths.append(length)

        temp = _safe_float(attrs.get(temp_attr))
        if temp is not None:
            temps.append(temp)

    if not lengths:
        raise ValueError(
            f"No valid positive values found for length attribute {length_attr!r}."
        )

    if not temps:
        raise ValueError(
            f"No valid values found for temperature attribute {temp_attr!r}."
        )

    stats = {
        "length_min": min(lengths),
        "length_max": max(lengths),
        "temp_min": min(temps),
        "temp_max": max(temps),
    }

    log.info(
        "Normalization stats | %s in [%.3f, %.3f] | %s in [%.3f, %.3f]",
        length_attr,
        stats["length_min"],
        stats["length_max"],
        temp_attr,
        stats["temp_min"],
        stats["temp_max"],
    )

    return stats


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _edge_weight(
    u: str,
    v: str,
    edges: dict[int, dict[str, Any]],
    weight_attr: str,
    weight_hooks: list[WeightHook],
) -> float:
    """
    Return the effective routing weight for the edge u->v.
    """
    attrs = _canonical_edge(edges)

    raw = attrs.get(weight_attr)
    try:
        weight = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        weight = 1.0

    if not math.isfinite(weight) or weight <= 0:
        weight = 1.0

    log.info(
        "Edge %s -> %s | base %r=%r | parsed_weight=%.3f",
        u, v, weight_attr, raw, weight
    )

    for hook in weight_hooks:
        before = weight
        weight = hook(u, v, attrs, weight)

        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Hook {hook.__name__!r} returned an invalid weight "
                f"({weight}) for edge ({u} -> {v})."
            )

        log.info(
            "Edge %s -> %s | hook=%s | before=%.3f | after=%.3f",
            u, v, getattr(hook, "__name__", repr(hook)), before, weight
        )

    log.info("Edge %s -> %s | final_weight=%.3f", u, v, weight)
    return weight


# ---------------------------------------------------------------------------
# Built-in hook factory: temperature + length blend
# ---------------------------------------------------------------------------

def make_temp_hook(
    graph: nx.MultiDiGraph,
    hour: int,
    alpha: float = 0.5,
    length_attr: str = "length",
) -> WeightHook:
    """
    Return a hook that adjusts length using hourly temperature.

    The graph is scanned once to derive normalization ranges for:
    - edge length
    - temp_mean_{hour}

    alpha:
      0.0 -> pure shortest distance
      1.0 -> strongest temperature penalty
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be 0-23, got {hour}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    temp_attr = f"temp_mean_{hour}"
    stats = _compute_temp_normalization_stats(
        graph,
        length_attr=length_attr,
        temp_attr=temp_attr,
    )

    length_min = stats["length_min"]
    length_max = stats["length_max"]
    temp_min = stats["temp_min"]
    temp_max = stats["temp_max"]

    def temp_hook(u: str, v: str, attrs: dict[str, Any], base_weight: float) -> float:
        norm_length = _minmax_norm(base_weight, length_min, length_max)

        raw_temp = attrs.get(temp_attr, temp_min)
        temp = _safe_float(raw_temp, temp_min)
        assert temp is not None

        norm_temp = _minmax_norm(temp, temp_min, temp_max)

        # Distance remains the backbone.
        # Temperature adds a penalty scaled by alpha.
        adjusted = base_weight * (1.0 + alpha * norm_temp)

        log.info(
            "temp_hook on %s -> %s | hour=%d | base_weight=%.3f | %s=%r | "
            "norm_length=%.3f | norm_temp=%.3f | adjusted=%.3f",
            u, v, hour, base_weight, temp_attr, raw_temp,
            norm_length, norm_temp, adjusted
        )

        return max(adjusted, 1e-6)

    temp_hook.__name__ = f"temp_hook(hour={hour}, alpha={alpha})"
    return temp_hook


# ---------------------------------------------------------------------------
# Core Dijkstra
# ---------------------------------------------------------------------------

def dijkstra(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    weight: str = "length",
    weight_hooks: list[WeightHook] | None = None,
) -> list[str]:
    """
    Dijkstra's algorithm on a NetworkX MultiDiGraph.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Source node {source!r} not in graph.")
    if target not in graph:
        raise nx.NodeNotFound(f"Target node {target!r} not in graph.")

    hooks: list[WeightHook] = weight_hooks or []

    log.info(
        "Starting dijkstra | source=%s | target=%s | weight=%s | hooks=%s",
        source,
        target,
        weight,
        [getattr(h, "__name__", repr(h)) for h in hooks],
    )

    dist: dict[str, float] = {source: 0.0}
    prev: dict[str, str | None] = {source: None}
    heap: list[tuple[float, str]] = [(0.0, source)]
    visited: set[str] = set()

    while heap:
        cost, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        log.info("Visiting node %s | current_cost=%.3f", u, cost)

        if u == target:
            log.info("Reached target %s with total_cost=%.3f", target, cost)
            break

        for v, edges in graph[u].items():
            if v in visited:
                continue

            log.info("Considering edge %s -> %s", u, v)

            edge_cost = _edge_weight(u, v, edges, weight, hooks)
            new_cost = cost + edge_cost
            old_cost = dist.get(v, math.inf)

            log.info(
                "Candidate path %s -> %s | edge_cost=%.3f | new_cost=%.3f | old_cost=%.3f",
                u, v, edge_cost, new_cost, old_cost
            )

            if new_cost < old_cost:
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(heap, (new_cost, v))

                log.info(
                    "Updated best path to %s | predecessor=%s | best_cost=%.3f",
                    v, u, new_cost
                )

    if target not in dist:
        raise nx.NetworkXNoPath(
            f"No path found between {source!r} and {target!r}."
        )

    path: list[str] = []
    node: str | None = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()

    if path[0] != source:
        raise nx.NetworkXNoPath(
            f"Path reconstruction failed between {source!r} and {target!r}."
        )

    log.info("Final path: %s", path)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shortest_path(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    weight: str = "length",
    weight_hooks: list[WeightHook] | None = None,
) -> list[str]:
    return dijkstra(graph, source, target, weight=weight, weight_hooks=weight_hooks)