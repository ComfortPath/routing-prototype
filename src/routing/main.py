"""
FastAPI server — serves a pedestrian street network at GET /route/network
and computes shortest paths at POST /route/path.

The graph is loaded once on startup from a local GraphML file, then cached
in memory for fast responses.

GET /route/network response shape:
{
    "geojson":    { "type": "FeatureCollection", "features": [...] },
    "center":     [lon, lat],
    "node_count": int,
    "edge_count":  int
}

POST /route/path request shape:
{
    "origin":      { "lon": float, "lat": float },
    "destination": { "lon": float, "lat": float }
}

POST /route/path response shape:
{
    "geojson":    { "type": "FeatureCollection", "features": [...] },
    "distance_m": float,
    "duration_s": float   (None if no travel-time attribute available)
}

Usage
-----
pip install fastapi uvicorn networkx

# Default path (expects big_network.graphml two levels above this file):
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Custom path via env var:
GRAPHML_PATH="/absolute/path/to/big_network.graphml" uvicorn main:app --port 8001
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import networkx as nx
from fastapi import FastAPI, HTTPException
from api.routing import shortest_path, make_temp_hook, WeightHook
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default: <repo_root>/data/big_network.graphml
_DEFAULT_GRAPHML = Path(__file__).resolve().parents[1] / "data" / "big_network.graphml"
GRAPHML_PATH: Path = Path(os.getenv("GRAPHML_PATH", str(_DEFAULT_GRAPHML)))

# Edge attribute used as routing weight. Falls back to straight-line distance
# if not present in the GraphML.
WEIGHT_ATTR = os.getenv("ROUTE_WEIGHT", "length")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global variables to use throughout
# ---------------------------------------------------------------------------

_network_payload: dict | None = None
_graph: nx.MultiDiGraph | None = None          # kept in memory for routing
_node_pos: dict[str, tuple[float, float]] = {} # node_id -> (lon, lat)


# ---------------------------------------------------------------------------
# Custom classes to keep track of route elements using pydantic (allows for minimal instanstiation)
# ---------------------------------------------------------------------------

class Coordinate(BaseModel):
    lon: float
    lat: float

class RouteTime(BaseModel):
    hour: int

class RouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    time: RouteTime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_value(v: Any) -> Any:
    """Sanitise a single property value for JSON serialisation."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, list):
        return v[0] if len(v) == 1 else str(v)
    if isinstance(v, bool):
        return v
    return v


def _load_graphml(path: Path) -> nx.MultiDiGraph:
    """Load a GraphML file into a NetworkX MultiDiGraph."""
    log.info("Loading GraphML from %s …", path)
    graph = nx.read_graphml(path, node_type=str, force_multigraph=True)
    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    log.info("Loaded %d nodes / %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _build_payload(graph: nx.MultiDiGraph) -> tuple[dict, dict[str, tuple[float, float]]]:
    """
    Convert a NetworkX MultiDiGraph into the API payload and a node-position index.

    Returns (payload_dict, node_pos_dict).
    """
    # ------------------------------------------------------------------
    # 1. Index node positions
    # ------------------------------------------------------------------
    node_pos: dict[str, tuple[float, float]] = {}
    xs: list[float] = []
    ys: list[float] = []

    for node_id, attrs in graph.nodes(data=True):
        try:
            x = float(attrs["x"])   # longitude
            y = float(attrs["y"])   # latitude
        except (KeyError, TypeError, ValueError):
            continue
        node_pos[node_id] = (x, y)
        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError(
            "No node positions found — GraphML must have 'x' and 'y' node attributes."
        )

    center_lon = (min(xs) + max(xs)) / 2
    center_lat = (min(ys) + max(ys)) / 2

    # ------------------------------------------------------------------
    # 2. Build GeoJSON features from edges
    # ------------------------------------------------------------------
    features: list[dict[str, Any]] = []

    for u, v, _key, edge_attrs in graph.edges(data=True, keys=True):
        src = node_pos.get(str(u))
        dst = node_pos.get(str(v))

        if src is None or dst is None:
            continue

        geometry = {
            "type": "LineString",
            "coordinates": [list(src), list(dst)],
        }

        props: dict[str, Any] = {"u": u, "v": v, "key": _key, "edge_attrs": edge_attrs}
        for k, val in edge_attrs.items():
            props[k] = _clean_value(val)

        features.append({"type": "Feature", "geometry": geometry, "properties": props})

    log.info("Built GeoJSON with %d edge features.", len(features))

    payload = {
        "geojson": {"type": "FeatureCollection", "features": features},
        "center": [round(center_lon, 6), round(center_lat, 6)],
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }
    return payload, node_pos


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two (lon, lat) points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlamb / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_node(lon: float, lat: float) -> str:
    """Return the node_id of the graph node closest to (lon, lat)."""
    best_id = min(
        _node_pos,
        key=lambda nid: _haversine_m(lon, lat, _node_pos[nid][0], _node_pos[nid][1]),
    )
    return best_id


def _path_to_geojson(
    graph: nx.MultiDiGraph,
    node_path: list[str],
) -> tuple[dict, float, float | None]:
    """
    Convert a list of node IDs to a GeoJSON FeatureCollection of edge LineStrings.

    Returns (geojson, total_distance_m, total_duration_s_or_None).
    """
    features: list[dict] = []
    total_distance_m = 0.0
    total_duration_s: float | None = 0.0

    for u, v in zip(node_path[:-1], node_path[1:]):
        # Pick the lightest parallel edge (by WEIGHT_ATTR)
        edges = graph[u][v]
        best_key = min(
            edges,
            key=lambda k: float(edges[k].get(WEIGHT_ATTR, 1) or 1),
        )
        attrs = edges[best_key]

        src = _node_pos.get(str(u))
        dst = _node_pos.get(str(v))
        if src is None or dst is None:
            continue

        seg_len = float(attrs.get("length") or 0) or _haversine_m(*src, *dst)
        total_distance_m += seg_len

        # Duration — use travel_time if present, otherwise None
        if total_duration_s is not None:
            travel_time = attrs.get("travel_time") or attrs.get("duration")
            if travel_time is not None:
                try:
                    total_duration_s += float(travel_time)
                except (TypeError, ValueError):
                    total_duration_s = None
            else:
                total_duration_s = None

        props: dict[str, Any] = {"u": u, "v": v}
        for k, val in attrs.items():
            props[k] = _clean_value(val)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(src), list(dst)],
            },
            "properties": props,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, total_distance_m, total_duration_s


# ---------------------------------------------------------------------------
# Lifespan — load once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _network_payload, _graph, _node_pos

    if not GRAPHML_PATH.exists():
        log.error("GraphML file not found: %s", GRAPHML_PATH)
        log.error("Set the GRAPHML_PATH environment variable to the correct path.")
        yield
        return

    try:
        graph = _load_graphml(GRAPHML_PATH)
        payload, node_pos = _build_payload(graph)

        # graph is stored in memory to use again for routing
        _graph = graph
        _node_pos = node_pos
        _network_payload = payload

        log.info(
            "Network ready — center %s, %d features.",
            _network_payload["center"],
            len(_network_payload["geojson"]["features"]),
        )
    except Exception as exc:
        log.error("Failed to load network: %s", exc, exc_info=True)

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Temp Network API",
    description="Serves a pedestrian street network (+ temperature data) as GeoJSON from a GraphML file.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/route/network", summary="Return the full pedestrian network as GeoJSON")
async def get_network() -> dict:
    """
    Returns:
    - **geojson**: GeoJSON FeatureCollection of network edges (LineStrings).
    - **center**: `[longitude, latitude]` derived from the node bounding box.
    - **node_count**: number of graph nodes.
    - **edge_count**: number of graph edges.
    """
    if _network_payload is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Network data is not available. "
                f"Check that GRAPHML_PATH points to a valid file (currently: {GRAPHML_PATH})."
            ),
        )
    return _network_payload


@app.post("/route/path", summary="Compute shortest path between two coordinates")
async def post_route(body: RouteRequest) -> dict:
    """
    Snaps both coordinates to the nearest graph nodes and returns the
    shortest path (weighted by `ROUTE_WEIGHT`, default: `length`).

    Request body:
    ```json
    {
        "origin":      { "lon": 4.47, "lat": 51.91 },
        "destination": { "lon": 4.49, "lat": 51.93 }
        "time": { "hour" : 10}
    }
    ```

    Returns:
    - **geojson**: GeoJSON FeatureCollection of the route edges.
    - **distance_m**: total route length in metres.
    - **duration_s**: travel time in seconds (null if not in graph data).
    - **origin_node**: snapped origin node ID.
    - **destination_node**: snapped destination node ID.
    - **node_path**: ordered list of node IDs along the route.
    """
    if _graph is None or not _node_pos:
        raise HTTPException(status_code=503, detail="Network not loaded.")

    origin_node = _nearest_node(body.origin.lon, body.origin.lat)
    dest_node = _nearest_node(body.destination.lon, body.destination.lat)

    time = body.time.hour
    WEIGHT_ATTR = f"temp_mean_{time}"

    if origin_node == dest_node:
        raise HTTPException(
            status_code=400,
            detail="Origin and destination snap to the same node. Move the points further apart.",
        )

    try:

        # To add extra weight penalties, define hooks and pass them here:
        # thsi can be used to add extra pentalties down the line.
        """ 
        def temperature_hook(u: str, v: str, attrs: dict, base_weight: float) -> float:
            temp = attrs.get("temperature", 0)
            return base_weight * (1 + 0.01 * temp)

        weight_hooks: list[WeightHook] = [temperature_hook]
        """
        #   weight_hooks: list[WeightHook] = [my_hook]
        # alpha gives the importance of the weight in [0-1]
        weight_hooks = [make_temp_hook(_graph, hour=time, alpha=1)]  # time is already body.time.hour

        node_path = shortest_path(
            _graph,
            source=origin_node,
            target=dest_node,
            weight="length",
            weight_hooks=weight_hooks,
        )
    except nx.NetworkXNoPath:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between nodes {origin_node} and {dest_node}.",
        )
    except nx.NodeNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    geojson, distance_m, duration_s = _path_to_geojson(_graph, node_path)

    return {
        "geojson": geojson,
        "distance_m": round(distance_m, 2),
        "duration_s": round(duration_s, 1) if duration_s is not None else None,
        "origin_node": origin_node,
        "destination_node": dest_node,
        "node_path": node_path,
    }


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {
        "status": "ok",
        "network_loaded": _network_payload is not None,
        "graphml_path": str(GRAPHML_PATH),
        "node_count": _network_payload["node_count"] if _network_payload else None,
        "edge_count": _network_payload["edge_count"] if _network_payload else None,
    }
