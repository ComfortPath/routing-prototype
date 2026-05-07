"""
FastAPI server for the pedestrian routing prototype.

The network is loaded once on startup from a persisted NetworkSchema folder:

    network/
    ├── nodes.parquet
    ├── edges.parquet
    └── metadata.json

Routing is intentionally fixed for this prototype:

POST /route/path
    Snap origin and destination coordinates to the nearest network nodes.
    Select the requested hour from the hourly utci_category edge column.
    Compute the route with NumpyRoutingNetwork.

Run from the project root with:

    uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload

Optional environment variables:

    NETWORK_FOLDER=/absolute/path/to/network_folder
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import mapping

from src.routing.routing import NumpyRoutingNetwork
from src.schema import NetworkSchema


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_NETWORK_FOLDER = Path(__file__).resolve().parents[2] / "data" / "big_network"
NETWORK_FOLDER = Path(os.getenv("NETWORK_FOLDER", str(_DEFAULT_NETWORK_FOLDER)))
UTCI_CATEGORY_COL = "utci_category"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------------

_schema: NetworkSchema | None = None
_routing_network: NumpyRoutingNetwork | None = None
_network_payload: dict[str, Any] | None = None
_node_pos: dict[Any, tuple[float, float]] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Coordinate(BaseModel):
    """A geographic coordinate in longitude/latitude order."""

    lon: float
    lat: float


class RouteTime(BaseModel):
    """Hour used to select the UTCI category for each edge."""

    hour: int = Field(..., ge=0, le=23)


class RouteRequest(BaseModel):
    """Request body for POST /route/path."""

    origin: Coordinate
    destination: Coordinate
    time: RouteTime


# ---------------------------------------------------------------------------
# JSON and geometry helpers
# ---------------------------------------------------------------------------

def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy/geopandas values into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, np.ndarray):
        return [_clean_value(v) for v in value.tolist()]

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}

    return value


def _schema_to_wgs84(schema: NetworkSchema) -> NetworkSchema:
    """
    Return a schema whose geometries are in EPSG:4326 when CRS information exists.

    The web client expects longitude/latitude coordinates. If the persisted
    parquet files are already in EPSG:4326 or have no CRS, they are left as-is.
    """
    nodes = schema.nodes.copy()
    edges = schema.edges.copy()

    if nodes.crs is not None and nodes.crs.to_epsg() != 4326:
        nodes = nodes.to_crs(4326)

    if edges.crs is not None and edges.crs.to_epsg() != 4326:
        edges = edges.to_crs(4326)

    return NetworkSchema(
        nodes=nodes,
        edges=edges,
        metadata=dict(schema.metadata),
    )


def _node_id_column(schema: NetworkSchema) -> str | None:
    """Return the node-id column name if one exists; otherwise use the index."""
    return "node_id" if "node_id" in schema.nodes.columns else None


def _build_node_positions(schema: NetworkSchema) -> dict[Any, tuple[float, float]]:
    """
    Build a dictionary that maps original node IDs to (lon, lat) positions.

    Geometry is preferred over x/y columns because geometries are reprojected by
    _schema_to_wgs84, while x/y columns may still contain their original CRS.
    """
    nodes = schema.nodes
    node_id_col = _node_id_column(schema)
    node_pos: dict[Any, tuple[float, float]] = {}

    for idx, row in nodes.iterrows():
        node_id = row[node_id_col] if node_id_col is not None else idx

        geom = row.geometry if "geometry" in nodes.columns else None
        if geom is not None and not geom.is_empty:
            node_pos[node_id] = (float(geom.x), float(geom.y))
            continue

        if "x" in nodes.columns and "y" in nodes.columns:
            node_pos[node_id] = (float(row["x"]), float(row["y"]))

    if not node_pos:
        raise ValueError("No node positions found. Expected node geometry or x/y columns.")

    return node_pos


def _edge_properties(edge_row: Any, edge_row_idx: int) -> dict[str, Any]:
    """Convert one edge row into JSON-safe GeoJSON feature properties."""
    props: dict[str, Any] = {"edge_row": edge_row_idx}

    for key, value in edge_row.items():
        if key == "geometry":
            continue
        props[str(key)] = _clean_value(value)

    return props


def _edge_geometry(edge_row: Any, node_pos: dict[Any, tuple[float, float]]) -> dict[str, Any] | None:
    """
    Return a GeoJSON geometry for one edge row.

    The stored edge geometry is preferred. If it is missing, a simple straight
    line between the edge's u and v node positions is returned.
    """
    geom = edge_row.geometry if "geometry" in edge_row.index else None
    if geom is not None and not geom.is_empty:
        return mapping(geom)

    src = node_pos.get(edge_row["u"])
    dst = node_pos.get(edge_row["v"])
    if src is None or dst is None:
        return None

    return {
        "type": "LineString",
        "coordinates": [list(src), list(dst)],
    }


def _edge_feature(
    edge_row: Any,
    edge_row_idx: int,
    node_pos: dict[Any, tuple[float, float]],
) -> dict[str, Any] | None:
    """Convert one edge row into a GeoJSON Feature."""
    geometry = _edge_geometry(edge_row, node_pos)
    if geometry is None:
        return None

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": _edge_properties(edge_row, edge_row_idx),
    }


def _build_network_payload(
    schema: NetworkSchema,
    routing_network: NumpyRoutingNetwork,
    node_pos: dict[Any, tuple[float, float]],
) -> dict[str, Any]:
    """Build the GET /route/network response from the persisted schema."""
    features: list[dict[str, Any]] = []

    for edge_row_idx, edge_row in routing_network.edges.iterrows():
        feature = _edge_feature(edge_row, edge_row_idx, node_pos)
        if feature is not None:
            features.append(feature)

    xs = [coord[0] for coord in node_pos.values()]
    ys = [coord[1] for coord in node_pos.values()]

    return {
        "geojson": {
            "type": "FeatureCollection",
            "features": features,
        },
        "center": [
            round((min(xs) + max(xs)) / 2, 6),
            round((min(ys) + max(ys)) / 2, 6),
        ],
        "node_count": routing_network.n_nodes,
        "edge_count": routing_network.n_edges,
        "metadata": _clean_value(schema.metadata),
    }


# ---------------------------------------------------------------------------
# Spatial routing helpers
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in metres between two lon/lat coordinates."""
    radius_m = 6_371_000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_node(lon: float, lat: float) -> Any:
    """Return the node ID closest to the given lon/lat coordinate."""
    if not _node_pos:
        raise RuntimeError("Node-position index is empty.")

    return min(
        _node_pos,
        key=lambda node_id: _haversine_m(
            lon,
            lat,
            _node_pos[node_id][0],
            _node_pos[node_id][1],
        ),
    )


def _edge_rows_from_node_path(
    routing_network: NumpyRoutingNetwork,
    node_path: Any,
) -> list[int]:
    """Convert a returned node path into edge row indices."""
    node_ids = np.asarray(node_path).tolist()
    edge_rows: list[int] = []

    for u_id, v_id in zip(node_ids[:-1], node_ids[1:]):
        u_idx = routing_network.node_to_idx[u_id]
        v_idx = routing_network.node_to_idx[v_id]

        for neighbor_idx, edge_idx in routing_network.adjacency[u_idx]:
            if neighbor_idx == v_idx:
                edge_rows.append(int(edge_idx))
                break
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Could not find edge between routed nodes {u_id!r} and {v_id!r}.",
            )

    return edge_rows


def _route_to_geojson(
    routing_network: NumpyRoutingNetwork,
    edge_rows: list[int],
) -> tuple[dict[str, Any], float, float | None]:
    """Convert route edge row indices into route GeoJSON and summary statistics."""
    features: list[dict[str, Any]] = []
    total_distance_m = 0.0
    total_duration_s: float | None = 0.0

    for edge_row_idx in edge_rows:
        edge_row = routing_network.edges.iloc[edge_row_idx]

        feature = _edge_feature(edge_row, edge_row_idx, _node_pos)
        if feature is not None:
            features.append(feature)

        if "length" in routing_network.edges.columns:
            length = _clean_value(edge_row["length"])
            if length is not None:
                total_distance_m += float(length)

        if total_duration_s is not None:
            duration = None
            if "travel_time" in routing_network.edges.columns:
                duration = edge_row["travel_time"]
            elif "duration" in routing_network.edges.columns:
                duration = edge_row["duration"]

            duration = _clean_value(duration)
            if duration is None:
                total_duration_s = None
            else:
                total_duration_s += float(duration)

    return (
        {"type": "FeatureCollection", "features": features},
        total_distance_m,
        total_duration_s,
    )


def _route_cost(routing_network: NumpyRoutingNetwork, edge_rows: list[int], hour: int) -> float:
    """Return the weighted cost for the selected route."""
    weights = routing_network.add_weights([UTCI_CATEGORY_COL], routing_network.edge_cost, hour)
    return float(np.sum(weights[edge_rows]))


def _serialise_node_path(node_ids: Any) -> list[Any]:
    """Convert a NumPy array of node IDs into a JSON-safe list."""
    return [_clean_value(node_id) for node_id in np.asarray(node_ids).tolist()]


# ---------------------------------------------------------------------------
# Lifespan: load the parquet network once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NetworkSchema, build the routing network, and cache API payloads."""
    global _schema, _routing_network, _network_payload, _node_pos

    if not NETWORK_FOLDER.exists():
        log.error("Network folder not found: %s", NETWORK_FOLDER)
        yield
        return

    try:
        log.info("Loading network schema from %s", NETWORK_FOLDER)
        schema = NetworkSchema.from_folder(NETWORK_FOLDER)
        schema = _schema_to_wgs84(schema)

        routing_network = NumpyRoutingNetwork(
            schema=schema,
            directed=False,
        )

        if UTCI_CATEGORY_COL not in routing_network.edges.columns:
            raise ValueError(f"Edge table does not contain required column {UTCI_CATEGORY_COL!r}.")


        node_pos = _build_node_positions(schema)
        payload = _build_network_payload(schema, routing_network, node_pos)

        _schema = schema
        _routing_network = routing_network
        _node_pos = node_pos
        _network_payload = payload

        log.info(
            "Network ready: %d nodes / %d edges / %d GeoJSON features",
            routing_network.n_nodes,
            routing_network.n_edges,
            len(payload["geojson"]["features"]),
        )

    except Exception as exc:
        log.error("Failed to load network: %s", exc, exc_info=True)

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pedestrian Routing API",
    description="Serves a parquet-based pedestrian network and computes UTCI-category routes.",
    version="2.1.0",
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
async def get_network() -> dict[str, Any]:
    """
    Return the full edge network as a GeoJSON FeatureCollection.

    Array-valued edge properties such as utci_category are returned as JSON lists.
    The response also includes the map center, node count, edge count, and
    persisted metadata from metadata.json.
    """
    if _network_payload is None:
        raise HTTPException(
            status_code=503,
            detail=f"Network is not available. Check NETWORK_FOLDER: {NETWORK_FOLDER}",
        )

    return _network_payload


@app.post("/route/path", summary="Compute a UTCI-aware route between two coordinates")
async def post_route(body: RouteRequest) -> dict[str, Any]:
    """
    Snap origin and destination coordinates to the nearest network nodes and
    compute a shortest path using only utci_category at the selected hour.

    Example request:

    ```json
    {
      "origin": {"lon": 4.47, "lat": 51.91},
      "destination": {"lon": 4.49, "lat": 51.93},
      "time": {"hour": 10}
    }
    ```
    """
    if _routing_network is None:
        raise HTTPException(status_code=503, detail="Network not loaded.")

    origin_node = _nearest_node(body.origin.lon, body.origin.lat)
    destination_node = _nearest_node(body.destination.lon, body.destination.lat)

    if origin_node == destination_node:
        raise HTTPException(
            status_code=400,
            detail="Origin and destination snap to the same node. Move the points further apart.",
        )

    hour = body.time.hour

    node_path = _routing_network.shortest_path(
        source_node_id=origin_node,
        target_node_id=destination_node,
        hour=hour,
    )

    if node_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between nodes {origin_node!r} and {destination_node!r}.",
        )

    edge_rows = _edge_rows_from_node_path(_routing_network, node_path)
    geojson, distance_m, duration_s = _route_to_geojson(_routing_network, edge_rows)

    return {
        "geojson": geojson,
        "distance_m": round(distance_m, 2),
        "duration_s": round(duration_s, 1) if duration_s is not None else None,
        "cost": round(_route_cost(_routing_network, edge_rows, hour), 3),
        "weight_variable": UTCI_CATEGORY_COL,
        "weight_hour": hour,
        "origin_node": _clean_value(origin_node),
        "destination_node": _clean_value(destination_node),
        "node_path": _serialise_node_path(node_path),
        "edge_rows": edge_rows,
    }


@app.get("/health", summary="Health check")
async def health() -> dict[str, Any]:
    """Return server status and network-loading information."""
    return {
        "status": "ok",
        "network_loaded": _routing_network is not None,
        "network_folder": str(NETWORK_FOLDER),
        "routing_column": UTCI_CATEGORY_COL,
        "node_count": _routing_network.n_nodes if _routing_network else None,
        "edge_count": _routing_network.n_edges if _routing_network else None,
    }
