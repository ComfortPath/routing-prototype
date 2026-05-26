"""
FastAPI server for the pedestrian routing prototype.

The network is loaded once on startup from a persisted NetworkSchema folder:

    network/
    ├── nodes.parquet
    ├── edges.parquet
    └── metadata.json

POST /route/path
    Snap origin and destination coordinates to the nearest network nodes.
    Build a routing weight_config from the selected environmental variables.
    Compute the route with NumpyRoutingNetwork.

Run from the project root with:

    uvicorn src.routing.main:app --host 0.0.0.0 --port 8001 --reload

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

_DEFAULT_NETWORK_FOLDER = Path(__file__).resolve().parents[2] / "data" / "network_final" / "red_bbox"
NETWORK_FOLDER = Path(os.getenv("NETWORK_FOLDER", str(_DEFAULT_NETWORK_FOLDER)))
DEFAULT_WEIGHT_VARIABLE = "utci_category"
UTCI_MEDIAN_VARIABLE = "utci_median"


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
_available_weight_variable_names: set[str] = set()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Coordinate(BaseModel):
    """A geographic coordinate in longitude/latitude order."""

    lon: float
    lat: float


class RouteTime(BaseModel):
    """Hour used for selected hourly edge variables."""

    hour: int = Field(..., ge=0, le=23)


class WeightVariable(BaseModel):
    """One variable entry in the routing.py weight_config dictionary."""

    name: str
    w: float = Field(..., ge=0.0)
    hour: int | None = Field(default=None, ge=0, le=23)
    column: str | None = None
    gamma_state_factor: float | None = Field(default=None, ge=0.0)
    hot_categories: list[int] | None = None
    counts_as_hot: bool | None = None
    hot_threshold: float | None = None


class WeightConfig(BaseModel):
    """Routing.py-compatible weight configuration."""

    variables: list[WeightVariable] = Field(default_factory=list)
    hot_state_increment: int = Field(default=3, ge=0)
    cold_state_recovery: int = Field(default=1, ge=0)
    hot_edge_routing: bool | None = None


class RouteRequest(BaseModel):
    """Request body for POST /route/path."""

    origin: Coordinate
    destination: Coordinate
    time: RouteTime
    weight_config: WeightConfig | None = None


# ---------------------------------------------------------------------------
# JSON, metadata, and geometry helpers
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


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a pydantic model as a dictionary for both pydantic v1 and v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    return model.dict(exclude_none=True)


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


def _build_available_weight_variables(
    routing_network: NumpyRoutingNetwork,
) -> list[dict[str, Any]]:
    """Describe the fixed routing variable exposed to the front-end.

    Routing intentionally uses only UTCI category. The variables are therefore
    not derived from metadata.json anymore.
    """
    if DEFAULT_WEIGHT_VARIABLE not in routing_network.edges.columns:
        log.warning(
            "Routing variable %r is not present in the edge table and will not be exposed.",
            DEFAULT_WEIGHT_VARIABLE,
        )
        return []

    return [{
        "name": DEFAULT_WEIGHT_VARIABLE,
        "label": "UTCI category",
        "hourly": True,
        "hours": 24,
        "default_selected": True,
        "default_weight": 1.0,
        "min_weight": 0.0,
        "max_weight": 2.0,
        "step": 0.1,
    }]


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
    available_weight_variables = _build_available_weight_variables(routing_network)

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
        "available_weight_variables": available_weight_variables,
    }


# ---------------------------------------------------------------------------
# Weight-config helpers
# ---------------------------------------------------------------------------

def _default_weight_config(hour: int) -> dict[str, Any]:
    """Fallback config when a client does not send environmental preferences."""
    return {"variables": []}


def _utci_gamma(gamma_state_factor: float):
    """Return the route-state sensitivity function used by routing.py."""
    return lambda state, factor=gamma_state_factor: 1.0 + factor * state


def _normalise_weight_variable(raw_variable: dict[str, Any], fallback_hour: int) -> dict[str, Any] | None:
    """Validate and convert the fixed UTCI-category route variable."""
    variable = dict(raw_variable)
    name = variable.get("name")

    if name != DEFAULT_WEIGHT_VARIABLE:
        raise HTTPException(
            status_code=422,
            detail=f"Only {DEFAULT_WEIGHT_VARIABLE!r} can be used as a routing variable.",
        )

    if name not in _available_weight_variable_names:
        raise HTTPException(
            status_code=422,
            detail=f"Routing variable {DEFAULT_WEIGHT_VARIABLE!r} is not available in the edge table.",
        )

    weight = float(variable.get("w", 0.0))
    if weight <= 0.0:
        return None

    hour = int(variable.get("hour", fallback_hour))
    gamma_state_factor = float(variable.get("gamma_state_factor", 0.05) or 0.0)
    hot_categories = variable.get("hot_categories") or [7, 8, 9]

    return {
        "name": DEFAULT_WEIGHT_VARIABLE,
        "column": DEFAULT_WEIGHT_VARIABLE,
        "w": weight,
        "hour": hour,
        "gamma": _utci_gamma(gamma_state_factor),
        "gamma_state_factor": gamma_state_factor,
        "hot_categories": [int(category) for category in hot_categories],
    }


def _request_weight_config(body: RouteRequest) -> dict[str, Any]:
    """Convert the request body into the routing.py weight_config dictionary."""
    hour = body.time.hour

    if body.weight_config is None:
        return _default_weight_config(hour)

    config = _model_to_dict(body.weight_config)
    normalised_variables: list[dict[str, Any]] = []

    for raw_variable in config.get("variables", []):
        variable = _normalise_weight_variable(raw_variable, fallback_hour=hour)
        if variable is not None:
            normalised_variables.append(variable)

    if not normalised_variables:
        return {"variables": []}

    return {
        "variables": normalised_variables,
        "hot_state_increment": int(config.get("hot_state_increment", 3)),
        "cold_state_recovery": int(config.get("cold_state_recovery", 1)),
        "hot_edge_routing": bool(config.get("hot_edge_routing", True)),
    }


def _prepare_weight_config(
    routing_network: NumpyRoutingNetwork,
    weight_config: dict[str, Any],
) -> dict[str, Any]:
    """Attach extracted edge arrays so add_weight can evaluate a route."""
    prepared = {
        **weight_config,
        "variables": [dict(variable) for variable in weight_config.get("variables", [])],
    }

    for variable in prepared["variables"]:
        variable["_values"] = routing_network.extract_edge_values(
            column=variable.get("column", variable["name"]),
            hour=variable.get("hour"),
        )

    return prepared


def _route_cost(
    routing_network: NumpyRoutingNetwork,
    edge_rows: list[int],
    weight_config: dict[str, Any],
) -> float:
    """Return the dynamic weighted route cost for the selected edge sequence."""
    prepared_config = _prepare_weight_config(routing_network, weight_config)
    route_state = 0
    total_cost = 0.0

    for edge_row in edge_rows:
        edge_cost, route_state = routing_network.add_weight(
            edge_idx=edge_row,
            weight_config=prepared_config,
            route_state=route_state,
        )
        total_cost += edge_cost

    return float(total_cost)


def _serialisable_weight_config(weight_config: dict[str, Any]) -> dict[str, Any]:
    """Remove internal arrays/callables before returning config to the client."""
    clean_config = {
        **weight_config,
        "variables": [],
    }

    for variable in weight_config.get("variables", []):
        clean_variable = {
            key: value
            for key, value in variable.items()
            if key != "_values" and not callable(value)
        }
        clean_config["variables"].append(_clean_value(clean_variable))

    return _clean_value(clean_config)


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


def _hourly_edge_value(edge_row: Any, column: str, hour: int) -> float | None:
    """Return a scalar or selected-hour edge value as a float."""
    if column not in edge_row.index:
        return None

    value = edge_row[column]
    if value is None:
        return None

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if arr.size == 0:
        return None

    if arr.size == 1:
        selected = arr[0]
    elif hour < arr.size:
        selected = arr[hour]
    else:
        return None

    if math.isnan(float(selected)) or math.isinf(float(selected)):
        return None
    return float(selected)


def _route_to_geojson(
    routing_network: NumpyRoutingNetwork,
    edge_rows: list[int],
    hour: int,
) -> tuple[dict[str, Any], float, float | None, float | None]:
    """Convert route edge row indices into route GeoJSON and summary statistics."""
    features: list[dict[str, Any]] = []
    total_distance_m = 0.0
    total_duration_s: float | None = 0.0
    weighted_utci_sum = 0.0
    utci_weight_sum = 0.0

    for edge_row_idx in edge_rows:
        edge_row = routing_network.edges.iloc[edge_row_idx]

        feature = _edge_feature(edge_row, edge_row_idx, _node_pos)
        if feature is not None:
            features.append(feature)

        length = None
        if "length" in routing_network.edges.columns:
            length = _clean_value(edge_row["length"])
            if length is not None:
                total_distance_m += float(length)

        utci_median = _hourly_edge_value(edge_row, UTCI_MEDIAN_VARIABLE, hour)
        if utci_median is not None:
            length_weight = float(length) if length is not None and float(length) > 0 else 1.0
            weighted_utci_sum += utci_median * length_weight
            utci_weight_sum += length_weight

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

    average_utci_median = (
        weighted_utci_sum / utci_weight_sum
        if utci_weight_sum > 0.0
        else None
    )

    return (
        {"type": "FeatureCollection", "features": features},
        total_distance_m,
        total_duration_s,
        average_utci_median,
    )


def _serialise_node_path(node_ids: Any) -> list[Any]:
    """Convert a NumPy array of node IDs into a JSON-safe list."""
    return [_clean_value(node_id) for node_id in np.asarray(node_ids).tolist()]


# ---------------------------------------------------------------------------
# Lifespan: load the parquet network once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NetworkSchema, build the routing network, and cache API payloads."""
    global _schema, _routing_network, _network_payload, _node_pos, _available_weight_variable_names

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

        node_pos = _build_node_positions(schema)
        payload = _build_network_payload(schema, routing_network, node_pos)
        available_names = {
            variable["name"]
            for variable in payload.get("available_weight_variables", [])
        }

        _schema = schema
        _routing_network = routing_network
        _node_pos = node_pos
        _network_payload = payload
        _available_weight_variable_names = available_names

        log.info(
            "Network ready: %d nodes / %d edges / %d GeoJSON features / weight variables: %s",
            routing_network.n_nodes,
            routing_network.n_edges,
            len(payload["geojson"]["features"]),
            sorted(available_names),
        )

    except Exception as exc:
        log.error("Failed to load network: %s", exc, exc_info=True)

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pedestrian Routing API",
    description="Serves a parquet-based pedestrian network and computes configurable weighted routes.",
    version="3.0.0",
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

    Array-valued edge properties such as utci_category and utci_median are
    returned as JSON lists. The response also includes the fixed routing
    variable used by the front-end.
    """
    if _network_payload is None:
        raise HTTPException(
            status_code=503,
            detail=f"Network is not available. Check NETWORK_FOLDER: {NETWORK_FOLDER}",
        )

    return _network_payload


@app.post("/route/path", summary="Compute a configurable weighted route between two coordinates")
async def post_route(body: RouteRequest) -> dict[str, Any]:
    """
    Snap origin and destination coordinates to the nearest network nodes and
    compute a shortest path using the supplied routing.py weight_config.

    Example request:

    ```json
    {
      "origin": {"lon": 4.47, "lat": 51.91},
      "destination": {"lon": 4.49, "lat": 51.93},
      "time": {"hour": 10},
      "weight_config": {
        "variables": [
          {"name": "utci_category", "column": "utci_category", "w": 1.0, "hour": 10}
        ]
      }
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

    weight_config = _request_weight_config(body)

    node_path = _routing_network.shortest_path(
        source_node_id=origin_node,
        target_node_id=destination_node,
        weight_config=weight_config,
    )

    if node_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between nodes {origin_node!r} and {destination_node!r}.",
        )

    edge_rows = _edge_rows_from_node_path(_routing_network, node_path)
    geojson, distance_m, duration_s, average_utci_median = _route_to_geojson(
        _routing_network,
        edge_rows,
        body.time.hour,
    )
    route_cost = _route_cost(_routing_network, edge_rows, weight_config)

    return {
        "geojson": geojson,
        "distance_m": round(distance_m, 2),
        "duration_s": round(duration_s, 1) if duration_s is not None else None,
        "cost": round(route_cost, 3),
        "average_utci_median": round(average_utci_median, 2) if average_utci_median is not None else None,
        "average_median_utci": round(average_utci_median, 2) if average_utci_median is not None else None,
        "weight_config": _serialisable_weight_config(weight_config),
        "weight_variables": [variable["name"] for variable in weight_config.get("variables", [])],
        "weight_hour": body.time.hour,
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
        "available_weight_variables": sorted(_available_weight_variable_names),
        "node_count": _routing_network.n_nodes if _routing_network else None,
        "edge_count": _routing_network.n_edges if _routing_network else None,
    }
