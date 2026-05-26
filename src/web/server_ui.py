"""Server-side logic for the Shiny UTCI route planner."""

from __future__ import annotations

from copy import deepcopy
import math
import statistics
from typing import Any
import asyncio

from shiny import reactive, render, ui
from maplibre import Layer, LayerType, Map, MapContext, MapOptions
from maplibre import render_maplibregl
from maplibre.basemaps import Carto
from maplibre.controls import NavigationControl, ScaleControl
from maplibre.sources import GeoJSONSource

from src.web.api_client import fetch_network, fetch_route


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOURS = list(range(24))
UTCI_CATEGORY_VARIABLE = "utci_category"
UTCI_MEDIAN_VARIABLE = "utci_median"
HOT_EDGE_GAMMA_STATE_FACTOR = 0.02
NETWORK_LAYER_ID = "network-edges"
ROUTE_LAYER_ID = "route-path"
MARKER_LAYER_ID = "route-markers"
FALLBACK_CENTER = (4.48, 51.92)  # Rotterdam-ish default
EMPTY_GEOJSON: dict[str, Any] = {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# Routing preference helpers
# ---------------------------------------------------------------------------

def _variable_label(variable_name: str) -> str:
    """Return a readable label for the fixed routing variable."""
    if variable_name == UTCI_CATEGORY_VARIABLE:
        return "UTCI category"
    return variable_name.replace("_", " ").title()


def weight_controls_ui() -> Any:
    """Build the fixed UTCI-category route-preference controls."""
    return ui.div(
        ui.p(
            "Routing uses UTCI category only. Set importance to 0 for the "
            "ordinary shortest path.",
            style="font-size: 0.9em; color: #aaa;",
        ),
        ui.input_slider(
            "weight_utci_category",
            "UTCI category importance",
            min=0.0,
            max=1.0,
            value=1.0,
            step=0.1,
        ),
        ui.input_checkbox(
            "use_hot_edge_routing",
            "Avoid consecutive hot edges",
            value=True,
        ),
        ui.p(
            "When disabled, the route still uses the UTCI category penalty, "
            "but it does not add extra penalty after consecutive hot edges.",
            style="font-size: 0.8em; color: #888; margin-top: -0.25rem;",
        ),
    )


def build_weight_config(input, selected_hour: int) -> dict[str, Any]:
    """Build the routing.py weight_config dictionary from sidebar inputs."""
    importance = float(input.weight_utci_category() or 0.0)
    if importance <= 0.0:
        return {"variables": []}

    use_hot_edge_routing = bool(input.use_hot_edge_routing())

    return {
        "variables": [
            {
                "name": UTCI_CATEGORY_VARIABLE,
                "column": UTCI_CATEGORY_VARIABLE,
                "w": importance,
                "hour": int(selected_hour),
                "gamma_state_factor": HOT_EDGE_GAMMA_STATE_FACTOR if use_hot_edge_routing else 0.0,
                "hot_categories": [7, 8, 9],
            }
        ],
        "hot_state_increment": 3 if use_hot_edge_routing else 0,
        "cold_state_recovery": 1 if use_hot_edge_routing else 0,
        "hot_edge_routing": use_hot_edge_routing,
    }


# ---------------------------------------------------------------------------
# Hourly array-value helpers
# ---------------------------------------------------------------------------

def _clean_float(value: Any) -> float | None:
    """Return a finite float value, or None when conversion is impossible."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    return value


def _hourly_value(props: dict[str, Any], variable: str, hour: int) -> float | None:
    """Return props[variable][hour] as a finite float when it is available."""
    values = props.get(variable)

    if not isinstance(values, list) or hour >= len(values):
        return None

    return _clean_float(values[hour])


def _hour_values(network_data: dict[str, Any], hour: int, variable: str) -> list[float]:
    """Collect valid selected-hour values from all network edge features."""
    values: list[float] = []

    for feature in network_data.get("geojson", {}).get("features", []):
        props = feature.get("properties", {})
        value = _hourly_value(props, variable, hour)
        if value is not None:
            values.append(value)

    return values


def build_hour_stats(network_data: dict[str, Any], variable: str) -> dict[int, dict[str, float]]:
    """Build min/median/max statistics for each hour of an array-valued variable."""
    stats: dict[int, dict[str, float]] = {}

    for hour in HOURS:
        values = _hour_values(network_data, hour, variable)
        if not values:
            continue

        stats[hour] = {
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }

    return stats


def scale_bounds(hour: int, hour_stats: dict[int, dict[str, float]]) -> tuple[float, float, float] | None:
    """Return per-hour colour scale bounds for the selected routing hour."""
    if not hour_stats or hour not in hour_stats:
        return None

    s = hour_stats[hour]
    lo, mid, hi = s["min"], s["median"], s["max"]

    if abs(hi - lo) < 0.01:
        lo -= 0.5
        hi += 0.5

    return lo, mid, hi


def geojson_for_hour(network_data: dict[str, Any], hour: int, variable: str) -> dict[str, Any]:
    """Copy API GeoJSON and expose props[variable][hour] as a scalar temp property."""
    geojson = deepcopy(network_data.get("geojson", EMPTY_GEOJSON))

    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        props["temp"] = _hourly_value(props, variable, hour)

    return geojson


# ---------------------------------------------------------------------------
# Map and display helpers
# ---------------------------------------------------------------------------

def color_expression(t_min: float, t_mid: float, t_max: float) -> list:
    """Return an Inferno MapLibre expression for the selected-hour UTCI value."""
    span = t_max - t_min

    return [
        "case",
        ["==", ["get", "temp"], None],
        "#aaaaaa",
        [
            "interpolate",
            ["linear"],
            ["get", "temp"],
            t_min,
            "#000004",
            t_min + span * 0.1,
            "#160b39",
            t_min + span * 0.2,
            "#420a68",
            t_min + span * 0.3,
            "#6a176e",
            t_min + span * 0.4,
            "#932667",
            t_min + span * 0.5,
            "#bc3754",
            t_min + span * 0.6,
            "#dd513a",
            t_min + span * 0.7,
            "#f37819",
            t_min + span * 0.8,
            "#fca50a",
            t_min + span * 0.9,
            "#f6d746",
            t_max,
            "#fcffa4",
        ],
    ]


def gradient_legend(bounds: tuple[float, float, float] | None) -> Any:
    """Build the compact colour-gradient legend shown below the time slider."""
    if bounds is None:
        return ui.p(
            "Hourly UTCI median data unavailable from API.",
            style="font-size: 0.8em; color: #888; margin-top: 0.25rem;",
        )

    lo, mid, hi = bounds
    return ui.div(
        ui.tags.div(
            style=(
                "height: 10px; border-radius: 999px; margin-top: 0.35rem; "
                "background: linear-gradient(to right, "
                    "#000004, #160b39, #420a68, #6a176e, #932667, "
                    "#bc3754, #dd513a, #f37819, #fca50a, #f6d746, #fcffa4);"
            )
        ),
        ui.tags.div(
            ui.span(f"{lo:.1f} °C"),
            ui.span(f"{mid:.1f} °C"),
            ui.span(f"{hi:.1f} °C"),
            style=(
                "display: flex; justify-content: space-between; font-size: 0.75em; "
                "color: #aaa; margin-top: 0.2rem;"
            ),
        ),
    )


def build_map(center: tuple[float, float], geojson: dict[str, Any], line_color: Any) -> Map:
    """Create the base MapLibre map with network, route, and marker layers."""
    m = Map(MapOptions(center=center, zoom=13, style=Carto.POSITRON))
    m.add_control(NavigationControl(), position="bottom-right")
    m.add_control(ScaleControl(), position="bottom-left")

    m.add_layer(
        Layer(
            id=NETWORK_LAYER_ID,
            type=LayerType.LINE,
            source=GeoJSONSource(data=geojson),
            paint={
                "line-color": line_color,
                "line-width": 2,
                "line-opacity": 0.9,
            },
        )
    )

    m.add_layer(
        Layer(
            id=ROUTE_LAYER_ID,
            type=LayerType.LINE,
            source=GeoJSONSource(data=EMPTY_GEOJSON),
            paint={
                "line-color": "#ff0000",
                "line-width": 4,
                "line-opacity": 0.95,
            },
        )
    )

    m.add_layer(
        Layer(
            id=MARKER_LAYER_ID,
            type=LayerType.CIRCLE,
            source=GeoJSONSource(data=EMPTY_GEOJSON),
            paint={
                "circle-radius": 8,
                "circle-color": [
                    "match",
                    ["get", "marker"],
                    "origin", "#44dd88",
                    "destination", "#ff5555",
                    "#ffffff",
                ],
                "circle-stroke-width": 2,
                "circle-stroke-color": "#ffffff",
            },
        )
    )

    return m


def _marker_geojson(
    origin: tuple[float, float] | None,
    destination: tuple[float, float] | None,
) -> dict[str, Any]:
    """Build a FeatureCollection for the selected origin/destination markers."""
    features: list[dict[str, Any]] = []

    if origin:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(origin)},
            "properties": {"marker": "origin"},
        })

    if destination:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(destination)},
            "properties": {"marker": "destination"},
        })

    return {"type": "FeatureCollection", "features": features}


def _route_preference_summary(weight_config: dict[str, Any]) -> str:
    """Return a compact text summary of selected route preference controls."""
    variables = weight_config.get("variables", [])
    if not variables:
        return "Shortest path: no environmental variables selected."

    parts: list[str] = []
    for variable in variables:
        name = variable.get("name", "unknown")
        weight = variable.get("w", 0)
        hour = variable.get("hour")
        label = _variable_label(name)
        if hour is None:
            parts.append(f"{label} × {float(weight):.1f}")
        else:
            parts.append(f"{label}[{int(hour):02d}:00] × {float(weight):.1f}")

    hot_edge_routing = bool(weight_config.get("hot_edge_routing", False))
    parts.append("hot-edge routing on" if hot_edge_routing else "hot-edge routing off")

    return "Routing preferences: " + ", ".join(parts)

def compact_stat(label: str, value: str) -> Any:
    """Return a compact sidebar statistic row."""
    return ui.div(
        ui.span(label, style="font-size: 0.85em; color: #666;"),
        ui.span(value, style="font-size: 0.95em; font-weight: 600;"),
        style=(
            "display: flex; "
            "justify-content: space-between; "
            "align-items: center; "
            "padding: 0.25rem 0; "
            "border-bottom: 1px solid #eee;"
        ),
    )

def compact_status(label: str, value: str, dot_color: str | None = None) -> Any:
    """Return a compact sidebar status row."""
    label_content = [ui.span(label, style="font-size: 0.85em; color: #666;")]

    if dot_color is not None:
        label_content.insert(
            0,
            ui.span(
                "●",
                style=f"color: {dot_color}; margin-right: 6px; font-size: 0.8em;",
            ),
        )

    return ui.div(
        ui.span(*label_content),
        ui.span(
            value,
            style=(
                "font-size: 0.9em; "
                "font-weight: 600; "
                "text-align: right; "
                "max-width: 60%; "
                "overflow-wrap: anywhere;"
            ),
        ),
        style=(
            "display: flex; "
            "justify-content: space-between; "
            "align-items: center; "
            "padding: 0.25rem 0; "
            "border-bottom: 1px solid #eee;"
        ),
    )

# ---------------------------------------------------------------------------
# Shiny server
# ---------------------------------------------------------------------------

def server(input, output, session):
    """Register Shiny reactives, map rendering, and API route requests."""
    network_data: reactive.value[dict[str, Any] | None] = reactive.value(None)
    load_error: reactive.value[str | None] = reactive.value(None)

    route_origin: reactive.value[tuple[float, float] | None] = reactive.value(None)
    route_destination: reactive.value[tuple[float, float] | None] = reactive.value(None)
    route_result_data: reactive.value[dict[str, Any] | None] = reactive.value(None)
    route_error: reactive.value[str | None] = reactive.value(None)
    route_loading: reactive.value[bool] = reactive.value(False)
    route_request_time: reactive.value[int | None] = reactive.value(None)

    @reactive.Calc
    def hour_stats_data() -> dict[int, dict[str, float]]:
        """Compute hourly stats for the UTCI median display variable."""
        data = network_data()
        if data is None:
            return {}
        return build_hour_stats(data, UTCI_MEDIAN_VARIABLE)

    @reactive.Calc
    def selected_hour() -> int:
        """Return the selected routing hour from the only visible hour slider."""
        return int(input.time())

    @reactive.Effect
    async def _load_network():
        """Load the network from the API, retrying while the backend starts."""
        if network_data() is not None:
            return

        last_error = None

        for _ in range(20):
            try:
                data = await fetch_network()
                network_data.set(data)
                load_error.set(None)
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)

        load_error.set(f"Failed to load network after retries: {last_error}")

    @render_maplibregl
    def map() -> Map:
        """Render the initial MapLibre map and selected-hour network layer."""
        data = network_data()
        stats = hour_stats_data()

        center = FALLBACK_CENTER
        geojson = EMPTY_GEOJSON
        line_color: Any = "#4da3ff"

        if data is not None:
            center = tuple(data.get("center", FALLBACK_CENTER))

            if stats:
                hour = selected_hour()
                bounds = scale_bounds(hour, stats)
                geojson = geojson_for_hour(data, hour, UTCI_MEDIAN_VARIABLE)
                line_color = color_expression(*bounds) if bounds else "#4da3ff"
            else:
                geojson = data.get("geojson", EMPTY_GEOJSON)

        return build_map(center, geojson, line_color)

    @reactive.Effect
    async def _update_map() -> None:
        """Update network colours whenever the route-planning hour changes."""
        data = network_data()
        if data is None:
            return

        stats = hour_stats_data()
        if stats:
            hour = selected_hour()
            geojson = geojson_for_hour(data, hour, UTCI_MEDIAN_VARIABLE)
            bounds = scale_bounds(hour, stats)
            line_color = color_expression(*bounds) if bounds else "#4da3ff"
        else:
            geojson = data.get("geojson", EMPTY_GEOJSON)
            line_color = "#4da3ff"

        async with MapContext("map") as m:
            m.set_data(NETWORK_LAYER_ID, geojson)
            m.set_paint_property(NETWORK_LAYER_ID, "line-color", line_color)

    @reactive.Effect
    @reactive.event(input.map_clicked)
    async def _handle_map_click() -> None:
        """Use the first two map clicks as route origin and destination."""
        click = input.map_clicked()
        if click is None:
            return

        coords = click.get("coords", {})
        lon = coords.get("lng", coords.get("lon"))
        lat = coords.get("lat")

        if lon is None or lat is None:
            return

        coord = (float(lon), float(lat))

        if route_origin() is None:
            route_origin.set(coord)
        elif route_destination() is None:
            route_destination.set(coord)
        else:
            return

        async with MapContext("map") as m:
            m.set_data(MARKER_LAYER_ID, _marker_geojson(route_origin(), route_destination()))

    @reactive.Effect
    @reactive.event(input.clear_points)
    async def _clear_points() -> None:
        """Clear selected points, route output, and map route geometry."""
        route_origin.set(None)
        route_destination.set(None)
        route_result_data.set(None)
        route_error.set(None)
        route_request_time.set(None)

        async with MapContext("map") as m:
            m.set_data(MARKER_LAYER_ID, EMPTY_GEOJSON)
            m.set_data(ROUTE_LAYER_ID, EMPTY_GEOJSON)

    @reactive.Effect
    @reactive.event(input.find_route)
    async def _find_route() -> None:
        """Request a route for the selected origin, destination, hour, and weights."""
        origin = route_origin()
        dest = route_destination()

        if origin is None or dest is None:
            route_error.set("Please click two points on the map first.")
            return

        hour = selected_hour()
        weight_config = build_weight_config(input, hour)
        route_request_time.set(hour)
        route_error.set(None)
        route_loading.set(True)

        try:
            result = await fetch_route(
                origin,
                dest,
                hour,
                weight_config=weight_config,
            )
            result["requested_hour"] = hour
            result["requested_weight_config"] = weight_config
            route_result_data.set(result)

            async with MapContext("map") as m:
                m.set_data(ROUTE_LAYER_ID, result.get("geojson", EMPTY_GEOJSON))

        except Exception as exc:
            route_error.set(str(exc))
            route_result_data.set(None)
        finally:
            route_loading.set(False)

    @render.text
    def time_label() -> str:
        """Show the selected route-planning hour as HH:00."""
        return f"{selected_hour():02d}:00"

    @render.ui
    def time_gradient() -> Any:
        """Show the selected-hour UTCI colour gradient below the time slider."""
        if load_error() is not None:
            return ui.p("Colour scale unavailable because the network failed to load.")

        return gradient_legend(scale_bounds(selected_hour(), hour_stats_data()))

    @render.ui
    def stats():
        """Render compact network and selected-hour UTCI statistics."""
        if load_error() is not None:
            return ui.p(f"Failed to load network: {load_error()}")

        data = network_data()
        if data is None:
            return ui.p("Loading stats...")

        stat_rows = [
            compact_stat("Nodes", f"{data['node_count']:,}"),
            compact_stat("Edges", f"{data['edge_count']:,}"),
        ]

        hour = int(input.time())
        hour_stats = hour_stats_data()

        if hour_stats and hour in hour_stats:
            s = hour_stats[hour]
            stat_rows.extend(
                [
                    compact_stat("UTCI min", f"{s['min']:.1f} °C"),
                    compact_stat("UTCI median", f"{s['median']:.1f} °C"),
                    compact_stat("UTCI max", f"{s['max']:.1f} °C"),
                ]
            )
        else:
            stat_rows.append(
                ui.p(
                    "Selected-hour UTCI data unavailable.",
                    style="font-size: 0.85em; color: #888;",
                )
            )

        return ui.div(*stat_rows)

    @render.ui
    def weight_controls() -> Any:
        """Show route-preference controls once the network has loaded."""
        if load_error() is not None:
            return ui.p("Route preferences unavailable because the network failed to load.")

        if network_data() is None:
            return ui.p("Loading route preferences...")

        return weight_controls_ui()

    @render.ui
    def locations_display():
        """Render selected origin and destination as one compact sidebar element."""
        origin = route_origin()
        destination = route_destination()

        if origin is None:
            origin_value = "Click map"
        else:
            origin_value = f"{origin[1]:.5f}, {origin[0]:.5f}"

        if origin is None:
            destination_value = "Set origin first"
        elif destination is None:
            destination_value = "Click map"
        else:
            destination_value = f"{destination[1]:.5f}, {destination[0]:.5f}"

        return ui.div(
            compact_status("Origin", origin_value, "#44dd88"),
            compact_status("Destination", destination_value, "#ff5555"),
        )

    @render.ui
    def route_result():
        """Render route results as compact sidebar rows."""
        if route_loading():
            return compact_status("Route", "Fetching…")

        err = route_error()
        if err:
            return ui.div(
                compact_status("Route", "Error"),
                ui.p(
                    err,
                    style="font-size: 0.85em; color: #ff6666; margin-top: 0.4rem;",
                ),
            )

        result = route_result_data()
        if result is None:
            return compact_status("Route", "No route yet")

        distance_m = result.get("distance_m")
        avg_utci_median = result.get("average_utci_median")
        if avg_utci_median is None:
            avg_utci_median = result.get("average_median_utci")

        returned_config = result.get("weight_config") or result.get("requested_weight_config") or {}

        rows = []

        if distance_m is not None:
            rows.append(compact_status("Distance", f"{distance_m / 1000:.2f} km"))

        if avg_utci_median is not None:
            rows.append(compact_status("Avg. UTCI", f"{avg_utci_median:.1f} °C"))

        rows.append(
            ui.p(
                _route_preference_summary(returned_config),
                style=(
                    "font-size: 0.8em; "
                    "color: #777; "
                    "line-height: 1.25; "
                    "margin-top: 0.5rem;"
                ),
            )
        )

        return ui.div(*rows)