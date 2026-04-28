from __future__ import annotations

from copy import deepcopy
import math
import statistics

from shiny import reactive, render, ui
from maplibre import Layer, LayerType, Map, MapContext, MapOptions
from maplibre import render_maplibregl
from maplibre.basemaps import Carto
from maplibre.controls import NavigationControl, ScaleControl
from maplibre.sources import GeoJSONSource

from src.web.api_client import fetch_network, fetch_route


HOURS = list(range(24))
NETWORK_LAYER_ID = "network-edges"
ROUTE_LAYER_ID = "route-path"
MARKER_LAYER_ID = "route-markers"
FALLBACK_CENTER = (4.48, 51.92)  # Rotterdam-ish default
EMPTY_GEOJSON: dict = {"type": "FeatureCollection", "features": []}


# ── Temperature helpers ────────────────────────────────────────────────────────

def _clean_temp(value) -> float | None:
    """Return a usable float temperature or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(value) or math.isclose(value, 0.0, abs_tol=0.01):
        return None

    return value


def _hour_values(network_data: dict, hour: int) -> list[float]:
    """Collect valid temperature values for one hour from API geojson."""
    attr = f"temp_mean_{hour}"
    values: list[float] = []

    for feature in network_data.get("geojson", {}).get("features", []):
        props = feature.get("properties", {})
        temp = _clean_temp(props.get(attr))
        if temp is not None:
            values.append(temp)

    return values


def build_hour_stats(network_data: dict) -> dict[int, dict[str, float]]:
    """
    Build per-hour stats if temp_mean_* properties are present in the API payload.
    Returns an empty dict when the API only provides plain network geometry.
    """
    stats: dict[int, dict[str, float]] = {}

    for hour in HOURS:
        values = _hour_values(network_data, hour)
        if not values:
            continue

        stats[hour] = {
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }

    return stats


def global_bounds(hour_stats: dict[int, dict[str, float]]) -> tuple[float, float, float] | None:
    """Compute global min/median/max across all available hours."""
    if not hour_stats:
        return None

    mins = [s["min"] for s in hour_stats.values()]
    maxs = [s["max"] for s in hour_stats.values()]
    meds = [s["median"] for s in hour_stats.values()]

    return min(mins), statistics.median(meds), max(maxs)


def scale_bounds(
    hour: int,
    norm_mode: str,
    hour_stats: dict[int, dict[str, float]],
) -> tuple[float, float, float] | None:
    """Return colour scale bounds for the current hour/mode."""
    if not hour_stats or hour not in hour_stats:
        return None

    if norm_mode == "global":
        return global_bounds(hour_stats)

    s = hour_stats[hour]
    lo, mid, hi = s["min"], s["median"], s["max"]

    if abs(hi - lo) < 0.01:
        lo -= 0.5
        hi += 0.5

    return lo, mid, hi


def geojson_for_hour(network_data: dict, hour: int) -> dict:
    """
    Copy API geojson and map temp_mean_<hour> -> temp so the existing
    MapLibre colour expression can use a single 'temp' property.
    """
    geojson = deepcopy(network_data.get("geojson", EMPTY_GEOJSON))
    attr = f"temp_mean_{hour}"

    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        props["temp"] = _clean_temp(props.get(attr))

    return geojson


def color_expression(t_min: float, t_mid: float, t_max: float) -> list:
    """Blue -> white -> red temperature colour scale."""
    return [
        "case",
        ["==", ["get", "temp"], None],
        "#aaaaaa",
        [
            "interpolate",
            ["linear"],
            ["get", "temp"],
            t_min,
            "#2166ac",
            t_mid,
            "#f7f7f7",
            t_max,
            "#d6191b",
        ],
    ]


def build_map(center: tuple[float, float], geojson: dict, line_color) -> Map:
    """Create the base MapLibre map with network + empty route/marker layers."""
    m = Map(MapOptions(center=center, zoom=13, style=Carto.DARK_MATTER))
    m.add_control(NavigationControl(), position="bottom-right")
    m.add_control(ScaleControl(), position="bottom-left")

    # Network layer
    m.add_layer(
        Layer(
            id=NETWORK_LAYER_ID,
            type=LayerType.LINE,
            source=GeoJSONSource(data=geojson),
            paint={
                "line-color": line_color,
                "line-width": 1.5,
                "line-opacity": 0.9,
            },
        )
    )

    # Route overlay layer (starts empty)
    m.add_layer(
        Layer(
            id=ROUTE_LAYER_ID,
            type=LayerType.LINE,
            source=GeoJSONSource(data=EMPTY_GEOJSON),
            paint={
                "line-color": "#f0c040",
                "line-width": 4,
                "line-opacity": 0.95,
            },
        )
    )

    # Marker layer for origin / destination points
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
) -> dict:
    """Build a FeatureCollection for the two click markers."""
    features = []
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


# ── Server ─────────────────────────────────────────────────────────────────────

def server(input, output, session):
    network_data = reactive.value(None)
    hour_stats_data = reactive.value({})
    load_error = reactive.value(None)

    # Route planner state
    # Each is (lon, lat) or None
    route_origin: reactive.value[tuple[float, float] | None] = reactive.value(None)
    route_destination: reactive.value[tuple[float, float] | None] = reactive.value(None)
    route_result_data: reactive.value[dict | None] = reactive.value(None)
    route_error: reactive.value[str | None] = reactive.value(None)
    route_loading: reactive.value[bool] = reactive.value(False)
    route_request_time: reactive.value[int | None] = reactive.value(None)

    # ── Load network ──────────────────────────────────────────────────────────

    @reactive.Effect
    async def _load_network():
        if network_data() is not None or load_error() is not None:
            return

        try:
            data = await fetch_network()
        except Exception as exc:
            load_error.set(str(exc))
            return

        network_data.set(data)
        hour_stats_data.set(build_hour_stats(data))

    # ── Initial map render ────────────────────────────────────────────────────

    @render_maplibregl
    def map():
        data = network_data()
        stats = hour_stats_data()

        center = FALLBACK_CENTER
        geojson = EMPTY_GEOJSON
        line_color = "#4da3ff"

        if data is not None:
            center = tuple(data.get("center", FALLBACK_CENTER))

            if stats:
                bounds = scale_bounds(0, "per_hour", stats)
                geojson = geojson_for_hour(data, 0)
                line_color = color_expression(*bounds) if bounds else "#4da3ff"
            else:
                geojson = data.get("geojson", EMPTY_GEOJSON)

        return build_map(center, geojson, line_color)

    # ── Temperature network updates ───────────────────────────────────────────

    @reactive.Effect
    async def _update_map():
        data = network_data()
        if data is None:
            return

        stats = hour_stats_data()

        if stats:
            geojson = geojson_for_hour(data, input.hour())
            bounds = scale_bounds(input.hour(), input.norm_mode(), stats)
            line_color = color_expression(*bounds) if bounds else "#4da3ff"
        else:
            geojson = data.get("geojson", EMPTY_GEOJSON)
            line_color = "#4da3ff"

        async with MapContext("map") as m:
            m.set_data(NETWORK_LAYER_ID, geojson)
            m.set_paint_property(NETWORK_LAYER_ID, "line-color", line_color)

    # ── Map click handler — collect route points ──────────────────────────────

    @reactive.Effect
    @reactive.event(input.map_clicked)
    async def _handle_map_click():
        if input.app_mode() != "route":
            return

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
            m.set_data(
                MARKER_LAYER_ID,
                _marker_geojson(route_origin(), route_destination()),
            )

    # ── Clear points button ───────────────────────────────────────────────────

    @reactive.Effect
    @reactive.event(input.clear_points)
    async def _clear_points():
        route_origin.set(None)
        route_destination.set(None)
        route_result_data.set(None)
        route_error.set(None)
        route_request_time.set(None)

        async with MapContext("map") as m:
            m.set_data(MARKER_LAYER_ID, EMPTY_GEOJSON)
            m.set_data(ROUTE_LAYER_ID, EMPTY_GEOJSON)

    # ── Find route button ─────────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.find_route)
    async def _find_route():
        origin = route_origin()
        dest = route_destination()

        if origin is None or dest is None:
            route_error.set("Please click two points on the map first.")
            return

        selected_hour = int(input.time())
        route_request_time.set(selected_hour)

        route_error.set(None)
        route_loading.set(True)

        try:
            result = await fetch_route(origin, dest, selected_hour)
            result["requested_hour"] = selected_hour
            route_result_data.set(result)

            route_geojson = result.get("geojson", EMPTY_GEOJSON)
            async with MapContext("map") as m:
                m.set_data(ROUTE_LAYER_ID, route_geojson)

        except Exception as exc:
            route_error.set(str(exc))
            route_result_data.set(None)
        finally:
            route_loading.set(False)

    # ── Sidebar outputs ───────────────────────────────────────────────────────

    @render.text
    def hour_label():
        return f"{input.hour():02d}:00"

    @render.ui
    def network_stats():
        if load_error() is not None:
            return ui.p(f"Failed to load network: {load_error()}")

        data = network_data()
        if data is None:
            return ui.p("Loading network...")

        return ui.layout_columns(
            ui.value_box("Nodes", f"{data['node_count']:,}"),
            ui.value_box("Edges", f"{data['edge_count']:,}"),
            col_widths=[6, 6],
        )

    @render.ui
    def temp_stats():
        stats = hour_stats_data()

        if load_error() is not None:
            return ui.p("Temperature data unavailable because the network failed to load.")

        if not stats or input.hour() not in stats:
            return ui.p("No temperature fields exposed by the API.")

        s = stats[input.hour()]
        return ui.layout_columns(
            ui.value_box("Min", f"{s['min']:.1f} C"),
            ui.value_box("Median", f"{s['median']:.1f} C"),
            ui.value_box("Max", f"{s['max']:.1f} C"),
            col_widths=[4, 4, 4],
        )

    @render.ui
    def scale_range():
        stats = hour_stats_data()

        if load_error() is not None:
            return ui.p("Unavailable.")

        bounds = scale_bounds(input.hour(), input.norm_mode(), stats)
        if bounds is None:
            return ui.p("Temperature data unavailable from API.")

        lo, _, hi = bounds
        return ui.p(f"{lo:.1f} C (blue)  ->  {hi:.1f} C (red)")

    # ── Route planner sidebar outputs ─────────────────────────────────────────

    @render.ui
    def origin_display():
        o = route_origin()
        if o is None:
            return ui.p(ui.em("Click the map to set origin"), style="color: #888;")
        return ui.p(
            ui.span("●", style="color: #44dd88; margin-right: 6px;"),
            f"{o[1]:.5f} N,  {o[0]:.5f} E",
        )

    @render.ui
    def destination_display():
        o = route_origin()
        d = route_destination()
        if o is None:
            return ui.p(ui.em("Set origin first"), style="color: #888;")
        if d is None:
            return ui.p(ui.em("Click the map to set destination"), style="color: #888;")
        return ui.p(
            ui.span("●", style="color: #ff5555; margin-right: 6px;"),
            f"{d[1]:.5f} N,  {d[0]:.5f} E",
        )

    @render.ui
    def route_result():
        if route_loading():
            return ui.p("Fetching route…")

        err = route_error()
        if err:
            return ui.div(
                ui.p(err, style="color: #ff6666;"),
            )

        result = route_result_data()
        if result is None:
            return ui.p(ui.em("No route yet."), style="color: #888;")

        distance_m = result.get("distance_m")
        duration_s = result.get("duration_s")

        parts = []
        if distance_m is not None:
            km = distance_m / 1000
            parts.append(ui.value_box("Distance", f"{km:.2f} km"))
        if duration_s is not None:
            mins = int(duration_s // 60)
            secs = int(duration_s % 60)
            parts.append(ui.value_box("Duration", f"{mins}m {secs:02d}s"))

        if not parts:
            return ui.p("Route received — no distance/duration in response.")

        return ui.layout_columns(*parts, col_widths=[6, 6] if len(parts) == 2 else [12])
