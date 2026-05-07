"""Shiny app entry point for the UTCI network and route planner UI."""

from shiny import App, ui
from maplibre import output_maplibregl

from src.web.server_ui import server


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_radio_buttons(
            "app_mode",
            None,
            choices={"network": "🌡  UTCI Network", "route": "🗺  Route Planner"},
            selected="network",
            inline=True,
        ),
        ui.hr(),

        ui.panel_conditional(
            "input.app_mode === 'network'",
            ui.h4("UTCI Network"),
            ui.p("Pedestrian network fetched from the parquet-based API."),
            ui.h6("Hour"),
            ui.output_text("hour_label"),
            ui.input_slider("hour", None, min=0, max=23, value=0, step=1, ticks=True),
            ui.hr(),
            ui.h6("Selected-hour UTCI categories"),
            ui.output_ui("temp_stats"),
            ui.hr(),
            ui.h6("Colour scale range"),
            ui.output_ui("scale_range"),
            ui.h6("Normalisation"),
            ui.input_radio_buttons(
                "norm_mode",
                None,
                choices={"per_hour": "Per hour", "global": "Global (all hours)"},
                selected="per_hour",
            ),
            ui.hr(),
            ui.h6("Network"),
            ui.output_ui("network_stats"),
        ),

        ui.panel_conditional(
            "input.app_mode === 'route'",
            ui.h4("Route Planner"),
            ui.p(
                "Click two points on the map to set your origin and destination, "
                "then press the button below to request the route."
            ),
            ui.hr(),
            ui.h6("Origin"),
            ui.output_ui("origin_display"),
            ui.h6("Destination"),
            ui.output_ui("destination_display"),
            ui.hr(),
            ui.h6("Time"),
            ui.output_text("time_label"),
            ui.input_slider("time", None, min=0, max=23, value=0, step=1, ticks=True),
            ui.input_action_button(
                "find_route",
                "Find Route",
                class_="btn-primary w-100",
            ),
            ui.input_action_button(
                "clear_points",
                "Clear Points",
                class_="btn-outline-secondary w-100 mt-2",
            ),
            ui.hr(),
            ui.h6("Route result"),
            ui.output_ui("route_result"),
        ),
        width=300,
    ),
    output_maplibregl("map", height="100%"),
    title="UTCI Network Visualiser",
    fillable=True,
)


# ---------------------------------------------------------------------------
# App object
# ---------------------------------------------------------------------------

app = App(app_ui, server)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
