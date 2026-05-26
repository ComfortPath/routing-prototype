"""Shiny app entry point for the thermally conscious route planner."""

from shiny import App, ui
from maplibre import output_maplibregl

from src.web.server_ui import server


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.p(
            "Click two points on the map, choose the routing hour, and calculate "
            "a route!"
        ),
        ui.hr(),
        ui.h6("Route points"),
        ui.output_ui("locations_display"),
        ui.hr(),
        ui.h6("Time"),
        ui.output_text("time_label"),
        ui.input_slider("time", None, min=0, max=23, value=0, step=1, ticks=True),
        ui.output_ui("time_gradient"),
        ui.hr(),
        ui.h6("Route preferences"),
        ui.output_ui("weight_controls"),
        ui.layout_columns(
            ui.input_action_button(
                "find_route",
                "Find Route",
                class_="btn-primary w-100",
            ),
            ui.input_action_button(
                "clear_points",
                "Clear Points",
                class_="btn-outline-secondary w-100",
            ),
            col_widths=[6, 6],
            gap="0.5rem",
        ),  
        ui.hr(),
        ui.h6("Route result"),
        ui.output_ui("route_result"),
        ui.hr(),
        ui.h6("Network Stats"),
        ui.output_ui("stats"),
        width=330
        
    ),
    output_maplibregl("map", height="100%"),
    title="UTCI Route Planner",
    fillable=True,
)


# ---------------------------------------------------------------------------
# App object
# ---------------------------------------------------------------------------

app = App(app_ui, server)
