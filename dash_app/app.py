# filename: dash_app/app.py
# purpose:  Dash multi-page app entrypoint — sidebar nav from page_registry,
#           page_container, and a path-traversal-safe /charts/<file> route.
# version:  1.0

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html
from flask import abort, send_from_directory

from config import CHARTS_DIR, FAST_MODE
from dash_app.utils.charts import ALLOWED_CHART_EXTENSIONS

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    pages_folder=str(Path(__file__).parent / "pages"),
)
server = app.server


@server.route("/charts/<path:filename>")
def serve_chart(filename: str):
    if Path(filename).suffix.lower() not in ALLOWED_CHART_EXTENSIONS:
        abort(404)
    return send_from_directory(CHARTS_DIR, filename)


def _sidebar() -> dbc.Nav:
    links = [
        dbc.NavLink(page["name"], href=page["path"], active="exact")
        for page in sorted(dash.page_registry.values(), key=lambda p: p["order"])
    ]
    return dbc.Nav(links, vertical=True, pills=True, className="bg-light p-3 h-100")


app.layout = dbc.Container(
    [
        html.H2("Customer Support Intelligence Platform", className="my-3"),
        dbc.Row(
            [
                dbc.Col(_sidebar(), width=2),
                dbc.Col(dash.page_container, width=10),
            ]
        ),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run(debug=FAST_MODE, host="0.0.0.0", port=8050)
