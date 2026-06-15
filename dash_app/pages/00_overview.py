# filename: dash_app/pages/00_overview.py
# purpose:  Overview page — project summary, KPI cards from the model registry,
#           MLflow consolidation summary, and artifact notes with dashboard corrections.
# version:  1.0

import dash_bootstrap_components as dbc
from dash import html, register_page

from dash_app.utils.data_sources import (
    get_dashboard_notes,
    load_model_registry,
    load_section10_consolidation,
)

register_page(__name__, path="/", name="Overview", order=0)


def _kpi_card(title: str, algo: str, metric_label: str, metric_value: float, extra: str | None = None) -> dbc.Card:
    body = [
        html.H6(title, className="card-subtitle text-muted"),
        html.H4(algo.upper(), className="card-title"),
        html.P(f"{metric_label}: {metric_value:.4f}", className="mb-0"),
    ]
    if extra:
        body.append(html.P(extra, className="mb-0 text-success small"))
    return dbc.Card(dbc.CardBody(body), className="h-100")


def layout():
    registry = load_model_registry()
    consolidation = load_section10_consolidation()
    artifact_notes, dashboard_corrections = get_dashboard_notes()

    best_models = consolidation.get("best_models", {})
    ticket_type = registry.get("ticket_type", {})
    ticket_priority = registry.get("ticket_priority", {})
    resolution = best_models.get("resolution_time", {})

    kpi_row = dbc.Row(
        [
            dbc.Col(
                _kpi_card(
                    "Ticket Type Classifier",
                    ticket_type.get("algo", "n/a"),
                    "Val F1-macro",
                    ticket_type.get("val_f1_macro", 0.0),
                    f"+{ticket_type['vs_baseline']:.4f} vs. baseline" if "vs_baseline" in ticket_type else None,
                ),
                md=4,
            ),
            dbc.Col(
                _kpi_card(
                    "Ticket Priority Classifier",
                    ticket_priority.get("algo", "n/a"),
                    "Val F1-macro",
                    ticket_priority.get("val_f1_macro", 0.0),
                    f"+{ticket_priority['vs_baseline']:.4f} vs. baseline" if "vs_baseline" in ticket_priority else None,
                ),
                md=4,
            ),
            dbc.Col(
                _kpi_card(
                    "Resolution Time Regressor",
                    resolution.get("algo", "n/a"),
                    "Val RMSE (hours)",
                    resolution.get("val_rmse", 0.0),
                ),
                md=4,
            ),
        ],
        className="mb-4 g-3",
    )

    mlflow_summary = dbc.Card(
        dbc.CardBody(
            [
                html.H5("MLflow Consolidation (Section 10)"),
                html.P(f"Total runs logged: {consolidation.get('total_mlflow_runs', 'n/a')}"),
                html.P(f"Registered models: {len(consolidation.get('registered_models', []))}"),
                html.Ul(
                    [
                        html.Li(f"{exp}: {n} runs")
                        for exp, n in consolidation.get("runs_per_experiment", {}).items()
                    ]
                ),
            ]
        ),
        className="mb-4",
    )

    notes_section = html.Div(
        [
            html.H5("Artifact Notes"),
            *[dbc.Alert(note, color="info", className="mb-2") for note in artifact_notes],
            *[
                dbc.Alert(f"Dashboard note: {note}", color="secondary", className="mb-2")
                for note in dashboard_corrections
            ],
        ]
    )

    return dbc.Container(
        [
            html.P(
                "End-to-end ML system automating ticket classification, priority "
                "prediction, and resolution-time estimation for SaaS/e-commerce "
                "customer support operations.",
                className="lead",
            ),
            kpi_row,
            mlflow_summary,
            notes_section,
        ],
        fluid=True,
        className="py-3",
    )
