# filename: dash_app/pages/04_drift.py
# purpose:  Drift Monitoring page — live Prometheus PSI gauges when available, a
#           "no check yet" state, or a static Section 12 report fallback.
# version:  1.0

import os
from datetime import datetime

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Input, Output, callback, dcc, html, register_page

from config import DRIFT_PSI_THRESHOLD
from dash_app.utils import api_client
from dash_app.utils.charts import multiclass_colors
from dash_app.utils.data_sources import load_drift_metrics

register_page(__name__, path="/drift", name="Drift Monitoring", order=4)

DASH_ADMIN_ENABLED = os.getenv("DASH_ADMIN_ENABLED", "").lower() == "true"


def layout():
    return dbc.Container(
        [
            html.H4("Drift Monitoring", className="my-3"),
            html.Div(id="drift-content"),
            html.Div(
                [
                    html.Hr(),
                    dbc.Button("Run drift check now", id="drift-check-btn", color="secondary"),
                    html.Div(id="drift-check-result", className="mt-2"),
                    html.P(
                        "Normally automated by the csip_drift_monitor Airflow DAG "
                        "(03:00 UTC daily).",
                        className="text-muted small mt-2",
                    ),
                ],
                hidden=not DASH_ADMIN_ENABLED,
            ),
            dcc.Interval(id="drift-interval", interval=30_000, n_intervals=0),
        ],
        fluid=True,
        className="py-3",
    )


def _metric_card(title: str, value: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([html.H6(title, className="card-subtitle text-muted"), html.H4(value, className="card-title")]),
        className="h-100 text-center",
    )


def _psi_chart(features: list[str], scores: list[float], threshold: float, title: str):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    features_sorted = [features[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    fig = px.bar(
        x=features_sorted, y=scores_sorted, color=features_sorted,
        color_discrete_sequence=multiclass_colors(len(features_sorted)),
        title=title,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold})")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="PSI")
    return dcc.Graph(figure=fig)


def _render_live(last_check_ts: float):
    psi_data, psi_error = api_client.query_prometheus("csip_feature_drift_psi")
    detected_data, _ = api_client.query_prometheus("csip_drift_detected")

    if psi_error or not psi_data.get("result"):
        return dbc.Alert("Drift gauges unavailable from Prometheus.", color="warning")

    features, scores = [], []
    for series in psi_data["result"]:
        features.append(series["metric"].get("feature", "unknown"))
        scores.append(float(series["value"][1]))

    drift_detected = False
    if detected_data and detected_data.get("result"):
        drift_detected = float(detected_data["result"][0]["value"][1]) > 0

    max_psi = max(scores) if scores else 0.0
    n_drifted = sum(1 for s in scores if s > DRIFT_PSI_THRESHOLD)
    timestamp_str = datetime.fromtimestamp(last_check_ts).strftime("%Y-%m-%d %H:%M:%S")

    summary = dbc.Row(
        [
            dbc.Col(_metric_card("Max PSI", f"{max_psi:.4f}"), md=4),
            dbc.Col(_metric_card("Features Drifted", f"{n_drifted} / {len(features)}"), md=4),
            dbc.Col(_metric_card("Drift Detected", "Yes" if drift_detected else "No"), md=4),
        ],
        className="mb-3 g-3",
    )

    return html.Div(
        [
            dbc.Badge("Live (Prometheus)", color="success", className="mb-2 me-2"),
            dbc.Badge(f"Last checked: {timestamp_str}", color="light", text_color="dark", className="mb-2"),
            summary,
            _psi_chart(features, scores, DRIFT_PSI_THRESHOLD, "Feature Drift (PSI) vs. Training Baseline"),
        ]
    )


def _render_scenario(scenario: dict, threshold: float, label: str):
    feature_scores = scenario.get("feature_scores", {})
    features = list(feature_scores.keys())
    scores = list(feature_scores.values())

    summary = dbc.Row(
        [
            dbc.Col(_metric_card("Max PSI", f"{scenario['max_psi']:.4f}"), md=4),
            dbc.Col(_metric_card("Features Drifted", f"{scenario['n_drifted']} / {len(features)}"), md=4),
            dbc.Col(_metric_card("Drift Detected", "Yes" if scenario["drift_detected"] else "No"), md=4),
        ],
        className="my-3 g-3",
    )

    return html.Div([summary, _psi_chart(features, scores, threshold, f"{label} — Feature PSI vs. Training Baseline")])


def _render_static_fallback():
    metrics = load_drift_metrics()
    scenarios = metrics.get("scenarios", {})
    if not scenarios:
        return dbc.Alert("Prometheus unreachable and no static drift report found.", color="warning")

    threshold = metrics.get("drift_psi_threshold", DRIFT_PSI_THRESHOLD)
    tabs = []
    for key, label in [
        ("train_vs_test", "Train vs. Test (no drift)"),
        ("train_vs_shifted_test", "Train vs. Shifted Test (+2σ Customer Age)"),
    ]:
        scenario = scenarios.get(key, {}).get("custom_psi", {})
        if scenario:
            tabs.append(dbc.Tab(_render_scenario(scenario, threshold, label), label=label, className="pt-3"))

    return html.Div(
        [
            dbc.Badge("Historical (Section 12 notebook run)", color="warning", className="mb-2"),
            dbc.Tabs(tabs),
        ]
    )


@callback(
    Output("drift-content", "children"),
    Input("drift-interval", "n_intervals"),
)
def update_drift(_n_intervals):
    data, error = api_client.query_prometheus("csip_drift_last_check_timestamp")
    if error or not data or not data.get("result"):
        return _render_static_fallback()

    last_check_ts = float(data["result"][0]["value"][1])
    if last_check_ts < 0:
        return dbc.Alert(
            "No drift check has run yet — the csip_drift_monitor Airflow DAG runs "
            "daily at 03:00 UTC, or trigger one manually below.",
            color="info",
        )

    return _render_live(last_check_ts)


@callback(
    Output("drift-check-result", "children"),
    Input("drift-check-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_drift_check(_n_clicks):
    if not DASH_ADMIN_ENABLED:
        return dbc.Alert("Admin actions are disabled on this deployment.", color="secondary")

    data, error = api_client.trigger_drift_check()
    if error:
        return dbc.Alert(f"Drift check failed: {error}", color="danger")

    color = "warning" if data["drift_detected"] else "success"
    return dbc.Alert(
        f"Drift check complete — max_psi={data['max_psi']:.4f}, "
        f"drift_detected={data['drift_detected']}, n_drifted={data['n_drifted']}.",
        color=color,
    )
