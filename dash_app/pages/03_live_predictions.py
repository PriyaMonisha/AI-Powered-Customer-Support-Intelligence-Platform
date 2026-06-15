# filename: dash_app/pages/03_live_predictions.py
# purpose:  Live Predictions page — submits a ticket to the 4 FastAPI prediction/
#           explanation endpoints in parallel and renders each result independently.
# version:  1.0

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update, register_page

from dash_app.utils import api_client
from dash_app.utils.charts import binary_colors, multiclass_colors
from dash_app.utils.data_sources import load_confidence_thresholds

log = logging.getLogger(__name__)

register_page(__name__, path="/predict", name="Live Predictions", order=3)

_HARDCODED_CHANNELS = ["Chat", "Email", "Phone", "Social media"]
_HARDCODED_GENDERS = ["Female", "Male", "Other"]
_HARDCODED_PRODUCTS = [
    "Adobe Photoshop", "Amazon Echo", "Amazon Kindle", "Apple AirPods", "Asus ROG",
    "Autodesk AutoCAD", "Bose QuietComfort", "Bose SoundLink Speaker", "Canon DSLR Camera",
    "Canon EOS", "Dell XPS", "Dyson Vacuum Cleaner", "Fitbit Charge",
    "Fitbit Versa Smartwatch", "Garmin Forerunner", "GoPro Action Camera", "GoPro Hero",
    "Google Nest", "Google Pixel", "HP Pavilion", "LG OLED", "LG Smart TV",
    "LG Washing Machine", "Lenovo ThinkPad", "MacBook Pro", "Microsoft Office",
    "Microsoft Surface", "Microsoft Xbox Controller", "Nest Thermostat", "Nikon D",
    "Nintendo Switch", "Nintendo Switch Pro Controller", "Philips Hue Lights",
    "PlayStation", "Roomba Robot Vacuum", "Samsung Galaxy", "Samsung Soundbar",
    "Sony 4K HDR TV", "Sony PlayStation", "Sony Xperia", "Xbox", "iPhone",
]


def _load_dropdown_options() -> dict[str, list[str]]:
    try:
        from config import TABULAR_ENCODER_PATH
        from src.features.tabular_features import ORDINAL_FEATURES, TabularEncoder

        enc = TabularEncoder.load(TABULAR_ENCODER_PATH)
        channel_idx = ORDINAL_FEATURES.index("Ticket Channel")
        gender_idx = ORDINAL_FEATURES.index("Customer Gender")
        return {
            "channels": list(enc.ordinal_enc.categories_[channel_idx]),
            "genders": list(enc.ordinal_enc.categories_[gender_idx]),
            "products": list(enc.target_enc.categories_[0]),
        }
    except Exception as e:
        log.warning("TabularEncoder dropdown sourcing failed: %s", e)

    try:
        import pandas as pd

        from config import PREPROCESSED_DATA_PATH

        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        return {
            "channels": sorted(df["Ticket Channel"].dropna().unique().tolist()),
            "genders": sorted(df["Customer Gender"].dropna().unique().tolist()),
            "products": sorted(df["Product Purchased"].dropna().unique().tolist()),
        }
    except Exception as e:
        log.warning("CSV dropdown sourcing failed: %s", e)

    return {"channels": _HARDCODED_CHANNELS, "genders": _HARDCODED_GENDERS, "products": _HARDCODED_PRODUCTS}


_DROPDOWN_OPTIONS = _load_dropdown_options()

_RESPONSE_HOUR_OPTIONS = [{"label": "Not yet responded (-1)", "value": -1}] + [
    {"label": f"{h:02d}:00", "value": h} for h in range(24)
]


def layout():
    return dbc.Container(
        [
            html.H4("Live Predictions", className="my-3"),
            dbc.Alert(id="lp-validation-alert", color="danger", is_open=False, dismissable=True),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Ticket Subject"),
                            dcc.Input(id="lp-subject", type="text", className="form-control mb-2",
                                      placeholder="e.g. Issue with my order"),
                            dbc.Label("Ticket Description"),
                            dcc.Textarea(id="lp-description", className="form-control mb-2", rows=4,
                                         placeholder="Describe the issue..."),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Customer Age"),
                            dcc.Input(id="lp-age", type="number", value=30.0, min=0, className="form-control mb-2"),
                            dbc.Label("Customer Gender"),
                            dcc.Dropdown(
                                id="lp-gender",
                                options=_DROPDOWN_OPTIONS["genders"],
                                value="Other" if "Other" in _DROPDOWN_OPTIONS["genders"] else _DROPDOWN_OPTIONS["genders"][0],
                                clearable=False,
                                className="mb-2",
                            ),
                            dbc.Label("Product Purchased"),
                            dcc.Dropdown(
                                id="lp-product",
                                options=["Unknown"] + _DROPDOWN_OPTIONS["products"],
                                value="Unknown",
                                clearable=False,
                                className="mb-2",
                            ),
                            dbc.Label("Ticket Channel"),
                            dcc.Dropdown(
                                id="lp-channel",
                                options=_DROPDOWN_OPTIONS["channels"],
                                value="Email" if "Email" in _DROPDOWN_OPTIONS["channels"] else _DROPDOWN_OPTIONS["channels"][0],
                                clearable=False,
                                className="mb-2",
                            ),
                            dbc.Label("Days Since Purchase (optional)"),
                            dcc.Input(id="lp-days-since-purchase", type="number", min=0,
                                      className="form-control mb-2", placeholder="leave blank if unknown"),
                            dbc.Label("First Response Hour"),
                            dcc.Dropdown(
                                id="lp-response-hour",
                                options=_RESPONSE_HOUR_OPTIONS,
                                value=-1,
                                clearable=False,
                                className="mb-2",
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
            dbc.Button("Predict", id="lp-submit-btn", color="primary", className="my-3"),
            dbc.Row(
                [
                    dbc.Col([html.H5("Ticket Type"), html.Div(id="lp-type-output")], md=6, className="mb-4"),
                    dbc.Col([html.H5("Ticket Priority"), html.Div(id="lp-priority-output")], md=6, className="mb-4"),
                    dbc.Col([html.H5("Resolution Time"), html.Div(id="lp-resolution-output")], md=6, className="mb-4"),
                    dbc.Col([html.H5("Priority Explanation (SHAP)"), html.Div(id="lp-explain-output")], md=6, className="mb-4"),
                ]
            ),
        ],
        fluid=True,
        className="py-3",
    )


def _probability_chart(probabilities: dict, title: str):
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    fig = px.bar(
        x=labels, y=values, color=labels,
        color_discrete_sequence=multiclass_colors(len(labels)),
        title=title,
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Probability")
    return dcc.Graph(figure=fig)


def _render_type(data: dict | None, error: str | None):
    if error:
        return dbc.Alert(f"FastAPI unreachable: {error}", color="danger")
    if data is None:
        return dbc.Alert("Request timed out.", color="danger")

    badges = [
        dbc.Badge("Auto-route" if data["auto_route"] else "No auto-route",
                  color="success" if data["auto_route"] else "secondary", className="me-2"),
        dbc.Badge("Flag for review" if data["flag_for_review"] else "Not flagged",
                  color="warning" if data["flag_for_review"] else "secondary"),
    ]
    note = load_confidence_thresholds().get("ticket_type", {}).get("note")

    children = []
    reliability_note = data.get("reliability_note")
    if data.get("model_status") == "below_quality_bar" and reliability_note:
        children.append(dbc.Alert(reliability_note, color="warning", className="mb-2"))

    children += [
        html.P(f"Predicted: {data['predicted_label']} (confidence {data['confidence']:.1%})"),
        html.Div(badges, className="mb-2"),
        _probability_chart(data["probabilities"], "Ticket Type Probabilities"),
    ]
    if note:
        children.append(dbc.Alert(note, color="light", className="small mt-2"))
    return html.Div(children)


def _render_priority(data: dict | None, error: str | None):
    if error:
        return dbc.Alert(f"FastAPI unreachable: {error}", color="danger")
    if data is None:
        return dbc.Alert("Request timed out.", color="danger")

    note = load_confidence_thresholds().get("ticket_priority", {}).get("note")
    children = [
        html.P(f"Predicted: {data['predicted_label']} (confidence {data['confidence']:.1%})"),
        _probability_chart(data["probabilities"], "Ticket Priority Probabilities"),
    ]
    if note:
        children.append(dbc.Alert(note, color="light", className="small mt-2"))
    return html.Div(children)


def _render_resolution(data: dict | None, error: str | None):
    if error:
        return dbc.Alert(f"FastAPI unreachable: {error}", color="danger")
    if data is None:
        return dbc.Alert("Request timed out.", color="danger")

    children = [html.P(f"Predicted resolution time: {data['predicted_hours']:.2f} hours")]
    if data.get("warning"):
        children.append(dbc.Alert(data["warning"], color="warning"))
    return html.Div(children)


def _render_explain(data: dict | None, error: str | None):
    if error:
        return dbc.Alert(f"FastAPI unreachable: {error}", color="danger")
    if data is None:
        return dbc.Alert("Request timed out.", color="danger")

    features = data["top_features"]
    if not features:
        return dbc.Alert("No SHAP feature exceeded the |value| > 0.001 cutoff.", color="info")

    names = [f["feature"] for f in features]
    values = [f["shap_value"] for f in features]
    neg_color, pos_color = binary_colors()
    bar_colors = [pos_color if v >= 0 else neg_color for v in values]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=bar_colors))
    fig.update_layout(
        title=f"Top SHAP Features (predicted: {data['predicted_label']})",
        xaxis_title="SHAP value",
        yaxis=dict(autorange="reversed"),
    )
    return dcc.Graph(figure=fig)


@callback(
    Output("lp-validation-alert", "children"),
    Output("lp-validation-alert", "is_open"),
    Output("lp-type-output", "children"),
    Output("lp-priority-output", "children"),
    Output("lp-resolution-output", "children"),
    Output("lp-explain-output", "children"),
    Input("lp-submit-btn", "n_clicks"),
    State("lp-subject", "value"),
    State("lp-description", "value"),
    State("lp-age", "value"),
    State("lp-gender", "value"),
    State("lp-product", "value"),
    State("lp-channel", "value"),
    State("lp-days-since-purchase", "value"),
    State("lp-response-hour", "value"),
    prevent_initial_call=True,
)
def on_submit(_n_clicks, subject, description, age, gender, product, channel, days_since_purchase, response_hour):
    errors = []
    if not subject:
        errors.append("Ticket Subject is required.")
    if not description:
        errors.append("Ticket Description is required.")
    if age is None or age < 0:
        errors.append("Customer Age must be >= 0.")
    if days_since_purchase is not None and days_since_purchase < 0:
        errors.append("Days Since Purchase must be >= 0 if provided.")

    if errors:
        return " ".join(errors), True, no_update, no_update, no_update, no_update

    ticket = {
        "ticket_subject": subject,
        "ticket_description": description,
        "customer_age": float(age),
        "customer_gender": gender,
        "product_purchased": product,
        "ticket_channel": channel,
        "days_since_purchase": float(days_since_purchase) if days_since_purchase is not None else None,
        "response_hour_of_day": float(response_hour),
    }

    calls = {
        "type": api_client.predict_type,
        "priority": api_client.predict_priority,
        "resolution": api_client.predict_resolution,
        "explain": api_client.explain_priority,
    }
    results: dict[str, tuple[dict | None, str | None]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn, ticket): key for key, fn in calls.items()}
        try:
            for future in as_completed(futures, timeout=10):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = (None, str(e))
        except TimeoutError:
            pass

    for key in calls:
        results.setdefault(key, (None, None))

    return (
        "",
        False,
        _render_type(*results["type"]),
        _render_priority(*results["priority"]),
        _render_resolution(*results["resolution"]),
        _render_explain(*results["explain"]),
    )
