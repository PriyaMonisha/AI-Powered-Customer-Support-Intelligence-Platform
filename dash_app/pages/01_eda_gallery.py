# filename: dash_app/pages/01_eda_gallery.py
# purpose:  EDA Gallery page — static gallery of the 14 Section 2 EDA charts.
# version:  1.0

import dash_bootstrap_components as dbc
from dash import html, register_page

from dash_app.utils.charts import chart_exists, chart_url

register_page(__name__, path="/eda", name="EDA Gallery", order=1)

EDA_CHARTS = [
    ("eda_01_ticket_type_dist.png", "Ticket Type Distribution (5-class primary target) — nearly balanced, 1.07:1 ratio"),
    ("eda_02_ticket_priority_dist.png", "Ticket Priority Distribution (4 classes: Low / Medium / High / Critical)"),
    ("eda_03_ticket_status_dist.png", "Ticket Status Distribution"),
    ("eda_04_type_priority_heatmap.png", "Ticket Priority by Type (row-normalized — proportion within each type)"),
    ("eda_05_channel_dist.png", "Ticket Channel Distribution"),
    ("eda_06_age_histogram.png", "Customer Age Distribution"),
    ("eda_07_hours_to_resolve_dist.png", "Resolution Time Distribution (closed tickets)"),
    ("eda_08_csat_distribution.png", "CSAT Distribution (closed tickets) — mean 2.99, 67.3% null"),
    ("eda_09_days_since_purchase_dist.png", "Customer Tenure Distribution (Days Since Purchase)"),
    ("eda_10_hours_by_type.png", "Resolution Time by Ticket Type (closed tickets)"),
    ("eda_11_hours_by_priority.png", "Resolution Time by Priority (closed tickets) — Critical SLA breach ~80%"),
    ("eda_12_csat_by_priority.png", "CSAT Score by Priority (closed tickets)"),
    ("eda_13_response_hour_dist.png", "First Response Hour Distribution"),
    ("eda_14_subject_top20.png", "Ticket Subject Distribution (all unique subjects) — subject_word_count constant at 2.0"),
]


def _chart_card(filename: str, caption: str) -> dbc.Card:
    if chart_exists(filename):
        media = html.Img(src=chart_url(filename), className="card-img-top")
    else:
        media = dbc.Alert("Chart not yet generated", color="warning", className="m-2")
    return dbc.Card([media, dbc.CardBody(html.P(caption, className="card-text small"))], className="h-100")


def layout():
    cards = [dbc.Col(_chart_card(filename, caption), md=4, className="mb-3") for filename, caption in EDA_CHARTS]
    return dbc.Container(
        [
            html.H4("EDA Gallery", className="my-3"),
            dbc.Row(cards, className="g-3"),
        ],
        fluid=True,
        className="py-3",
    )
