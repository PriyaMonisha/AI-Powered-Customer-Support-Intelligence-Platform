# filename: dash_app/pages/05_clustering_shap.py
# purpose:  Clustering & Explainability page — static galleries of the Section 8b
#           K-Means/PCA/t-SNE charts, Section 8c SHAP, and Phase C fairness breakdown.
# version:  1.1

import dash_bootstrap_components as dbc
from dash import html, register_page

from dash_app.utils.charts import chart_exists, chart_url

register_page(__name__, path="/clustering-shap", name="Clustering & Explainability", order=5)

CLUSTERING_CHARTS = [
    ("clustering_elbow.png", "Elbow Method (inertia vs. K)"),
    ("clustering_silhouette.png", "Silhouette & Davies-Bouldin Index by K — best K=2 (silhouette=0.1573, DBI=2.1902)"),
    ("clustering_pca_scatter.png", "PCA Projection of Clusters (K=2)"),
    ("clustering_tsne.png", "t-SNE Projection of Clusters (stratified subsample)"),
]

SHAP_CHARTS = [
    ("shap_priority_bar.png", "Priority Classifier — Mean |SHAP| Feature Importance (top: days_since_purchase)"),
    ("shap_priority_beeswarm.png", "Priority Classifier — SHAP Beeswarm"),
    ("shap_priority_waterfall_correct.png", "Priority Classifier — Waterfall (high-confidence correct prediction)"),
    ("shap_priority_waterfall_wrong.png", "Priority Classifier — Waterfall (mid-confidence wrong prediction)"),
    ("shap_regressor_bar.png", "Resolution Regressor — Mean |SHAP| Feature Importance (top: response_hour_of_day)"),
    ("shap_regressor_beeswarm.png", "Resolution Regressor — SHAP Beeswarm"),
]

ATTENTION_CHARTS = [
    ("distilbert_attention_correct.png", "DistilBERT [CLS] Attention — correctly-classified example (token-level, by layer)"),
    ("distilbert_attention_wrong.png", "DistilBERT [CLS] Attention — misclassified example (token-level, by layer)"),
]

FAIRNESS_CHARTS = [
    ("phase_c_fairness_gender.png",  "Priority F1 by Customer Gender (Female/Male/Other)"),
    ("phase_c_fairness_age.png",     "Priority F1 by Customer Age Band (18-30 / 31-45 / 46-60 / 61+)"),
    ("phase_c_fairness_channel.png", "Priority F1 by Ticket Channel (Chat / Email / Phone / Social media)"),
]

_FAIRNESS_CAVEAT = (
    "Overall XGB priority F1-macro ≈ 0.25 (test set, 4-class noise floor). "
    "Segment differences shown here reflect statistical noise, not model discrimination. "
    "This is a reproducible baseline for future comparison once a stronger model is available."
)


def _chart_card(filename: str, caption: str) -> dbc.Card:
    if chart_exists(filename):
        media = html.Img(src=chart_url(filename), className="card-img-top")
    else:
        media = dbc.Alert("Chart not yet generated", color="warning", className="m-2")
    return dbc.Card([media, dbc.CardBody(html.P(caption, className="card-text small"))], className="h-100")


def _gallery(charts: list[tuple[str, str]]) -> dbc.Row:
    return dbc.Row(
        [dbc.Col(_chart_card(filename, caption), md=6, className="mb-3") for filename, caption in charts],
        className="g-3",
    )


def layout():
    return dbc.Container(
        [
            html.H4("Clustering & Explainability", className="my-3"),
            html.H5("K-Means Clustering (Section 8b)", className="mt-3"),
            _gallery(CLUSTERING_CHARTS),
            html.H5("SHAP Explainability (Section 8c)", className="mt-4"),
            _gallery(SHAP_CHARTS),
            html.H5("DistilBERT Attention Visualization (Section 9b)", className="mt-4"),
            _gallery(ATTENTION_CHARTS),
            html.H5("Fairness / Segment Error Breakdown (Phase C)", className="mt-4"),
            dbc.Alert(_FAIRNESS_CAVEAT, color="info", className="mb-3"),
            _gallery(FAIRNESS_CHARTS),
        ],
        fluid=True,
        className="py-3",
    )
