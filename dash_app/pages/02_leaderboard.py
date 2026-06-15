# filename: dash_app/pages/02_leaderboard.py
# purpose:  Model Leaderboard page — MLflow-backed classifier/regressor leaderboards,
#           refreshed on an interval, with a static-fallback badge.
# version:  1.0

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Input, Output, callback, dash_table, dcc, html, register_page

from dash_app.utils.charts import multiclass_colors
from dash_app.utils.mlflow_client import get_leaderboard

register_page(__name__, path="/leaderboard", name="Model Leaderboard", order=2)


def _build_tab_content(df, metric_col: str, chart_title: str):
    if df.empty:
        return dbc.Alert("No data available.", color="warning")

    df = df.reset_index(drop=True)
    algos = sorted(df["algo"].unique())
    color_map = dict(zip(algos, multiclass_colors(len(algos))))

    chart_df = df.head(10).copy()
    chart_df["label"] = chart_df["algo"] + " · " + chart_df["run_id"].astype(str).str[:8]
    fig = px.bar(
        chart_df,
        x="label",
        y=metric_col,
        color="algo",
        color_discrete_map=color_map,
        title=chart_title,
    )
    fig.update_layout(xaxis_title=None)

    display_df = df.copy()
    if "start_time" in display_df.columns:
        display_df["start_time"] = display_df["start_time"].astype(str)
    display_df = display_df.fillna("")

    table = dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        page_size=10,
        sort_action="native",
        style_table={"overflowX": "auto"},
    )

    return html.Div([dcc.Graph(figure=fig), table])


def layout():
    return dbc.Container(
        [
            html.H4("Model Leaderboard", className="my-3"),
            dbc.Badge("Loading...", id="leaderboard-source-badge", color="secondary", className="mb-3"),
            dbc.Tabs(
                [
                    dbc.Tab(html.Div(id="leaderboard-clf-content"), label="Classifiers", tab_id="clf"),
                    dbc.Tab(html.Div(id="leaderboard-reg-content"), label="Regressors", tab_id="reg"),
                ],
                active_tab="clf",
            ),
            dcc.Interval(id="leaderboard-interval", interval=60_000, n_intervals=0),
        ],
        fluid=True,
        className="py-3",
    )


@callback(
    Output("leaderboard-clf-content", "children"),
    Output("leaderboard-reg-content", "children"),
    Output("leaderboard-source-badge", "children"),
    Output("leaderboard-source-badge", "color"),
    Input("leaderboard-interval", "n_intervals"),
)
def update_leaderboard(_n_intervals):
    clf_df, reg_df, source = get_leaderboard()
    clf_df = clf_df.sort_values("val_f1_macro", ascending=False)
    reg_df = reg_df.sort_values("val_rmse", ascending=True)

    clf_content = _build_tab_content(clf_df, "val_f1_macro", "Validation F1-macro by Run")
    reg_content = _build_tab_content(reg_df, "val_rmse", "Validation RMSE by Run")

    if source == "live":
        return clf_content, reg_content, "Live (MLflow)", "success"
    return clf_content, reg_content, "Static fallback (model_report_*.json)", "warning"
