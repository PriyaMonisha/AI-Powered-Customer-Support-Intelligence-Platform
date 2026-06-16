# filename: tests/test_dash_app.py
# purpose:  dash_app — page registry/layouts, data_sources/charts loaders, and
#           api_client/mlflow_client graceful-degradation paths.
# version:  1.0

import sys
from unittest.mock import MagicMock

import dash
import pandas as pd
import pytest
import requests
from mlflow.exceptions import MlflowException

import dash_app.app as dash_app_module
from dash_app.utils import api_client, charts, data_sources, mlflow_client

PAGE_KEYS = [
    "pages.00_overview",
    "pages.01_eda_gallery",
    "pages.02_leaderboard",
    "pages.03_live_predictions",
    "pages.04_drift",
    "pages.05_clustering_shap",
]


# --- page registry + layouts --------------------------------------------------------

def test_page_registry_has_six_pages_with_unique_paths_and_orders():
    pages = list(dash.page_registry.values())
    assert len(pages) == 6
    assert sorted(p["order"] for p in pages) == list(range(6))
    assert len({p["path"] for p in pages}) == 6


@pytest.mark.parametrize("page_key", PAGE_KEYS)
def test_page_layout_renders_without_error(page_key):
    page = dash.page_registry[page_key]
    mod = sys.modules[page["module"]]
    assert mod.layout() is not None


def test_charts_route_serves_existing_png_and_blocks_traversal(tmp_path, monkeypatch):
    (tmp_path / "chart.png").write_bytes(b"fake-png-bytes")
    monkeypatch.setattr(dash_app_module, "CHARTS_DIR", tmp_path)

    client = dash_app_module.server.test_client()

    resp = client.get("/charts/chart.png")
    assert resp.status_code == 200

    resp = client.get("/charts/nonexistent.png")
    assert resp.status_code == 404

    resp = client.get("/charts/../config.py")
    assert resp.status_code == 404


# --- data_sources --------------------------------------------------------------------

def test_load_model_registry_has_expected_keys():
    registry = data_sources.load_model_registry()
    assert "ticket_type" in registry
    assert "ticket_priority" in registry


def test_load_section10_consolidation_has_expected_keys():
    consolidation = data_sources.load_section10_consolidation()
    assert "best_models" in consolidation
    assert "notes" in consolidation


def test_load_confidence_thresholds_has_expected_keys():
    thresholds = data_sources.load_confidence_thresholds()
    assert "ticket_type" in thresholds
    assert "ticket_priority" in thresholds


def test_load_drift_metrics_has_scenarios():
    metrics = data_sources.load_drift_metrics()
    assert "scenarios" in metrics


def test_load_model_report_has_leaderboard():
    report = data_sources.load_model_report()
    assert "classifiers" in report["leaderboard"]
    assert "regressors" in report["leaderboard"]


def test_get_dashboard_notes_returns_two_lists():
    artifact_notes, dashboard_corrections = data_sources.get_dashboard_notes()
    assert isinstance(artifact_notes, list)
    assert isinstance(dashboard_corrections, list)
    assert len(dashboard_corrections) >= 1


def test_load_json_missing_file_returns_empty_dict(tmp_path):
    assert data_sources._load_json(tmp_path / "does_not_exist.json") == {}


def test_load_json_malformed_file_returns_empty_dict(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    assert data_sources._load_json(bad_file) == {}


# --- charts ----------------------------------------------------------------------------

def test_multiclass_colors_returns_n_colors():
    colors = charts.multiclass_colors(5)
    assert len(colors) == 5
    assert all(isinstance(c, str) and c for c in colors)


def test_binary_colors_returns_two_hex_colors():
    colors = charts.binary_colors()
    assert len(colors) == 2
    assert all(c.startswith("#") for c in colors)


def test_histogram_color_returns_one_hex_color():
    assert charts.histogram_color().startswith("#")


def test_chart_exists_returns_bool_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "CHARTS_DIR", tmp_path)
    (tmp_path / "existing.png").touch()
    assert charts.chart_exists("existing.png") is True
    assert charts.chart_exists("does_not_exist.png") is False


# --- api_client --------------------------------------------------------------------------

def test_predict_type_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"predicted_label": "Billing inquiry"}
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr(api_client._session, "post", lambda *a, **k: mock_resp)

    data, error = api_client.predict_type({"ticket_subject": "x"})
    assert error is None
    assert data["predicted_label"] == "Billing inquiry"


def test_predict_type_connection_error_returns_none_and_error(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(api_client._session, "post", raise_conn_error)

    data, error = api_client.predict_type({"ticket_subject": "x"})
    assert data is None
    assert "refused" in error


def test_query_prometheus_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "success", "data": {"result": []}}
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr(api_client._session, "get", lambda *a, **k: mock_resp)

    data, error = api_client.query_prometheus("up")
    assert error is None
    assert data == {"result": []}


def test_query_prometheus_connection_error_returns_none_and_error(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(api_client._session, "get", raise_conn_error)

    data, error = api_client.query_prometheus("up")
    assert data is None
    assert error is not None


# --- mlflow_client ------------------------------------------------------------------------

def test_get_runs_df_returns_empty_df_on_connection_error(monkeypatch):
    def raise_conn_error(name):
        raise ConnectionError("refused")

    monkeypatch.setattr(mlflow_client.mlflow, "get_experiment_by_name", raise_conn_error)

    assert mlflow_client.get_runs_df("csip-baseline-classifiers").empty


def test_get_runs_df_returns_empty_df_on_mlflow_exception(monkeypatch):
    def raise_mlflow_error(name):
        raise MlflowException("boom")

    monkeypatch.setattr(mlflow_client.mlflow, "get_experiment_by_name", raise_mlflow_error)

    assert mlflow_client.get_runs_df("csip-baseline-classifiers").empty


def test_get_runs_df_success_strips_prefixes_and_fills_algo(monkeypatch):
    class FakeExperiment:
        experiment_id = "123"

    fake_df = pd.DataFrame({
        "run_id": ["abc"],
        "metrics.val_f1_macro": [0.25],
        "params.algo": ["lgbm"],
    })
    monkeypatch.setattr(mlflow_client.mlflow, "get_experiment_by_name", lambda name: FakeExperiment())
    monkeypatch.setattr(mlflow_client.mlflow, "search_runs", lambda experiment_ids: fake_df)

    df = mlflow_client.get_runs_df("csip-advanced-classifiers")
    assert df["val_f1_macro"].iloc[0] == 0.25
    assert df["algo"].iloc[0] == "lgbm"
    assert df["experiment_name"].iloc[0] == "csip-advanced-classifiers"


def test_get_leaderboard_falls_back_to_static_when_unreachable(monkeypatch):
    monkeypatch.setattr(mlflow_client, "_mlflow_reachable", lambda: False)

    clf_df, reg_df, source = mlflow_client.get_leaderboard()
    assert source == "static"
    assert not clf_df.empty
    assert not reg_df.empty
