# filename: dash_app/utils/mlflow_client.py
# purpose:  MLflow-backed model leaderboard, with a static-artifact fallback when
#           MLflow is unreachable or has no comparable runs.
# version:  1.0

import logging
import os

import mlflow
import pandas as pd
import requests
from mlflow.exceptions import MlflowException

from config import MLFLOW_TRACKING_URI
from dash_app.utils.data_sources import load_model_report

log = logging.getLogger(__name__)

# Bound MLflow's default retry/backoff (5 retries, 120s timeout per request) so that
# get_leaderboard() fails fast into the static fallback when MLflow is unreachable —
# otherwise a single dcc.Interval tick could block for minutes across 5 experiments.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiments restricted to those with comparable metrics (CE-4) — csip-clustering
# (silhouette, not F1), csip-explainability (no training metrics) and
# csip-model-registry are excluded, otherwise they'd inject NaN val_f1_macro/val_rmse
# rows and break the sort.
CLASSIFIER_EXPERIMENTS = [
    "csip-baseline-classifiers",
    "csip-advanced-classifiers",
    "csip-distilbert-text",
    "csip-bilstm-text",
]
REGRESSOR_EXPERIMENTS = ["csip-regression-models"]

CLF_COLS = ["experiment_name", "run_id", "algo", "task", "start_time",
            "val_f1_macro", "test_f1_macro", "val_f1_weighted"]
REG_COLS = ["experiment_name", "run_id", "algo", "task", "start_time",
            "val_rmse", "val_mae", "val_r2", "val_mape"]


def get_runs_df(experiment_name: str) -> pd.DataFrame:
    """search_runs() for one experiment, with metrics./params. prefixes stripped.
    Returns an empty DataFrame if the experiment doesn't exist or MLflow is down."""
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    except (MlflowException, ConnectionError) as e:
        log.warning("MLflow query for %s failed: %s", experiment_name, e)
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.rename(columns=lambda c: c.removeprefix("metrics.").removeprefix("params."))
    df["experiment_name"] = experiment_name
    # csip-baseline-classifiers logs params.algorithm, not params.algo
    if "algo" not in df.columns:
        df["algo"] = "unknown"
    else:
        df["algo"] = df["algo"].fillna("unknown")
    return df


def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def _static_leaderboard() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    leaderboard = load_model_report().get("leaderboard", {})
    clf_df = _select(pd.DataFrame(leaderboard.get("classifiers", [])), CLF_COLS)
    reg_df = _select(pd.DataFrame(leaderboard.get("regressors", [])), REG_COLS)
    return clf_df.dropna(subset=["val_f1_macro"]), reg_df.dropna(subset=["val_rmse"]), "static"


def _mlflow_reachable() -> bool:
    """Cheap precheck so a fully-down MLflow doesn't pay 5x retry/backoff
    (MLFLOW_HTTP_REQUEST_MAX_RETRIES) before falling back to the static report."""
    try:
        requests.get(MLFLOW_TRACKING_URI, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def get_leaderboard() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (clf_df, reg_df, source) where source is "live" (MLflow) or
    "static" (artifacts/reports/model_report_*.json fallback)."""
    if not _mlflow_reachable():
        log.warning("MLflow tracking server unreachable at %s", MLFLOW_TRACKING_URI)
        return _static_leaderboard()

    clf_frames = [f for f in (get_runs_df(name) for name in CLASSIFIER_EXPERIMENTS) if not f.empty]
    reg_frames = [f for f in (get_runs_df(name) for name in REGRESSOR_EXPERIMENTS) if not f.empty]

    clf_df = pd.concat(clf_frames, ignore_index=True) if clf_frames else pd.DataFrame()
    reg_df = pd.concat(reg_frames, ignore_index=True) if reg_frames else pd.DataFrame()

    if clf_df.empty and reg_df.empty:
        return _static_leaderboard()

    return _select(clf_df, CLF_COLS).dropna(subset=["val_f1_macro"]), \
        _select(reg_df, REG_COLS).dropna(subset=["val_rmse"]), "live"
