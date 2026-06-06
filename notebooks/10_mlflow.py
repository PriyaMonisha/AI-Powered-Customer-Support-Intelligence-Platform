# filename: notebooks/10_mlflow.py
# purpose:  MLflow consolidation + Model Registry — reads S5-S9 runs, registers best artifacts
# version:  1.0

# %% [1] Constants
FAST_MODE = True

# %% [2] PROJECT_ROOT detection (Colab-safe boilerplate)
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    for _cand in [
        Path("/content/drive/MyDrive/csip"),
        Path("/content/csip"),
        Path.cwd(),
        Path.cwd().parent,
    ]:
        if (_cand / "config.py").exists():
            PROJECT_ROOT = _cand
            break
    else:
        raise FileNotFoundError("config.py not found — verify path.")

# %% [3] Imports
import json
import logging
import sys
import tempfile
import shutil
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    MODELS_DIR, ARTIFACTS_DIR, CHARTS_DIR, REPORTS_DIR,
    RF_TYPE_PATH, RF_PRIORITY_PATH,
    XGB_TYPE_PATH, XGB_PRIORITY_PATH,
    LGBM_TYPE_PATH, LGBM_PRIORITY_PATH,
    RF_REGRESSOR_PATH, LGBM_REGRESSOR_PATH, XGBOOST_REGRESSOR_PATH,
    RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# %% [4] MLflow URI + client + dirs
# Path.as_uri() produces file:///C:/... with %20-encoded spaces — correct on Windows
MLFLOW_URI = (PROJECT_ROOT / "mlruns").as_uri()
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)
logger.info("MLflow tracking URI: %s", MLFLOW_URI)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [5] Helper functions

def get_exp_id(exp_name: str) -> Optional[str]:
    """Return experiment_id or None if experiment does not exist."""
    exp = client.get_experiment_by_name(exp_name)
    return exp.experiment_id if exp else None


def load_runs(exp_name: str) -> pd.DataFrame:
    """Load all runs from one experiment as a flat DataFrame.
    Returns empty DataFrame if experiment not found or has 0 runs.
    All metric/param values remain as-is (params = strings; metrics = floats).
    """
    exp_id = get_exp_id(exp_name)
    if exp_id is None:
        logger.warning("Experiment '%s' not found — skipping.", exp_name)
        return pd.DataFrame()
    runs = client.search_runs(experiment_ids=[exp_id])
    if not runs:
        logger.info("Experiment '%s': 0 runs (pending Colab?)", exp_name)
        return pd.DataFrame()
    rows = []
    for r in runs:
        row: dict = {"run_id": r.info.run_id, "experiment": exp_name}
        row.update({f"p_{k}": v for k, v in r.data.params.items()})
        row.update({f"m_{k}": v for k, v in r.data.metrics.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("Loaded %d runs from '%s'.", len(df), exp_name)
    return df


def _algo_col(df: pd.DataFrame) -> pd.Series:
    """Unify 'p_algo' (advanced) and 'p_algorithm' (baseline) into one series."""
    if "p_algo" in df.columns and "p_algorithm" in df.columns:
        return df["p_algo"].fillna(df["p_algorithm"])
    if "p_algo" in df.columns:
        return df["p_algo"]
    if "p_algorithm" in df.columns:
        return df["p_algorithm"]
    return pd.Series(dtype=str, index=df.index)


def best_per_algo(df: pd.DataFrame, task: str, metric: str, maximize: bool = True) -> pd.DataFrame:
    """Return best metric value per algo for a given task.
    Handles unified 'p_algo'/'p_algorithm' column naming.
    """
    if df.empty or metric not in df.columns or "p_task" not in df.columns:
        return pd.DataFrame(columns=["algo", metric])
    sub = df[df["p_task"] == task].copy()
    if sub.empty:
        return pd.DataFrame(columns=["algo", metric])
    sub["_algo"] = _algo_col(sub)
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    agg_fn = "max" if maximize else "min"
    result = (
        sub.groupby("_algo")[metric]
        .agg(agg_fn)
        .reset_index()
        .rename(columns={"_algo": "algo"})
        .sort_values(metric, ascending=not maximize)
    )
    return result


def set_alias_safe(model_name: str, alias: str, version: str) -> None:
    """Set model alias; falls back to a tag if alias API is unavailable."""
    try:
        client.set_registered_model_alias(model_name, alias, version)
        logger.info("  alias '%s' -> %s v%s", alias, model_name, version)
    except AttributeError:
        # MLflow < 2.3 — alias API not available
        client.set_model_version_tag(model_name, version, f"alias_{alias}", "true")
        logger.info("  tag 'alias_%s' set on %s v%s (alias API unavailable)", alias, model_name, version)
    except Exception as exc:
        logger.warning("  Could not set alias '%s' on %s v%s: %s", alias, model_name, version, exc)


# %% [6] Load all runs from all trained experiments
baseline_df   = load_runs("csip-baseline-classifiers")
advanced_df   = load_runs("csip-advanced-classifiers")
regression_df = load_runs("csip-regression-models")
cluster_df    = load_runs("csip-clustering")
explain_df    = load_runs("csip-explainability")
distilbert_df = load_runs("csip-distilbert-text")   # empty until Colab run
bilstm_df     = load_runs("csip-bilstm-text")        # empty until Colab run

total_runs = sum(
    len(df) for df in [
        baseline_df, advanced_df, regression_df, cluster_df,
        explain_df, distilbert_df, bilstm_df,
    ]
)
logger.info("Total existing MLflow runs: %d", total_runs)

# %% [7] Build per-task leaderboards
clf_df = pd.concat([baseline_df, advanced_df], ignore_index=True)

type_lb     = best_per_algo(clf_df,   task="ticket_type",     metric="m_val_f1_macro", maximize=True)
priority_lb = best_per_algo(clf_df,   task="ticket_priority", metric="m_val_f1_macro", maximize=True)

# Regression: unify task column (regression runs have no p_task filter needed — only 1 task)
if not regression_df.empty and "m_val_rmse" in regression_df.columns:
    regression_df["p_task"] = "resolution"   # synthetic — only one task in this experiment
    reg_lb = best_per_algo(regression_df, task="resolution", metric="m_val_rmse", maximize=False)
else:
    reg_lb = pd.DataFrame(columns=["algo", "m_val_rmse"])

logger.info("Type leaderboard:\n%s", type_lb.to_string(index=False))
logger.info("Priority leaderboard:\n%s", priority_lb.to_string(index=False))
logger.info("Regression leaderboard:\n%s", reg_lb.to_string(index=False))

# Best model per task (used for consolidation JSON)
best_type     = type_lb.iloc[0].to_dict()     if not type_lb.empty     else {}
best_priority = priority_lb.iloc[0].to_dict() if not priority_lb.empty else {}
best_reg      = reg_lb.iloc[0].to_dict()      if not reg_lb.empty      else {}

# %% [8] Register models in MLflow Model Registry
#
# Pattern per model type:
#   1. Load production pkl + challenger pkl from disk
#   2. New run in 'csip-model-registry' experiment — log both as sklearn models
#   3. mlflow.register_model() → ModelVersion (auto-increments version)
#   4. set_alias_safe() → Production / Challenger
#
# NOTE: S5-S9 training experiments are NOT re-logged. This creates ONE new experiment
#       (csip-model-registry) with 3 registration runs — one per model type.

REGISTRY_SPEC = [
    {
        "registered_name": "csip-ticket-type-classifier",
        "run_name":        "register_ticket_type",
        "task":            "ticket_type",
        "production":      {"path": LGBM_TYPE_PATH,          "algo": "lgbm"},
        "challenger":      {"path": XGB_TYPE_PATH,           "algo": "xgb"},
        "metric":          "val_f1_macro",
        "direction":       "maximize",
    },
    {
        "registered_name": "csip-priority-classifier",
        "run_name":        "register_priority",
        "task":            "ticket_priority",
        "production":      {"path": XGB_PRIORITY_PATH,       "algo": "xgb"},
        "challenger":      {"path": RF_PRIORITY_PATH,        "algo": "rf"},
        "metric":          "val_f1_macro",
        "direction":       "maximize",
    },
    {
        "registered_name": "csip-resolution-regressor",
        "run_name":        "register_resolution",
        "task":            "hours_to_resolve",
        "production":      {"path": RF_REGRESSOR_PATH,       "algo": "rf"},
        "challenger":      {"path": XGBOOST_REGRESSOR_PATH,  "algo": "xgb"},
        "metric":          "val_rmse",
        "direction":       "minimize",
    },
]

mlflow.set_experiment("csip-model-registry")

registry_results: list[dict] = []

for spec in REGISTRY_SPEC:
    name = spec["registered_name"]
    logger.info("Registering '%s'...", name)

    prod_path = spec["production"]["path"]
    chal_path = spec["challenger"]["path"]

    if not prod_path.exists():
        logger.error("  Production pkl not found: %s — skipping.", prod_path)
        continue
    if not chal_path.exists():
        logger.error("  Challenger pkl not found: %s — skipping.", chal_path)
        continue

    prod_wrapper = joblib.load(prod_path)
    chal_wrapper = joblib.load(chal_path)

    with mlflow.start_run(run_name=spec["run_name"]) as run:
        mlflow.log_params({
            "registered_name":   name,
            "task":              spec["task"],
            "production_algo":   spec["production"]["algo"],
            "challenger_algo":   spec["challenger"]["algo"],
            "metric":            spec["metric"],
            "direction":         spec["direction"],
        })
        # Log the underlying sklearn-compatible model object (not the dataclass wrapper)
        mlflow.sklearn.log_model(prod_wrapper.model, artifact_path="production")
        mlflow.sklearn.log_model(chal_wrapper.model, artifact_path="challenger")
        run_id = run.info.run_id

    # Register from run artifacts — auto-increments version number
    prod_uri = f"runs:/{run_id}/production"
    chal_uri = f"runs:/{run_id}/challenger"

    prod_mv = mlflow.register_model(prod_uri, name)
    chal_mv = mlflow.register_model(chal_uri, name)

    set_alias_safe(name, "Production", prod_mv.version)
    set_alias_safe(name, "Challenger",  chal_mv.version)

    # Add human-readable description tags
    try:
        client.update_registered_model(
            name=name,
            description=(
                f"CSIP {spec['task']} model. "
                f"Production={spec['production']['algo']}, "
                f"Challenger={spec['challenger']['algo']}. "
                f"Best metric: {spec['metric']} ({spec['direction']})."
            ),
        )
    except Exception as exc:
        logger.warning("  Could not set model description: %s", exc)

    registry_results.append({
        "registered_name":  name,
        "run_id":           run_id,
        "production_version": prod_mv.version,
        "challenger_version": chal_mv.version,
        "production_algo":  spec["production"]["algo"],
        "challenger_algo":  spec["challenger"]["algo"],
    })
    logger.info("  Registered: production=v%s  challenger=v%s", prod_mv.version, chal_mv.version)

logger.info("Model Registry: %d models registered.", len(registry_results))

# %% [9] Leaderboard charts (one per task — type / priority / regression)

palette = sns.color_palette("Set2")

def _bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    xlabel: str,
    save_path: Path,
    lower_is_better: bool = False,
    champion_algo: Optional[str] = None,
) -> None:
    """Horizontal bar chart of algo vs metric. Champion bar highlighted."""
    if df.empty:
        logger.warning("Skipping chart '%s' — empty DataFrame.", title)
        return

    df = df.copy().reset_index(drop=True)
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.sort_values(metric_col, ascending=lower_is_better)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(df))))
    colors = [
        palette[0] if row["algo"] == champion_algo else palette[2]
        for _, row in df.iterrows()
    ]
    bars = ax.barh(df["algo"], df[metric_col], color=colors, edgecolor="white", height=0.55)

    # Value labels on bars
    for bar, val in zip(bars, df[metric_col]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(axis="y", labelsize=9)
    if not lower_is_better:
        ax.set_xlim(0, min(1.0, df[metric_col].max() * 1.18))
    else:
        ax.set_xlim(0, df[metric_col].max() * 1.15)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=palette[0], label="Production (champion)"),
        Patch(color=palette[2], label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Chart saved: %s", save_path)


_bar_chart(
    df=type_lb,
    metric_col="m_val_f1_macro",
    title="Ticket Type Classifier — Val F1-Macro Leaderboard\n(5-class; noise floor ≈ 0.20; DistilBERT S9 is primary)",
    xlabel="Val F1-Macro",
    save_path=CHARTS_DIR / "s10_type_leaderboard.png",
    lower_is_better=False,
    champion_algo=best_type.get("algo"),
)

_bar_chart(
    df=priority_lb,
    metric_col="m_val_f1_macro",
    title="Ticket Priority Classifier — Val F1-Macro Leaderboard\n(4-class; XGB/RF lead tabular models)",
    xlabel="Val F1-Macro",
    save_path=CHARTS_DIR / "s10_priority_leaderboard.png",
    lower_is_better=False,
    champion_algo=best_priority.get("algo"),
)

_bar_chart(
    df=reg_lb,
    metric_col="m_val_rmse",
    title="Resolution Time Regressor — Val RMSE Leaderboard\n(hours; all R²≈0 — uniform synthetic data)",
    xlabel="Val RMSE (hours; lower is better)",
    save_path=CHARTS_DIR / "s10_regression_leaderboard.png",
    lower_is_better=True,
    champion_algo=best_reg.get("algo"),
)

# %% [10] Consolidated summary JSON
def _run_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for exp_name, df in [
        ("csip-baseline-classifiers",  baseline_df),
        ("csip-advanced-classifiers",  advanced_df),
        ("csip-regression-models",     regression_df),
        ("csip-clustering",            cluster_df),
        ("csip-explainability",        explain_df),
        ("csip-distilbert-text",       distilbert_df),
        ("csip-bilstm-text",           bilstm_df),
        ("csip-model-registry",        pd.DataFrame()),  # just logged — count from client
    ]:
        if exp_name == "csip-model-registry":
            eid = get_exp_id(exp_name)
            cnt = len(client.search_runs(experiment_ids=[eid])) if eid else 0
        else:
            cnt = len(df)
        counts[exp_name] = cnt
    return counts


run_counts = _run_counts()
all_total  = sum(run_counts.values())

consolidation = {
    "section":       10,
    "total_mlflow_runs": all_total,
    "runs_per_experiment": run_counts,
    "registered_models": registry_results,
    "best_models": {
        "ticket_type": {
            "algo":        best_type.get("algo"),
            "val_f1_macro": round(float(best_type.get("m_val_f1_macro", 0)), 6),
            "registered_name": "csip-ticket-type-classifier",
        },
        "ticket_priority": {
            "algo":        best_priority.get("algo"),
            "val_f1_macro": round(float(best_priority.get("m_val_f1_macro", 0)), 6),
            "registered_name": "csip-priority-classifier",
        },
        "resolution_time": {
            "algo":     best_reg.get("algo"),
            "val_rmse": round(float(best_reg.get("m_val_rmse", 0)), 6),
            "registered_name": "csip-resolution-regressor",
        },
    },
    "notes": [
        "S9 (DistilBERT) and S8a (BiLSTM) pending Colab run — not yet in registry.",
        "Classical ML on Ticket Type hits noise floor F1≈0.20; DistilBERT is primary classifier.",
        "All R²≈0 for regression — synthetic uniform timestamps carry no tabular signal.",
    ],
}

out_path = REPORTS_DIR / "section_10_consolidation.json"
# Atomic write to avoid corrupt JSON on interrupt
with tempfile.NamedTemporaryFile(
    mode="w", dir=REPORTS_DIR, delete=False, suffix=".tmp", encoding="utf-8"
) as tmp:
    json.dump(consolidation, tmp, indent=2)
    tmp_name = tmp.name
shutil.move(tmp_name, out_path)
logger.info("Consolidation JSON saved: %s", out_path)

# %% [11] Final summary table
print("\n" + "=" * 62)
print("CSIP — Section 10: MLflow Consolidation Summary")
print("=" * 62)
print(f"\nTotal MLflow runs across all experiments: {all_total}")
print("\nRuns per experiment:")
for exp_name, cnt in run_counts.items():
    print(f"  {exp_name:<40} {cnt:>3}")

print("\nBest models (tabular only; DistilBERT S9 pending):")
if best_type:
    print(f"  Ticket Type      : {best_type.get('algo'):<6}  val_f1_macro={float(best_type.get('m_val_f1_macro', 0)):.4f}")
if best_priority:
    print(f"  Ticket Priority  : {best_priority.get('algo'):<6}  val_f1_macro={float(best_priority.get('m_val_f1_macro', 0)):.4f}")
if best_reg:
    print(f"  Resolution Time  : {best_reg.get('algo'):<6}  val_rmse    ={float(best_reg.get('m_val_rmse', 0)):.4f}")

print("\nModel Registry:")
for r in registry_results:
    print(f"  {r['registered_name']}")
    print(f"    Production v{r['production_version']}: {r['production_algo']}")
    print(f"    Challenger v{r['challenger_version']}: {r['challenger_algo']}")

print("\nCharts saved to:", CHARTS_DIR)
print("JSON saved to  :", out_path)
print("=" * 62)
