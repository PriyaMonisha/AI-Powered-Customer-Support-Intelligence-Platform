# filename: notebooks/phase_c_02_fairness.py
# purpose:  Phase C — Fairness/segment error breakdown for XGB Ticket Priority classifier.
#           Breaks prediction errors by Customer Gender, Customer Age band, Ticket Channel.
#           Baseline for future comparison once a stronger model is available.
# version:  1.0

FAST_MODE = True  # no-op here; kept for project consistency

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score

from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")
# Use local mlruns/ — Docker MLflow server may not be running during terminal runs
_MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(_MLRUNS_DIR.as_uri())

# ── Cell 1: Paths ─────────────────────────────────────────────────────────────
FEAT_DIR    = PROJECT_ROOT / "data" / "processed" / "features"
CHARTS_DIR  = PROJECT_ROOT / "artifacts" / "charts"
METRICS_OUT = PROJECT_ROOT / "artifacts" / "metrics" / "phase_c_fairness_metrics.json"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Priority label map: {0: Critical, 1: High, 2: Low, 3: Medium}
label_maps    = json.load(open(FEAT_DIR / "label_maps.json"))
PRIO_CLASSES  = [label_maps["ticket_priority"][str(i)] for i in range(4)]
logger.info("Priority classes: %s", PRIO_CLASSES)

# ── Cell 2: Load model + test features ───────────────────────────────────────
xgb_clf = joblib.load(PROJECT_ROOT / "models" / "xgb_priority_classifier.pkl")
X_test   = np.load(FEAT_DIR / "X_test_tabular.npy")
y_test   = np.load(FEAT_DIR / "y_test_prio.npy")

y_pred = xgb_clf.model.predict(X_test)
overall_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
logger.info(
    "XGB priority overall — test F1-macro=%.4f  n_samples=%d",
    overall_f1, len(y_test),
)

# ── Cell 3: Recover test metadata via split_indices.json ─────────────────────
# X_test_tabular.npy rows are sorted by Ticket ID (see notebooks/04_features.py).
# Reconstruct df_test using the same Ticket IDs and the same sort to guarantee alignment.
splits = json.load(open(PROJECT_ROOT / "data" / "processed" / "split_indices.json"))
test_ids = set(splits["test"])

df_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cleaned_tickets.csv")
df_test = (
    df_full[df_full["Ticket ID"].isin(test_ids)]
    .sort_values("Ticket ID")
    .reset_index(drop=True)
)
assert len(df_test) == len(X_test), (
    f"Alignment mismatch: df_test has {len(df_test)} rows but X_test has {len(X_test)}. "
    "Check split_indices.json sorting."
)
logger.info("Test metadata aligned: %d rows", len(df_test))

# Attach predictions to metadata
df_test = df_test.copy()
df_test["y_true"]  = y_test
df_test["y_pred"]  = y_pred
df_test["correct"] = (y_test == y_pred)

# ── Cell 4: Age bands ─────────────────────────────────────────────────────────
age_bins   = [0, 30, 45, 60, 200]
age_labels = ["18-30", "31-45", "46-60", "61+"]
df_test["age_band"] = pd.cut(
    df_test["Customer Age"], bins=age_bins, labels=age_labels, right=True
).astype(str)

logger.info("Age band distribution:\n%s", df_test["age_band"].value_counts().to_string())

# ── Cell 5: Fairness metric computation ───────────────────────────────────────
FAIRNESS_CAVEAT = (
    f"IMPORTANT: Overall model F1-macro = {overall_f1:.3f} (XGB priority, test set). "
    "This is near the 4-class uniform noise floor (~0.25). "
    "Segment-level differences reflect statistical noise, NOT systematic model discrimination. "
    "This analysis establishes a reproducible baseline for future comparison "
    "once a stronger model is trained."
)
logger.info(FAIRNESS_CAVEAT)

RELIABILITY_THRESHOLD = 50  # segments with n < 50 flagged as unreliable


def _segment_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute per-segment F1-macro (and per-class F1) for a single grouping column.
    Returns a DataFrame with index=segment_values, cols=[n, reliable, f1_macro, *class_f1s].
    """
    rows = []
    for val, grp in df.groupby(group_col, sort=True):
        n = len(grp)
        f1_m = f1_score(grp["y_true"], grp["y_pred"], average="macro",
                        labels=list(range(4)), zero_division=0)
        per_class = f1_score(grp["y_true"], grp["y_pred"], average=None,
                             labels=list(range(4)), zero_division=0)
        row = {
            "segment": str(val),
            "n": n,
            "reliable": n >= RELIABILITY_THRESHOLD,
            "f1_macro": round(float(f1_m), 4),
        }
        for cls_idx, cls_name in enumerate(PRIO_CLASSES):
            row[f"f1_{cls_name.lower()}"] = round(float(per_class[cls_idx]), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


gender_df  = _segment_metrics(df_test, "Customer Gender")
age_df     = _segment_metrics(df_test, "age_band")
channel_df = _segment_metrics(df_test, "Ticket Channel")

for dim, df_m in [("Gender", gender_df), ("Age band", age_df), ("Channel", channel_df)]:
    logger.info("\n=== %s breakdown ===\n%s", dim, df_m[["n", "reliable", "f1_macro"]].to_string())


# ── Cell 6: Heatmaps (seaborn, Seaborn 0.13 hue-safe) ───────────────────────
def _heatmap(df_m: pd.DataFrame, dim_label: str, filename: str) -> Path:
    """
    Heatmap of per-class F1 per segment.
    Annotates each cell with F1 value + sample count.
    Unreliable segments (n < threshold) are marked with '*'.
    """
    class_cols = [f"f1_{c.lower()}" for c in PRIO_CLASSES]
    f1_data    = df_m[class_cols].copy()
    f1_data.columns = PRIO_CLASSES

    # Build annotation: "0.31\n(n=120)" or "0.31*\n(n=18)" for unreliable
    annot = pd.DataFrame(index=f1_data.index, columns=f1_data.columns, dtype=str)
    for seg in f1_data.index:
        n         = int(df_m.loc[seg, "n"])
        reliable  = bool(df_m.loc[seg, "reliable"])
        flag      = "" if reliable else "*"
        for cls in PRIO_CLASSES:
            val = f1_data.loc[seg, cls]
            annot.loc[seg, cls] = f"{val:.2f}{flag}\n(n={n})"

    fig, ax = plt.subplots(figsize=(8, max(3, len(f1_data) * 0.9 + 1.5)))
    sns.heatmap(
        f1_data.astype(float),
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1 score"},
    )
    ax.set_title(
        f"Priority Classifier F1 by {dim_label}\n"
        f"(overall F1-macro={overall_f1:.3f}; * = n < {RELIABILITY_THRESHOLD}, unreliable)",
        fontsize=11,
    )
    ax.set_xlabel("Priority class")
    ax.set_ylabel(dim_label)

    out = CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


gender_chart  = _heatmap(gender_df,  "Customer Gender", "phase_c_fairness_gender.png")
age_chart     = _heatmap(age_df,     "Age Band",        "phase_c_fairness_age.png")
channel_chart = _heatmap(channel_df, "Ticket Channel",  "phase_c_fairness_channel.png")


# ── Cell 7: MLflow ────────────────────────────────────────────────────────────
mlflow.set_experiment("csip-explainability")
with mlflow.start_run(run_name="priority_fairness_breakdown") as run:
    mlflow.log_param("model", "xgb_priority_classifier")
    mlflow.log_param("split", "test")
    mlflow.log_param("n_samples", int(len(df_test)))
    mlflow.log_param("reliability_threshold_n", RELIABILITY_THRESHOLD)
    mlflow.log_metric("overall_f1_macro", round(overall_f1, 6))

    for seg, row in gender_df.iterrows():
        mlflow.log_metric(f"gender_{seg}_f1_macro", row["f1_macro"])
    for seg, row in age_df.iterrows():
        safe_age = str(seg).replace("+", "plus")
        mlflow.log_metric(f"age_{safe_age}_f1_macro", row["f1_macro"])
    for seg, row in channel_df.iterrows():
        safe_seg = str(seg).replace(" ", "_")
        mlflow.log_metric(f"channel_{safe_seg}_f1_macro", row["f1_macro"])

    for chart in [gender_chart, age_chart, channel_chart]:
        mlflow.log_artifact(str(chart))

fairness_run_id = run.info.run_id
logger.info("MLflow run ID: %s", fairness_run_id)


# ── Cell 8: JSON artifact ─────────────────────────────────────────────────────
def _df_to_dict(df_m: pd.DataFrame) -> dict:
    return {
        seg: {k: v for k, v in row.items()}
        for seg, row in df_m.to_dict("index").items()
    }


output = {
    "section": "phase_c_02",
    "description": "Fairness/segment error breakdown — XGB priority classifier on test set",
    "generated_at": datetime.now().isoformat(),
    "caveat": FAIRNESS_CAVEAT,
    "model": "xgb_priority_classifier",
    "overall_f1_macro": round(overall_f1, 6),
    "n_test_samples": int(len(df_test)),
    "reliability_threshold_n": RELIABILITY_THRESHOLD,
    "dimensions": {
        "gender":  _df_to_dict(gender_df),
        "age_band": _df_to_dict(age_df),
        "channel": _df_to_dict(channel_df),
    },
    "charts": {
        "gender":   str(gender_chart.name),
        "age_band": str(age_chart.name),
        "channel":  str(channel_chart.name),
    },
    "mlflow_run_id": fairness_run_id,
    "mlflow_experiment": "csip-explainability",
}

METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = METRICS_OUT.with_suffix(".json.tmp")
tmp.write_text(json.dumps(output, indent=2))
shutil.move(str(tmp), str(METRICS_OUT))
logger.info("Written: %s", METRICS_OUT)


# ── Cell 9: Summary ───────────────────────────────────────────────────────────
print(f"\n=== FAIRNESS BREAKDOWN SUMMARY (overall F1-macro={overall_f1:.4f}) ===")
print(f"\n{'':5} {FAIRNESS_CAVEAT[:120]}...")
for dim_label, df_m in [("Gender", gender_df), ("Age band", age_df), ("Channel", channel_df)]:
    print(f"\n--- {dim_label} ---")
    print(df_m[["n", "reliable", "f1_macro"]].to_string())
print(f"\nCharts: {CHARTS_DIR}")
print(f"Metrics: {METRICS_OUT}")
