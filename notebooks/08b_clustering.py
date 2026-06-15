# filename: notebooks/08b_clustering.py
# purpose:  K-Means clustering on tabular ticket features (Section 8b)
# version:  1.0

FAST_MODE = True   # FIRST LINE — set True for dev/smoke test, False for full sweep
K_RANGE = range(2, 7) if FAST_MODE else range(2, 11)

# ---------------------------------------------------------------------------
# Imports + PROJECT_ROOT
# ---------------------------------------------------------------------------
import datetime
import gc
import json
import logging
import sys
import warnings
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    FEATURES_DIR,
    FAST_MODE as CFG_FAST_MODE,
    KMEANS_MODEL_PATH,
    KMEANS_SCALER_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.utils.helpers import NumpyEncoder

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section8b")
logger.info("FAST_MODE=%s  K_RANGE=%d-%d", FAST_MODE, min(K_RANGE), max(K_RANGE))

# ---------------------------------------------------------------------------
# Cell 3 — Load data
# ---------------------------------------------------------------------------
logger.info("Loading tabular features from %s ...", FEATURES_DIR)

X_train_raw = np.load(FEATURES_DIR / "X_train_tabular.npy")
y_train_type = np.load(FEATURES_DIR / "y_train_type.npy")
y_train_prio = np.load(FEATURES_DIR / "y_train_prio.npy")

with open(FEATURES_DIR / "tabular_columns.json") as fh:
    tab_col_names: list[str] = json.load(fh)

with open(FEATURES_DIR / "label_maps.json") as fh:
    label_maps: dict = json.load(fh)

ticket_type_names = [label_maps["ticket_type"][str(i)] for i in range(len(set(y_train_type)))]
priority_names    = [label_maps["ticket_priority"][str(i)] for i in range(len(set(y_train_prio)))]

logger.info("X_train shape: %s  |  %d features", X_train_raw.shape, len(tab_col_names))
logger.info("Ticket types: %s", ticket_type_names)
logger.info("Priorities:   %s", priority_names)

# ---------------------------------------------------------------------------
# Cell 4 — StandardScaler
# ---------------------------------------------------------------------------
def fit_scaler(X_train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on training data. Returns (scaler, X_scaled)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    logger.info("StandardScaler fitted: mean=%s std=%s",
                np.round(scaler.mean_[:3], 3), np.round(scaler.scale_[:3], 3))
    return scaler, X_scaled

scaler, X_train_scaled = fit_scaler(X_train_raw)

# ---------------------------------------------------------------------------
# Cell 5 — K-Means sweep (no model objects stored — memory-safe for large datasets)
# ---------------------------------------------------------------------------
def sweep_kmeans(
    X: np.ndarray,
    k_range: range,
    random_state: int,
) -> dict[int, dict]:
    """
    Sweep K-Means across k_range. Returns {k: {inertia, silhouette, davies_bouldin}}.
    Models are NOT stored in the dict to keep memory footprint small.
    Uses n_init='auto' which is 1 for k-means++ (sklearn 1.4+).
    """
    results: dict[int, dict] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto", init="k-means++")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        results[k] = {"inertia": float(km.inertia_), "silhouette": float(sil), "davies_bouldin": float(dbi)}
        logger.info("K=%d  inertia=%.2f  silhouette=%.4f  davies_bouldin=%.4f", k, km.inertia_, sil, dbi)
    return results

logger.info("=" * 60)
logger.info("K-Means sweep  K_RANGE=%d-%d", min(K_RANGE), max(K_RANGE))
logger.info("=" * 60)
sweep_results = sweep_kmeans(X_train_scaled, K_RANGE, RANDOM_STATE)

# ---------------------------------------------------------------------------
# Cell 6 — Select best K + fit final model
# ---------------------------------------------------------------------------
def select_best_k(sweep_results: dict[int, dict]) -> int:
    """Return K with highest silhouette score."""
    best_k = max(sweep_results, key=lambda k: sweep_results[k]["silhouette"])
    logger.info(
        "Best K=%d  (silhouette=%.4f  inertia=%.2f  davies_bouldin=%.4f)",
        best_k, sweep_results[best_k]["silhouette"], sweep_results[best_k]["inertia"],
        sweep_results[best_k]["davies_bouldin"],
    )
    return best_k

best_k = select_best_k(sweep_results)

final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto", init="k-means++")
final_model.fit(X_train_scaled)
cluster_labels: np.ndarray = final_model.predict(X_train_scaled)
logger.info("Cluster sizes: %s", {k: int((cluster_labels == k).sum()) for k in range(best_k)})

# ---------------------------------------------------------------------------
# Cell 7 — PCA for visualization
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_train_scaled)
explained_var = pca.explained_variance_ratio_
logger.info("PCA explained variance: PC1=%.2f%%  PC2=%.2f%%",
            explained_var[0] * 100, explained_var[1] * 100)

# ---------------------------------------------------------------------------
# Cell 8 — Charts
# ---------------------------------------------------------------------------
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Discrete color palettes
CLUSTER_PALETTE = sns.color_palette("Set2",  n_colors=best_k)
TYPE_MARKERS    = ["o", "s", "^", "D", "P"]   # up to 5 ticket types
PRIO_MARKERS    = ["o", "s", "^", "D"]          # up to 4 priority levels
TYPE_PALETTE    = sns.color_palette("Paired_r", n_colors=len(ticket_type_names))
PRIO_PALETTE    = sns.color_palette("Dark2_r",  n_colors=len(priority_names))

def plot_elbow_curve(
    sweep_results: dict[int, dict],
    best_k: int,
    save_path: Path,
) -> Path:
    ks      = list(sweep_results.keys())
    inertia = [sweep_results[k]["inertia"] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertia, marker="o", color="#4e79a7", linewidth=2)
    ax.axvline(x=best_k, color="#e15759", linestyle="--", linewidth=1.5,
               label=f"Best K={best_k}")
    ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Inertia (SSE)")
    ax.set_title("K-Means Elbow Curve"); ax.set_xticks(ks); ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return save_path

def plot_silhouette_scores(
    sweep_results: dict[int, dict],
    best_k: int,
    save_path: Path,
) -> Path:
    ks  = list(sweep_results.keys())
    sil = [sweep_results[k]["silhouette"] for k in ks]
    dbi = [sweep_results[k]["davies_bouldin"] for k in ks]
    colors = ["#e15759" if k == best_k else "#4e79a7" for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ks, sil, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Silhouette Score")
    ax.set_xticks(ks)

    ax2 = ax.twinx()
    ax2.plot(ks, dbi, marker="o", color="#59a14f", linewidth=2, label="Davies-Bouldin Index")
    ax2.set_ylabel("Davies-Bouldin Index")
    ax2.legend(loc="upper right", fontsize=8)

    ax.set_title("Silhouette (bars, higher=better) & Davies-Bouldin Index (line, lower=better) by K")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return save_path

def plot_pca_scatter(
    X_pca: np.ndarray,
    cluster_labels: np.ndarray,
    y_type: np.ndarray,
    y_prio: np.ndarray,
    type_names: list[str],
    prio_names: list[str],
    best_k: int,
    explained_var: np.ndarray,
    save_path: Path,
) -> Path:
    """
    2-panel: left = cluster color × ticket-type marker; right = cluster color × priority marker.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pc1_label = f"PC1 ({explained_var[0]*100:.1f}%)"
    pc2_label = f"PC2 ({explained_var[1]*100:.1f}%)"

    for ax, (y_label, label_names, markers, title) in zip(
        axes,
        [
            (y_type, type_names, TYPE_MARKERS, "Clusters × Ticket Type"),
            (y_prio, prio_names, PRIO_MARKERS, "Clusters × Priority"),
        ],
    ):
        for c_id in range(best_k):
            for t_id, t_name in enumerate(label_names):
                mask = (cluster_labels == c_id) & (y_label == t_id)
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    color=CLUSTER_PALETTE[c_id],
                    marker=markers[t_id % len(markers)],
                    s=30, alpha=0.6, linewidths=0,
                    label=f"C{c_id}·{t_name}" if c_id == 0 else None,
                )
        ax.set_xlabel(pc1_label); ax.set_ylabel(pc2_label)
        ax.set_title(title)
        if ax == axes[0]:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                      title="Cluster·Type", ncol=1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return save_path

def plot_tsne_scatter(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    best_k: int,
    random_state: int,
    save_path: Path,
    n_sample: int = 500,
) -> Path:
    """t-SNE 2D scatter. Stratified subsample in FAST_MODE to keep runtime under 30s."""
    if FAST_MODE and len(X_scaled) > n_sample:
        _, X_tsne_raw, _, labels_tsne = train_test_split(
            X_scaled, cluster_labels,
            test_size=n_sample, stratify=cluster_labels, random_state=random_state,
        )
        logger.info("t-SNE: sampled %d/%d points (stratified by cluster)", n_sample, len(X_scaled))
    else:
        X_tsne_raw, labels_tsne = X_scaled, cluster_labels

    perplexity = max(5, min(50, len(X_tsne_raw) // 20))
    logger.info("t-SNE: n=%d  perplexity=%d", len(X_tsne_raw), perplexity)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity,
                n_jobs=1, max_iter=300 if FAST_MODE else 1000)
    X_2d = tsne.fit_transform(X_tsne_raw)

    fig, ax = plt.subplots(figsize=(8, 7))
    for c_id in range(best_k):
        mask = labels_tsne == c_id
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=CLUSTER_PALETTE[c_id], s=20, alpha=0.6, linewidths=0,
                   label=f"Cluster {c_id}")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE Clustering (K={best_k})")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return save_path

elbow_path = plot_elbow_curve(sweep_results, best_k, CHARTS_DIR / "clustering_elbow.png")
sil_path   = plot_silhouette_scores(sweep_results, best_k, CHARTS_DIR / "clustering_silhouette.png")
pca_path   = plot_pca_scatter(
    X_pca, cluster_labels, y_train_type, y_train_prio,
    ticket_type_names, priority_names, best_k, explained_var,
    CHARTS_DIR / "clustering_pca_scatter.png",
)
tsne_path  = plot_tsne_scatter(
    X_train_scaled, cluster_labels, best_k, RANDOM_STATE,
    CHARTS_DIR / "clustering_tsne.png",
)
logger.info("Saved 4 charts to %s", CHARTS_DIR)

# ---------------------------------------------------------------------------
# Cell 9 — MLflow logging
# ---------------------------------------------------------------------------
_mlflow_uri = MLFLOW_TRACKING_URI
try:
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-clustering")
    logger.info("MLflow connected: %s", _mlflow_uri)
except Exception:
    _mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-clustering")
    logger.warning("MLflow server unavailable -- using local file store: %s", _mlflow_uri)

with mlflow.start_run(run_name="kmeans_sweep"):
    mlflow.log_params({
        "best_k":       str(best_k),
        "k_range":      f"{min(K_RANGE)}-{max(K_RANGE)}",
        "fast_mode":    str(FAST_MODE),
        "scaler":       "StandardScaler",
        "init":         "k-means++",
        "n_init":       "auto",
        "random_state": str(RANDOM_STATE),
        "n_features":   str(len(tab_col_names)),
        "n_samples":    str(X_train_raw.shape[0]),
    })
    for k in K_RANGE:
        mlflow.log_metric(f"silhouette_k{k}",     round(sweep_results[k]["silhouette"],     6))
        mlflow.log_metric(f"inertia_k{k}",        round(sweep_results[k]["inertia"],        6))
        mlflow.log_metric(f"davies_bouldin_k{k}", round(sweep_results[k]["davies_bouldin"], 6))
    mlflow.log_metric("best_silhouette",     round(sweep_results[best_k]["silhouette"],     6))
    mlflow.log_metric("best_inertia",        round(sweep_results[best_k]["inertia"],        6))
    mlflow.log_metric("best_davies_bouldin", round(sweep_results[best_k]["davies_bouldin"], 6))
    for chart_path in [elbow_path, sil_path, pca_path, tsne_path]:
        mlflow.log_artifact(str(chart_path))
    mlflow.sklearn.log_model(final_model, "kmeans_model")
    logger.info("MLflow run logged to experiment 'csip-clustering'")

# ---------------------------------------------------------------------------
# Cell 10 — Save model artifacts (atomic tmp → replace)
# ---------------------------------------------------------------------------
def _atomic_dump(obj, path: Path) -> None:
    """joblib.dump via tmp file → atomic rename (Windows-safe)."""
    tmp = path.with_suffix(".tmp")
    joblib.dump(obj, tmp)
    tmp.replace(path)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
_atomic_dump(scaler,      KMEANS_SCALER_PATH)
_atomic_dump(final_model, KMEANS_MODEL_PATH)
logger.info("Saved scaler   → %s", KMEANS_SCALER_PATH)
logger.info("Saved KMeans   → %s", KMEANS_MODEL_PATH)

# ---------------------------------------------------------------------------
# Cell 11 — Save metrics JSON
# ---------------------------------------------------------------------------
section_08b_metrics = {
    "section":           "8b",
    "fast_mode":         FAST_MODE,
    "generated_at":      datetime.datetime.now().isoformat(),
    "k_range":           list(K_RANGE),
    "best_k":            best_k,
    "best_silhouette":     round(sweep_results[best_k]["silhouette"], 6),
    "best_inertia":        round(sweep_results[best_k]["inertia"],    6),
    "best_davies_bouldin": round(sweep_results[best_k]["davies_bouldin"], 6),
    "cluster_sizes":     {str(k): int((cluster_labels == k).sum()) for k in range(best_k)},
    "pca_explained_var": [round(float(v), 6) for v in explained_var],
    "all_k_results": {
        str(k): {
            "silhouette":     round(sweep_results[k]["silhouette"],     6),
            "inertia":        round(sweep_results[k]["inertia"],        6),
            "davies_bouldin": round(sweep_results[k]["davies_bouldin"], 6),
        }
        for k in K_RANGE
    },
    "artifacts": {
        "model":  str(KMEANS_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "scaler": str(KMEANS_SCALER_PATH.relative_to(PROJECT_ROOT)),
    },
}

metrics_dir = ARTIFACTS_DIR / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_path = metrics_dir / "section_08b_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(section_08b_metrics, fh, indent=2, cls=NumpyEncoder)
logger.info("Saved metrics → %s", metrics_path)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SECTION 8b COMPLETE — K-Means Clustering")
logger.info("Best K=%d  silhouette=%.4f  davies_bouldin=%.4f",
            best_k, sweep_results[best_k]["silhouette"], sweep_results[best_k]["davies_bouldin"])
logger.info("Models: %s | %s", KMEANS_MODEL_PATH.name, KMEANS_SCALER_PATH.name)
logger.info("Charts: %s", " | ".join(p.name for p in [elbow_path, sil_path, pca_path, tsne_path]))
logger.info("MLflow experiment: csip-clustering (1 run)")
logger.info("=" * 60)

gc.collect()
