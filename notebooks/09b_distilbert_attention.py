# filename: notebooks/09b_distilbert_attention.py
# purpose:  Token-level attention visualization for the fine-tuned DistilBERT
#           ticket-type classifier (Section 9 follow-up — mentor checklist item 14)
# version:  1.0

FAST_MODE = True   # FIRST LINE — True: subsample test set for the prediction pass

# ---------------------------------------------------------------------------
# Imports + PROJECT_ROOT
# ---------------------------------------------------------------------------
import json
import logging
import string
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

import numpy as np
import pandas as pd
import torch
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_PATH,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from src.utils.helpers import NumpyEncoder

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section9b")
torch.manual_seed(RANDOM_STATE)

N_TEST_SAMPLE = 300 if FAST_MODE else None   # None = full test set for the prediction pass
TOP_K_TOKENS  = 16                            # tokens shown per attention heatmap
logger.info("FAST_MODE=%s  N_TEST_SAMPLE=%s  TOP_K_TOKENS=%d", FAST_MODE, N_TEST_SAMPLE, TOP_K_TOKENS)

# ---------------------------------------------------------------------------
# Cell — Load test split + label maps (mirrors notebooks/09_distilbert.py)
# ---------------------------------------------------------------------------
PREPROCESSED_PATH  = PROJECT_ROOT / "data" / "processed" / "preprocessed_tickets.csv"
SPLIT_INDICES_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"
LABEL_MAPS_PATH    = PROJECT_ROOT / "data" / "processed" / "features" / "label_maps.json"

df = pd.read_csv(PREPROCESSED_PATH)
df["Ticket ID"] = df["Ticket ID"].astype(int)
df = df.set_index("Ticket ID")

with open(SPLIT_INDICES_PATH) as f:
    splits = json.load(f)
with open(LABEL_MAPS_PATH) as f:
    label_maps = json.load(f)

label2id = {v: int(k) for k, v in label_maps["ticket_type"].items()}
id2label = {int(k): v for k, v in label_maps["ticket_type"].items()}

test_ids = [int(i) for i in splits["test"] if int(i) in df.index]
test_df  = df.loc[test_ids].copy()
test_df["label_id"] = test_df["Ticket Type"].map(label2id)

if FAST_MODE and len(test_df) > N_TEST_SAMPLE:
    test_df = test_df.sample(n=N_TEST_SAMPLE, random_state=RANDOM_STATE)

logger.info("Loaded %d test tickets  classes=%s", len(test_df), id2label)

# ---------------------------------------------------------------------------
# Cell — Load fine-tuned model + tokenizer (CPU inference, output_attentions=True)
# ---------------------------------------------------------------------------
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_PATH))
model = AutoModelForSequenceClassification.from_pretrained(
    str(DISTILBERT_PATH), output_attentions=True,
).to(device)
model.eval()
logger.info("Loaded fine-tuned DistilBERT from %s", DISTILBERT_PATH)

# ---------------------------------------------------------------------------
# Cell — Prediction pass (no attentions) to find a correct + a misclassified example
# ---------------------------------------------------------------------------
def batch_predict(df_split: pd.DataFrame, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    subjects     = df_split["Ticket Subject"].fillna("[NO SUBJECT]").astype(str).tolist()
    descriptions = df_split["Ticket Description"].fillna("[NO DESCRIPTION]").astype(str).tolist()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(subjects), batch_size):
            enc = tokenizer(
                subjects[i:i + batch_size], descriptions[i:i + batch_size],
                max_length=DISTILBERT_MAX_LENGTH, truncation=True,
                padding="max_length", return_tensors="pt",
            ).to(device)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            probs  = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.tolist())
            confs.extend(conf.tolist())
    return np.array(preds), np.array(confs)

logger.info("Running prediction pass over %d test tickets...", len(test_df))
preds, confs = batch_predict(test_df)
y_true = test_df["label_id"].to_numpy()
sample_accuracy = float((preds == y_true).mean())
logger.info("Prediction pass complete. Sample accuracy: %.4f", sample_accuracy)

# ---------------------------------------------------------------------------
# Cell — Select 1 correctly-classified + 1 misclassified example (highest confidence each)
# ---------------------------------------------------------------------------
correct_mask = preds == y_true
wrong_mask   = ~correct_mask

correct_idx = int(np.argmax(np.where(correct_mask, confs, -1))) if correct_mask.any() else 0
wrong_idx   = int(np.argmax(np.where(wrong_mask,   confs, -1))) if wrong_mask.any()   else 0

logger.info(
    "Correct example: idx=%d true=%s pred=%s conf=%.4f",
    correct_idx, id2label[y_true[correct_idx]], id2label[preds[correct_idx]], confs[correct_idx],
)
logger.info(
    "Wrong example:   idx=%d true=%s pred=%s conf=%.4f",
    wrong_idx, id2label[y_true[wrong_idx]], id2label[preds[wrong_idx]], confs[wrong_idx],
)

# ---------------------------------------------------------------------------
# Cell — Attention extraction + heatmap plot
# ---------------------------------------------------------------------------
def get_attention_heatmap(row: pd.Series) -> tuple[np.ndarray, list[str]]:
    """
    Tokenize one (subject, description) pair, run a forward pass with
    output_attentions=True, and return a (n_layers, top_k) matrix of
    [CLS]-token attention (averaged over heads) plus the top-k token strings,
    restored to left-to-right order.
    """
    subject     = str(row["Ticket Subject"])     if pd.notna(row["Ticket Subject"])     else "[NO SUBJECT]"
    description = str(row["Ticket Description"]) if pd.notna(row["Ticket Description"]) else "[NO DESCRIPTION]"

    enc = tokenizer(
        subject, description, max_length=DISTILBERT_MAX_LENGTH,
        truncation=True, padding="max_length", return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    attn_mask = enc["attention_mask"][0].bool()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    tokens = [t for t, m in zip(tokens, attn_mask.tolist()) if m]
    n_real = len(tokens)

    # [CLS]-token attention to every real (non-padding) token, averaged over heads, per layer
    layer_cls_attn = np.stack([
        out.attentions[layer][0, :, 0, :n_real].mean(dim=0).numpy()
        for layer in range(len(out.attentions))
    ])  # (n_layers, n_real)

    top_k = min(TOP_K_TOKENS, n_real)
    top_idx = np.sort(np.argsort(layer_cls_attn[-1])[::-1][:top_k])  # restore left-to-right order

    return layer_cls_attn[:, top_idx], [tokens[i] for i in top_idx]


def plot_attention_heatmap(matrix: np.ndarray, token_labels: list[str], title: str, save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(8, len(token_labels) * 0.55), 5.5))
    sns.heatmap(
        matrix, ax=ax, cmap="viridis", cbar_kws={"label": "[CLS] attention (mean over heads)"},
        xticklabels=token_labels,
        yticklabels=[f"Layer {i + 1}" for i in range(matrix.shape[0])],
        square=False, linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Token")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


CHARTS_DIR.mkdir(parents=True, exist_ok=True)

correct_row = test_df.iloc[correct_idx]
wrong_row   = test_df.iloc[wrong_idx]

correct_matrix, correct_tokens = get_attention_heatmap(correct_row)
wrong_matrix, wrong_tokens     = get_attention_heatmap(wrong_row)

correct_path = plot_attention_heatmap(
    correct_matrix, correct_tokens,
    f"DistilBERT [CLS] Attention -- Correct (true={id2label[y_true[correct_idx]]}, "
    f"pred={id2label[preds[correct_idx]]}, conf={confs[correct_idx]:.2f})",
    CHARTS_DIR / "distilbert_attention_correct.png",
)
wrong_path = plot_attention_heatmap(
    wrong_matrix, wrong_tokens,
    f"DistilBERT [CLS] Attention -- Misclassified (true={id2label[y_true[wrong_idx]]}, "
    f"pred={id2label[preds[wrong_idx]]}, conf={confs[wrong_idx]:.2f})",
    CHARTS_DIR / "distilbert_attention_wrong.png",
)
logger.info("Saved 2 attention heatmaps to %s", CHARTS_DIR)

# ---------------------------------------------------------------------------
# Cell — MLflow logging (experiment "csip-explainability")
# ---------------------------------------------------------------------------
_mlflow_uri = MLFLOW_TRACKING_URI
try:
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-explainability")
    logger.info("MLflow connected: %s", _mlflow_uri)
except Exception:
    _mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-explainability")
    logger.warning("MLflow server unavailable -- using local file store: %s", _mlflow_uri)

with mlflow.start_run(run_name="distilbert_attention"):
    mlflow.log_params({
        "model":          "distilbert_type_classifier",
        "n_layers":       str(correct_matrix.shape[0]),
        "top_k_tokens":   str(TOP_K_TOKENS),
        "n_test_sampled": str(len(test_df)),
        "fast_mode":      str(FAST_MODE),
        "random_state":   str(RANDOM_STATE),
    })
    mlflow.log_metrics({
        "sample_accuracy":    sample_accuracy,
        "correct_confidence": float(confs[correct_idx]),
        "wrong_confidence":   float(confs[wrong_idx]),
    })
    mlflow.log_artifact(str(correct_path))
    mlflow.log_artifact(str(wrong_path))
    logger.info("MLflow run logged to experiment 'csip-explainability'")

# ---------------------------------------------------------------------------
# Cell — section_09b_metrics.json
# ---------------------------------------------------------------------------
SPECIAL_OR_PUNCT = set(tokenizer.all_special_tokens) | set(string.punctuation)

def pct_special_or_punct(tokens: list[str]) -> float:
    return round(sum(1 for t in tokens if t in SPECIAL_OR_PUNCT or not any(c.isalnum() for c in t)) / len(tokens), 4)

pct_correct = pct_special_or_punct(correct_tokens)
pct_wrong   = pct_special_or_punct(wrong_tokens)

section_09b_metrics = {
    "section":         "9b",
    "fast_mode":       FAST_MODE,
    "n_test_sampled":  len(test_df),
    "sample_accuracy": round(sample_accuracy, 6),
    "top_k_tokens":    TOP_K_TOKENS,
    "correct_example": {
        "true_label": id2label[y_true[correct_idx]],
        "pred_label": id2label[preds[correct_idx]],
        "confidence": round(float(confs[correct_idx]), 6),
        "top_tokens": correct_tokens,
        "pct_top_tokens_special_or_punct": pct_correct,
    },
    "wrong_example": {
        "true_label": id2label[y_true[wrong_idx]],
        "pred_label": id2label[preds[wrong_idx]],
        "confidence": round(float(confs[wrong_idx]), 6),
        "top_tokens": wrong_tokens,
        "pct_top_tokens_special_or_punct": pct_wrong,
    },
    "interpretation": (
        f"{pct_correct:.0%} of the top-{TOP_K_TOKENS} [CLS]-attended tokens in the correct example "
        f"and {pct_wrong:.0%} in the misclassified example are special/punctuation tokens rather "
        f"than class-indicative content words. This is consistent with Section 9's finding "
        f"(test F1-macro=0.1954, tied with the 0.20 noise floor) that Ticket Type carries no "
        f"learnable signal from text."
    ),
    "artifacts": {
        "correct_heatmap": str(correct_path.relative_to(PROJECT_ROOT)),
        "wrong_heatmap":   str(wrong_path.relative_to(PROJECT_ROOT)),
    },
}

metrics_path = ARTIFACTS_DIR / "metrics" / "section_09b_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as fh:
    json.dump(section_09b_metrics, fh, indent=2, cls=NumpyEncoder, ensure_ascii=False)
logger.info("Saved metrics -> %s", metrics_path)

logger.info("=" * 60)
logger.info("SECTION 9b COMPLETE -- DistilBERT Attention Visualization")
logger.info("Sample accuracy: %.4f", sample_accuracy)
logger.info("Correct example pct special/punct: %.0f%%", pct_correct * 100)
logger.info("Wrong example pct special/punct:   %.0f%%", pct_wrong * 100)
logger.info("=" * 60)
