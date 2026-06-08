# filename: notebooks/09_distilbert.py
# purpose:  DistilBERT fine-tuning for 5-class Ticket Type classification (Colab T4)
# version:  1.0

# =============================================================================
# Cell 1 — Constants (NO imports, NO function calls — must run first, alone)
# =============================================================================
FAST_MODE    = False
EPOCHS       = 2 if FAST_MODE else 5
BATCH_SIZE   = 32
BACKBONE_LR  = 1e-5    # fine-tune DistilBERT backbone
HEAD_LR      = 1e-4    # train pre_classifier + classifier head from scratch
WARMUP_RATIO = 0.1     # fraction of total steps for linear warmup
N_CLASSES    = 5
PATIENCE     = 1 if FAST_MODE else 2   # FAST: patience=1 never fires in 2 epochs; Full: reasonable
SEED         = 42      # overridden by RANDOM_STATE from config in Cell 4

# =============================================================================
# Cell 2 — Colab guards (Drive mount + transformers version check)
# =============================================================================
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass  # not in Colab — running locally

try:
    import transformers
    assert transformers.__version__ >= "4.30.0"
except (ImportError, AssertionError):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers>=4.30.0", "accelerate", "huggingface_hub"])
    raise RuntimeError(
        "Restart runtime now (Runtime → Restart session), then re-run from Cell 1"
    )

import os
# MLflow >=3.0 blocks the plain filesystem backend ("file://...mlruns") by default —
# this project standardized on file-store URIs (Section 7+), so opt back in explicitly.
# Must be set BEFORE mlflow resolves its tracking store (set_experiment/start_run).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import mlflow  # noqa: F401  (Colab base image does not ship mlflow)
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mlflow"])
    import mlflow  # noqa: F401  (no torch/CUDA state involved — safe to import inline, no restart)

# =============================================================================
# Cell 3 — PROJECT_ROOT detection (searches for config.py — robust to any Drive path)
# =============================================================================
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    for _candidate in [
        Path("/content/drive/MyDrive/csip"),
        Path("/content/csip"),
        Path.cwd(),
        Path.cwd().parent,
    ]:
        if (_candidate / "config.py").exists():
            PROJECT_ROOT = _candidate
            break
    else:
        raise FileNotFoundError(
            "config.py not found. Verify the Drive mount path and repo location."
        )

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# =============================================================================
# Cell 4 — Imports + config + logger + MLflow URI + seed + dirs
# =============================================================================
import json
import logging
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    CHARTS_DIR,
    DISTILBERT_PATH,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_HF_REPO,
    HF_HUB_TOKEN,
    RANDOM_STATE,
)
MAX_LENGTH = DISTILBERT_MAX_LENGTH   # single source of truth — no hardcoded 128

# force=True: Colab/transformers pre-attach root handlers, making basicConfig a no-op
# without it — INFO logs (epoch progress, test eval) get silently swallowed
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

# Persist MLflow runs to PROJECT_ROOT/mlruns (Drive-backed) — survives Colab disconnects
mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_STATE)
logger.info("Seed: %d (from config.RANDOM_STATE)", RANDOM_STATE)

# =============================================================================
# Cell 5 — load_text_splits() with label-to-int conversion + int Ticket ID guard
# =============================================================================
SPLIT_INDICES_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"
PREPROCESSED_PATH  = PROJECT_ROOT / "data" / "processed" / "preprocessed_tickets.csv"
LABEL_MAPS_PATH    = PROJECT_ROOT / "data" / "processed" / "features" / "label_maps.json"


def load_text_splits(preprocessed_path, split_indices_path, label_maps_path):
    df = pd.read_csv(preprocessed_path)
    df["Ticket ID"] = df["Ticket ID"].astype(int)   # guard: CSV round-trip can produce float IDs
    df = df.set_index("Ticket ID")

    with open(split_indices_path) as f:
        splits = json.load(f)
    with open(label_maps_path) as f:
        label_maps = json.load(f)

    # label_maps JSON has string keys — convert to int
    label2id = {v: int(k) for k, v in label_maps["ticket_type"].items()}
    id2label  = {int(k): v for k, v in label_maps["ticket_type"].items()}

    results = {}
    for name, ids in splits.items():
        valid_ids = [int(i) for i in ids if int(i) in df.index]
        if len(valid_ids) < len(ids):
            logger.warning("%s: %d IDs missing from preprocessed CSV", name, len(ids) - len(valid_ids))
        df_split = df.loc[valid_ids].copy()
        df_split["label_id"] = df_split["Ticket Type"].map(label2id)
        results[name] = df_split

    return results, id2label, label2id


splits, id2label, label2id = load_text_splits(
    PREPROCESSED_PATH, SPLIT_INDICES_PATH, LABEL_MAPS_PATH
)

# Canonical label arrays — single source of truth used in Cells 6, 10, 14, 15, 19
y_train = np.array(splits["train"]["label_id"].tolist(), dtype=np.int64)
y_val   = np.array(splits["val"]["label_id"].tolist(),   dtype=np.int64)
y_test  = np.array(splits["test"]["label_id"].tolist(),  dtype=np.int64)

logger.info("Splits: train=%d val=%d test=%d", len(y_train), len(y_val), len(y_test))
logger.info("Classes: %s", id2label)

# =============================================================================
# Cell 5b — Pre-tokenization (batch, once per split — ~10x faster than per-__getitem__)
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_split(df):
    # fillna handles NaN subjects/descriptions before batch tokenization
    subjects     = df["Ticket Subject"].fillna("[NO SUBJECT]").astype(str).tolist()
    descriptions = df["Ticket Description"].fillna("[NO DESCRIPTION]").astype(str).tolist()
    return tokenizer(
        subjects,
        descriptions,
        max_length=MAX_LENGTH,
        truncation=True,       # truncates longer segment (description) first
        padding="max_length",  # uniform size required for DataLoader tensor stacking
        return_tensors="pt",
    )


train_enc = tokenize_split(splits["train"])
val_enc   = tokenize_split(splits["val"])
test_enc  = tokenize_split(splits["test"])
logger.info("Tokenization complete: max_length=%d", MAX_LENGTH)

# =============================================================================
# Cell 6 — TicketDataset (indexes pre-tokenized tensors)
# =============================================================================
class TicketDataset(Dataset):
    def __init__(self, encodings, labels: np.ndarray):
        self.input_ids      = encodings["input_ids"]       # tensor(N, MAX_LENGTH)
        self.attention_mask = encodings["attention_mask"]  # tensor(N, MAX_LENGTH)
        self.labels         = labels                       # numpy (N,) int64

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # NaN/empty handled by fillna in tokenize_split; indexing pre-computed tensors here
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


train_dataset = TicketDataset(train_enc, y_train)
val_dataset   = TicketDataset(val_enc,   y_val)
test_dataset  = TicketDataset(test_enc,  y_test)

# =============================================================================
# Cell 7 — DataLoaders (pin_memory gated on CUDA — avoids RuntimeError on CPU)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin    = device.type == "cuda"   # pin_memory=True on CPU raises RuntimeError on some systems

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=pin)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=pin)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=pin)
logger.info("Device: %s | pin_memory: %s | batches — train=%d val=%d test=%d",
            device, pin, len(train_loader), len(val_loader), len(test_loader))

# =============================================================================
# Cell 8 — Build model + parameter count log
# =============================================================================
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=N_CLASSES,
    id2label=id2label,   # {0: "Billing inquiry", ...} — int keys
    label2id=label2id,   # {"Billing inquiry": 0, ...}
).to(device)

total_p, trainable_p = count_parameters(model)
logger.info(
    "distilbert-base-uncased | total=%s trainable=%s (%.1f%%)",
    f"{total_p:,}", f"{trainable_p:,}", 100 * trainable_p / total_p,
)

# =============================================================================
# Cell 9 — Optimizer + scheduler
#   Param groups use name.startswith("distilbert.") to avoid the "classifier" ⊂
#   "pre_classifier" substring collision that would double-optimise head params.
# =============================================================================
def build_optimizer_and_scheduler(model, n_steps: int, warmup_ratio: float):
    no_decay = ["bias", "LayerNorm.weight"]

    backbone_decay,   backbone_nodecay   = [], []
    head_decay,       head_nodecay       = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("distilbert.")   # exact — never matches pre_classifier.*
        is_nodecay  = any(nd in name for nd in no_decay)
        if is_backbone:
            (backbone_nodecay if is_nodecay else backbone_decay).append(param)
        else:
            # pre_classifier.* and classifier.* both go here — no ambiguity
            (head_nodecay if is_nodecay else head_decay).append(param)

    all_grouped   = backbone_decay + backbone_nodecay + head_decay + head_nodecay
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(all_grouped) == len(all_trainable), (
        f"Param group mismatch: grouped={len(all_grouped)} total={len(all_trainable)}"
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_decay,   "lr": BACKBONE_LR, "weight_decay": 0.01},
        {"params": backbone_nodecay, "lr": BACKBONE_LR, "weight_decay": 0.0},
        {"params": head_decay,       "lr": HEAD_LR,     "weight_decay": 0.01},
        {"params": head_nodecay,     "lr": HEAD_LR,     "weight_decay": 0.0},
    ])
    warmup    = int(n_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=n_steps
    )
    logger.info(
        "Optimizer: backbone_lr=%s head_lr=%s warmup=%d total_steps=%d",
        BACKBONE_LR, HEAD_LR, warmup, n_steps,
    )
    return optimizer, scheduler


optimizer, scheduler = build_optimizer_and_scheduler(
    model,
    n_steps=len(train_loader) * EPOCHS,
    warmup_ratio=WARMUP_RATIO,
)

# =============================================================================
# Cell 10 — Class weights + criterion (with class count assertion)
# =============================================================================
def compute_class_weights_tensor(y: np.ndarray, n_classes: int, device):
    unique = np.unique(y)
    assert len(unique) == n_classes, (
        f"Missing classes: {set(range(n_classes)) - set(unique.tolist())}"
    )
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    logger.info("Class weights: %s", np.round(weights, 4))
    return torch.FloatTensor(weights).to(device)


class_weights = compute_class_weights_tensor(y_train, N_CLASSES, device)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

# =============================================================================
# Cell 11 — train_one_epoch()
# =============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(
            model(input_ids=input_ids, attention_mask=attention_mask).logits,
            labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # transformer std
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# =============================================================================
# Cell 12 — evaluate()
# =============================================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss    += criterion(outputs.logits, labels).item()
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), f1, acc, np.array(all_preds), np.array(all_labels)

# =============================================================================
# Cell 13 — Training loop (atomic checkpoint save + early stopping)
# =============================================================================
checkpoint_path                         = MODELS_DIR / "distilbert_checkpoint_best.pt"
best_val_f1, best_epoch, patience_ctr   = -1.0, 0, 0
history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

for epoch in range(EPOCHS):
    t0         = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
    val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)
    logger.info(
        "Epoch %d/%d | %.1fs | train_loss=%.4f val_loss=%.4f val_f1=%.4f val_acc=%.4f",
        epoch + 1, EPOCHS, time.time() - t0, train_loss, val_loss, val_f1, val_acc,
    )

    if val_f1 > best_val_f1:
        best_val_f1, best_epoch, patience_ctr = val_f1, epoch, 0
        # Atomic save: write to tmp then rename — prevents corrupt .pt on Colab disconnect
        with tempfile.NamedTemporaryFile(dir=MODELS_DIR, delete=False, suffix=".tmp") as tmp:
            torch.save(model.state_dict(), tmp.name)
        shutil.move(tmp.name, checkpoint_path)
        logger.info("  => Checkpoint saved (val_f1=%.4f, epoch=%d)", best_val_f1, epoch + 1)
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, PATIENCE)
            break

if FAST_MODE:
    logger.info(
        "NOTE: FAST_MODE=True (EPOCHS=%d). Set FAST_MODE=False for production F1.", EPOCHS
    )

# =============================================================================
# Cell 14 — Load best checkpoint → final test evaluation
# =============================================================================
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
test_loss, test_f1, test_acc, test_preds, _ = evaluate(model, test_loader, criterion, device)
test_f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)
logger.info(
    "TEST | f1_macro=%.4f f1_weighted=%.4f accuracy=%.4f best_epoch=%d",
    test_f1, test_f1_weighted, test_acc, best_epoch + 1,
)

# =============================================================================
# Cell 15 — Charts (3 total)
# =============================================================================

# --- Chart 1: Training history (dual-axis, anchored scales, mark best epoch) ---
epochs_range = range(1, len(history["val_f1"]) + 1)
fig, ax1 = plt.subplots(figsize=(10, 5))
c_loss, c_f1 = "steelblue", "darkorange"

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color=c_loss)
ax1.plot(epochs_range, history["train_loss"], "o-", color=c_loss, label="Train Loss", lw=2)
ax1.plot(epochs_range, history["val_loss"],   "s--", color=c_loss, alpha=0.6, label="Val Loss")
ax1.tick_params(axis="y", labelcolor=c_loss)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.set_ylabel("Val F1-Macro", color=c_f1)
ax2.plot(epochs_range, history["val_f1"], "^-", color=c_f1, label="Val F1-Macro", lw=2)
ax2.tick_params(axis="y", labelcolor=c_f1)
ax2.set_ylim(0, 1)   # fixed scale — F1 is always [0, 1]
ax2.axvline(best_epoch + 1, color="gray", linestyle=":", alpha=0.7,
            label=f"Best Epoch ({best_epoch + 1})")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
plt.title(f"DistilBERT Training History ({'FAST_MODE' if FAST_MODE else 'Full'})")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "distilbert_training_history.png", dpi=150, bbox_inches="tight")
plt.close("all")
logger.info("Saved distilbert_training_history.png")

# --- Chart 2: Confusion matrix (2-panel: counts + normalized) ---
cm          = confusion_matrix(y_test, test_preds)
cm_norm     = cm.astype(float) / cm.sum(axis=1, keepdims=True)
class_names = [id2label[i] for i in range(N_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm],
    ["d", ".2f"],
    ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
):
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("DistilBERT — Test Set Confusion Matrix", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "distilbert_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")
logger.info("Saved distilbert_confusion_matrix.png")

# --- Chart 3: Confidence histogram (explicit model.eval + no_grad — dropout must be off) ---
model.eval()
all_probs: list[float] = []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        all_probs.extend(
            torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
        )

plt.figure(figsize=(8, 5))
plt.hist(all_probs, bins=30, edgecolor="black", color="steelblue", alpha=0.8)
plt.axvline(float(np.mean(all_probs)), color="red", linestyle="--",
            label=f"Mean={float(np.mean(all_probs)):.3f}")
plt.xlabel("Max Softmax Probability (Confidence)")
plt.ylabel("Count")
plt.title("DistilBERT Prediction Confidence Distribution (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "distilbert_confidence_hist.png", dpi=150, bbox_inches="tight")
plt.close("all")
logger.info("Saved distilbert_confidence_hist.png")

# =============================================================================
# Cell 16 — save_pretrained (model + tokenizer)
# =============================================================================
DISTILBERT_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(DISTILBERT_PATH)
tokenizer.save_pretrained(DISTILBERT_PATH)
logger.info("Model + tokenizer saved to %s", DISTILBERT_PATH)
# NOTE for Section 11 FastAPI: config.json writes id2label with STRING keys ("0" not 0).
# Load with: id2label = {int(k): v for k, v in model.config.id2label.items()}

# =============================================================================
# Cell 17 — upload_with_retry() (ImportError distinguished from network errors)
# =============================================================================
def upload_with_retry(local_dir, repo_id, token, max_retries: int = 3) -> bool:
    if not token or not repo_id:
        logger.warning("HF_HUB_TOKEN or DISTILBERT_HF_REPO not set — skipping Hub upload")
        return False
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed: pip install huggingface_hub")
        return False
    api = HfApi()
    for attempt in range(max_retries):
        try:
            api.create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)
            api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, token=token)
            logger.info("Uploaded to https://huggingface.co/%s", repo_id)
            return True
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(
                "Upload attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1, max_retries, e, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
    logger.warning("HF Hub upload failed after %d retries", max_retries)
    return False


upload_with_retry(DISTILBERT_PATH, DISTILBERT_HF_REPO, HF_HUB_TOKEN)

# =============================================================================
# Cell 18 — MLflow (set_experiment BEFORE start_run; step= for per-epoch curves)
# =============================================================================
# CRITICAL: set_experiment must come before start_run — inside the context manager is TOO LATE
mlflow.set_experiment("csip-distilbert-text")

with mlflow.start_run(run_name="distilbert_finetuned") as run:
    mlflow.log_params({
        "backbone":       "distilbert-base-uncased",
        "max_length":     MAX_LENGTH,
        "backbone_lr":    str(BACKBONE_LR),
        "head_lr":        str(HEAD_LR),
        "epochs_trained": len(history["val_f1"]),
        "batch_size":     BATCH_SIZE,
        "warmup_ratio":   WARMUP_RATIO,
        "best_val_epoch": best_epoch + 1,
        "fast_mode":      str(FAST_MODE),
        "n_classes":      N_CLASSES,
        "patience":       PATIENCE,
    })

    # Per-epoch metrics with step= — renders as time-series chart in MLflow UI
    for i, (vf1, vl, tl) in enumerate(
        zip(history["val_f1"], history["val_loss"], history["train_loss"])
    ):
        mlflow.log_metric("val_f1_macro", vf1, step=i)
        mlflow.log_metric("val_loss",     vl,  step=i)
        mlflow.log_metric("train_loss",   tl,  step=i)

    mlflow.log_metrics({
        "best_val_f1":      best_val_f1,
        "test_f1_macro":    test_f1,
        "test_f1_weighted": test_f1_weighted,
        "test_accuracy":    test_acc,
    })

    mlflow.log_artifact(str(CHARTS_DIR / "distilbert_training_history.png"))
    mlflow.log_artifact(str(CHARTS_DIR / "distilbert_confusion_matrix.png"))
    logger.info("MLflow run logged: %s (experiment: csip-distilbert-text)", run.info.run_id)

# =============================================================================
# Cell 19 — section_09_metrics.json (all metrics + per-class F1)
# =============================================================================
report = classification_report(
    y_test,
    test_preds,
    target_names=[id2label[i] for i in range(N_CLASSES)],
    output_dict=True,
    zero_division=0,
)

metrics = {
    "best_val_f1_macro":  round(float(best_val_f1),        6),
    "best_val_epoch":     best_epoch + 1,
    "test_f1_macro":      round(float(test_f1),             6),
    "test_f1_weighted":   round(float(test_f1_weighted),    6),
    "test_accuracy":      round(float(test_acc),            6),
    "epochs_trained":     len(history["val_f1"]),
    "fast_mode":          FAST_MODE,
    "per_class_f1":       {
        k: round(float(v["f1-score"]), 6)
        for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    },
    "history":            {
        k: [round(float(v), 6) for v in vs]
        for k, vs in history.items()
    },
}

metrics_path = ARTIFACTS_DIR / "metrics" / "section_09_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics, fh, indent=2)

logger.info("=" * 60)
logger.info("SECTION 9 COMPLETE")
logger.info("Test F1-macro:    %.4f", test_f1)
logger.info("Test F1-weighted: %.4f", test_f1_weighted)
logger.info("Test accuracy:    %.4f", test_acc)
logger.info("Best val epoch:   %d", best_epoch + 1)
logger.info("=" * 60)
logger.info("Artifacts saved:")
logger.info("  %s", DISTILBERT_PATH)
logger.info("  %s", checkpoint_path)
logger.info("  %s", CHARTS_DIR / "distilbert_training_history.png")
logger.info("  %s", CHARTS_DIR / "distilbert_confusion_matrix.png")
logger.info("  %s", CHARTS_DIR / "distilbert_confidence_hist.png")
logger.info("  %s", metrics_path)
logger.info("=" * 60)
logger.info("Next step: commit section-9, then Section 10 MLflow consolidation")
