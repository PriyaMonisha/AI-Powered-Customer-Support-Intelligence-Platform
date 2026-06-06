# filename: notebooks/08a_bilstm.py
# purpose:  BiLSTM ticket-type classifier with GloVe 100d embeddings (Section 8a)
#           Run on Google Colab T4 GPU for production training.
# version:  1.0

FAST_MODE  = True    # FIRST LINE — 2 epochs; set False for 5 epochs on Colab
EPOCHS     = 2 if FAST_MODE else 5
BATCH_SIZE = 64
MAX_SEQ_LEN = 128    # 99th percentile of token count is ~65; 128 is generous
MIN_FREQ    = 2      # minimum word frequency to include in vocabulary

# ---------------------------------------------------------------------------
# Cell 2 — Colab setup (Google Drive mount + optional installs)
# ---------------------------------------------------------------------------
try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount("/content/drive")
    print("Colab detected — Google Drive mounted")
except ImportError:
    IN_COLAB = False
    print("Running locally (not Colab)")

try:
    import torch
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "torch==2.3.0"], check=True)
    import torch

# ---------------------------------------------------------------------------
# Cell 3 — Imports + PROJECT_ROOT
# ---------------------------------------------------------------------------
import datetime
import json
import logging
import sys
import warnings
import zipfile
from collections import Counter
from pathlib import Path

try:
    import urllib.request
except ImportError:
    pass

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Colab: try standard mount location, then cwd
    PROJECT_ROOT = Path("/content/drive/MyDrive/csip")
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

try:
    from config import (
        ARTIFACTS_DIR,
        BILSTM_EMBEDDING_MATRIX_PATH,
        BILSTM_PATH,
        BILSTM_VOCAB_PATH,
        CHARTS_DIR,
        FEATURES_DIR,
        FAST_MODE as CFG_FAST_MODE,
        MLFLOW_TRACKING_URI,
        MODELS_DIR,
        PREPROCESSED_DATA_PATH,
        RANDOM_STATE,
        SPLIT_INDICES_PATH,
    )
except ImportError as exc:
    raise ImportError(f"Could not import config.py — check PROJECT_ROOT={PROJECT_ROOT}") from exc

try:
    from src.models.bilstm import BiLSTMClassifier
except ImportError:
    # Inline fallback if src/ not on path (e.g., fresh Colab session)
    class BiLSTMClassifier(nn.Module):  # type: ignore[no-redef]
        def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64,
                     n_layers=2, n_classes=5, dropout=0.3, pad_idx=0, bidirectional=True):
            super().__init__()
            self.vocab_size = vocab_size; self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim; self.bidirectional = bidirectional
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                                bidirectional=bidirectional,
                                dropout=dropout if n_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            lstm_out = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(lstm_out, n_classes)
        def forward(self, x):
            emb = self.dropout(self.embedding(x))
            _, (h, _) = self.lstm(emb)
            h_cat = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
            return self.fc(self.dropout(h_cat))
        def load_pretrained_embeddings(self, matrix, freeze=False):
            self.embedding.weight.data.copy_(matrix)
            self.embedding.weight.requires_grad = not freeze

try:
    from src.utils.helpers import NumpyEncoder
except ImportError:
    import json as _json
    class NumpyEncoder(_json.JSONEncoder):  # type: ignore[no-redef]
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("section8a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("FAST_MODE=%s  EPOCHS=%d  device=%s", FAST_MODE, EPOCHS, device)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Cell 5 — Download GloVe 6B 100d
# ---------------------------------------------------------------------------
def download_glove(glove_dir: Path) -> Path:
    """Download glove.6B.100d.txt if not already present. ~822 MB zip."""
    glove_dir.mkdir(parents=True, exist_ok=True)
    glove_txt = glove_dir / "glove.6B.100d.txt"
    if glove_txt.exists():
        logger.info("GloVe already present: %s", glove_txt)
        return glove_txt
    glove_zip = glove_dir / "glove.6B.zip"
    if not glove_zip.exists():
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        logger.info("Downloading GloVe from %s (~822 MB, may take several minutes) ...", url)
        urllib.request.urlretrieve(url, glove_zip)
        logger.info("Download complete: %s", glove_zip)
    logger.info("Extracting glove.6B.100d.txt ...")
    with zipfile.ZipFile(glove_zip, "r") as z:
        z.extract("glove.6B.100d.txt", glove_dir)
    logger.info("GloVe ready: %s", glove_txt)
    return glove_txt

glove_dir  = PROJECT_ROOT / "data" / "glove"
glove_path = download_glove(glove_dir)

# ---------------------------------------------------------------------------
# Cell 6 — Load text + labels (Ticket Subject + [SEP] + Ticket Description)
# ---------------------------------------------------------------------------
def load_text_splits(
    preprocessed_path: Path,
    split_indices_path: Path,
    label_maps_path: Path,
) -> tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: (train_texts, val_texts, test_texts, y_train, y_val, y_test)
    Joins on Ticket ID from split_indices.json. Warns on any missing IDs.
    """
    df = pd.read_csv(preprocessed_path).set_index("Ticket ID")
    df["text"] = (
        df["Ticket Subject"].fillna("") + " [SEP] " + df["Ticket Description"].fillna("")
    )

    with open(label_maps_path) as fh:
        label_maps = json.load(fh)
    type_map = {v: int(k) for k, v in label_maps["ticket_type"].items()}
    df["label"] = df["Ticket Type"].map(type_map)

    with open(split_indices_path) as fh:
        split_indices = json.load(fh)

    results = {}
    for name, ids in split_indices.items():
        valid_ids = [i for i in ids if i in df.index]
        missing   = len(ids) - len(valid_ids)
        if missing > 0:
            logger.warning("%s split: %d IDs missing from preprocessed CSV", name, missing)
        sub = df.loc[valid_ids]
        results[name] = (sub["text"].tolist(), sub["label"].values)

    (train_texts, y_train), (val_texts, y_val), (test_texts, y_test) = (
        results["train"], results["val"], results["test"],
    )

    # Cross-verify: label distribution should match y_train_type.npy
    y_train_check = np.load(FEATURES_DIR / "y_train_type.npy")
    for i in range(5):
        assert (y_train == i).sum() == (y_train_check == i).sum(), \
            f"Label mismatch for class {i}: text={( y_train==i).sum()} npy={(y_train_check==i).sum()}"

    logger.info("Text splits: train=%d  val=%d  test=%d", len(train_texts), len(val_texts), len(test_texts))
    return train_texts, val_texts, test_texts, y_train, y_val, y_test

train_texts, val_texts, test_texts, y_train, y_val, y_test = load_text_splits(
    PREPROCESSED_DATA_PATH, SPLIT_INDICES_PATH, FEATURES_DIR / "label_maps.json"
)

# ---------------------------------------------------------------------------
# Cell 7 — Build vocabulary (frequency-sorted, reserved-token collision guard)
# ---------------------------------------------------------------------------
def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    """
    <PAD>=0, <UNK>=1 are reserved and never overwritten by training words.
    Sort by frequency desc then alpha for determinism across retrains.
    """
    RESERVED = ["<PAD>", "<UNK>"]
    counter = Counter(w for t in texts for w in t.lower().split())
    valid_words = sorted(
        [w for w, c in counter.items() if c >= min_freq and w not in RESERVED],
        key=lambda w: (-counter[w], w),
    )
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(RESERVED)}
    vocab.update({w: i + len(RESERVED) for i, w in enumerate(valid_words)})
    logger.info("Vocab: %d tokens (%d words + %d special)",
                len(vocab), len(valid_words), len(RESERVED))
    return vocab

vocab = build_vocab(train_texts, min_freq=MIN_FREQ)

# ---------------------------------------------------------------------------
# Cell 8 — GloVe embedding matrix
# ---------------------------------------------------------------------------
def build_embedding_matrix(
    vocab: dict[str, int],
    glove_path: Path,
    embedding_dim: int = 100,
) -> torch.Tensor:
    """
    (vocab_size, embedding_dim) FloatTensor.
    OOV tokens initialised with mean±std of loaded GloVe vectors.
    <PAD> stays zero (embedding layer handles via padding_idx).
    Raises RuntimeError if no vectors loaded (corrupt file guard).
    """
    embeddings    = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    loaded_vectors: list[np.ndarray] = []

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if parts[0] in vocab:
                vec = np.array(parts[1:], dtype=np.float32)
                if vec.shape[0] == embedding_dim:
                    embeddings[vocab[parts[0]]] = vec
                    loaded_vectors.append(vec)

    if not loaded_vectors:
        raise RuntimeError(
            f"No GloVe embeddings loaded from {glove_path}. "
            "File may be corrupt or wrong embedding dimension."
        )

    loaded_arr = np.array(loaded_vectors)
    mean_v, std_v = loaded_arr.mean(0), loaded_arr.std(0)

    for word, idx in vocab.items():
        if np.allclose(embeddings[idx], 0) and word != "<PAD>":
            embeddings[idx] = np.random.normal(mean_v, std_v)

    embeddings[vocab["<PAD>"]] = 0.0

    coverage = len(loaded_vectors) / len(vocab) * 100
    logger.info(
        "GloVe: loaded %d/%d embeddings (%.1f%% coverage)",
        len(loaded_vectors), len(vocab), coverage,
    )
    return torch.FloatTensor(embeddings)

embedding_matrix = build_embedding_matrix(vocab, glove_path, embedding_dim=100)

# ---------------------------------------------------------------------------
# Cell 9 — Dataset + DataLoaders
# ---------------------------------------------------------------------------
class TicketTextDataset(Dataset):
    """Tokenises text, maps to vocab indices, pads/truncates to max_len."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        vocab: dict[str, int],
        max_len: int = 128,
    ) -> None:
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len
        self.unk_idx = vocab["<UNK>"]
        self.pad_idx = vocab["<PAD>"]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens  = self.texts[idx].lower().split()
        indices = [self.vocab.get(w, self.unk_idx) for w in tokens]
        indices = (indices + [self.pad_idx] * self.max_len)[: self.max_len]
        return (
            torch.LongTensor(indices),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )

train_ds = TicketTextDataset(train_texts, y_train, vocab, MAX_SEQ_LEN)
val_ds   = TicketTextDataset(val_texts,   y_val,   vocab, MAX_SEQ_LEN)
test_ds  = TicketTextDataset(test_texts,  y_test,  vocab, MAX_SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

logger.info("DataLoaders: train=%d  val=%d  test=%d batches",
            len(train_loader), len(val_loader), len(test_loader))

# ---------------------------------------------------------------------------
# Cell 10 — Class weights
# ---------------------------------------------------------------------------
def compute_class_weights_tensor(
    y_train: np.ndarray,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Balanced class weights as FloatTensor for CrossEntropyLoss(weight=...)."""
    assert y_train.min() >= 0 and y_train.max() < n_classes, \
        f"Labels must be in [0, {n_classes-1}], got [{y_train.min()}, {y_train.max()}]"
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    logger.info("Class weights: %s", np.round(weights, 4))
    return torch.FloatTensor(weights).to(device)

class_weights = compute_class_weights_tensor(y_train, n_classes=5, device=device)

# ---------------------------------------------------------------------------
# Cell 11 — Build model + load pretrained embeddings
# ---------------------------------------------------------------------------
model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=64,
    n_layers=2,
    n_classes=5,
    dropout=0.3,
    bidirectional=True,
).to(device)

model.load_pretrained_embeddings(embedding_matrix.to(device), freeze=False)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("BiLSTMClassifier: vocab=%d  hidden=64  layers=2  params=%s",
            len(vocab), f"{n_params:,}")

# ---------------------------------------------------------------------------
# Cell 12 — Training loop (differential LR + gradient clipping)
# ---------------------------------------------------------------------------
def evaluate_split(
    model: BiLSTMClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (avg_loss, accuracy, f1_macro)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc      = float(accuracy_score(all_labels, all_preds))
    f1       = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, f1

def train_bilstm(
    model: BiLSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    class_weights: torch.Tensor,
) -> dict[str, list]:
    """
    AdamW with differential LR:
      embedding layer: 1e-4 (preserve GloVe semantics)
      LSTM + FC:       1e-3 (train from scratch)
    Gradient clipping max_norm=5.0 (standard for LSTMs).
    Best checkpoint saved by val F1-macro.
    """
    emb_params   = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    optimizer = torch.optim.AdamW([
        {"params": emb_params,   "lr": 1e-4},
        {"params": other_params, "lr": 1e-3},
    ])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    ckpt_path   = MODELS_DIR / "bilstm_checkpoint_best.pt"
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    logger.info("Training: %d epochs  |  embedding_lr=1e-4  lstm_lr=1e-3  clip=5.0", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(texts), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate_split(model, val_loader, criterion, device)

        history["train_loss"].append(round(avg_train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_f1"].append(round(val_f1, 6))
        history["val_acc"].append(round(val_acc, 6))

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f",
            epoch, epochs, avg_train_loss, val_loss, val_acc, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               val_f1,
                "val_loss":             val_loss,
                "vocab_size":           len(vocab),
            }, ckpt_path)
            logger.info("  -> Best checkpoint saved (val_f1=%.4f)", val_f1)

    # Restore best weights for final evaluation
    best_ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Restored best checkpoint from epoch %d (val_f1=%.4f)",
                best_ckpt["epoch"], best_ckpt["val_f1"])
    return history

history = train_bilstm(model, train_loader, val_loader, EPOCHS, device, class_weights)

# ---------------------------------------------------------------------------
# Cell 13 — Training history chart
# ---------------------------------------------------------------------------
def plot_training_history(
    history: dict[str, list],
    save_path: Path,
) -> Path:
    """2-panel: train+val loss (left) and val F1-macro (right)."""
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history["train_loss"], label="train", marker="o")
    axes[0].plot(epochs_range, history["val_loss"],   label="val",   marker="s")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("BiLSTM Training/Validation Loss")
    axes[0].legend(); axes[0].set_xticks(list(epochs_range))

    axes[1].plot(epochs_range, history["val_f1"], color="#2ca02c", marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1-macro")
    axes[1].set_title("Validation F1-macro per Epoch")
    axes[1].set_xticks(list(epochs_range))

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path

history_chart = plot_training_history(history, CHARTS_DIR / "bilstm_training_history.png")

# ---------------------------------------------------------------------------
# Cell 14 — Final evaluation on test set
# ---------------------------------------------------------------------------
with open(FEATURES_DIR / "label_maps.json") as fh:
    _lm = json.load(fh)
ticket_type_names = [_lm["ticket_type"][str(i)] for i in range(5)]

criterion_eval = nn.CrossEntropyLoss(weight=class_weights)

_, val_acc_final,  val_f1_final  = evaluate_split(model, val_loader,  criterion_eval, device)
_, test_acc_final, test_f1_final = evaluate_split(model, test_loader, criterion_eval, device)

logger.info("Final val:  f1_macro=%.4f  acc=%.4f", val_f1_final, val_acc_final)
logger.info("Final test: f1_macro=%.4f  acc=%.4f", test_f1_final, test_acc_final)

# Confusion matrix on test set
model.eval()
all_preds_test, all_labels_test = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        preds = model(texts).argmax(dim=1).cpu().numpy()
        all_preds_test.extend(preds)
        all_labels_test.extend(labels.numpy())

try:
    from src.models.baseline import plot_confusion_matrix
    cm_chart = plot_confusion_matrix(
        np.array(all_labels_test), np.array(all_preds_test),
        ticket_type_names, CHARTS_DIR / "bilstm_confusion_matrix.png",
    )
except ImportError:
    # Inline fallback
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels_test, all_preds_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ticket_type_names, yticklabels=ticket_type_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix: BiLSTM on Test Set")
    plt.tight_layout()
    cm_path = CHARTS_DIR / "bilstm_confusion_matrix.png"
    fig.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    cm_chart = cm_path

# ---------------------------------------------------------------------------
# Cell 15 — Save all required artifacts
# ---------------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model state dict (for inference — no optimizer state)
torch.save(model.state_dict(), BILSTM_PATH)
logger.info("Saved model weights → %s", BILSTM_PATH)

# Vocabulary
with open(BILSTM_VOCAB_PATH, "w") as fh:
    json.dump(vocab, fh, indent=2)
logger.info("Saved vocab (%d tokens) → %s", len(vocab), BILSTM_VOCAB_PATH)

# Embedding matrix — required for FastAPI serving (avoids GloVe re-download)
np.save(BILSTM_EMBEDDING_MATRIX_PATH,
        model.embedding.weight.detach().cpu().numpy())
logger.info("Saved embedding matrix → %s", BILSTM_EMBEDDING_MATRIX_PATH)

# ---------------------------------------------------------------------------
# Cell 16 — MLflow logging
# ---------------------------------------------------------------------------
_mlflow_uri = MLFLOW_TRACKING_URI
try:
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-bilstm-text")
    logger.info("MLflow connected: %s", _mlflow_uri)
except Exception:
    _mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-bilstm-text")
    logger.warning("MLflow server unavailable -- using local file store: %s", _mlflow_uri)

with mlflow.start_run(run_name="bilstm_glove100d"):
    mlflow.log_params({
        "architecture":    "BiLSTM",
        "embedding":       "GloVe-100d",
        "hidden_dim":      "64",
        "n_layers":        "2",
        "dropout":         "0.3",
        "max_seq_len":     str(MAX_SEQ_LEN),
        "batch_size":      str(BATCH_SIZE),
        "epochs_trained":  str(EPOCHS),
        "embedding_lr":    "1e-4",
        "lstm_fc_lr":      "1e-3",
        "vocab_size":      str(len(vocab)),
        "class_weight":    "balanced",
        "grad_clip":       "5.0",
        "fast_mode":       str(FAST_MODE),
        "optimizer":       "AdamW",
    })
    mlflow.log_metrics({
        "val_f1_macro":   round(val_f1_final,   6),
        "val_accuracy":   round(val_acc_final,   6),
        "test_f1_macro":  round(test_f1_final,  6),
        "test_accuracy":  round(test_acc_final, 6),
    })
    for chart in [history_chart, cm_chart]:
        mlflow.log_artifact(str(chart))
    logger.info("MLflow run 'bilstm_glove100d' logged")

# ---------------------------------------------------------------------------
# Cell 17 — Metrics JSON
# ---------------------------------------------------------------------------
section_08a_metrics = {
    "section":          "8a",
    "fast_mode":        FAST_MODE,
    "generated_at":     datetime.datetime.now().isoformat(),
    "architecture":     "BiLSTM-GloVe-100d",
    "vocab_size":       len(vocab),
    "max_seq_len":      MAX_SEQ_LEN,
    "hidden_dim":       64,
    "n_layers":         2,
    "dropout":          0.3,
    "epochs_trained":   EPOCHS,
    "device":           str(device),
    "val_f1_macro":     round(val_f1_final,   6),
    "val_accuracy":     round(val_acc_final,   6),
    "test_f1_macro":    round(test_f1_final,  6),
    "test_accuracy":    round(test_acc_final, 6),
    "training_history": history,
    "artifacts": {
        "model":            str(BILSTM_PATH.relative_to(PROJECT_ROOT)),
        "vocab":            str(BILSTM_VOCAB_PATH.relative_to(PROJECT_ROOT)),
        "embedding_matrix": str(BILSTM_EMBEDDING_MATRIX_PATH.relative_to(PROJECT_ROOT)),
    },
}

metrics_dir = ARTIFACTS_DIR / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_path = metrics_dir / "section_08a_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(section_08a_metrics, fh, indent=2, cls=NumpyEncoder)
logger.info("Saved metrics → %s", metrics_path)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SECTION 8a COMPLETE — BiLSTM (GloVe 100d)")
logger.info("val  F1-macro=%.4f  accuracy=%.4f", val_f1_final,  val_acc_final)
logger.info("test F1-macro=%.4f  accuracy=%.4f", test_f1_final, test_acc_final)
logger.info("MLflow experiment: csip-bilstm-text (1 run)")
logger.info("=" * 60)
