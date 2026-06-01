# filename: notebooks/05_diagnostic.py
# purpose:  Diagnose Section 5 below-random F1 — 5 ordered checks
# version:  1.0

import json
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from config import (
    BASELINE_TYPE_PATH,
    FEATURES_DIR,
    LE_TYPE_PATH,
    PREPROCESSED_DATA_PATH,
    SPLIT_INDICES_PATH,
)
from src.models.baseline import BaselineClassifier

SEP = "=" * 60


# ── CHECK 1 — Label encoder consistency ───────────────────────
print(SEP)
print("CHECK 1 — Label encoder consistency")
print(SEP)

le: LabelEncoder = joblib.load(LE_TYPE_PATH)
y_val = np.load(FEATURES_DIR / "y_val_type.npy")

print("LabelEncoder classes:       ", le.classes_)
print("Unique values in y_val_type:", np.unique(y_val))
print("Value counts in y_val_type: ", np.bincount(y_val.astype(int)))
print("n_classes (le):             ", len(le.classes_))
print("n_classes (y_val unique):   ", len(np.unique(y_val)))

classes_consistent = len(le.classes_) == len(np.unique(y_val))
print("Classes consistent:         ", classes_consistent)


# ── CHECK 2 — Row order contract verification ──────────────────
print()
print(SEP)
print("CHECK 2 — Row order contract (X_val / y_val alignment)")
print(SEP)

with open(SPLIT_INDICES_PATH) as fh:
    splits = json.load(fh)
val_ids = splits["val"]
print("Val split ticket IDs (first 5):", val_ids[:5])
print("Total val IDs in split file:   ", len(val_ids))

df = pd.read_csv(PREPROCESSED_DATA_PATH)
print("Preprocessed CSV shape:        ", df.shape)
print("Ticket ID dtype in CSV:        ", df["Ticket ID"].dtype)
print("val_ids element type:          ", type(val_ids[0]))

df_val = df[df["Ticket ID"].isin(val_ids)].reset_index(drop=True)
print("df_val rows after .isin():     ", len(df_val))

y_val_reconstructed = le.transform(df_val["Ticket Type"])
y_val_saved = np.load(FEATURES_DIR / "y_val_type.npy")

print("y_val_saved shape:             ", y_val_saved.shape)
print("y_val_reconstructed shape:     ", y_val_reconstructed.shape)

match = np.array_equal(y_val_reconstructed, y_val_saved)
print("Row order MATCH:               ", match)
if not match:
    n_mismatches = (y_val_reconstructed != y_val_saved).sum()
    print(f"[!] MISMATCHES: {n_mismatches} / {len(y_val_saved)}")
    # Show first 10 discrepancies
    bad_idx = np.where(y_val_reconstructed != y_val_saved)[0][:10]
    print("First mismatch indices:", bad_idx)
    print("reconstructed at those idx:", y_val_reconstructed[bad_idx])
    print("saved       at those idx:", y_val_saved[bad_idx])
else:
    print("[OK] y_val row order matches reconstructed from split_indices.json")


# ── CHECK 3 — Class distribution in val set ───────────────────
print()
print(SEP)
print("CHECK 3 — Class distribution in val set")
print(SEP)

print("Val set class distribution (Ticket Type):")
for i, cls in enumerate(le.classes_):
    count = int((y_val_saved == i).sum())
    pct   = 100.0 * count / len(y_val_saved)
    print(f"  {i}: {cls:<30s} -> {count:4d} samples ({pct:.1f}%)")

print()
print("Val set class distribution (Ticket Priority):")
from config import LE_PRIORITY_PATH
le_prio = joblib.load(LE_PRIORITY_PATH)
y_val_prio = np.load(FEATURES_DIR / "y_val_prio.npy")
for i, cls in enumerate(le_prio.classes_):
    count = int((y_val_prio == i).sum())
    pct   = 100.0 * count / len(y_val_prio)
    print(f"  {i}: {cls:<30s} -> {count:4d} samples ({pct:.1f}%)")


# ── CHECK 4 — Trivial prediction sanity check ─────────────────
print()
print(SEP)
print("CHECK 4 — Trivial prediction sanity check")
print(SEP)

clf = BaselineClassifier.load(BASELINE_TYPE_PATH)
print("Loaded model:", clf)
print("feature_schema:", clf.feature_schema)

# Load the feature matrix that matches this model's schema
if clf.feature_schema == "tfidf_only":
    X_vl_check = sp.load_npz(str(FEATURES_DIR / "X_val_tfidf.npz"))
elif clf.feature_schema == "tabular_only":
    X_vl_check = np.load(FEATURES_DIR / "X_val_tabular.npy")
elif clf.feature_schema in ("combined_sparse", "combined_dense"):
    X_tab = np.load(FEATURES_DIR / "X_val_tabular.npy")
    X_tfidf = sp.load_npz(str(FEATURES_DIR / "X_val_tfidf.npz"))
    X_combined = sp.hstack([sp.csr_matrix(X_tab), X_tfidf])
    X_vl_check = X_combined.toarray() if clf.feature_schema == "combined_dense" else X_combined
else:
    raise ValueError(f"Unknown feature_schema: {clf.feature_schema}")

preds = clf.model.predict(X_vl_check)
print("Prediction distribution:", np.bincount(preds.astype(int)))
print("Unique predicted classes:", np.unique(preds))
print("Is model predicting only 1 class?", len(np.unique(preds)) == 1)

majority_class = int(np.bincount(y_val_saved.astype(int)).argmax())
majority_pred  = np.full_like(y_val_saved, majority_class)
majority_f1    = f1_score(y_val_saved, majority_pred, average="macro", zero_division=0)
print(f"Majority class (class {majority_class}) F1-macro floor: {majority_f1:.4f}")

actual_f1 = f1_score(y_val_saved, preds, average="macro", zero_division=0)
print(f"Saved model actual val F1-macro:                       {actual_f1:.4f}")
print(f"Above majority baseline?  {actual_f1 > majority_f1}")


# ── CHECK 5 — TF-IDF signal with sufficient iterations ────────
print()
print(SEP)
print("CHECK 5 — TF-IDF signal (fresh LR, 2000 iterations)")
print(SEP)

X_tr = sp.load_npz(str(FEATURES_DIR / "X_train_tfidf.npz"))
y_tr = np.load(FEATURES_DIR / "y_train_type.npy")
X_vl = sp.load_npz(str(FEATURES_DIR / "X_val_tfidf.npz"))
y_vl = np.load(FEATURES_DIR / "y_val_type.npy")

print(f"Train: X={X_tr.shape}, y={y_tr.shape}")
print(f"Val:   X={X_vl.shape}, y={y_vl.shape}")

lr_test = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
    C=1.0,
    solver="saga",
    multi_class="multinomial",
)
print("Fitting fresh LR (max_iter=2000) on TF-IDF train ...")
lr_test.fit(X_tr, y_tr)

preds_fresh = lr_test.predict(X_vl)
f1_fresh    = f1_score(y_vl, preds_fresh, average="macro", zero_division=0)
converged   = lr_test.n_iter_[0] < 2000

print(f"Fresh LR val F1-macro:   {f1_fresh:.4f}")
print(f"Iterations used:         {lr_test.n_iter_[0]} / 2000")
print(f"Converged:               {converged}")
print(f"Prediction distribution: {np.bincount(preds_fresh.astype(int))}")

print()
print(SEP)
print("DIAGNOSTIC SUMMARY")
print(SEP)
print(f"CHECK 1 — Classes consistent:           {classes_consistent}")
print(f"CHECK 2 — Row order match:              {match}")
print(f"CHECK 3 — See distribution above")
print(f"CHECK 4 — Model above majority baseline:{actual_f1 > majority_f1}")
print(f"CHECK 5 — Fresh LR F1 (2000 iter):      {f1_fresh:.4f}  (converged={converged})")
print(SEP)

if not match:
    print("[ROOT CAUSE] Row order mismatch between X_val and y_val arrays.")
    print("            Feature-label pairs are shuffled -> below-chance performance.")
elif f1_fresh > 0.40:
    print("[ROOT CAUSE] LR_MAX_ITER=200 was insufficient. Features have real signal.")
    print(f"            Fresh LR at 2000 iter: {f1_fresh:.4f} vs saved: {actual_f1:.4f}")
elif f1_fresh < 0.22:
    print("[INFO] Features genuinely have near-zero signal for Ticket Type.")
    print("       Consistent with EDA: identical top TF-IDF terms across classes.")
    print("       This is expected for this synthetic dataset -- DistilBERT is the fix.")
else:
    print(f"[INFO] Partial signal detected (F1={f1_fresh:.4f}). Check class distribution.")
