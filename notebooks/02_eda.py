# filename: notebooks/02_eda.py
# purpose:  Exploratory data analysis on cleaned_tickets.csv. Produces 14 PNG charts
#           in artifacts/charts/. Reveals class imbalance, SLA patterns, CSAT drivers.
# version:  1.0
# run:      python notebooks/02_eda.py   (terminal, no GPU needed)

# %% ── Notebook compatibility (terminal + Colab) ──────────────────────────────
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from google.colab import drive
    drive.mount("/content/drive")
    sys.path.insert(0, "/content/drive/MyDrive/CSIP")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %% ── Imports ────────────────────────────────────────────────────────────────
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    CHARTS_DIR,
    PROCESSED_DATA_DIR,
    SLA_CRITICAL_HOURS,
    SLA_HIGH_HOURS,
    SLA_LOW_HOURS,
    SLA_MEDIUM_HOURS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("02_eda")

sns.set_theme(style="whitegrid")

# %% ── Chart save helper ──────────────────────────────────────────────────────
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
saved_charts: list = []


def save_chart(path: Path) -> None:
    """Save current figure to path, close it, and append to saved_charts tracker."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_charts.append(path)
    log.info("Saved: %s", path.name)


# %% ── 0. Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_tickets.csv")
print(f"\nLoaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}\n")

# %% ── 1. Target distributions ────────────────────────────────────────────────
print("=" * 60)
print("BLOCK 1 — TARGET DISTRIBUTIONS")
print("=" * 60)

# ── eda_01: Ticket Type (5-class primary target) ──────────────────────────
type_counts = df["Ticket Type"].value_counts()
# Reverse so largest bar appears at the top (barh plots first element at bottom)
tc_plot = type_counts[::-1]
colors_type = sns.color_palette("Set2", len(tc_plot))[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(tc_plot.index, tc_plot.values, color=colors_type)
for bar, count in zip(bars, tc_plot.values):
    pct = count / len(df) * 100
    ax.text(
        bar.get_width() + 15,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({pct:.1f}%)",
        va="center",
        fontsize=9,
    )
ax.set_xlim(0, type_counts.max() * 1.22)
ax.set_xlabel("Count")
ax.set_title("Ticket Type Distribution (5-class primary target)")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_01_ticket_type_dist.png")

print(f"\nTicket Type counts:\n{type_counts.to_string()}")
print(f"Imbalance ratio: {type_counts.max() / type_counts.min():.2f}:1")

# ── eda_02: Ticket Priority ────────────────────────────────────────────────
priority_counts = df["Ticket Priority"].value_counts()
priority_display_order = [
    p for p in ["Critical", "High", "Medium", "Low"] if p in priority_counts.index
]
priority_counts = priority_counts.reindex(priority_display_order, fill_value=0)
# Reverse so Critical (most urgent) appears at the top
pc_plot = priority_counts[::-1]
colors_pri = sns.color_palette("Set2", len(pc_plot))[::-1]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(pc_plot.index, pc_plot.values, color=colors_pri)
for bar, count in zip(bars, pc_plot.values):
    pct = count / len(df) * 100
    ax.text(
        bar.get_width() + 10,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({pct:.1f}%)",
        va="center",
        fontsize=9,
    )
ax.set_xlim(0, priority_counts.max() * 1.22)
ax.set_xlabel("Count")
ax.set_title("Ticket Priority Distribution")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_02_ticket_priority_dist.png")

print(f"\nTicket Priority counts:\n{priority_counts.to_string()}")

# ── eda_03: Ticket Status ──────────────────────────────────────────────────
status_counts = df["Ticket Status"].value_counts()
colors_status = sns.color_palette("Dark2_r", len(status_counts))

fig, ax = plt.subplots(figsize=(7, 4))
bars_v = ax.bar(status_counts.index, status_counts.values, color=colors_status)
for bar, (label, count) in zip(bars_v, status_counts.items()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 25,
        f"{count:,}\n({count / len(df):.1%})",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("Count")
ax.set_title("Ticket Status Distribution")
ax.set_ylim(0, status_counts.max() * 1.18)
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_03_ticket_status_dist.png")

print(f"\nTicket Status counts:\n{status_counts.to_string()}")

# %% ── 2. Cross-analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 2 — CROSS-ANALYSIS")
print("=" * 60)

# ── eda_04: Ticket Type × Priority heatmap (row-normalized) ───────────────
ct = pd.crosstab(df["Ticket Type"], df["Ticket Priority"], normalize="index")
col_order = [p for p in ["Critical", "High", "Medium", "Low"] if p in ct.columns]
ct = ct.reindex(columns=col_order)

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(ct, annot=True, fmt=".1%", cmap="Blues", ax=ax, linewidths=0.5, linecolor="white",
            vmin=0, vmax=0.35)
ax.set_title("Ticket Priority by Type\n(row-normalized — proportion within each type)")
ax.set_xlabel("Ticket Priority")
ax.set_ylabel("Ticket Type")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_04_type_priority_heatmap.png")

# ── eda_05: Ticket Channel ─────────────────────────────────────────────────
channel_counts = df["Ticket Channel"].value_counts()
colors_ch = sns.color_palette("Set2", len(channel_counts))

fig, ax = plt.subplots(figsize=(8, 4))
bars_ch = ax.bar(channel_counts.index, channel_counts.values, color=colors_ch)
for bar, count in zip(bars_ch, channel_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 15,
        f"{count:,}",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("Count")
ax.set_title("Ticket Channel Distribution")
ax.set_ylim(0, channel_counts.max() * 1.12)
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_05_channel_dist.png")

print(f"\nTicket Channel counts:\n{channel_counts.to_string()}")

# %% ── 3. Numeric distributions ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 3 — NUMERIC DISTRIBUTIONS")
print("=" * 60)

# Define closed subset once — reused in Blocks 4 and 6
closed = df[df["is_resolved"] == 1].copy()
print(f"\nClosed tickets: {len(closed):,} / {len(df):,} ({len(closed) / len(df):.1%})")

accent = sns.color_palette("Accent_r")

# ── eda_06: Customer Age ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["Customer Age"].dropna(), bins=20, color=accent[0], edgecolor="white")
ax.set_xlabel("Customer Age")
ax.set_ylabel("Count")
ax.set_title("Customer Age Distribution")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_06_age_histogram.png")

print(f"\nCustomer Age: mean={df['Customer Age'].mean():.1f}, "
      f"std={df['Customer Age'].std():.1f}, "
      f"range [{df['Customer Age'].min():.0f}, {df['Customer Age'].max():.0f}]")

# ── eda_07: Hours to resolve (closed tickets only) ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(closed["hours_to_resolve"], bins=25, color=accent[1], edgecolor="white")
ax.set_xlabel("Hours to Resolve")
ax.set_ylabel("Count")
ax.set_title(f"Resolution Time Distribution (closed tickets, n={len(closed):,})")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_07_hours_to_resolve_dist.png")

print(f"\nhours_to_resolve: range {closed['hours_to_resolve'].min():.2f}–"
      f"{closed['hours_to_resolve'].max():.2f}h, "
      f"median {closed['hours_to_resolve'].median():.2f}h")
print("  Note: near-uniform distribution is expected — synthetic data with random timestamps")

# ── eda_08: CSAT distribution (closed tickets, ratings 1–5) ───────────────
csat_counts = (
    closed["Customer Satisfaction Rating"]
    .dropna()
    .round()
    .astype(int)
    .value_counts()
    .sort_index()
)
colors_csat = sns.color_palette("Accent_r", len(csat_counts))

fig, ax = plt.subplots(figsize=(7, 4))
bars_csat = ax.bar(csat_counts.index, csat_counts.values, color=colors_csat, width=0.6)
for bar, count in zip(bars_csat, csat_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 8,
        f"{count:,}",
        ha="center",
        fontsize=9,
    )
ax.set_xlabel("Customer Satisfaction Rating (1–5)")
ax.set_ylabel("Count")
ax.set_title(f"CSAT Distribution (closed tickets, n={len(closed):,})")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0, csat_counts.max() * 1.14)
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_08_csat_distribution.png")

# ── eda_09: Days since purchase ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["days_since_purchase"].dropna(), bins=25, color=accent[2], edgecolor="white")
ax.set_xlabel("Days Since Purchase")
ax.set_ylabel("Count")
ax.set_title("Customer Tenure Distribution (Days Since Purchase)")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_09_days_since_purchase_dist.png")

print(f"\ndays_since_purchase: mean={df['days_since_purchase'].mean():.1f}, "
      f"std={df['days_since_purchase'].std():.1f}")

# %% ── 4. Resolution + CSAT deep-dive ────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 4 — RESOLUTION & CSAT DEEP-DIVE")
print("=" * 60)

PRIORITY_ORDER = [
    p for p in ["Critical", "High", "Medium", "Low"]
    if p in df["Ticket Priority"].unique()
]

# ── eda_10: Hours to resolve by Ticket Type ────────────────────────────────
type_order = closed["Ticket Type"].value_counts().index.tolist()

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(
    data=closed,
    x="hours_to_resolve",
    y="Ticket Type",
    order=type_order,
    hue="Ticket Type",
    palette="Set2",
    legend=False,
    ax=ax,
)
ax.set_xlabel("Hours to Resolve")
ax.set_title("Resolution Time by Ticket Type (closed tickets)")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_10_hours_by_type.png")

# ── eda_11: Hours to resolve by Priority ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(
    data=closed,
    x="hours_to_resolve",
    y="Ticket Priority",
    order=PRIORITY_ORDER,
    hue="Ticket Priority",
    palette="Set2",
    legend=False,
    ax=ax,
)
ax.set_xlabel("Hours to Resolve")
ax.set_title("Resolution Time by Priority (closed tickets)")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_11_hours_by_priority.png")

# ── eda_12: CSAT by Priority ───────────────────────────────────────────────
csat_closed = closed.dropna(subset=["Customer Satisfaction Rating"])

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(
    data=csat_closed,
    x="Customer Satisfaction Rating",
    y="Ticket Priority",
    order=PRIORITY_ORDER,
    hue="Ticket Priority",
    palette="Paired_r",
    legend=False,
    ax=ax,
)
ax.set_xlabel("Customer Satisfaction Rating (1–5)")
ax.set_title("CSAT Score by Priority (closed tickets)")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_12_csat_by_priority.png")

# ── eda_13: First response hour-of-day (exclude -1 sentinel) ──────────────
no_response = (df["response_hour_of_day"] == -1).sum()
print(f"\nresponse_hour_of_day: {no_response:,} rows excluded "
      f"(open/pending — no first response yet)")

responded = df.loc[df["response_hour_of_day"] != -1, "response_hour_of_day"]

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(responded, bins=24, range=(0, 24), color=accent[3], edgecolor="white")
ax.set_xlabel("Hour of Day (0–23)")
ax.set_ylabel("Count")
ax.set_title(f"First Response Hour Distribution (n={len(responded):,} tickets with response)")
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_13_response_hour_dist.png")

# %% ── 5. Text analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 5 — TEXT ANALYSIS")
print("=" * 60)

# ── eda_14: Ticket Subject distribution (show all unique subjects) ─────────
top20 = df["Ticket Subject"].value_counts()   # all subjects — dataset has 16 unique
top20.index = top20.index.str[:40]            # truncate labels to prevent overflow
n_subjects = len(top20)

fig, ax = plt.subplots(figsize=(11, 9))
color_subject = sns.color_palette("Set2")[0]
ax.barh(range(n_subjects), top20.values[::-1], color=color_subject)
ax.set_yticks(range(n_subjects))
ax.set_yticklabels(top20.index[::-1], fontsize=9)
ax.set_xlabel("Count")
ax.set_title(f"Ticket Subject Distribution (all {n_subjects} unique subjects)")
plt.tight_layout()
save_chart(CHARTS_DIR / "eda_14_subject_top20.png")

print(f"\nAll {n_subjects} Ticket Subjects:\n{top20.to_string()}")

# %% ── 6. Findings summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 6 — FINDINGS SUMMARY")
print("=" * 60)

# Imbalance ratio
print(f"\nTicket Type imbalance ratio: {type_counts.max() / type_counts.min():.2f}:1")

# SLA breach — closed tickets only; severity order Critical → Low
SLA_MAP = {
    "Critical": SLA_CRITICAL_HOURS,   # 4h
    "High":     SLA_HIGH_HOURS,        # 24h
    "Medium":   SLA_MEDIUM_HOURS,      # 72h
    "Low":      SLA_LOW_HOURS,         # 168h
}
print("\nSLA breach rate (closed tickets only — open/pending excluded):")
for priority, sla_hours in SLA_MAP.items():
    subset = closed[closed["Ticket Priority"] == priority]
    if len(subset) == 0:
        print(f"  {priority}: no data in this split")
        continue
    breach_rate = (subset["hours_to_resolve"] > sla_hours).mean()
    print(f"  {priority}: {breach_rate:.1%} breach  (n={len(subset):,}, SLA={sla_hours}h)")

# CSAT correlation — filter to rows with CSAT available; exclude flag/ID columns
exclude_cols = {
    "Ticket ID", "is_resolved", "has_first_response", "csat_available",
    "Customer Satisfaction Rating",
}
numeric_cols = df.select_dtypes(include="number").columns.difference(exclude_cols)
csat_df = df.loc[
    df["csat_available"] == 1,
    list(numeric_cols) + ["Customer Satisfaction Rating"],
].copy()
corr = (
    csat_df[numeric_cols]
    .corrwith(csat_df["Customer Satisfaction Rating"])
    .abs()
    .sort_values(ascending=False)
)
print(f"\nTop-5 |Pearson| correlations with CSAT (n={len(csat_df):,} closed tickets):")
print(corr.head(5).round(4).to_string())

# CSAT summary
print(f"\nCSAT: mean={df['Customer Satisfaction Rating'].mean():.2f}, "
      f"std={df['Customer Satisfaction Rating'].std():.2f}, "
      f"null rate={df['Customer Satisfaction Rating'].isna().mean():.1%}")

# Top-10 most-ticketed products
print("\nTop-10 products by ticket volume:")
print(df["Product Purchased"].value_counts().head(10).to_string())

# %% ── 7. Verify all charts written this run ───────────────────────────────────
assert len(saved_charts) == 14, (
    f"Expected 14 charts written this run, got {len(saved_charts)}: "
    f"{[p.name for p in saved_charts]}"
)
print(f"\n[OK] {len(saved_charts)} charts saved to {CHARTS_DIR}")
print("\nNext step: python notebooks/03_preprocessing.py")
