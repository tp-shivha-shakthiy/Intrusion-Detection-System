"""
evaluation.py
=============
Output Layer — Evaluation, Visualisation & Results Persistence

Provides:
  - plot_confusion_matrix()     binary and multi-class CM figures
  - plot_feature_importance()   XGBoost feature-importance bar chart
  - save_results()              write metrics.csv and model_comparison.csv
  - print_final_summary()       formatted console table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, classification_report,
)


# ---------------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    normal_class_idx: int,
    save_dir: str = "assets",
    prefix: str = "",
) -> None:
    """
    Save both a binary and a multi-class confusion matrix to *save_dir*.

    Parameters
    ----------
    y_true           : ground-truth multi-class labels
    y_pred           : predicted multi-class labels
    class_names      : list of str  (le.classes_)
    normal_class_idx : int
    save_dir         : directory for PNG output
    prefix           : filename prefix  (e.g. "xgboost_")
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Binary CM ---
    y_true_bin = np.where(y_true == normal_class_idx, 0, 1)
    y_pred_bin = np.where(y_pred == normal_class_idx, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
    ConfusionMatrixDisplay(cm_bin, display_labels=['Normal', 'Attack']).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f"{prefix}Binary Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}binary_cm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Multi-class CM ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_multi = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm_multi, display_labels=class_names).plot(
        ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45
    )
    ax.set_title(f"{prefix}Multi-Class Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}multiclass_cm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Feature Importance (XGBoost)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    n_components: int = 10,
    save_dir: str = "assets",
) -> None:
    """
    Plot XGBoost feature importance scores for the PCA components.

    Parameters
    ----------
    model        : fitted XGBClassifier
    n_components : number of PCA components (x-axis labels)
    save_dir     : output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        importances = model.feature_importances_
    except AttributeError:
        print("  [feature_importance] Model has no feature_importances_ — skipping.")
        return

    labels = [f"PC{i+1}" for i in range(len(importances))]
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(importances)),
           importances[indices],
           color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([labels[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Feature Importance Score")
    ax.set_title("XGBoost Feature Importance (PCA Components)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Results Persistence
# ---------------------------------------------------------------------------

def save_results(
    all_test_results: list,
    cv_results: dict,
    results_dir: str = "results",
) -> None:
    """
    Write model_comparison.csv (blind-test) and metrics.csv (CV).

    Parameters
    ----------
    all_test_results : list of single-row DataFrames from each trainer
    cv_results       : dict  {'HGB': df, 'XGBoost': df, 'LogReg': df}
    results_dir      : output directory
    """
    os.makedirs(results_dir, exist_ok=True)

    # model_comparison.csv — one row per model, blind test metrics
    comparison_df = pd.concat(all_test_results, ignore_index=True)
    path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(path, index=False, float_format='%.4f')
    print(f"  Saved: {path}")

    # metrics.csv — all CV fold metrics concatenated
    rows = []
    for model_name, df in cv_results.items():
        df_copy = df.copy()
        df_copy.insert(0, 'Model', model_name)
        rows.append(df_copy)

    if rows:
        metrics_df = pd.concat(rows, ignore_index=True)
        path = os.path.join(results_dir, "metrics.csv")
        metrics_df.to_csv(path, index=False, float_format='%.4f')
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------

def print_final_summary(all_test_results: list) -> None:
    """Print a formatted comparison table to stdout."""
    df = pd.concat(all_test_results, ignore_index=True)
    print("\n" + "=" * 72)
    print("  FINAL MODEL COMPARISON — Blind 20 % Test Set")
    print("=" * 72)
    float_cols = [c for c in df.columns if c != 'Model']
    fmt = {c: '{:.4f}'.format for c in float_cols}
    print(df.to_string(index=False, formatters=fmt))
    print("=" * 72)
