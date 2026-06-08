"""
main.py
=======
INTRUSION DETECTION SYSTEM — Full Pipeline Orchestrator
========================================================

Executes the complete IDS pipeline in order:

  Phase 3  Preprocessing       (preprocessing.py)
  Phase 4a Feature Selection   (feature_selection.py)
  Phase 4b Dimensionality Red. (dimensionality_reduction.py)
  Phase 5  80/20 Split         (dimensionality_reduction.py)
  Phase 6  CV + SMOTE Balance  (balancing.py)
  Phase 7  HGB Training        (train_hgb.py)
  Phase 8  XGBoost CV          (train_xgboost.py)
  Phase 9  XGBoost Test Eval   (train_xgboost.py)
  Phase 10 Logistic Regression (train_logistic.py)
  Output   Evaluation + Plots  (evaluation.py)

Usage
-----
  python main.py                        # default data/raw/
  python main.py --data-dir /path/csv   # custom raw data path
  python main.py --balancer kmeans      # use MiniBatchKMeans+SMOTE
  python main.py --skip-plots           # skip matplotlib output
"""

import argparse
import numpy as np

from src.preprocessing          import load_and_preprocess
from src.feature_selection      import select_features
from src.dimensionality_reduction import reduce_and_split
from src.balancing              import smote_folds, kmeans_smote_folds
from src.train_hgb              import train_hgb
from src.train_xgboost          import train_xgboost
from src.train_logistic         import train_logistic
from src.evaluation             import (
    plot_confusion_matrix,
    plot_feature_importance,
    save_results,
    print_final_summary,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="UNSW-NB15 Intrusion Detection System"
    )
    parser.add_argument(
        '--data-dir', default='data/raw',
        help='Directory containing UNSW-NB15_1.csv … UNSW-NB15_4.csv'
    )
    parser.add_argument(
        '--balancer', choices=['smote', 'kmeans'], default='smote',
        help='Class-balancing strategy (default: smote)'
    )
    parser.add_argument(
        '--n-splits', type=int, default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--mi-k', type=int, default=15,
        help='Top-k MI features to retain (default: 15)'
    )
    parser.add_argument(
        '--pca-components', type=int, default=10,
        help='PCA components (default: 10)'
    )
    parser.add_argument(
        '--skip-plots', action='store_true',
        help='Skip saving confusion matrices and feature importance plots'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Phase 3 — Preprocessing
    # ------------------------------------------------------------------
    X_processed, y_multi, le = load_and_preprocess(data_dir=args.data_dir)
    normal_class_idx = int(np.where(le.classes_ == 'Normal')[0][0])
    class_names = list(le.classes_)

    # ------------------------------------------------------------------
    # Phase 4a — MI Feature Selection
    # ------------------------------------------------------------------
    X_mi, selector = select_features(
        X_processed, y_multi, k=args.mi_k
    )
    del X_processed

    # ------------------------------------------------------------------
    # Phase 4b + 5 — PCA + 80/20 Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler, pca = reduce_and_split(
        X_mi, y_multi, n_components=args.pca_components
    )
    del X_mi

    # ------------------------------------------------------------------
    # Phase 6 — Balanced CV Folds
    # ------------------------------------------------------------------
    if args.balancer == 'kmeans':
        print("\nUsing MiniBatchKMeans + SMOTE balancing strategy.")
        balanced_folds = kmeans_smote_folds(
            X_train, y_train, n_splits=args.n_splits
        )
    else:
        print("\nUsing standard SMOTE balancing strategy.")
        balanced_folds = smote_folds(
            X_train, y_train, n_splits=args.n_splits
        )

    # ------------------------------------------------------------------
    # Phase 7 — HistGradientBoosting
    # ------------------------------------------------------------------
    hgb_cv_df, hgb_model = train_hgb(
        balanced_folds, normal_class_idx, class_names
    )

    # ------------------------------------------------------------------
    # Phase 8 + 9 — XGBoost
    # ------------------------------------------------------------------
    xgb_cv_df, xgb_test_df, xgb_model = train_xgboost(
        balanced_folds, X_test, y_test,
        normal_class_idx, class_names,
        num_class=len(class_names),
    )

    # ------------------------------------------------------------------
    # Phase 10 — Logistic Regression
    # ------------------------------------------------------------------
    lr_cv_df, lr_test_df, lr_model = train_logistic(
        balanced_folds, X_test, y_test,
        normal_class_idx, class_names,
    )

    # ------------------------------------------------------------------
    # Output Layer — Evaluation
    # ------------------------------------------------------------------
    all_test_results = [xgb_test_df, lr_test_df]
    cv_results = {
        'HGB':     hgb_cv_df,
        'XGBoost': xgb_cv_df,
        'LogReg':  lr_cv_df,
    }

    print_final_summary(all_test_results)
    save_results(all_test_results, cv_results)

    if not args.skip_plots:
        print("\n=== Generating evaluation plots ===")

        # XGBoost confusion matrices
        y_xgb_pred = xgb_model.predict(X_test)
        plot_confusion_matrix(
            y_test, y_xgb_pred, class_names, normal_class_idx,
            save_dir='assets', prefix='xgboost_',
        )

        # Logistic Regression confusion matrices
        y_lr_pred = lr_model.predict(X_test)
        plot_confusion_matrix(
            y_test, y_lr_pred, class_names, normal_class_idx,
            save_dir='assets', prefix='logreg_',
        )

        # Feature importance
        plot_feature_importance(
            xgb_model,
            n_components=args.pca_components,
            save_dir='assets',
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
