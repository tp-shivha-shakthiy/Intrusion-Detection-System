"""
cross_validation.py
===================
Generic cross-validation runner.

Takes a list of pre-built balanced folds (from balancing.py) and an
sklearn-compatible estimator, runs training + validation on each fold,
and returns a results DataFrame.

Used by train_hgb.py, train_xgboost.py, and train_logistic.py to avoid
duplicated loop logic.
"""

import numpy as np
import pandas as pd
import gc
from sklearn.metrics import accuracy_score, f1_score


def run_cv(
    model,
    balanced_folds: list,
    normal_class_idx: int,
    model_name: str = "Model",
) -> tuple:
    """
    Train *model* on each fold, compute binary + multi-class metrics.

    Parameters
    ----------
    model             : sklearn estimator (unfitted)
    balanced_folds    : list of dicts from balancing.py
    normal_class_idx  : int   index of the 'Normal' class in LabelEncoder
    model_name        : str   label used in progress output

    Returns
    -------
    results_df  : pd.DataFrame   per-fold + mean metrics
    trained_model : last fitted model (fold 5)
    """
    results_log = []

    for fold_idx, fold_data in enumerate(balanced_folds):
        print(f"\n  [{model_name}] Training fold {fold_idx + 1} / "
              f"{len(balanced_folds)} ...")

        X_tr  = fold_data['X_train_fold']
        y_tr  = fold_data['y_train_fold']
        X_val = fold_data['X_val_fold']
        y_val = fold_data['y_val_fold']

        model.fit(X_tr, y_tr)
        y_pred_multi = model.predict(X_val)

        # Multi-class metrics
        multi_acc   = accuracy_score(y_val, y_pred_multi)
        macro_f1    = f1_score(y_val, y_pred_multi, average='macro',
                                zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred_multi, average='weighted',
                                zero_division=0)

        # Binary metrics  (Normal=0, Attack=1)
        y_val_bin  = np.where(y_val == normal_class_idx, 0, 1)
        y_pred_bin = np.where(y_pred_multi == normal_class_idx, 0, 1)
        binary_acc = accuracy_score(y_val_bin, y_pred_bin)
        binary_f1  = f1_score(y_val_bin, y_pred_bin, average='binary',
                               zero_division=0)

        results_log.append({
            'Fold':        f"Fold {fold_idx + 1}",
            'Binary Acc':  binary_acc,
            'Binary F1':   binary_f1,
            'Multi-Acc':   multi_acc,
            'Macro F1':    macro_f1,
            'Weighted F1': weighted_f1,
        })

        print(f"    Multi-Acc: {multi_acc:.4f}  |  Binary F1: {binary_f1:.4f}")
        del X_tr, y_tr, X_val, y_val; gc.collect()

    # Append mean row
    df = pd.DataFrame(results_log)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['Fold'] = 'Mean'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df, model   # returns the last fitted model
