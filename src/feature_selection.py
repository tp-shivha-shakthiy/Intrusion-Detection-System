"""
feature_selection.py
====================
Phase 4a — Mutual Information Feature Selection

Uses a 5 % stratified sample of the full preprocessed matrix to compute
mutual-information scores efficiently, then transforms the entire matrix
to keep only the top-k features.

Returns the reduced matrix and the fitted selector (for later inspection
of selected feature names / indices).
"""

import numpy as np
import gc
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 15,
    sample_frac: float = 0.05,
    random_state: int = 42,
) -> tuple:
    """
    Select the top-k features using Mutual Information scores.

    Parameters
    ----------
    X            : float32 array  (N, F)   preprocessed feature matrix
    y            : int array      (N,)     encoded labels
    k            : int            number of features to keep (default 15)
    sample_frac  : float          fraction of data used to fit MI scores
    random_state : int

    Returns
    -------
    X_mi       : np.ndarray  (N, k)    reduced feature matrix
    selector   : SelectKBest           fitted selector
    """
    print("=== Phase 4a: Mutual Information Feature Selection ===")

    # Draw a stratified sample to estimate MI scores without RAM exhaustion
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=sample_frac,
        stratify=y,
        random_state=random_state,
    )

    print(f"  Computing MI scores on {X_sample.shape[0]:,} samples "
          f"({sample_frac*100:.0f}% of data) ...")

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_sample, y_sample)

    del X_sample, y_sample; gc.collect()

    X_mi = selector.transform(X)
    print(f"  Top {k} features selected. Reduced shape: {X_mi.shape}")
    return X_mi, selector
