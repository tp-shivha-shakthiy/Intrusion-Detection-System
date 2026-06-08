"""
dimensionality_reduction.py
============================
Phase 4b + 5 — PCA Feature Extraction & 80/20 Holdout Split

Steps performed here (in order, to prevent data leakage):
  1. StandardScaler normalisation of MI-selected features
  2. PCA reduction to n_components principal components
  3. Stratified 80/20 train / test split

The scaler and PCA objects are returned so they can be reused at
inference time (fit on training data only in a real pipeline; here
they are fit on the full set before the split, which matches the
original notebook approach).
"""

import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def reduce_and_split(
    X_mi: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    Scale → PCA → stratified split.

    Parameters
    ----------
    X_mi         : float array  (N, k)   MI-selected feature matrix
    y            : int array    (N,)     encoded labels
    n_components : int          PCA components to keep
    test_size    : float        fraction held out for final evaluation
    random_state : int

    Returns
    -------
    X_train  : np.ndarray  (0.8N, n_components)
    X_test   : np.ndarray  (0.2N, n_components)
    y_train  : np.ndarray
    y_test   : np.ndarray
    scaler   : fitted StandardScaler
    pca      : fitted PCA
    """
    print("=== Phase 4b: StandardScaler + PCA Feature Extraction ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mi)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA reduction complete. Shape: {X_pca.shape}")
    print(f"  Explained variance ratio sum: "
          f"{pca.explained_variance_ratio_.sum():.4f}")

    del X_mi, X_scaled; gc.collect()

    print("\n=== Phase 5: Stratified 80/20 Holdout Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    del X_pca; gc.collect()

    return X_train, X_test, y_train, y_test, scaler, pca
