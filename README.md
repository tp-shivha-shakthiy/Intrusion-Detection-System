# Intrusion Detection System — UNSW-NB15

Network intrusion detection pipeline trained on the UNSW-NB15 dataset.
Produces both **binary** (Normal / Attack) and **multi-class** (10 attack
categories) predictions.

---

## Project structure

```
INTRUSION-DETECTION-SYSTEM/
├── assets/                    Generated plots (confusion matrices, etc.)
├── data/
│   ├── raw/                   UNSW-NB15_1.csv … UNSW-NB15_4.csv go here
│   └── processed/             (reserved for cached intermediate arrays)
├── src/
│   ├── preprocessing.py       Phase 3  — load, clean, encode, log1p
│   ├── feature_selection.py   Phase 4a — Mutual Information (SelectKBest)
│   ├── dimensionality_reduction.py  Phase 4b+5 — StandardScaler, PCA, split
│   ├── balancing.py           Phase 6  — SMOTE / MiniBatchKMeans+SMOTE folds
│   ├── cross_validation.py    Shared CV loop used by all trainers
│   ├── train_hgb.py           Phase 7  — HistGradientBoostingClassifier
│   ├── train_xgboost.py       Phase 8+9 — XGBoost (CV + blind test)
│   ├── train_logistic.py      Phase 10 — Logistic Regression (saga)
│   └── evaluation.py          Output layer — plots, CSV results
├── notebooks/
│   └── Intrusion_Detection.ipynb   Original exploratory notebook
├── results/
│   ├── model_comparison.csv   Blind-test metrics for all models
│   └── metrics.csv            Per-fold CV metrics for all models
├── main.py                    Full pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Architecture overview

```
Dataset
  └─► Preprocessing (log1p, encoding, normalisation)
        └─► Feature Selection (MI → top 15)
              └─► PCA (10 components)
                    ├─► 80 % Training set
                    │     └─► Stratified 5-Fold CV
                    │           └─► SMOTE balancing (per fold)
                    │                 ├─► HGB  ──┐
                    │                 ├─► XGB  ──┼─► IDS Model
                    │                 └─► LR   ──┘
                    │                               ├─► Binary  (Normal / Attack)
                    └─► 20 % Test set (blind) ──────└─► Multi-class (10 categories)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download UNSW-NB15 from the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
and place the four CSV files in `data/raw/`:

```
data/raw/UNSW-NB15_1.csv
data/raw/UNSW-NB15_2.csv
data/raw/UNSW-NB15_3.csv
data/raw/UNSW-NB15_4.csv
```

### 3. Run the full pipeline

```bash
python main.py
```

### 4. Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/raw` | Path to raw CSV files |
| `--balancer` | `smote` | `smote` or `kmeans` (MiniBatchKMeans+SMOTE) |
| `--n-splits` | `5` | Number of CV folds |
| `--mi-k` | `15` | Top-k MI features |
| `--pca-components` | `10` | PCA output dimensions |
| `--skip-plots` | off | Skip saving PNG plots |

Example:

```bash
python main.py --balancer kmeans --pca-components 12 --skip-plots
```

---

## Output classes

**Binary:**  `Normal` / `Attack`

**Multi-class (10):**
`Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`,
`Generic`, `Normal`, `Reconnaissance`, `Shellcode`, `Worms`

---

## Notes on class balancing

The architecture specifies K-means SMOTE. Standard SMOTE is used by default
because the `Worms` class (~111 samples across 1.4 M rows) is too sparse for
KMeansSMOTE's cluster-geometry requirement — it triggers a `RuntimeError`.
Pass `--balancer kmeans` to use the MiniBatchKMeans + SMOTE hybrid, which
achieves a similar effect without crashing.
