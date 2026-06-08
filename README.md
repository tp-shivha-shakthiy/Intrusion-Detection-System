# Intrusion Detection System — UNSW-NB15

A multi-model network intrusion detection pipeline trained on the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
Produces both **binary** (Normal / Attack) and **multi-class** (10 attack categories) predictions.

Implemented as part of the research paper:
> *A novel intrusion detection system for class imbalance datasets using hybrid sampling with deep learning techniques* — Kasina et al., Information Sciences 741 (2026)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Models](#models)
- [Quick Start](#quick-start)
- [Output Classes](#output-classes)
- [Notes on Class Balancing](#notes-on-class-balancing)
- [Results](#results)

---

## Project Structure

```
INTRUSION-DETECTION-SYSTEM/
│
├── src/                                Core pipeline modules
│   ├── __init__.py
│   ├── preprocessing.py                Phase 3  — load, clean, encode, log1p normalisation
│   ├── feature_selection.py            Phase 4a — Mutual Information (SelectKBest, top-15)
│   ├── dimensionality_reduction.py     Phase 4b+5 — StandardScaler → PCA → 80/20 split
│   ├── balancing.py                    Phase 6  — SMOTE / MiniBatchKMeans+SMOTE per fold
│   ├── cross_validation.py             Shared stratified CV loop (used by sklearn trainers)
│   └── evaluation.py                   Output layer — confusion matrices, CSV results
│
├── models/                             Standalone model training scripts
│   ├── train_hgb.py                    Phase 7  — HistGradientBoostingClassifier
│   ├── train_xgboost.py                Phase 8+9 — XGBoost (CV + blind holdout test)
│   ├── train_logistic.py               Phase 10 — Logistic Regression (multinomial/saga)
│   ├── train_dnn.py                    Deep Neural Network (weighted cross-entropy)
│   ├── train_lstm.py                   Bidirectional LSTM (MI + PCA + KMeansSMOTE)
│   ├── train_bilstm.py                 Weighted Bi-LSTM vs XGBoost dual-pipeline
│   ├── train_bilstm_multitask.py       Multi-task DNN — shared backbone, binary + multi-class 
│
├── notebooks/
│   └── Intrusion_Detection.ipynb       Original exploratory notebook
│
├── data/
│   ├── raw/                            Place UNSW-NB15_1.csv … UNSW-NB15_4.csv here
│   └── processed/                      Reserved for cached intermediate arrays
│
├── assets/ 
      |__Architecture.jpeg                
├── results/
│
├── main.py                             Full pipeline orchestrator (sklearn models)
├── requirements.txt
└── README.md
```

---

## Architecture Overview

The pipeline follows four sequential layers — Preprocessing, Data Balancing, Training, and Output — matching the framework described in the paper.

```
 PREPROCESSING LAYER          DATA BALANCING              TRAINING       OUTPUT LAYER
 ══════════════════════════╦══════════════════════════╦═════════════╦══════════════════════
                           ║                          ║             ║
   ┌──────────┐            ║  ┌────────────────────┐  ║             ║  ┌─────────────────┐
   │ Dataset  │            ║  │  Class Imbalance   │  ║             ║  │  Binary Class   │
   └────┬─────┘            ║  │  Techniques        │◄─╫─Training   ║  │                 │
        │                  ║  │  SMOTE / KMeans    │  ║   Fold      ║  │  1. Normal      │
        ▼                  ║  └─────────┬──────────┘  ║             ║  │  2. Attack      │
  ┌─────────────────────┐  ║            │              ║             ║  └────────┬────────┘
  │  Data Preprocessing │  ║  ┌─────────▼──────────┐  ║             ║           │
  │  log transformation │──╫─►│  Stratified K-Fold │  ║  ┌────────┐ ║           │
  │  Cleaning / Encoding│  ║  │  CV (training set) │  ║  │  IDS   │◄╫───────────┤
  │  Normalisation      │  ║  └─────────┬──────────┘  ║  │ Model  │ ║           │
  └─────────┬───────────┘  ║            │              ║  └────────┘ ║           ▼
            │              ║            │ Validation    ║      ▲      ║  ┌─────────────────┐
            ▼              ║            ▼ Fold          ║      │      ║  │  Multi Class    │
  ┌─────────────────────┐  ║  ┌────────────────────┐   ║      │      ║  │                 │
  │  Feature Selection  │  ║  │  Classifiers       │───╫──────┘      ║  │  1. Fuzzers     │
  │  MI → top 15        │  ║  │  ML and DL models  │   ║             ║  │  2. Analysis    │
  │                     │  ║  └────────────────────┘   ║             ║  │  3. Backdoors   │
  │  Feature Extraction │  ║            ▲               ║             ║  │  4. DoS         │
  │  PCA → 10 components│  ║            │               ║             ║  │  5. Exploits    │
  └─────────┬───────────┘  ║            │               ║             ║  │  6. Generic     │
            │   80%        ║            │               ║             ║  │  7. Recon       │
            ├──────────────╫────────────┘               ║             ║  │  8. Shellcode   │
            │   20%        ║                             ║             ║  │  9. Worms       │
            ▼              ║  ┌────────────────────┐    ║             ║  └─────────────────┘
      ┌───────────┐        ║  │     Test Data      │────╫─────────────╫──────────►
      │ Test Data │────────╫─►│   (blind 20%)      │    ║             ║
      └───────────┘        ║  └────────────────────┘    ║             ║
                           ║                             ║             ║
```

---

## Models

Two model families are included. The **sklearn models** run through the shared `src/` pipeline and `main.py`. The **deep learning models** are self-contained scripts that each include their own preprocessing, feature selection, and balancing internally.

### Sklearn models (via `main.py`)

| Phase | Model | Notes |
|-------|-------|-------|
| 7 | `HistGradientBoostingClassifier` | Fast LightGBM-style baseline |
| 8+9 | `XGBoostClassifier` | CV + blind holdout evaluation |
| 10 | `LogisticRegression` | Multinomial / saga solver |

### Deep learning models (standalone scripts)

| Script | Architecture | Pipeline |
|--------|-------------|---------|
| `train_dnn.py` | 3-layer DNN (weighted loss) | Log + raw scaling |
| `train_lstm.py` | Bidirectional LSTM | MI + PCA + KMeansSMOTE |
| `train_bilstm.py` | Weighted Bi-LSTM + XGBoost | Dual pipeline comparison |
| `train_bilstm_multitask.py` | Multi-task DNN — shared backbone, separate binary and multi-class heads | MI + PCA, joint 40/60 loss |
| `train_1dcnn.py` | 1D-CNN | MI + PCA + KMeansSMOTE |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download UNSW-NB15 from the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place all four CSV files in `data/raw/`:

```
data/raw/UNSW-NB15_1.csv
data/raw/UNSW-NB15_2.csv
data/raw/UNSW-NB15_3.csv
data/raw/UNSW-NB15_4.csv
```

### 3. Run the full sklearn pipeline

```bash
python main.py
```

### 4. Run a specific deep learning model

Each deep learning script is self-contained and can be run independently:

```bash
python models/train_lstm.py
python models/train_bilstm_multitask.py
```

### 5. Pipeline options (`main.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/raw` | Path to raw CSV files |
| `--balancer` | `smote` | `smote` or `kmeans` (MiniBatchKMeans+SMOTE) |
| `--n-splits` | `5` | Number of CV folds |
| `--mi-k` | `15` | Top-k MI features |
| `--pca-components` | `10` | PCA output dimensions |
| `--skip-plots` | off | Skip saving PNG plots |

```bash
python main.py --balancer kmeans --pca-components 12 --skip-plots
```

---

## Output Classes

**Binary:** `Normal` / `Attack`

**Multi-class (10 categories):**

| Label | Description |
|-------|-------------|
| Normal | Benign traffic |
| Fuzzers | Fuzzing attempts |
| Analysis | Network scanning / analysis |
| Backdoor | Backdoor access |
| DoS | Denial of Service |
| Exploits | Exploit-based attacks |
| Generic | Generic protocol attacks |
| Reconnaissance | Reconnaissance sweeps |
| Shellcode | Shellcode injection |
| Worms | Worm propagation |

---

## Notes on Class Balancing

The pipeline uses **SMOTE by default** rather than KMeansSMOTE.

The `Worms` class has only ~111 samples across 1.4 million rows. KMeansSMOTE requires minority samples to form dense geometric clusters — at this sparsity level it raises a `RuntimeError`. Standard SMOTE (k_neighbors=3) handles ultra-minority classes safely and is applied inside each fold to prevent data leakage.

Pass `--balancer kmeans` to use the MiniBatchKMeans + SMOTE hybrid instead, which approximates the original K-means SMOTE architecture without the sparsity crash.

---

## Results

Generated outputs are written to:

- `results/model_comparison.csv` — blind 20% holdout metrics for all sklearn models
- `results/metrics.csv` — per-fold CV metrics for all sklearn models
- `assets/` — confusion matrix PNGs and XGBoost feature importance plot

Metrics reported: Binary Accuracy, Binary F1, Multi-class Accuracy, Macro F1, Weighted F1.

## Comparison with Kasina et al. (2026)

The paper reports 99.95% binary F1 and 97.92% weighted F1 on UNSW-NB15
using SMOTE-ENN + DNN. This implementation matches the weighted F1 (0.9792,
Bi-LSTM) and improves Macro F1 through the Weighted Bi-LSTM configuration
(0.4932 vs unreported in paper), reflecting stronger minority class detection
on rare attack types (Worms, Shellcode, Analysis). The multi-task hierarchical
DNN with shared feature extractor is an architectural addition not present in
the original paper.
