import pandas as pd
import numpy as np
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

# Safe automated installation/import layer for imbalanced-learn utilities
try:
    from imblearn.over_sampling import KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("Installing missing imbalanced-learn packages...")
    os.system('pip install imbalanced-learn')
    from imblearn.over_sampling import KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler

# Set deterministic seeds for deep learning reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using execution device: {device}")

"""log+raw scaling"""
# =========================================================================
# === PHASE 3: PRODUCTION PREPROCESSING LAYER ===
# =========================================================================
print("\n=== Phase 3: Commencing Preprocessing Layer ===")

col_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss',
    'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_src_ltm', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm_d', 'ct_srv_dst', 'ct_state_ttl', 'ct_src_user_ltm',
    'ct_src_zone_ltm', 'ct_dst_host_ltm', 'ct_srv_src', 'ct_dst_sport_ltm_d', 'ct_dst_src_ltm_d',
    'ct_src_ltm_d_d', 'ct_src_ltm_d_s', 'ct_dst_ltm_d_d', 'ct_dst_ltm_d_s', 'ct_srv_dst_d', 'ct_srv_src_d',
    'ct_state_ttl_d', 'ct_src_user_ltm_d', 'ct_src_zone_ltm_d', 'ct_dst_host_ltm_d', 'ct_srv_dst_d_d',
    'ct_srv_src_d_d', 'ct_state_ttl_d_d', 'ct_src_user_ltm_d_d', 'ct_src_zone_ltm_d_d', 'ct_dst_host_ltm_d_d',
    'ct_srv_dst_d_d_d', 'ct_srv_src_d_d_d', 'ct_state_ttl_d_d_d', 'ct_src_user_ltm_d_d_d',
    'ct_src_zone_ltm_d_d_d', 'ct_dst_host_ltm_d_d_d', 'id', 'attack_cat', 'label'
]

files = [f'UNSW-NB15_{i}.csv' for i in range(1, 4 + 1)]
df_list = []

for f in files:
    try:
        print(f"Loading {f}...")
        df_temp = pd.read_csv(f, header=None, low_memory=False)
        if df_temp.shape[1] == 49:
            df_temp.columns = col_names[:47] + ['attack_cat', 'label']
        else:
            df_temp.columns = col_names[:df_temp.shape[1]]
        df_list.append(df_temp)
    except FileNotFoundError:
        print(f"⚠️ Warning: {f} not found.")

df = pd.concat(df_list, ignore_index=True)
print(f"Data Ingestion Complete. Combined Shape: {df.shape}")

# Target Processing & Standardization Mapping
target_col = 'attack_cat'
df[target_col] = df[target_col].fillna('Normal').astype(str).str.strip().str.lower()

category_mapping = {
    'normal': 'Normal', 'fuzzers': 'Fuzzers', 'analysis': 'Analysis',
    'backdoor': 'Backdoor', 'dos': 'DoS', 'exploits': 'Exploits',
    'generic': 'Generic', 'reconnaissance': 'Reconnaissance',
    'shellcode': 'Shellcode', 'worms': 'Worms'
}
df[target_col] = df[target_col].map(category_mapping).fillna('Normal')

target_encoder = LabelEncoder()
y_all = target_encoder.fit_transform(df[target_col])
num_classes = len(target_encoder.classes_)
normal_class_idx = list(target_encoder.classes_).index('Normal')

# Compute balanced class weights to fix the 45.89% Macro F1 issue natively
class_counts = np.bincount(y_all)
total_samples = len(y_all)
# Adaptive inverse frequency weighting string
class_weights = total_samples / (num_classes * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

drop_cols = ['id', 'label', 'stime', 'ltime', 'srcip', 'dstip']
X_raw = df.drop([c for c in drop_cols if c in df.columns] + [target_col], axis=1)
del df, df_list; gc.collect()

# --- Continuous Feature Log-Transformation Layer ---
for port_col in ['sport', 'dsport']:
    if port_col in X_raw.columns:
        X_raw[port_col] = pd.to_numeric(X_raw[port_col], errors='coerce').fillna(-1).astype('int32')

true_cat_features = ['proto', 'state', 'service']
binary_features = ['is_ftp_login', 'is_sm_ips_ports']
continuous_features = [col for col in X_raw.columns if col not in true_cat_features + binary_features]

X_continuous = np.log1p(X_raw[continuous_features].clip(lower=0)).fillna(0).astype('float32')

X_categorical = pd.DataFrame(index=X_raw.index)
for col in true_cat_features:
    if col in X_raw.columns:
        X_categorical[col] = LabelEncoder().fit_transform(X_raw[col].astype(str)).astype('float32')

X_binary = X_raw[binary_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

X_processed = np.hstack([X_continuous.values, X_categorical.values, X_binary.values])
print("🎉 Base feature space extraction successful.")
del X_raw, X_continuous, X_categorical, X_binary; gc.collect()


# =========================================================================
# === PHASE 11: LEAKAGE-SAFE PYTORCH DNN ENGINE ===
# =========================================================================
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate_safe_fold(X_train, y_train, X_val, y_val, num_features):
    # Scale inside the fold to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_va_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True)
    
    model = DeepNeuralNetwork(input_dim=num_features, output_dim=num_classes).to(device)
    # Balanced weights are passed directly to the loss function to protect minority classes
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    model.train()
    for epoch in range(5):  # Efficient training epochs for fast CPU execution
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_va_t.to(device))
        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
        
    y_val_bin = (y_val != normal_class_idx).astype(int)
    val_preds_bin = (val_preds != normal_class_idx).astype(int)
    
    return [
        accuracy_score(y_val_bin, val_preds_bin),
        f1_score(y_val_bin, val_preds_bin, average='binary'),
        accuracy_score(y_val, val_preds),
        f1_score(y_val, val_preds, average='macro'),
        f1_score(y_val, val_preds, average='weighted')
    ]

# =========================================================================
# === RUNNING STRATIFIED 5-FOLD VERIFICATION ===
# =========================================================================
print("\n=== Phase 11: Running Leakage-Safe Stratified Cross-Validation ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_all), 1):
    print(f"🧠 Processing Safe Fold {fold} / 5 directly on Core Data Matrix...")
    X_tr, y_tr = X_processed[train_idx], y_all[train_idx]
    X_val, y_val = X_processed[val_idx], y_all[val_idx]
    
    metrics = train_and_evaluate_safe_fold(X_tr, y_tr, X_val, y_val, X_processed.shape[1])
    fold_metrics.append([fold] + metrics)

# =========================================================================
# === OUTPUT SUMMARY MATRIX DISPLAY ===
# =========================================================================
print("\n=== Cross-Validation Matrix (Leakage-Safe Native Weighted DNN) ===")
cv_df = pd.DataFrame(fold_metrics, columns=['Fold', 'Binary Acc', 'Binary F1', 'Multi-Acc', 'Multi-F1 (Macro)', 'Weighted F1'])
print(cv_df.to_string(index=False))

print("-" * 85)
print(f"Mean Average |  {cv_df['Binary Acc'].mean():.6f}  |  {cv_df['Binary F1'].mean():.6f}  |  {cv_df['Multi-Acc'].mean():.6f}  |  {cv_df['Multi-F1 (Macro)'].mean():.6f}  |  {cv_df['Weighted F1'].mean():.6f}")
print("-" * 85)