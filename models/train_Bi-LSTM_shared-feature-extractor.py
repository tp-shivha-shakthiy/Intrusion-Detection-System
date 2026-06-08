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

# Set deterministic seeds for deep learning reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using execution device: {device}")

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
y_multi = target_encoder.fit_transform(df[target_col])
num_classes = len(target_encoder.classes_)
normal_class_idx = list(target_encoder.classes_).index('Normal')

# Deriving explicit binary targets directly mapping to multi-class ground truths
y_binary = (y_multi != normal_class_idx).astype(int)

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
# === HYBRID CONFIGURATION LAYER: MI + PCA ===
# =========================================================================
print("\n=== Executing Configuration Pipeline (MI + PCA Only) ===")
sample_size = min(50000, len(X_processed))
idx_sample = np.random.choice(len(X_processed), sample_size, replace=False)
mi_scores = mutual_info_classif(X_processed[idx_sample], y_multi[idx_sample], random_state=42)
top_feature_indices = np.argsort(mi_scores)[-30:]
X_mi = X_processed[:, top_feature_indices]

pca = PCA(n_components=15, random_state=42)
X_pca = pca.fit_transform(X_mi)
del X_processed, X_mi; gc.collect()


# =========================================================================
# === PHASE 11: MULTI-TASK HIERARCHICAL NEURAL NETWORK ARCHITECTURE ===
# =========================================================================
class MultiTaskHierarchicalDNN(nn.Module):
    def __init__(self, input_dim, num_multi_classes):
        super(MultiTaskHierarchicalDNN, self).__init__()
        
        # Shared Feature Extractor Base
        self.shared_base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Binary Classification Head (Outputs 2 logits)
        self.binary_head = nn.Linear(64, 2)
        
        # Multi-Class Classification Head (Outputs num_multi_classes logits)
        self.multi_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_multi_classes)
        )
        
    def forward(self, x):
        shared_features = self.shared_base(x)
        binary_logits = self.binary_head(shared_features)
        multi_logits = self.multi_head(shared_features)
        return binary_logits, multi_logits


# =========================================================================
# === STRATIFIED CROSS-VALIDATION LOOP ===
# =========================================================================
print("\n=== Running Multi-Task Hierarchical Stratified Cross-Validation ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y_multi), 1):
    X_tr, y_tr_bin, y_tr_mul = X_pca[train_idx], y_binary[train_idx], y_multi[train_idx]
    X_val, y_val_bin, y_val_mul = X_pca[val_idx], y_binary[val_idx], y_multi[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_val)
    
    # Bundle features with dual target arrays
    dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_bin, dtype=torch.long),
        torch.tensor(y_tr_mul, dtype=torch.long)
    )
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = MultiTaskHierarchicalDNN(input_dim=X_pca.shape[1], num_multi_classes=num_classes).to(device)
    
    loss_fn_bin = nn.CrossEntropyLoss()
    loss_fn_mul = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    model.train()
    for epoch in range(8):
        for batch_x, batch_y_bin, batch_y_mul in train_loader:
            batch_x, batch_y_bin, batch_y_mul = batch_x.to(device), batch_y_bin.to(device), batch_y_mul.to(device)
            
            optimizer.zero_grad()
            out_bin, out_mul = model(batch_x)
            
            # Multi-Task Joint Optimization Layer
            # Weight allocation: 40% Binary focus protection, 60% Multi-class separation push
            loss = (0.40 * loss_fn_bin(out_bin, batch_y_bin)) + (0.60 * loss_fn_mul(out_mul, batch_y_mul))
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_x_t = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        pred_bin_logits, pred_mul_logits = model(val_x_t)
        
        preds_bin = torch.argmax(pred_bin_logits, dim=1).cpu().numpy()
        preds_mul = torch.argmax(pred_mul_logits, dim=1).cpu().numpy()
        
    fold_metrics.append([
        fold,
        accuracy_score(y_val_bin, preds_bin),
        f1_score(y_val_bin, preds_bin, average='binary'),
        accuracy_score(y_val_mul, preds_mul),
        f1_score(y_val_mul, preds_mul, average='macro'),
        f1_score(y_val_mul, preds_mul, average='weighted')
    ])
    print(f"🧠 Fold {fold} / 5 Hierarchical Optimization Complete.")
    del X_tr_s, X_va_s, train_loader, model; gc.collect()

# =========================================================================
# === METRIC DISPLAY MATRIX ===
# =========================================================================
print("\n=== Cross-Validation Matrix (Hierarchical Multi-Task DNN) ===")
df_res = pd.DataFrame(fold_metrics, columns=['Fold', 'Binary Acc', 'Binary F1', 'Multi-Acc', 'Multi-F1 (Macro)', 'Weighted F1'])
print(df_res.to_string(index=False))

print("-" * 85)
print(f"Mean Average |  {df_res['Binary Acc'].mean():.6f}  |  {df_res['Binary F1'].mean():.6f}  |  {df_res['Multi-Acc'].mean():.6f}  |  {df_res['Multi-F1 (Macro)'].mean():.6f}  |  {df_res['Weighted F1'].mean():.6f}")
print("-" * 85)