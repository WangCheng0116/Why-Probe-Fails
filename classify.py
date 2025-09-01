import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# --------------------------------------------------
# 1.  CONFIGURATION
# -------------------------------------------------
seed            = 42
base_directory = "DIRECTORY_TO_YOUR_NPY_FILE"  
id_malicious_datasets = ["NAME_OF_NPY_FILE"]
id_benign_datasets    = ["NAME_OF_NPY_FILE"]
# --------------------------------------------------
def load_npy(path: str) -> np.ndarray:
    """Loads an .npy file and returns a 2‑D array (N, dim)."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} is not shaped (N, dim) – got {arr.shape}")
    return arr


def list_all_datasets(directory: str) -> dict[str, str]:
    """Return dict {dataset_name: full_path} for every .npy file."""
    files = glob.glob(os.path.join(directory, "*.npy"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in files}


def metrics(y_true, y_pred, y_prob):
    """Return a dict of common classification metrics."""
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred,  zero_division=0),
        "f1":   f1_score(y_true, y_pred,     zero_division=0),
        "auc":  roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    }


# --------------------------------------------------
# 2.  LOAD DATA
# --------------------------------------------------
all_files = list_all_datasets(base_directory)

# Build ID pool
def paths_for(names):
    return [all_files[n] for n in names if n in all_files]

id_paths = paths_for(id_malicious_datasets) + paths_for(id_benign_datasets)
if len(id_paths) == 0:
    raise ValueError("No ID datasets found. Check your names.")

X_id, y_id = [], []
for p in id_paths:
    arr = load_npy(p)
    label = 1 if p.split(os.sep)[-1].startswith("malicious_") else 0
    X_id.append(arr)
    y_id.append(np.full(arr.shape[0], label, dtype=int))

X_id = np.vstack(X_id)
y_id = np.concatenate(y_id)

# --------------------------------------------------
# 3.  PREPROCESS, TRAIN, VALIDATE (ID)
# --------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_id, y_id, test_size=0.20, stratify=y_id, random_state=seed
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

svm = SVC(kernel="linear", random_state=seed, probability=True)
svm.fit(X_tr_scaled, y_tr)

y_val_pred = svm.predict(X_val_scaled)
y_val_prob = svm.predict_proba(X_val_scaled)[:, 1]

results = []
results.append(
    {"dataset": "ID (20 % split)"} | metrics(y_val, y_val_pred, y_val_prob)
)

# --------------------------------------------------
# 4.  OOD EVALUATION
# --------------------------------------------------
ood_datasets = [name for name in all_files if name not in
                (id_malicious_datasets + id_benign_datasets)]

for name in sorted(ood_datasets):
    X_ood = load_npy(all_files[name])
    y_ood = np.full(X_ood.shape[0], 1 if name.startswith("malicious_") else 0)
    X_ood_scaled = scaler.transform(X_ood)  # same scaler!
    y_pred = svm.predict(X_ood_scaled)
    y_prob = svm.predict_proba(X_ood_scaled)[:, 1]
    results.append({"dataset": f"OOD – {name}"} | metrics(y_ood, y_pred, y_prob))

# --------------------------------------------------
# 5.  PRINT SUMMARY
# --------------------------------------------------
df = pd.DataFrame(results)
pd.set_option("display.precision", 4)
print("\n==========  SVM Results (ID vs. OOD)  ==========")
print(df.to_string(index=False))
