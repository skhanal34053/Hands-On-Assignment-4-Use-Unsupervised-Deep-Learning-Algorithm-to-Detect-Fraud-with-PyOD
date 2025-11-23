"""
AutoEncoder Fraud Detection (PyOD)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from pyod.models.auto_encoder import AutoEncoder

# LOAD DATA
DATA_FILE = "creditcard.csv"
print(f"\nLoading dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# NEW: check for NaNs 
print("\nChecking for NaNs in raw data...")
print(df.isna().sum().sum(), "NaNs found total")
if df.isna().sum().sum() > 0:
    print("Dropping rows with NaN...")
    df = df.dropna()

X = df.drop("Class", axis=1).values
y = df["Class"].values


# PREPROCESSING
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- NEW: check for NaNs after scaling ---
if np.isnan(X_scaled).any():
    print("\nWARNING: NaNs found after scaling! Replacing with 0.")
    X_scaled = np.nan_to_num(X_scaled)


# AE trains only on normal transactions
X_train = X_scaled[y == 0]
X_test  = X_scaled
y_test  = y


# MODEL: AUTOENCODER
print("\nTraining AutoEncoder...")

clf = AutoEncoder(
    hidden_neuron_list=[16, 8, 16],
    batch_size=128,
    epoch_num=10,
    contamination=df["Class"].mean(),
    verbose=1
)

clf.fit(X_train)


# SCORING & EVALUATION
print("\nEvaluating model...")

y_scores = clf.decision_function(X_test)
y_pred   = clf.predict(X_test)

roc  = roc_auc_score(y_test, y_scores)
pr   = average_precision_score(y_test, y_scores)

print(f"\nROC-AUC: {roc:.4f}")
print(f"AUPRC : {pr:.4f}\n")
print(classification_report(y_test, y_pred))


# PLOT SCORE DISTRIBUTION
plt.figure(figsize=(10, 4))
plt.hist(y_scores[y_test == 0], bins=50, alpha=0.6, label="Normal")
plt.hist(y_scores[y_test == 1], bins=50, alpha=0.6, label="Fraud")
plt.title("AutoEncoder Anomaly Scores")
plt.xlabel("Score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")
