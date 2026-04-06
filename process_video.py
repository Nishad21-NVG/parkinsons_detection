import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_DIR   = os.path.join("ml_model", "video_model")
MODEL_PATH  = os.path.join(MODEL_DIR, "video_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "video_scaler.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Generating synthetic video features...")
np.random.seed(42)
n_each = 150

healthy   = np.random.randn(n_each, 10) * 0.3 + np.array([2.0, 0.4, 4.0, 0.5, 0.3, 0.1, 0.6, 1.5, 2.5, 1.0])
parkinson = np.random.randn(n_each, 10) * 1.2 + np.array([8.0, 3.5, 18.0, 0.2, 2.5, 1.2, 5.5, 5.0, 11.0, 6.0])

X = np.vstack([healthy, parkinson])
y = np.array([0] * n_each + [1] * n_each)

print(f"Dataset Shape  : {X.shape}")
print(f"Class Distribution: Healthy={n_each}, Parkinson={n_each}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Video Model Accuracy: {acc:.4f}")
print(classification_report(y_test, clf.predict(X_test), target_names=["Healthy", "Parkinson's"]))

joblib.dump(clf,    MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"[INFO] video_model.pkl  saved to → {MODEL_PATH}")
print(f"[INFO] video_scaler.pkl saved to → {SCALER_PATH}")
print("[DONE] Video model training complete.")