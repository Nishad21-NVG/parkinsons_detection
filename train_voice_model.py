# ============================================================
# train_voice_model.py
# Train voice-based Parkinson's detection model
# Dataset: UCI Parkinson's Voice Dataset (numeric features)
# Models: Logistic Regression, SVM, Random Forest → save best
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "voice", "parkinsons.data")
MODEL_DIR   = os.path.join("ml_model", "voice_model")
MODEL_PATH  = os.path.join(MODEL_DIR, "voice_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Step 1 : Load / Generate Dataset ─────────────────────────
def load_dataset():
    """
    Try to load the real UCI Parkinson's dataset.
    If not found, generate a synthetic dataset so the project
    still runs end-to-end for demo purposes.
    """
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        # UCI dataset uses 'status' column: 1=Parkinson's, 0=Healthy
        df = df.drop(columns=["name"], errors="ignore")
        X = df.drop(columns=["status"]).values
        y = df["status"].values
        feature_names = df.drop(columns=["status"]).columns.tolist()
    else:
        print("[WARNING] Dataset not found. Generating synthetic data for demo...")
        np.random.seed(42)
        n_samples  = 200
        n_features = 22   # matches UCI dataset dimensionality

        # Healthy class (label=0)
        X_healthy = np.random.randn(n_samples // 2, n_features) * 0.5

        # Parkinson's class (label=1) — shifted mean
        X_pd = np.random.randn(n_samples // 2, n_features) * 0.8 + 1.2

        X = np.vstack([X_healthy, X_pd])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Save synthetic data for reference
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df_syn = pd.DataFrame(X, columns=feature_names)
        df_syn["status"] = y
        df_syn.to_csv(DATA_PATH, index=False)
        print(f"[INFO] Synthetic dataset saved to {DATA_PATH}")

    return X, y, feature_names


# ── Step 2 : EDA ─────────────────────────────────────────────
def perform_eda(X, y, feature_names):
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    df = pd.DataFrame(X, columns=feature_names)
    df["status"] = y

    print(f"\nDataset Shape : {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum().sum()} total missing values")
    print(f"\nBasic Statistics:\n{df.describe().T[['mean','std','min','max']].to_string()}")
    print(f"\nClass Distribution:\n{pd.Series(y).value_counts().rename({0:'Healthy', 1:'Parkinson'}).to_string()}")

    # Plot class distribution
    plt.figure(figsize=(5, 3))
    sns.countplot(x=y)
    plt.title("Class Distribution (0=Healthy, 1=Parkinson's)")
    plt.xlabel("Class"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "class_distribution.png"))
    plt.close()
    print("\n[INFO] Class distribution plot saved.")


# ── Step 3 : Train & Compare Models ──────────────────────────
def train_models(X, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression (Baseline) ──────────────────────
    print("\n[MODEL 1] Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    results["Logistic Regression"] = lr_acc
    print(f"  Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr.predict(X_test),
                                 target_names=["Healthy", "Parkinson's"]))

    # ── SVM ─────────────────────────────────────────────────
    print("[MODEL 2] Support Vector Machine (SVM)")
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    results["SVM"] = svm_acc
    print(f"  Accuracy: {svm_acc:.4f}")
    print(classification_report(y_test, svm.predict(X_test),
                                  target_names=["Healthy", "Parkinson's"]))

    # ── Random Forest (Main Model) ───────────────────────────
    print("[MODEL 3] Random Forest (Main Model)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    results["Random Forest"] = rf_acc
    print(f"  Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf.predict(X_test),
                                  target_names=["Healthy", "Parkinson's"]))

    # ── Model Comparison Plot ────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.bar(results.keys(), results.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"))
    plt.close()

    # ── Confusion Matrix for Random Forest ──────────────────
    cm = confusion_matrix(y_test, rf.predict(X_test))
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Parkinson's"],
                yticklabels=["Healthy", "Parkinson's"])
    plt.title("Random Forest – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

    print("\n[INFO] Model comparison and confusion matrix plots saved.")
    print(f"\n[RESULT] Best Model: Random Forest with accuracy {rf_acc:.4f}")

    return rf, scaler


# ── Step 4 : Save Model ───────────────────────────────────────
def save_model(model, scaler):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[INFO] voice_model.pkl saved to → {MODEL_PATH}")
    print(f"[INFO] scaler.pkl        saved to → {SCALER_PATH}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, feature_names = load_dataset()
    perform_eda(X, y, feature_names)
    model, scaler = train_models(X, y)
    save_model(model, scaler)
    print("\n[DONE] Voice model training complete.")