# ============================================================
# train_image_model.py
# Train CNN for spiral drawing classification
# Dataset: Spiral images in data/image/healthy/ and data/image/parkinson/
# Output : ml_model/image_model/image_model.h5
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────
HEALTHY_DIR   = os.path.join("data", "image", "healthy")
PARKINSON_DIR = os.path.join("data", "image", "parkinson")
MODEL_DIR     = os.path.join("ml_model", "image_model")
MODEL_PATH    = os.path.join(MODEL_DIR, "image_model.h5")
IMG_SIZE      = (128, 128)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HEALTHY_DIR, exist_ok=True)
os.makedirs(PARKINSON_DIR, exist_ok=True)


# ── Step 1 : Generate Synthetic Spiral Images (if needed) ────
def draw_spiral(path, noisy=False, n_turns=3):
    """
    Draw a simple Archimedean spiral and save as PNG.
    noisy=True  → simulates Parkinson's tremor (jagged lines)
    noisy=False → smooth spiral (healthy)
    """
    size = 128
    img  = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    prev = None
    for angle_deg in range(0, 360 * n_turns, 3):
        angle = np.radians(angle_deg)
        r     = (angle_deg / (360 * n_turns)) * (size // 2 - 5)

        if noisy:
            r += np.random.uniform(-4, 4)

        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))

        if prev:
            draw.line([prev, (x, y)], fill=0, width=2)
        prev = (x, y)

    img.save(path)


def generate_synthetic_images(n_per_class=100):
    """Generate synthetic spiral images if the data folder is empty."""
    healthy_files   = os.listdir(HEALTHY_DIR)
    parkinson_files = os.listdir(PARKINSON_DIR)

    if len(healthy_files) == 0:
        print(f"[INFO] Generating {n_per_class} healthy spiral images...")
        for i in range(n_per_class):
            draw_spiral(os.path.join(HEALTHY_DIR, f"healthy_{i:03d}.png"), noisy=False)

    if len(parkinson_files) == 0:
        print(f"[INFO] Generating {n_per_class} Parkinson's spiral images...")
        for i in range(n_per_class):
            draw_spiral(os.path.join(PARKINSON_DIR, f"parkinson_{i:03d}.png"), noisy=True)


# ── Step 2 : Load Images ─────────────────────────────────────
def load_images():
    X, y = [], []

    for fname in os.listdir(HEALTHY_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(HEALTHY_DIR, fname)).convert("L")
                img = img.resize(IMG_SIZE)
                X.append(np.array(img) / 255.0)
                y.append(0)  # Healthy
            except Exception:
                pass

    for fname in os.listdir(PARKINSON_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(PARKINSON_DIR, fname)).convert("L")
                img = img.resize(IMG_SIZE)
                X.append(np.array(img) / 255.0)
                y.append(1)  # Parkinson's
            except Exception:
                pass

    X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    y = np.array(y)
    return X, y


# ── Step 3 : EDA ─────────────────────────────────────────────
def perform_eda(X, y):
    print("\n" + "=" * 50)
    print("IMAGE DATASET EDA")
    print("=" * 50)
    print(f"Dataset Shape  : {X.shape}")
    print(f"Pixel Range    : [{X.min():.2f}, {X.max():.2f}]")
    print(f"Class Distribution:")
    print(f"  Healthy      : {(y == 0).sum()}")
    print(f"  Parkinson's  : {(y == 1).sum()}")

    # Show sample images
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].squeeze(), cmap="gray")
        ax.set_title("Healthy" if y[i] == 0 else "Parkinson's")
        ax.axis("off")
    plt.suptitle("Sample Spiral Images")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "sample_images.png"))
    plt.close()
    print("[INFO] Sample images plot saved.")


# ── Step 4 : Build CNN ────────────────────────────────────────
def build_cnn(input_shape=(128, 128, 1)):
    """
    Simple CNN architecture:
    Conv → Pool → Conv → Pool → Flatten → Dense → Output
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")   # Binary output
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ── Step 5 : Train CNN ────────────────────────────────────────
def train_cnn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ── Training Curves ──────────────────────────────────────
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"],     label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("CNN Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("CNN Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "cnn_training_curves.png"))
    plt.close()

    # ── Evaluation ───────────────────────────────────────────
    y_pred_prob = model.predict(X_test).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    acc         = (y_pred == y_test).mean()

    print(f"\n[RESULT] CNN Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Healthy", "Parkinson's"],
                yticklabels=["Healthy", "Parkinson's"])
    plt.title("CNN – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "cnn_confusion_matrix.png"))
    plt.close()

    return model


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_synthetic_images(n_per_class=100)
    X, y = load_images()
    perform_eda(X, y)
    model = train_cnn(X, y)
    model.save(MODEL_PATH)
    print(f"\n[INFO] image_model.h5 saved to → {MODEL_PATH}")
    print("[DONE] Image model training complete.")