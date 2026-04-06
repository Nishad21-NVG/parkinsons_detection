# ============================================================
# app.py  –  Flask Backend
# Endpoints:
#   POST /predict-voice
#   POST /predict-image
#   POST /predict-video
# ============================================================

import os
import sys
import tempfile
import numpy as np
import joblib
from PIL import Image as PILImage

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Add project root to path so utils/ is importable ────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.feature_extraction import (
    extract_voice_features_from_file,
    preprocess_spiral_image,
    extract_video_features,
)
from utils.explanation import (
    get_voice_explanation,
    get_image_explanation,
    get_video_explanation,
    get_combined_explanation,
)

app = Flask(__name__)
CORS(app)

# ── Model Paths ───────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(__file__))
VOICE_MODEL_PATH  = os.path.join(BASE_DIR, "ml_model", "voice_model", "voice_model.pkl")
VOICE_SCALER_PATH = os.path.join(BASE_DIR, "ml_model", "voice_model", "scaler.pkl")
IMAGE_MODEL_PATH  = os.path.join(BASE_DIR, "ml_model", "image_model", "image_model.h5")
VIDEO_MODEL_PATH  = os.path.join(BASE_DIR, "ml_model", "video_model", "video_model.pkl")
VIDEO_SCALER_PATH = os.path.join(BASE_DIR, "ml_model", "video_model", "video_scaler.pkl")

# ── Lazy-load models ─────────────────────────────────────────
_voice_model  = None
_voice_scaler = None
_image_model  = None
_video_model  = None
_video_scaler = None


def load_voice_model():
    global _voice_model, _voice_scaler
    if _voice_model is None:
        _voice_model  = joblib.load(VOICE_MODEL_PATH)
        _voice_scaler = joblib.load(VOICE_SCALER_PATH)
    return _voice_model, _voice_scaler


def load_image_model():
    global _image_model
    if _image_model is None:
        from tensorflow.keras.models import load_model  # type: ignore
        _image_model = load_model(IMAGE_MODEL_PATH)
    return _image_model


def load_video_model():
    global _video_model, _video_scaler
    if _video_model is None:
        _video_model  = joblib.load(VIDEO_MODEL_PATH)
        _video_scaler = joblib.load(VIDEO_SCALER_PATH)
    return _video_model, _video_scaler


# ── Helper ────────────────────────────────────────────────────
def build_response(label, probability, explanation):
    return jsonify({
        "prediction":     label,
        "probability":    round(float(probability), 4),
        "confidence_pct": round(float(probability) * 100, 1),
        "explanation":    explanation,
        "status":         "success"
    })


# ── /predict-voice ────────────────────────────────────────────
@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    try:
        model, scaler = load_voice_model()
        features = None

        if "file" in request.files:
            f = request.files["file"]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                f.save(tmp.name)
                features = extract_voice_features_from_file(tmp.name)
            os.unlink(tmp.name)

        elif "features" in request.form:
            raw  = request.form["features"]
            vals = [float(v) for v in raw.split(",")]
            features = np.array(vals).reshape(1, -1)

        if features is None:
            return jsonify({"status": "error", "message": "No input provided."}), 400

        features_scaled = scaler.transform(features)
        prob_pd         = model.predict_proba(features_scaled)[0][1]
        label           = "Parkinson's Detected" if prob_pd >= 0.5 else "Healthy"
        explanation     = get_voice_explanation(label, prob_pd)

        return build_response(label, prob_pd, explanation)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── /predict-image ────────────────────────────────────────────
@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No image file provided."}), 400

        f   = request.files["file"]
        ext = os.path.splitext(f.filename)[1].lower() or ".png"

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        # Preprocess image directly here (no separate function)
        img = PILImage.open(tmp_path).convert("L")
        img = img.resize((128, 128))
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 128, 128, 1)
        os.unlink(tmp_path)

        # Load model and predict
        import tensorflow as tf
        cnn     = load_image_model()
        pred    = cnn(arr, training=False).numpy()
        prob_pd = float(pred[0][0])
        label   = "Parkinson's Detected" if prob_pd >= 0.5 else "Healthy"
        explanation = get_image_explanation(label, prob_pd)

        return build_response(label, prob_pd, explanation)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── /predict-video ────────────────────────────────────────────
@app.route("/predict-video", methods=["POST"])
def predict_video():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No video file provided."}), 400

        f   = request.files["file"]
        ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            features = extract_video_features(tmp.name)
        os.unlink(tmp.name)

        if features is None:
            return jsonify({"status": "error", "message": "Video feature extraction failed."}), 400

        model, scaler   = load_video_model()
        features_scaled = scaler.transform(features)
        prob_pd         = model.predict_proba(features_scaled)[0][1]
        label           = "Parkinson's Detected" if prob_pd >= 0.5 else "Healthy"
        explanation     = get_video_explanation(label, prob_pd)

        return build_response(label, prob_pd, explanation)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── /predict-combined ─────────────────────────────────────────
@app.route("/predict-combined", methods=["POST"])
def predict_combined():
    try:
        data        = request.json
        voice_label = data.get("voice_label", "Healthy")
        image_label = data.get("image_label", "Healthy")
        video_label = data.get("video_label", "Healthy")

        labels         = [voice_label, image_label, video_label]
        positive_count = sum(1 for l in labels if l == "Parkinson's Detected")
        final_label    = "Parkinson's Detected" if positive_count >= 2 else "Healthy"
        prob           = positive_count / 3.0
        explanation    = get_combined_explanation(voice_label, image_label, video_label)

        return build_response(final_label, prob, explanation)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Health Check ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Parkinson's Detection API is running."})


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Flask backend on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)