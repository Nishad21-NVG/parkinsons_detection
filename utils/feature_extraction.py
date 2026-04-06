# ============================================================
# feature_extraction.py
# Utility functions for extracting features from all modalities
# ============================================================

import numpy as np
import cv2
from PIL import Image
import os


# ─────────────────────────────────────────
# VOICE FEATURE EXTRACTION
# ─────────────────────────────────────────

def extract_voice_features_from_file(wav_path):
    """
    Extract exactly 22 acoustic features from a .wav file.
    Matches UCI Parkinson's dataset dimensionality.
    """
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=None)

        # MFCC features (mean of 13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # 13 features

        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y))

        # Spectral centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Spectral bandwidth
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        # Spectral rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # Spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        # Chroma mean
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

        # Combine → exactly 22 features
        # 13 (mfcc) + 1 (zcr) + 1 (rms) + 1 (centroid) +
        # 1 (bandwidth) + 1 (rolloff) + 1 (contrast) + 1 (chroma)
        # + 2 (tempo + spectral_flatness) = 22
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        features = np.concatenate([
            mfcc_mean,                    # 13
            [zcr, rms, centroid,          # 3
             bandwidth, rolloff,           # 2
             contrast, chroma,             # 2
             float(tempo), flatness]       # 2
        ])  # Total = 22

        return features.reshape(1, -1)

    except Exception as e:
        print(f"[ERROR] Voice feature extraction failed: {e}")
        return None

def extract_voice_features_from_csv_row(row):
    """
    If user uploads numeric voice features (UCI-style dataset row),
    return them as a numpy array directly.
    row: list or numpy array of numeric features
    """
    return np.array(row).reshape(1, -1)


# ─────────────────────────────────────────
# IMAGE FEATURE EXTRACTION / PREPROCESSING
# ─────────────────────────────────────────

def preprocess_spiral_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess a spiral drawing image for CNN input.
    Returns a 4D numpy array: (1, H, W, 1) for grayscale CNN.
    """
    try:
        img = Image.open(image_path).convert("L")   # grayscale
        img = img.resize(target_size)
        arr = np.array(img) / 255.0                  # normalize [0,1]
        arr = arr.reshape(1, target_size[0], target_size[1], 1)
        return arr
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return None


# ─────────────────────────────────────────
# VIDEO FEATURE EXTRACTION
# ─────────────────────────────────────────

def extract_video_features(video_path, max_frames=30):
    """
    Extract motion/tremor features from a video using OpenCV.
    Steps:
      1. Read frames (up to max_frames)
      2. Convert to grayscale
      3. Compute frame-to-frame absolute difference (motion)
      4. Compute mean and std of motion per frame → feature vector
    Returns a 1D numpy array of features.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[ERROR] Cannot open video file.")
            return None

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        if len(frames) < 2:
            print("[ERROR] Not enough frames in video.")
            return None

        # Compute frame differences (motion magnitude)
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            diffs.append(diff)

        diffs = np.array(diffs, dtype=np.float32)

        # Feature: mean motion, std motion, max motion per frame
        mean_motion = np.mean(diffs, axis=(1, 2))   # shape: (n_frames-1,)
        std_motion  = np.std(diffs, axis=(1, 2))

        # Aggregate into a fixed-length feature vector
        features = np.array([
            np.mean(mean_motion),
            np.std(mean_motion),
            np.max(mean_motion),
            np.min(mean_motion),
            np.mean(std_motion),
            np.std(std_motion),
            np.max(std_motion),
            np.percentile(mean_motion, 25),
            np.percentile(mean_motion, 75),
            np.percentile(mean_motion, 75) - np.percentile(mean_motion, 25),  # IQR
        ])

        return features.reshape(1, -1)

    except Exception as e:
        print(f"[ERROR] Video feature extraction failed: {e}")
        return None