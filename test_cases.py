# ============================================================
# test_cases.py
# Complete Test Cases + Edge Cases for Parkinson's Detection
# Run: python test_cases.py
# ============================================================

import os
import sys
import numpy as np
import unittest
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────
# IMPORT PROJECT MODULES
# ─────────────────────────────────────────────────────────────
from utils.explanation import (
    get_voice_explanation,
    get_image_explanation,
    get_video_explanation,
    get_combined_explanation
)

# ─────────────────────────────────────────────────────────────
# TEST SPLIT SUMMARY
# Total Tests : 20
# Train/Test  : 80% training data | 20% test cases
# ─────────────────────────────────────────────────────────────


# =============================================================
# 1. VOICE MODEL TESTS (6 tests)
# =============================================================
class TestVoiceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import joblib
        cls.model  = joblib.load("ml_model/voice_model/voice_model.pkl")
        cls.scaler = joblib.load("ml_model/voice_model/scaler.pkl")
        print("\n[VOICE MODEL TESTS]")

    def predict(self, features):
        arr    = np.array(features).reshape(1, -1)
        scaled = self.scaler.transform(arr)
        prob   = self.model.predict_proba(scaled)[0][1]
        label  = "Parkinson's Detected" if prob >= 0.5 else "Healthy"
        return label, prob

    def test_01_healthy_voice_normal(self):
        """TC01: Normal healthy voice features → Healthy"""
        features = [189.96, 184.86, 169.26, 0.0045, 0.026,
                    0.0028, 0.018, 0.214, 0.0045, 0.028,
                    0.0018, 22.1, 0.31]
        label, prob = self.predict(features)
        print(f"  TC01 Healthy Voice        → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Healthy")

    def test_02_parkinsons_voice_high_jitter(self):
        """TC02: High jitter/shimmer → Parkinson's"""
        features = [145.23, 112.45, 98.34, 0.0089, 0.052,
                    0.0065, 0.048, 0.456, 0.0098, 0.078,
                    0.0054, 14.3, 0.58]
        label, prob = self.predict(features)
        print(f"  TC02 Parkinson's Voice    → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Parkinson's Detected")

    def test_03_healthy_voice_low_jitter(self):
        """TC03: Low jitter, high HNR → Healthy"""
        features = [195.0, 190.0, 175.0, 0.003, 0.019,
                    0.002, 0.015, 0.18, 0.003, 0.022,
                    0.0015, 24.0, 0.28]
        label, prob = self.predict(features)
        print(f"  TC03 Healthy Low Jitter   → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Healthy")

    def test_04_parkinsons_severe(self):
        """TC04: Severe Parkinson's features"""
        features = [120.0, 95.0, 80.0, 0.012, 0.075,
                    0.009, 0.065, 0.62, 0.015, 0.11,
                    0.008, 10.5, 0.72]
        label, prob = self.predict(features)
        print(f"  TC04 Severe Parkinson's   → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Parkinson's Detected")

    def test_05_edge_all_zeros(self):
        """EC01: All zero features — edge case"""
        features = [0.0] * 13
        label, prob = self.predict(features)
        print(f"  EC01 All Zeros            → {label} ({prob*100:.1f}%)")
        self.assertIn(label, ["Healthy", "Parkinson's Detected"])

    def test_06_edge_very_high_values(self):
        """EC02: Extremely high feature values — edge case"""
        features = [999.0, 999.0, 999.0, 0.99, 0.99,
                    0.99, 0.99, 0.99, 0.99, 0.99,
                    0.99, 99.0, 0.99]
        label, prob = self.predict(features)
        print(f"  EC02 Very High Values     → {label} ({prob*100:.1f}%)")
        self.assertIn(label, ["Healthy", "Parkinson's Detected"])
        
# =============================================================
# 2. IMAGE MODEL TESTS (5 tests)
# =============================================================
class TestImageModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load CNN image model once."""
        from tensorflow.keras.models import load_model  # type: ignore
        cls.model = load_model("ml_model/image_model/image_model.h5")
        print("\n[IMAGE MODEL TESTS]")

    def predict_array(self, arr):
        arr    = arr.reshape(1, 128, 128, 1)
        prob   = float(self.model(arr, training=False).numpy()[0][0])
        label  = "Parkinson's Detected" if prob >= 0.5 else "Healthy"
        return label, prob

    # ── 80% Normal Cases ─────────────────────────────────────

    def test_07_smooth_spiral_healthy(self):
        """TC05: Smooth spiral → Healthy"""
        from PIL import Image, ImageDraw
        img  = Image.new("L", (128, 128), 255)
        draw = ImageDraw.Draw(img)
        cx, cy = 64, 64
        prev = None
        for a in range(0, 1080, 3):
            angle = np.radians(a)
            r = (a / 1080) * 55
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if prev:
                draw.line([prev, (x, y)], fill=0, width=2)
            prev = (x, y)
        arr = np.array(img) / 255.0
        label, prob = self.predict_array(arr)
        print(f"  TC05 Smooth Spiral        → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Healthy")

    def test_08_noisy_spiral_parkinsons(self):
        """TC06: Noisy/shaky spiral → Parkinson's"""
        from PIL import Image, ImageDraw
        img  = Image.new("L", (128, 128), 255)
        draw = ImageDraw.Draw(img)
        cx, cy = 64, 64
        prev = None
        np.random.seed(1)
        for a in range(0, 1080, 3):
            angle = np.radians(a)
            r = (a / 1080) * 55 + np.random.uniform(-6, 6)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if prev:
                draw.line([prev, (x, y)], fill=0, width=2)
            prev = (x, y)
        arr = np.array(img) / 255.0
        label, prob = self.predict_array(arr)
        print(f"  TC06 Noisy Spiral         → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Parkinson's Detected")

    def test_09_image_shape_correct(self):
        """TC07: Model accepts correct input shape (1,128,128,1)"""
        arr = np.random.rand(1, 128, 128, 1).astype(np.float32)
        pred = self.model(arr, training=False).numpy()
        print(f"  TC07 Input Shape Check    → Output shape: {pred.shape}")
        self.assertEqual(pred.shape, (1, 1))

    def test_10_image_output_range(self):
        """TC08: Output probability between 0 and 1"""
        arr  = np.random.rand(1, 128, 128, 1).astype(np.float32)
        prob = float(self.model(arr, training=False).numpy()[0][0])
        print(f"  TC08 Output Range         → Prob: {prob:.4f}")
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    # ── 20% Edge Cases ───────────────────────────────────────

    def test_11_edge_blank_white_image(self):
        """EC03: All white image — edge case"""
        arr = np.ones((128, 128), dtype=np.float32)
        label, prob = self.predict_array(arr)
        print(f"  EC03 Blank White Image    → {label} ({prob*100:.1f}%)")
        self.assertIn(label, ["Healthy", "Parkinson's Detected"])


# =============================================================
# 3. VIDEO MODEL TESTS (5 tests)
# =============================================================
class TestVideoModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load video model once."""
        import joblib
        cls.model  = joblib.load("ml_model/video_model/video_model.pkl")
        cls.scaler = joblib.load("ml_model/video_model/video_scaler.pkl")
        print("\n[VIDEO MODEL TESTS]")

    def predict(self, features):
        arr    = np.array(features).reshape(1, -1)
        scaled = self.scaler.transform(arr)
        prob   = self.model.predict_proba(scaled)[0][1]
        label  = "Parkinson's Detected" if prob >= 0.5 else "Healthy"
        return label, prob

    # ── 80% Normal Cases ─────────────────────────────────────

    def test_12_healthy_low_motion(self):
        """TC09: Low motion features → Healthy"""
        features = [2.0, 0.4, 4.0, 0.5, 0.3, 0.1, 0.6, 1.5, 2.5, 1.0]
        label, prob = self.predict(features)
        print(f"  TC09 Healthy Low Motion   → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Healthy")

    def test_13_parkinsons_high_tremor(self):
        """TC10: High tremor features → Parkinson's"""
        features = [8.0, 3.5, 18.0, 0.2, 2.5, 1.2, 5.5, 5.0, 11.0, 6.0]
        label, prob = self.predict(features)
        print(f"  TC10 Parkinson's Tremor   → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Parkinson's Detected")

    def test_14_healthy_very_smooth(self):
        """TC11: Very smooth motion → Healthy"""
        features = [1.5, 0.2, 2.5, 0.8, 0.2, 0.05, 0.4, 1.2, 1.8, 0.6]
        label, prob = self.predict(features)
        print(f"  TC11 Very Smooth Motion   → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Healthy")

    def test_15_parkinsons_extreme_tremor(self):
        """TC12: Extreme tremor → Parkinson's"""
        features = [12.0, 5.0, 25.0, 0.1, 4.0, 2.0, 8.0, 8.0, 16.0, 8.0]
        label, prob = self.predict(features)
        print(f"  TC12 Extreme Tremor       → {label} ({prob*100:.1f}%)")
        self.assertEqual(label, "Parkinson's Detected")

    # ── 20% Edge Cases ───────────────────────────────────────

    def test_16_edge_zero_motion(self):
        """EC04: Zero motion — no movement at all"""
        features = [0.0] * 10
        label, prob = self.predict(features)
        print(f"  EC04 Zero Motion          → {label} ({prob*100:.1f}%)")
        self.assertIn(label, ["Healthy", "Parkinson's Detected"])


# =============================================================
# 4. EXPLANATION MODULE TESTS (4 tests)
# =============================================================
class TestExplanationModule(unittest.TestCase):

    print("\n[EXPLANATION MODULE TESTS]")

    # ── 80% Normal Cases ─────────────────────────────────────

    def test_17_voice_explanation_pd(self):
        """TC13: Parkinson's voice explanation not empty"""
        msg = get_voice_explanation("Parkinson's Detected", 0.92)
        print(f"  TC13 Voice PD Explanation → {msg[:50]}...")
        self.assertIn("Parkinson", msg)
        self.assertGreater(len(msg), 20)

    def test_18_image_explanation_healthy(self):
        """TC14: Healthy image explanation not empty"""
        msg = get_image_explanation("Healthy", 0.15)
        print(f"  TC14 Image Healthy Expl   → {msg[:50]}...")
        self.assertIn("normal", msg.lower())
        self.assertGreater(len(msg), 20)

    def test_19_combined_majority_pd(self):
        """TC15: 2/3 positive → Parkinson's combined"""
        msg = get_combined_explanation(
            "Parkinson's Detected",
            "Parkinson's Detected",
            "Healthy"
        )
        print(f"  TC15 Combined 2/3 PD      → {msg[:50]}...")
        self.assertIn("2/3", msg)

    # ── 20% Edge Cases ───────────────────────────────────────

    def test_20_combined_all_healthy(self):
        """EC05: All healthy → combined healthy message"""
        msg = get_combined_explanation("Healthy", "Healthy", "Healthy")
        print(f"  EC05 All Healthy Combined → {msg[:50]}...")
        self.assertIn("0/3", msg)


# =============================================================
# MAIN RUNNER
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PARKINSON'S DETECTION — TEST SUITE")
    print("  80% Normal Cases | 20% Edge Cases")
    print("=" * 60)

    # Run all tests
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVoiceModel))
    suite.addTests(loader.loadTestsFromTestCase(TestImageModel))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoModel))
    suite.addTests(loader.loadTestsFromTestCase(TestExplanationModule))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"  TOTAL TESTS  : {result.testsRun}")
    print(f"  PASSED       : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  FAILED       : {len(result.failures)}")
    print(f"  ERRORS       : {len(result.errors)}")
    print("=" * 60)

    if result.wasSuccessful():
        print("  ✅ ALL TESTS PASSED!")
    else:
        print("  ❌ SOME TESTS FAILED!")
        for fail in result.failures:
            print(f"\n  FAILED: {fail[0]}")
            print(f"  {fail[1]}")
    print("=" * 60)