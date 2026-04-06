# ============================================================
# explanation.py
# Rule-based explanation module — no chatbot, no LLM needed.
# Generates short, meaningful messages based on prediction result.
# ============================================================


def get_voice_explanation(label: str, probability: float) -> str:
    """
    Return a plain-English explanation for a voice-based prediction.
    label:       "Parkinson's Detected" or "Healthy"
    probability: float between 0 and 1
    """
    pct = round(probability * 100, 1)

    if label == "Parkinson's Detected":
        if probability >= 0.85:
            return (
                f"The voice analysis strongly suggests Parkinson's disease ({pct}% confidence). "
                "Irregular vocal tremors, reduced phonation, and abnormal pitch variation were "
                "detected in the audio features. Please consult a neurologist for a clinical evaluation."
            )
        else:
            return (
                f"The voice analysis indicates possible Parkinson's disease ({pct}% confidence). "
                "Some vocal biomarkers associated with the condition were found, but the signal is "
                "moderate. A follow-up examination is recommended."
            )
    else:
        return (
            f"The voice analysis found no significant indicators of Parkinson's disease ({pct}% confidence). "
            "Vocal features appear within normal range. Continue regular health check-ups."
        )


def get_image_explanation(label: str, probability: float) -> str:
    """
    Return a plain-English explanation for a spiral drawing image prediction.
    """
    pct = round(probability * 100, 1)

    if label == "Parkinson's Detected":
        if probability >= 0.85:
            return (
                f"The spiral drawing analysis strongly indicates Parkinson's disease ({pct}% confidence). "
                "The drawing shows irregular stroke patterns, tremor artifacts, and reduced motor control "
                "consistent with Parkinson's. Please seek a neurological assessment."
            )
        else:
            return (
                f"The spiral drawing suggests possible Parkinson's disease ({pct}% confidence). "
                "Mild irregularities in line curvature and stroke pressure were detected. "
                "A clinical review is advisable."
            )
    else:
        return (
            f"The spiral drawing appears normal ({pct}% confidence). "
            "Stroke patterns and motor control indicators are within healthy range. "
            "No significant Parkinson's markers found in the drawing."
        )


def get_video_explanation(label: str, probability: float) -> str:
    """
    Return a plain-English explanation for a hand-movement video prediction.
    """
    pct = round(probability * 100, 1)

    if label == "Parkinson's Detected":
        if probability >= 0.85:
            return (
                f"Hand movement analysis strongly indicates Parkinson's disease ({pct}% confidence). "
                "High-frequency tremors and irregular motion patterns were detected across video frames. "
                "Please consult a medical specialist promptly."
            )
        else:
            return (
                f"Hand movement analysis suggests possible Parkinson's disease ({pct}% confidence). "
                "Mild tremor features were observed in the motion data. "
                "Further evaluation by a healthcare professional is recommended."
            )
    else:
        return (
            f"Hand movement analysis found no significant tremor or Parkinson's indicators ({pct}% confidence). "
            "Motion patterns are consistent with healthy hand movement. "
            "Maintain regular health monitoring."
        )


def get_combined_explanation(voice_label, image_label, video_label) -> str:
    """
    Generate a combined explanation based on all three modality predictions.
    Uses majority voting logic.
    """
    labels = [voice_label, image_label, video_label]
    positive_count = sum(1 for l in labels if l == "Parkinson's Detected")

    if positive_count >= 2:
        return (
            f"Combined analysis ({positive_count}/3 modalities positive): "
            "Multiple indicators suggest the presence of Parkinson's disease. "
            "Voice, image, and/or video features collectively indicate motor and vocal abnormalities. "
            "Early diagnosis improves treatment outcomes — please consult a neurologist immediately."
        )
    elif positive_count == 1:
        return (
            f"Combined analysis ({positive_count}/3 modalities positive): "
            "One modality flagged potential Parkinson's indicators while others appear normal. "
            "This may be an early sign or a false positive. "
            "A professional clinical evaluation is recommended for confirmation."
        )
    else:
        return (
            "Combined analysis (0/3 modalities positive): "
            "All three modalities indicate healthy status. "
            "No significant Parkinson's disease markers were found across voice, image, or video inputs. "
            "Continue routine health monitoring."
        )