"""
classification.py
-------------------
Combines the signals produced by anomaly_detection.py into a simple,
human-readable EVENT CLASSIFICATION:

    GREEN  = Routine thermal activity
    YELLOW = Unusual thermal activity
    RED    = Potential abnormal industrial thermal event

IMPORTANT SCIENTIFIC HONESTY NOTE:
This classification does NOT and CANNOT confirm an actual fire or accident.
It only flags that current thermal behaviour looks statistically different
from what is normal for that specific facility. A human analyst should
always review RED/YELLOW facilities before taking any action.

We deliberately use a simple, transparent RULE-BASED combination of the two
anomaly signals (z-score + Isolation Forest) plus persistence and frequency
change, rather than a second black-box model. This keeps the system fully
explainable, as required by the project brief.

Every classification is accompanied by a short, plain-English explanation
generated directly from the underlying numbers (NOT from an LLM).
"""

import numpy as np
import pandas as pd

# Thresholds used by the rule-based classifier. These are intentionally
# simple and easy to tune / explain.
ZSCORE_YELLOW_THRESHOLD = 2.0
ZSCORE_RED_THRESHOLD = 3.5

FREQUENCY_RATIO_YELLOW = 2.0   # recent detections/day is 2x historical
FREQUENCY_RATIO_RED = 4.0      # recent detections/day is 4x historical


def _frequency_ratio(row):
    """
    Ratio of recent detection frequency to historical detection frequency.
    A ratio of 1.0 means "business as usual". A ratio of 4.0 means the
    facility is being detected 4x more often than its own history suggests.
    We add a small constant to the denominator to avoid divide-by-zero for
    facilities that historically had almost no activity.
    """
    return row["recent_detections_per_day"] / max(row["hist_detections_per_day"], 0.02)


def classify_facility(row):
    """
    Applies simple, explainable rules to a single facility's row (which must
    already contain z-score and Isolation Forest columns) and returns:
        (classification_label, list_of_explanation_strings)
    """
    explanations = []
    freq_ratio = _frequency_ratio(row)

    z = row["frp_zscore"]
    iso_anomaly = bool(row["isoforest_is_anomaly"])
    iso_score = row["isoforest_anomaly_score"]

    # --- Build up plain-language explanations based on actual numbers ---
    if row["recent_total_detections"] == 0:
        explanations.append(
            "No thermal detections recorded in the recent monitoring window."
        )
        return "GREEN", explanations

    if z >= ZSCORE_YELLOW_THRESHOLD:
        explanations.append(
            f"Recent FRP is {z:.1f} standard deviations above this facility's historical mean."
        )

    if freq_ratio >= FREQUENCY_RATIO_YELLOW:
        explanations.append(
            f"Current detection frequency is about {freq_ratio:.1f}x the facility's normal historical frequency."
        )

    if iso_anomaly:
        explanations.append(
            "Isolation Forest flagged this facility's overall thermal pattern (FRP, brightness, "
            "frequency, persistence, distance) as statistically unusual compared to other facilities."
        )

    if row["recent_persistence"] >= 0.8 and row["hist_persistence"] < 0.4:
        explanations.append(
            "Thermal activity has become persistent (detected on most recent days), "
            "which is unusual compared to this facility's historical pattern."
        )

    if row["recent_mean_distance_m"] > 0 and row["recent_mean_distance_m"] < 500:
        explanations.append(
            "Recent thermal detections are spatially concentrated close to the facility, "
            "supporting a genuine facility-level thermal signal rather than a stray detection."
        )

    # --- Decide the final classification level ---
    strong_signal_count = sum([
        z >= ZSCORE_RED_THRESHOLD,
        freq_ratio >= FREQUENCY_RATIO_RED,
        iso_anomaly,
    ])

    moderate_signal_count = sum([
        z >= ZSCORE_YELLOW_THRESHOLD,
        freq_ratio >= FREQUENCY_RATIO_YELLOW,
        iso_anomaly,
    ])

    if strong_signal_count >= 2 or z >= ZSCORE_RED_THRESHOLD:
        label = "RED"
    elif moderate_signal_count >= 1:
        label = "YELLOW"
    else:
        label = "GREEN"

    if label == "GREEN" and not explanations:
        explanations.append(
            "Current thermal activity is consistent with this facility's historical normal behaviour."
        )

    return label, explanations


def classify_all_facilities(scored_profiles_df):
    """
    Applies classify_facility() to every row and adds:
        - 'classification'        : "GREEN" / "YELLOW" / "RED"
        - 'explanation'           : a single combined explanation string
        - 'explanation_list'      : the list form (useful for bullet points in UI)
    """
    df = scored_profiles_df.copy()

    labels = []
    explanation_strings = []
    explanation_lists = []

    for _, row in df.iterrows():
        label, explanations = classify_facility(row)
        labels.append(label)
        explanation_lists.append(explanations)
        explanation_strings.append(" ".join(explanations))

    df["classification"] = labels
    df["explanation"] = explanation_strings
    df["explanation_list"] = explanation_lists

    return df


if __name__ == "__main__":
    from data_loader import load_or_generate_data
    from spatial_matching import match_detections_to_facilities
    from feature_engineering import build_facility_profiles
    from anomaly_detection import run_anomaly_detection

    facilities, detections = load_or_generate_data()
    matched = match_detections_to_facilities(detections, facilities)
    profiles = build_facility_profiles(matched, facilities)
    scored = run_anomaly_detection(profiles)
    classified = classify_all_facilities(scored)

    print(classified[[
        "facility_name", "facility_type", "classification", "explanation"
    ]].sort_values("classification").to_string(index=False))
