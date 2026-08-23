"""
anomaly_detection.py
---------------------
Implements TWO complementary anomaly-detection methods on top of the
facility profiles built by feature_engineering.py:

METHOD A - Simple statistical anomaly detection (z-score)
    z = (recent_mean_frp - hist_mean_frp) / hist_std_frp

    This directly answers: "how many standard deviations away from this
    facility's own historical normal is its current FRP?" It is simple,
    transparent, and easy to explain to a non-technical audience.

METHOD B - Isolation Forest (scikit-learn)
    A general-purpose, unsupervised anomaly detection model. Instead of
    looking at FRP alone, it considers a small feature vector per facility
    (FRP level, brightness, detection frequency, persistence, distance to
    facility) and flags facilities whose overall combination of features is
    unusual compared to the rest of the population.

    Isolation Forest works by randomly partitioning the feature space;
    anomalies are points that get "isolated" (separated from the rest) in
    fewer random splits than normal points, because they sit in sparse
    regions of the feature space. It requires NO deep learning, NO GPU, and
    is very fast even on small datasets - perfect for this prototype.

We deliberately keep both methods SIMPLE and EXPLAINABLE, as required by the
project brief. No black-box deep learning is used anywhere.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Isolation Forest configuration. `contamination` is our rough prior belief
# about what fraction of facilities might be behaving anomalously at any
# given time - we keep it modest since true industrial anomalies should be
# rare events.
ISOLATION_FOREST_CONTAMINATION = 0.15
RANDOM_STATE = 42


def compute_zscore_anomaly(profiles_df):
    """
    METHOD A: Statistical z-score anomaly detection.

    Adds a 'frp_zscore' column to the dataframe:
        z = (recent_mean_frp - hist_mean_frp) / hist_std_frp

    A large positive z-score means the facility's recent FRP is far ABOVE
    what is historically normal for that specific facility.
    """
    df = profiles_df.copy()
    df["frp_zscore"] = (df["recent_mean_frp"] - df["hist_mean_frp"]) / df["hist_std_frp"]

    # Also compute a z-score based on the single latest observation, which is
    # useful for showing "right now, how extreme is the latest hotspot?"
    df["latest_frp_zscore"] = (df["latest_frp"] - df["hist_mean_frp"]) / df["hist_std_frp"]

    return df


def compute_isolation_forest_anomaly(profiles_df):
    """
    METHOD B: Isolation Forest anomaly detection.

    Features used (all facility-level, all easy to explain):
        - recent_mean_frp        : current heat intensity
        - recent_mean_brightness : current brightness temperature
        - recent_detections_per_day : current detection frequency
        - recent_persistence     : how many recent days had activity
        - recent_mean_distance_m : how far detections are from the facility
                                    (large distances can indicate noisy /
                                    unreliable matches)

    Adds two columns:
        - 'isoforest_anomaly_score' : higher = more anomalous
        - 'isoforest_is_anomaly'    : boolean flag from the model itself
    """
    df = profiles_df.copy()

    feature_cols = [
        "recent_mean_frp",
        "recent_mean_brightness",
        "recent_detections_per_day",
        "recent_persistence",
        "recent_mean_distance_m",
    ]

    # Facilities with literally zero recent detections still need valid
    # (zero-filled) feature rows so the model doesn't crash.
    X = df[feature_cols].fillna(0.0).to_numpy()

    # Isolation Forest needs at least a couple of samples to be meaningful.
    if len(X) < 5:
        df["isoforest_anomaly_score"] = 0.0
        df["isoforest_is_anomaly"] = False
        return df

    model = IsolationForest(
        n_estimators=200,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X)

    # decision_function: higher = more "normal", lower/negative = more
    # anomalous. We flip the sign so that in OUR dataframe, higher always
    # means "more anomalous" (more intuitive for the UI).
    raw_scores = model.decision_function(X)
    df["isoforest_anomaly_score"] = -raw_scores

    # predict(): -1 means anomaly, 1 means normal (scikit-learn convention).
    predictions = model.predict(X)
    df["isoforest_is_anomaly"] = predictions == -1

    return df


def run_anomaly_detection(profiles_df):
    """
    Convenience wrapper that runs BOTH methods and returns a single combined
    dataframe, ready for the classification step.
    """
    df = compute_zscore_anomaly(profiles_df)
    df = compute_isolation_forest_anomaly(df)
    return df


if __name__ == "__main__":
    from data_loader import load_or_generate_data
    from spatial_matching import match_detections_to_facilities
    from feature_engineering import build_facility_profiles

    facilities, detections = load_or_generate_data()
    matched = match_detections_to_facilities(detections, facilities)
    profiles = build_facility_profiles(matched, facilities)
    scored = run_anomaly_detection(profiles)

    print(scored[[
        "facility_name", "facility_type", "frp_zscore",
        "isoforest_anomaly_score", "isoforest_is_anomaly"
    ]].sort_values("frp_zscore", ascending=False).head(10))
