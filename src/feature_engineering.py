"""
feature_engineering.py
-----------------------
This module builds the FACILITY-LEVEL HISTORICAL THERMAL PROFILE.

The central idea of this whole prototype is:

    "What is NORMAL for THIS SPECIFIC facility?"

A steel plant that shows 25 MW of FRP every day is behaving completely
normally. A small manufacturing unit that suddenly shows 25 MW is behaving
very abnormally. So instead of a single global threshold, we compute a
per-facility historical baseline and compare each facility's RECENT activity
against ITS OWN history.

We split each facility's detections into:
    - "historical" window  (everything except the last N days)
    - "recent" window      (the last N days - this is what we evaluate)

This mirrors how you'd operate in a real system: you have a training/baseline
period, and you monitor the newest incoming data against it.
"""

import numpy as np
import pandas as pd

# How many of the most recent days count as "current / recent" activity
# that we want to evaluate for anomalies. Everything before that is treated
# as the historical baseline.
RECENT_WINDOW_DAYS = 5


def build_facility_profiles(matched_detections_df, facilities_df,
                             recent_window_days=RECENT_WINDOW_DAYS):
    """
    Computes a historical thermal profile + recent activity snapshot for
    every facility.

    Parameters
    ----------
    matched_detections_df : pd.DataFrame
        Output of spatial_matching.match_detections_to_facilities(). Must
        contain: matched_facility_id, timestamp, frp, brightness_temperature,
        distance_to_facility_m.
    facilities_df : pd.DataFrame
        The facilities table (facility_id, facility_name, facility_type, ...).
    recent_window_days : int
        Number of most-recent days considered "current activity".

    Returns
    -------
    pd.DataFrame
        One row per facility, with historical baseline stats AND recent
        activity stats, ready to be fed into anomaly detection.
    """

    df = matched_detections_df.copy()
    df = df.dropna(subset=["matched_facility_id"])  # ignore unmatched/stray detections
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df.empty:
        return pd.DataFrame()

    max_date = df["timestamp"].max()
    cutoff = max_date - pd.Timedelta(days=recent_window_days)

    historical_df = df[df["timestamp"] < cutoff]
    recent_df = df[df["timestamp"] >= cutoff]

    total_history_days = max(
        (cutoff - df["timestamp"].min()).days, 1
    )

    profiles = []

    for _, facility in facilities_df.iterrows():
        fid = facility["facility_id"]

        hist = historical_df[historical_df["matched_facility_id"] == fid]
        recent = recent_df[recent_df["matched_facility_id"] == fid]

        # --- Historical baseline stats ---
        total_detections_hist = len(hist)
        detections_per_day_hist = total_detections_hist / total_history_days

        if total_detections_hist > 0:
            mean_frp_hist = hist["frp"].mean()
            median_frp_hist = hist["frp"].median()
            max_frp_hist = hist["frp"].max()
            # Use a small floor on std so we never divide by zero later.
            std_frp_hist = max(hist["frp"].std(ddof=0), 0.5)
            mean_brightness_hist = hist["brightness_temperature"].mean()
        else:
            # Facility had NO historical detections at all (e.g. a very
            # "quiet" facility). We still need safe defaults.
            mean_frp_hist = 0.0
            median_frp_hist = 0.0
            max_frp_hist = 0.0
            std_frp_hist = 1.0
            mean_brightness_hist = 0.0

        # Persistence: fraction of historical days that had at least one
        # detection. High persistence = facility is "always warm" (e.g. a
        # refinery running 24/7). Low persistence = intermittent activity.
        if total_detections_hist > 0:
            days_with_detection = hist["timestamp"].dt.date.nunique()
            persistence_hist = days_with_detection / total_history_days
        else:
            persistence_hist = 0.0

        # --- Recent / current activity stats ---
        total_detections_recent = len(recent)
        detections_per_day_recent = total_detections_recent / max(recent_window_days, 1)

        if total_detections_recent > 0:
            mean_frp_recent = recent["frp"].mean()
            max_frp_recent = recent["frp"].max()
            mean_brightness_recent = recent["brightness_temperature"].mean()
            mean_distance_recent = recent["distance_to_facility_m"].mean()
            latest_timestamp = recent["timestamp"].max()
            latest_frp = recent.sort_values("timestamp").iloc[-1]["frp"]
            latest_distance = recent.sort_values("timestamp").iloc[-1]["distance_to_facility_m"]
        else:
            mean_frp_recent = 0.0
            max_frp_recent = 0.0
            mean_brightness_recent = 0.0
            mean_distance_recent = 0.0
            latest_timestamp = pd.NaT
            latest_frp = 0.0
            latest_distance = 0.0

        days_with_detection_recent = recent["timestamp"].dt.date.nunique() if total_detections_recent > 0 else 0
        persistence_recent = days_with_detection_recent / max(recent_window_days, 1)

        profiles.append({
            "facility_id": fid,
            "facility_name": facility["facility_name"],
            "facility_type": facility["facility_type"],
            "latitude": facility["latitude"],
            "longitude": facility["longitude"],

            # historical baseline
            "hist_total_detections": total_detections_hist,
            "hist_detections_per_day": detections_per_day_hist,
            "hist_mean_frp": mean_frp_hist,
            "hist_median_frp": median_frp_hist,
            "hist_max_frp": max_frp_hist,
            "hist_std_frp": std_frp_hist,
            "hist_mean_brightness": mean_brightness_hist,
            "hist_persistence": persistence_hist,

            # recent / current activity
            "recent_total_detections": total_detections_recent,
            "recent_detections_per_day": detections_per_day_recent,
            "recent_mean_frp": mean_frp_recent,
            "recent_max_frp": max_frp_recent,
            "recent_mean_brightness": mean_brightness_recent,
            "recent_persistence": persistence_recent,
            "recent_mean_distance_m": mean_distance_recent,
            "latest_timestamp": latest_timestamp,
            "latest_frp": latest_frp,
            "latest_distance_m": latest_distance,
        })

    return pd.DataFrame(profiles)


if __name__ == "__main__":
    # Small self-test chaining the whole pipeline so far.
    from data_loader import load_or_generate_data
    from spatial_matching import match_detections_to_facilities

    facilities, detections = load_or_generate_data()
    matched = match_detections_to_facilities(detections, facilities)
    profiles = build_facility_profiles(matched, facilities)
    print(profiles[[
        "facility_name", "facility_type", "hist_mean_frp", "hist_std_frp",
        "recent_mean_frp", "recent_max_frp", "recent_persistence"
    ]].sort_values("recent_max_frp", ascending=False).head(10))
