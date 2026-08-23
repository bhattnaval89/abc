"""
data_loader.py
--------------
This file is responsible for two things:

1. Generating a REALISTIC SYNTHETIC dataset (facilities + thermal detections)
   that behaves like real NASA FIRMS + OpenStreetMap data would, but without
   needing any internet connection or API key.

2. Loading the data from local CSV files. If the CSV files do not already
   exist in the `data/` folder, this script will create them automatically.

Why synthetic data?
--------------------
Real FIRMS/OSM integration requires API keys, internet access, and careful
handling of rate limits. Since this is a research PROTOTYPE meant to validate
an idea (not a production system), we first prove the concept using
synthetic data that mimics the real schema. Section 13 of the project
explains how to swap this out for real data later.

Everything in this file is heavily commented because it is meant to be
readable by a beginner Data Science student.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# We simulate a small industrial belt (roughly the size of a real industrial
# city/region). These coordinates are loosely inspired by an Indian steel
# city, but they are NOT meant to represent any real facility's exact
# location - they are just a realistic small bounding box for the demo.
REGION_CENTER_LAT = 22.80
REGION_CENTER_LON = 86.20
REGION_SPREAD_DEG = 0.15  # roughly a 15-20 km wide region

# Reproducibility - very important so the demo behaves the same way every run.
RANDOM_SEED = 42

# How many days of "historical" FIRMS detections we simulate.
HISTORY_DAYS = 90

# Facility type -> "normal" FRP (Fire Radiative Power, in MW) behaviour.
# These ranges are illustrative, loosely based on the general idea that
# continuous, high-temperature industrial processes (steel, refineries,
# thermal power) produce stronger and more frequent thermal signatures than
# lighter industries (generic manufacturing, waste handling).
FACILITY_PROFILES = {
    "Steel Plant":              {"frp_range": (15, 30), "detections_per_day": 0.9, "brightness_range": (330, 360)},
    "Refinery":                  {"frp_range": (10, 22), "detections_per_day": 0.8, "brightness_range": (325, 350)},
    "Thermal Power Plant":       {"frp_range": (12, 25), "detections_per_day": 0.7, "brightness_range": (320, 345)},
    "Cement Plant":              {"frp_range": (6, 14),  "detections_per_day": 0.5, "brightness_range": (315, 335)},
    "Chemical Plant":            {"frp_range": (5, 12),  "detections_per_day": 0.3, "brightness_range": (310, 330)},
    "Waste Facility":            {"frp_range": (2, 8),   "detections_per_day": 0.15, "brightness_range": (305, 325)},
    "Generic Manufacturing":     {"frp_range": (1, 5),   "detections_per_day": 0.08, "brightness_range": (300, 320)},
}

FACILITY_TYPES = list(FACILITY_PROFILES.keys())

# Satellite/source names, similar to what real FIRMS data contains.
SATELLITES = ["VIIRS_SNPP", "VIIRS_NOAA20", "MODIS_Terra", "MODIS_Aqua"]


def _make_facility_names(rng, n_facilities):
    """
    Generates plausible facility names by combining a place-style prefix
    with the facility type. Purely cosmetic - makes the demo feel realistic.
    """
    prefixes = [
        "Shanti", "Bharat", "Ganga", "Suvarna", "Utkal", "Vindhya", "Prabha",
        "Nilgiri", "Kaveri", "Meghna", "Surya", "Vayu", "Agni", "Prithvi",
        "Sindhu", "Himalaya", "Adarsh", "Kanchan", "Rudra", "Vishwa",
        "Tarang", "Amber", "Kalinga", "Shakti", "Mangal", "Sagar", "Vajra",
        "Nirmal", "Dakshin", "Uttar",
    ]
    used = rng.choice(prefixes, size=n_facilities, replace=False if n_facilities <= len(prefixes) else True)
    return used


def generate_synthetic_facilities(n_facilities=30, seed=RANDOM_SEED):
    """
    Creates the FACILITIES dataframe.

    Columns:
        facility_id, facility_name, facility_type, latitude, longitude

    We deliberately spread facility types unevenly - in reality there are
    usually more "generic manufacturing" units than mega steel plants.
    """
    rng = np.random.default_rng(seed)

    # Weighted distribution of facility types (must sum to 1.0)
    type_weights = {
        "Steel Plant": 0.10,
        "Refinery": 0.07,
        "Thermal Power Plant": 0.10,
        "Cement Plant": 0.13,
        "Chemical Plant": 0.15,
        "Waste Facility": 0.15,
        "Generic Manufacturing": 0.30,
    }
    types = rng.choice(
        list(type_weights.keys()),
        size=n_facilities,
        p=list(type_weights.values()),
    )

    names = _make_facility_names(rng, n_facilities)

    facilities = []
    for i in range(n_facilities):
        # Random position inside the region bounding box.
        lat = REGION_CENTER_LAT + rng.uniform(-REGION_SPREAD_DEG, REGION_SPREAD_DEG)
        lon = REGION_CENTER_LON + rng.uniform(-REGION_SPREAD_DEG, REGION_SPREAD_DEG)

        facility_type = types[i]
        facility_name = f"{names[i]} {facility_type}"

        facilities.append({
            "facility_id": f"FAC_{i+1:03d}",
            "facility_name": facility_name,
            "facility_type": facility_type,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
        })

    return pd.DataFrame(facilities)


def generate_synthetic_detections(facilities_df, history_days=HISTORY_DAYS, seed=RANDOM_SEED):
    """
    Creates the FIRMS-style thermal DETECTIONS dataframe for every facility,
    based on that facility's type-specific "normal" behaviour profile.

    Columns:
        detection_id, facility_id, timestamp, latitude, longitude,
        frp, brightness_temperature, confidence, satellite

    Key realism ingredients:
    - Detections occur roughly `detections_per_day` times per day on average
      (modelled with a Poisson-ish process per day), so busier facilities
      naturally get more rows.
    - FRP values are drawn from the facility type's normal range, with some
      day-to-day noise.
    - Detection locations are jittered slightly around the facility's exact
      coordinates, because satellite thermal pixels are not pinpoint-precise
      (real VIIRS pixels are ~375m, MODIS ~1km).
    - We inject an ARTIFICIAL ANOMALY into one or two facilities (the last
      few days show a sudden, sustained FRP spike). This is what our anomaly
      detection system is supposed to catch.
    """
    rng = np.random.default_rng(seed + 1)  # different seed stream from facilities

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=history_days)
    all_days = pd.date_range(start_date, end_date, freq="D")

    # Choose which facilities will contain an injected anomaly.
    # We specifically prefer Chemical Plant / Steel Plant type facilities
    # since the problem statement's example anomaly is a Chemical Plant.
    candidate_ids = facilities_df[facilities_df["facility_type"].isin(
        ["Chemical Plant", "Steel Plant"]
    )]["facility_id"].tolist()
    n_anomalous = min(2, len(candidate_ids))
    anomalous_facility_ids = list(rng.choice(candidate_ids, size=n_anomalous, replace=False)) if candidate_ids else []

    # The anomaly appears in the final few days of the history window, so the
    # UI can clearly show "recent behaviour looks different from history".
    anomaly_start_day_index = len(all_days) - 5  # last 5 days of the dataset

    detection_rows = []
    detection_counter = 0

    for _, facility in facilities_df.iterrows():
        profile = FACILITY_PROFILES[facility["facility_type"]]
        base_lat, base_lon = facility["latitude"], facility["longitude"]
        is_anomalous_facility = facility["facility_id"] in anomalous_facility_ids

        for day_index, day in enumerate(all_days):
            in_anomaly_window = is_anomalous_facility and day_index >= anomaly_start_day_index

            # --- How many detections today? ---
            expected_detections = profile["detections_per_day"]
            if in_anomaly_window:
                # During the anomaly window, the facility "lights up" far more
                # often than normal (e.g. a runaway process, equipment fire).
                expected_detections *= 4
            n_today = rng.poisson(lam=expected_detections)

            for _ in range(n_today):
                # --- FRP value ---
                low, high = profile["frp_range"]
                normal_frp = rng.uniform(low, high)
                if in_anomaly_window:
                    # Sudden spike: several times the normal maximum.
                    frp = rng.uniform(high * 3.0, high * 6.0)
                else:
                    # Small random noise around the normal range.
                    frp = normal_frp * rng.normal(1.0, 0.08)
                frp = max(frp, 0.1)

                # --- Brightness temperature (Kelvin) ---
                b_low, b_high = profile["brightness_range"]
                brightness = rng.uniform(b_low, b_high)
                if in_anomaly_window:
                    brightness += rng.uniform(15, 35)  # hotter than usual

                # --- Confidence (FIRMS reports low/nominal/high, we use %) ---
                confidence = int(np.clip(rng.normal(78, 12), 30, 100))

                # --- Location jitter (simulates satellite pixel imprecision) ---
                jitter_deg = rng.normal(0, 0.003)  # roughly up to a few hundred metres
                det_lat = base_lat + jitter_deg
                det_lon = base_lon + rng.normal(0, 0.003)

                # --- Timestamp: random time during the day ---
                timestamp = day + pd.Timedelta(
                    hours=int(rng.integers(0, 24)),
                    minutes=int(rng.integers(0, 60)),
                )

                detection_counter += 1
                detection_rows.append({
                    "detection_id": f"DET_{detection_counter:06d}",
                    "facility_id": facility["facility_id"],
                    "timestamp": timestamp,
                    "latitude": round(det_lat, 5),
                    "longitude": round(det_lon, 5),
                    "frp": round(frp, 2),
                    "brightness_temperature": round(brightness, 2),
                    "confidence": confidence,
                    "satellite": rng.choice(SATELLITES),
                })

    detections_df = pd.DataFrame(detection_rows)
    detections_df = detections_df.sort_values("timestamp").reset_index(drop=True)
    return detections_df


def load_or_generate_data(data_dir="data", n_facilities=30):
    """
    Main entry point used by the Streamlit app.

    If facilities.csv / firms_detections.csv already exist in `data_dir`,
    they are loaded as-is. Otherwise, both files are generated fresh and
    saved to disk so the next run is instant and reproducible.
    """
    os.makedirs(data_dir, exist_ok=True)
    facilities_path = os.path.join(data_dir, "facilities.csv")
    detections_path = os.path.join(data_dir, "firms_detections.csv")

    if os.path.exists(facilities_path) and os.path.exists(detections_path):
        facilities_df = pd.read_csv(facilities_path)
        detections_df = pd.read_csv(detections_path, parse_dates=["timestamp"])
        return facilities_df, detections_df

    facilities_df = generate_synthetic_facilities(n_facilities=n_facilities)
    detections_df = generate_synthetic_detections(facilities_df)

    facilities_df.to_csv(facilities_path, index=False)
    detections_df.to_csv(detections_path, index=False)

    return facilities_df, detections_df


if __name__ == "__main__":
    # Allows running `python src/data_loader.py` directly to (re)generate data.
    fac, det = load_or_generate_data()
    print(f"Facilities generated: {len(fac)}")
    print(f"Detections generated: {len(det)}")
    print(fac.head())
    print(det.head())
