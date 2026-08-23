"""
real_data_loader.py
---------------------
Fetches LIVE data from NASA FIRMS and OpenStreetMap (via the Overpass API)
and reshapes it into the exact same schema that data_loader.py produces
synthetically. This means the rest of the pipeline
(spatial_matching -> feature_engineering -> anomaly_detection -> classification)
does not need to change AT ALL to work with real data.

WHAT YOU NEED
-------------
1. A free NASA FIRMS "MAP_KEY".
   Get one at: https://firms.modaps.eosdis.nasa.gov/api/map_key/
   (You just need a free NASA Earthdata login - no payment, instant.)

2. Nothing extra for OpenStreetMap - the Overpass API is public and free,
   but please be a considerate user (small bounding boxes, don't hammer it
   in a loop).

3. Internet access from wherever you run this (this sandbox's network is
   restricted and cannot reach these two APIs - see note in README).

HOW THE TWO REAL APIS WORK
---------------------------
NASA FIRMS - "Area" API (simplest option, no auth headers, just a URL):

    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west},{south},{east},{north}/{day_range}/{date}

    - SOURCE is a sensor name, e.g. VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, MODIS_NRT
      (NRT = Near Real Time, last ~2 months only). For older data use the
      "_SP" (Standard Processing) suffix instead of "_NRT".
    - day_range: 1-10 days of data ending on `date`.
    - date: YYYY-MM-DD, optional (defaults to today).
    - Returns a plain CSV with columns like: latitude, longitude, frp,
      bright_ti4 (VIIRS) / brightness (MODIS), confidence, acq_date,
      acq_time, satellite, daynight.

OpenStreetMap - Overpass API (a query language over OSM's database):

    POST OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
    body: an Overpass QL query string

    We ask it for nodes/ways tagged as industrial land use, industrial
    buildings, or specific facility types (power plants, works, etc.)
    inside a bounding box, and read back GeoJSON-ish JSON.

Both fetch functions below raise clear exceptions on failure. The single
public entry point `load_real_data()` catches those exceptions and falls
back to the synthetic dataset from data_loader.py, so the app never crashes
just because a live service is unreachable.
"""

import time
import requests
import pandas as pd
import numpy as np

import data_loader  # for the synthetic fallback
from spatial_matching import match_detections_to_facilities

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox}/{day_range}/{date}"
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

# FIRMS sensors to pull. Feel free to trim this list - more sensors means
# more detections but also more request time.
DEFAULT_FIRMS_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]

# Max days of history the free FIRMS Area API allows in one NRT request.
FIRMS_MAX_DAY_RANGE = 10

REQUEST_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# NASA FIRMS
# ---------------------------------------------------------------------------

def fetch_firms_detections(map_key, bbox, day_range=10, date=None, sources=None):
    """
    Fetches thermal hotspot detections from the NASA FIRMS Area API for one
    or more sensors, and combines them into a single dataframe shaped like
    our internal FIRMS schema.

    Parameters
    ----------
    map_key : str
        Your personal FIRMS MAP_KEY (see module docstring for how to get one).
    bbox : tuple(float, float, float, float)
        (west, south, east, north) in decimal degrees.
    day_range : int
        Number of days of history to pull, ending on `date`. Max 10 for the
        free NRT (near-real-time) feed.
    date : str or None
        'YYYY-MM-DD'. If None, FIRMS defaults to the most recent available date.
    sources : list[str] or None
        Which FIRMS sensors to query. Defaults to DEFAULT_FIRMS_SOURCES.

    Returns
    -------
    pd.DataFrame
        Columns: detection_id, timestamp, latitude, longitude, frp,
        brightness_temperature, confidence, satellite.
        (No facility_id yet - that comes from spatial matching.)

    Raises
    ------
    RuntimeError if the MAP_KEY is missing/invalid or every sensor request fails.
    """
    if not map_key:
        raise RuntimeError("No FIRMS MAP_KEY provided. Get a free one at "
                            "https://firms.modaps.eosdis.nasa.gov/api/map_key/")

    day_range = min(day_range, FIRMS_MAX_DAY_RANGE)
    sources = sources or DEFAULT_FIRMS_SOURCES
    bbox_str = ",".join(str(round(v, 4)) for v in bbox)  # west,south,east,north
    date_str = date or ""

    all_frames = []
    errors = []

    for source in sources:
        url = FIRMS_AREA_URL.format(
            map_key=map_key,
            source=source,
            bbox=bbox_str,
            day_range=day_range,
            date=date_str,
        ).rstrip("/")  # trailing slash if date_str is empty

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.RequestException as e:
            errors.append(f"{source}: request failed ({e})")
            continue

        # FIRMS returns a CSV body. If the key is invalid or the area/date
        # is malformed, it typically returns a short plain-text error
        # instead of CSV - detect that so we don't silently produce garbage.
        text = resp.text.strip()
        if not text or text.lower().startswith("invalid") or "error" in text[:200].lower():
            errors.append(f"{source}: unexpected response ({text[:120]!r})")
            continue

        try:
            from io import StringIO
            raw_df = pd.read_csv(StringIO(text))
        except Exception as e:
            errors.append(f"{source}: CSV parse failed ({e})")
            continue

        if raw_df.empty:
            continue  # no detections for this sensor in this window - not an error

        frame = _normalize_firms_frame(raw_df, source)
        all_frames.append(frame)

        time.sleep(0.3)  # be a polite API citizen between requests

    if not all_frames:
        if errors:
            raise RuntimeError("All FIRMS requests failed:\n" + "\n".join(errors))
        # No errors, but also no data anywhere - valid outcome (quiet region/date).
        return pd.DataFrame(columns=[
            "detection_id", "timestamp", "latitude", "longitude",
            "frp", "brightness_temperature", "confidence", "satellite",
        ])

    combined = pd.concat(all_frames, ignore_index=True)
    combined["detection_id"] = [f"DET_{i+1:06d}" for i in range(len(combined))]
    return combined


def _normalize_firms_frame(raw_df, source):
    """
    FIRMS' CSV column names differ slightly between VIIRS and MODIS. This
    helper maps either format onto our internal, consistent schema:
        timestamp, latitude, longitude, frp, brightness_temperature,
        confidence, satellite
    """
    df = raw_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Brightness temperature column differs by sensor ---
    if "bright_ti4" in df.columns:          # VIIRS
        brightness_col = "bright_ti4"
    elif "brightness" in df.columns:        # MODIS
        brightness_col = "brightness"
    else:
        brightness_col = None

    # --- Confidence differs: VIIRS uses letters (l/n/h), MODIS uses 0-100 ---
    confidence_map = {"l": 30, "n": 60, "h": 90, "low": 30, "nominal": 60, "high": 90}
    if "confidence" in df.columns:
        conf_raw = df["confidence"]
        # VIIRS uses letter codes (l/n/h); MODIS uses numeric 0-100. Try the
        # letter mapping first (works regardless of the exact pandas string
        # dtype), and only fall back to numeric parsing for values that
        # aren't recognised letter codes.
        as_letters = conf_raw.astype(str).str.strip().str.lower().map(confidence_map)
        as_numeric = pd.to_numeric(conf_raw, errors="coerce")
        confidence = as_letters.fillna(as_numeric).fillna(50)
    else:
        confidence = 50

    # --- Timestamp: combine acq_date + acq_time (HHMM, sometimes zero-padded) ---
    acq_date = df.get("acq_date")
    acq_time = df.get("acq_time")
    if acq_date is not None and acq_time is not None:
        time_str = acq_time.astype(str).str.zfill(4)
        timestamp = pd.to_datetime(
            acq_date.astype(str) + " " + time_str.str[:2] + ":" + time_str.str[2:],
            errors="coerce",
        )
    else:
        timestamp = pd.NaT

    normalized = pd.DataFrame({
        "timestamp": timestamp,
        "latitude": pd.to_numeric(df.get("latitude"), errors="coerce"),
        "longitude": pd.to_numeric(df.get("longitude"), errors="coerce"),
        "frp": pd.to_numeric(df.get("frp"), errors="coerce"),
        "brightness_temperature": pd.to_numeric(df.get(brightness_col), errors="coerce") if brightness_col else np.nan,
        "confidence": confidence,
        "satellite": df.get("satellite", source).fillna(source) if "satellite" in df.columns else source,
    })

    return normalized.dropna(subset=["timestamp", "latitude", "longitude", "frp"])


# ---------------------------------------------------------------------------
# OPENSTREETMAP (Overpass API)
# ---------------------------------------------------------------------------

# Very simple keyword-based mapping from OSM tag values / facility names to
# our internal facility_type categories. Real-world tagging is messy, so
# this is intentionally a best-effort heuristic, not a guarantee.
_TYPE_KEYWORDS = [
    ("Steel Plant", ["steel", "iron", "smelt"]),
    ("Refinery", ["refinery", "petroleum", "oil"]),
    ("Thermal Power Plant", ["power plant", "power_plant", "thermal power", "electricity"]),
    ("Cement Plant", ["cement", "clinker"]),
    ("Chemical Plant", ["chemical", "fertiliser", "fertilizer", "petrochemical"]),
    ("Waste Facility", ["waste", "landfill", "recycl", "incinerat"]),
]


def _guess_facility_type(tags):
    """
    Looks at an OSM element's tags (a dict) and guesses which of our 7
    facility categories it best fits, falling back to "Generic Manufacturing".
    """
    haystack = " ".join(str(v).lower() for v in tags.values())
    for facility_type, keywords in _TYPE_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            return facility_type
    return "Generic Manufacturing"


def fetch_osm_facilities(bbox, timeout=REQUEST_TIMEOUT_SECONDS):
    """
    Queries the Overpass API for industrial facilities inside a bounding box.

    Parameters
    ----------
    bbox : tuple(float, float, float, float)
        (west, south, east, north) in decimal degrees.

    Returns
    -------
    pd.DataFrame
        Columns: facility_id, facility_name, facility_type, latitude, longitude.

    Raises
    ------
    RuntimeError on network failure or an empty/invalid response.
    """
    west, south, east, north = bbox
    # Overpass wants (south,west,north,east) order for its bbox filter.
    overpass_bbox = f"{south},{west},{north},{east}"

    query = f"""
[out:json][timeout:25];
(
  node["landuse"="industrial"]({overpass_bbox});
);
out center tags;
"""

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "industrial-thermal-prototype/1.0"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Overpass API request failed: {e}")

    try:
        payload = resp.json()
    except ValueError as e:
        raise RuntimeError(f"Overpass API returned non-JSON response: {e}")

    elements = payload.get("elements", [])
    if not elements:
        raise RuntimeError(
            "Overpass API returned zero industrial facilities for this bounding "
            "box. Try a larger area, or check the coordinates."
        )

    rows = []
    for i, el in enumerate(elements):
        # Nodes have lat/lon directly; ways/relations have a "center" instead.
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue

        tags = el.get("tags", {})
        name = tags.get("name") or f"Unnamed Facility {i+1}"
        facility_type = _guess_facility_type(tags)

        rows.append({
            "facility_id": f"OSM_{el.get('id', i)}",
            "facility_name": name,
            "facility_type": facility_type,
            "latitude": lat,
            "longitude": lon,
        })

    facilities_df = pd.DataFrame(rows).drop_duplicates(subset=["facility_id"])

    if facilities_df.empty:
        raise RuntimeError("Overpass API results had no usable coordinates.")

    return facilities_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# COMBINED LOADER (with automatic fallback to synthetic data)
# ---------------------------------------------------------------------------

def load_real_data(map_key, bbox, day_range=10, date=None, sources=None,
                    fallback_to_synthetic=True, data_dir="data"):
    """
    The main entry point: fetches live FIRMS + OSM data for a region, spatially
    matches them, and returns (facilities_df, matched_detections_df) in
    EXACTLY the shape the rest of the pipeline expects.

    If anything fails (missing/invalid MAP_KEY, network error, empty
    Overpass result, etc.) and `fallback_to_synthetic=True`, this quietly
    falls back to the synthetic dataset from data_loader.py instead of
    crashing the app. The second element of the returned tuple in that case
    is a string describing what happened (None if real data was used
    successfully) - the caller can show this as a warning banner in the UI.

    Returns
    -------
    facilities_df, detections_df, warning_message (str or None)
    """
    warning_message = None

    try:
        facilities_df = fetch_osm_facilities(bbox)
        raw_detections_df = fetch_firms_detections(
            map_key=map_key, bbox=bbox, day_range=day_range, date=date, sources=sources
        )

        if raw_detections_df.empty:
            raise RuntimeError("FIRMS returned zero thermal detections for this "
                                "area/date range. Try a wider date range or a bigger area.")

        # Real FIRMS detections don't come with a facility_id - that's what
        # our existing spatial matching logic is for.
        detections_df = match_detections_to_facilities(raw_detections_df, facilities_df)

        return facilities_df, detections_df, None

    except Exception as e:
        if not fallback_to_synthetic:
            raise
        warning_message = (
            f"Could not load live FIRMS/OSM data ({e}). "
            f"Falling back to the synthetic demo dataset."
        )
        facilities_df, raw_detections_df = data_loader.load_or_generate_data(data_dir=data_dir)
        detections_df = match_detections_to_facilities(raw_detections_df, facilities_df)
        return facilities_df, detections_df, warning_message


if __name__ == "__main__":
    # Example usage / manual smoke test.
    # The MAP_KEY and bounding box are read from config.py at the project
    # root - edit that file to change them.
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from config import FIRMS_MAP_KEY, BBOX

    facilities, detections, warning = load_real_data(
        map_key=FIRMS_MAP_KEY,
        bbox=BBOX,
        day_range=10,
    )

    if warning:
        print("WARNING:", warning)

    print(f"Facilities: {len(facilities)}")
    print(f"Detections: {len(detections)}")
    print(facilities.head())
    print(detections.head())
