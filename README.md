# Industrial Thermal Intelligence Prototype

A small, working research prototype for the SIH problem statement:

> **"AI-Based Detection and Classification of Industrial Fires and Persistent
> Thermal Sources Using NASA FIRMS, OSM & Satellite Data."**

## The core idea

NASA FIRMS is already very good at detecting **thermal anomalies** from
satellites. But detecting a hotspot is not the same as **understanding what
it means**. A steel plant running at 25 MW of Fire Radiative Power (FRP)
every day is completely normal. The same 25 MW appearing at a small
warehouse would be alarming.

This prototype explores a simple idea:

1. Associate every thermal detection with the nearest **industrial
   facility** (using real spatial matching, not guesswork).
2. Learn each facility's own **historical normal thermal behaviour**.
3. Compare **current** activity to that facility's own baseline — not to a
   single global threshold.
4. Flag facilities whose current behaviour looks statistically unusual, and
   **explain why** in plain language.

The question this MVP is built to answer:

> **"Can we identify an industrial facility whose current thermal behaviour
> is significantly different from its historical normal behaviour?"**

Running the demo and inspecting the two facilities the system flags as
**RED** (out of 30, with two artificially-injected anomalies) demonstrates
that the answer is **yes**.

## Project structure

```
project/
│
├── app.py                      # Streamlit UI - the entry point
├── data/
│   ├── facilities.csv          # generated automatically on first run
│   └── firms_detections.csv    # generated automatically on first run
│
├── src/
│   ├── data_loader.py           # synthetic data generation + loading
│   ├── spatial_matching.py      # GeoPandas/Shapely nearest-facility matching
│   ├── feature_engineering.py   # per-facility historical thermal profile
│   ├── anomaly_detection.py     # z-score + Isolation Forest anomaly scoring
│   └── classification.py        # GREEN / YELLOW / RED rules + explanations
│
├── requirements.txt
└── README.md
```

## How to run it

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv

# On macOS / Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (usually at
`http://localhost:8501`). On the very first run, the app automatically
generates a reproducible synthetic dataset (30 facilities, ~90 days of
thermal detections) into the `data/` folder. Subsequent runs reuse the same
CSV files, so the demo is always identical.

If you want to force fresh data, simply delete the `data/` folder and
re-run the app.

## How the pipeline works

```
FIRMS-style thermal detections (synthetic, FIRMS-schema CSV)
        ↓
OSM-style industrial facilities (synthetic, OSM-schema CSV)
        ↓
Spatial matching (GeoPandas + Shapely, nearest-facility join in a metric CRS)
        ↓
Facility-level historical thermal profile (mean/median/max/std FRP, persistence, frequency)
        ↓
Anomaly detection (statistical z-score AND scikit-learn Isolation Forest)
        ↓
Simple, explainable event classification (GREEN / YELLOW / RED)
        ↓
Interactive Folium map + facility intelligence panel + Plotly time series
```

Each module in `src/` can also be run directly for debugging, e.g.:

```bash
python src/data_loader.py
python src/spatial_matching.py
python src/feature_engineering.py
python src/anomaly_detection.py
python src/classification.py
```

Each of these prints a small preview of its output, which is useful for
understanding the pipeline step-by-step.

## Synthetic data design

Because obtaining and cleaning live NASA FIRMS + OpenStreetMap data is a
project in itself, this prototype first validates the *idea* using a
carefully designed **synthetic dataset with the same schema** the real data
would have. This is a well-established approach when validating an
analytical pipeline before wiring up live data feeds.

The synthetic data includes:

- **30 industrial facilities** across 7 types (steel plant, cement plant,
  refinery, thermal power plant, chemical plant, waste facility, generic
  manufacturing), scattered across a small ~20km industrial region.
- **~90 days of thermal detections per facility**, with realistic,
  type-dependent behaviour:
  - Steel plants and refineries: frequent, moderately high FRP.
  - Thermal power plants: persistent moderate activity.
  - Generic manufacturing / waste facilities: sparse, low FRP.
  - **Two facilities contain an artificially injected anomaly**: a sudden,
    sustained FRP spike in the final 5 days of the dataset, simulating an
    abnormal industrial thermal event.
- Random location jitter around each facility (simulating satellite pixel
  imprecision), a reproducible random seed, and realistic confidence /
  brightness-temperature / satellite-source fields.

## Connecting real data later

Section "How this prototype would connect to REAL data sources" inside the
running app (an expandable panel) explains exactly how `data_loader.py`
would be swapped out for:

- **NASA FIRMS** (live thermal detections, same lat/lon/FRP/brightness/
  confidence schema).
- **OpenStreetMap / Overpass API** (real industrial facility polygons/points).
- **Sentinel/Landsat imagery** (optional future visual-verification layer,
  intentionally out of scope for this MVP).

Because the pipeline (`spatial_matching.py` → `feature_engineering.py` →
`anomaly_detection.py` → `classification.py`) only depends on the **CSV
schema**, not on where the data came from, real data can be substituted
without touching the analytical logic — and the app is designed to never
crash if those external services are unavailable, since it always falls
back to local CSV files.

## Important scientific limitations

Please read this before drawing any conclusions from the prototype:

- FIRMS-style satellite thermal detections indicate **thermal anomalies**,
  not confirmed fires.
- Satellite observations have real **spatial and temporal limitations**
  (revisit time, pixel size, cloud cover) and can miss or mis-locate events.
- **Persistent industrial heat can be completely normal** — this is exactly
  why the system compares a facility to its *own* history instead of one
  global threshold.
- A flagged thermal anomaly **does not prove** an industrial accident, fire,
  or safety incident occurred.
- This prototype's GREEN/YELLOW/RED classification is a **research signal
  for further investigation**, not an operational emergency warning system.
- The dataset used here is **synthetic**, generated to validate the overall
  approach before connecting real NASA FIRMS / OSM data.

## Tech stack

Pure Python data-science stack, no deep learning, no GPU required:

- pandas, numpy — data handling
- geopandas, shapely — real spatial matching with a metric (UTM) CRS
- scikit-learn — Isolation Forest anomaly detection
- streamlit — interactive web UI
- folium + streamlit-folium — interactive map
- plotly — time-series charts
