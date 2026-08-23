"""
app.py
------
Industrial Thermal Intelligence Prototype - Streamlit UI.

This is the single entry point for the whole application. Run it with:

    streamlit run app.py

WHAT THIS APP DOES (in plain English):
1. Loads (or generates) a synthetic dataset of industrial facilities and
   satellite thermal detections - OR fetches live NASA FIRMS + OpenStreetMap
   data if you enable "Use live FIRMS/OSM data" in the sidebar.
2. Spatially matches every thermal detection to its nearest facility using
   real GeoPandas/Shapely logic.
3. Builds a historical "normal behaviour" profile for every facility.
4. Compares each facility's RECENT thermal activity against ITS OWN history
   using a statistical z-score AND an Isolation Forest model.
5. Classifies every facility as GREEN / YELLOW / RED and explains why.
6. Displays everything on an interactive map + a facility detail panel with
   a time-series chart.

By default the app runs entirely from local CSV files - no internet
connection or API key is required unless you switch on live data.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Make sure we can import our own modules from src/, regardless of the
# current working directory the app was launched from.
# THIS MUST HAPPEN BEFORE any "from data_loader import ..." style imports.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if not os.path.isdir(_SRC_DIR):
    raise RuntimeError(
        f"Could not find the 'src' folder at: {_SRC_DIR}\n"
        f"Make sure app.py and the src/ folder are in the same directory."
    )
sys.path.insert(0, _SRC_DIR)

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

import config
from data_loader import load_or_generate_data
from real_data_loader import load_real_data
from spatial_matching import match_detections_to_facilities
from feature_engineering import build_facility_profiles, RECENT_WINDOW_DAYS
from anomaly_detection import run_anomaly_detection
from classification import classify_all_facilities


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Thermal Intelligence Prototype",
    page_icon="🔥",
    layout="wide",
)

CLASS_COLORS = {"GREEN": "#2ecc71", "YELLOW": "#f1c40f", "RED": "#e74c3c"}
CLASS_RADIUS = {"GREEN": 6, "YELLOW": 9, "RED": 13}


# ---------------------------------------------------------------------------
# DATA PIPELINE (cached so it only runs once per session / until inputs change)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running thermal intelligence pipeline...")
def run_pipeline(use_real_data=False):
    """
    Runs the full pipeline end-to-end and returns everything the UI needs:
    facilities, all detections (with spatial match info), the final
    classified facility profiles, and an optional warning message (used
    when live data was requested but fell back to synthetic data).
    """
    warning = None

    if use_real_data:
        facilities_df, matched_detections_df, warning = load_real_data(
            map_key=config.FIRMS_MAP_KEY,
            bbox=config.BBOX,
            day_range=10,
        )
    else:
        facilities_df, detections_df = load_or_generate_data(data_dir="data", n_facilities=30)
        matched_detections_df = match_detections_to_facilities(detections_df, facilities_df)

    profiles_df = build_facility_profiles(matched_detections_df, facilities_df)
    scored_df = run_anomaly_detection(profiles_df)
    classified_df = classify_all_facilities(scored_df)
    return facilities_df, matched_detections_df, classified_df, warning


# ---------------------------------------------------------------------------
# SIDEBAR - DATA SOURCE + RUN PIPELINE
# (this must happen before the filters below, since they read from the data)
# ---------------------------------------------------------------------------
st.sidebar.title("Data source")
use_real_data = st.sidebar.toggle("Use live FIRMS/OSM data", value=False)

facilities_df, detections_df, classified_df, pipeline_warning = run_pipeline(
    use_real_data=use_real_data
)

if pipeline_warning:
    st.sidebar.warning(pipeline_warning)


# ---------------------------------------------------------------------------
# SIDEBAR - FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

facility_types = sorted(facilities_df["facility_type"].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Facility type",
    options=facility_types,
    default=facility_types,
)

severity_options = ["GREEN", "YELLOW", "RED"]
selected_severities = st.sidebar.multiselect(
    "Anomaly severity",
    options=severity_options,
    default=severity_options,
)

min_date = detections_df["timestamp"].min().date()
max_date = detections_df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Detection date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
# st.date_input returns a single date until the user picks a range - guard against that.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Recent activity window used for anomaly comparison: last "
    f"**{RECENT_WINDOW_DAYS} days** vs. all prior history."
)

# Apply filters
filtered_facilities = classified_df[
    classified_df["facility_type"].isin(selected_types)
    & classified_df["classification"].isin(selected_severities)
]

filtered_detections = detections_df[
    (detections_df["timestamp"].dt.date >= start_date)
    & (detections_df["timestamp"].dt.date <= end_date)
    & (detections_df["matched_facility_id"].isin(filtered_facilities["facility_id"]))
]


# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
st.title("🔥 Industrial Thermal Intelligence Prototype")
st.caption(
    "AI-based association of NASA-FIRMS-style thermal detections with industrial facilities, "
    "to learn each facility's normal thermal behaviour and flag unusual activity. "
    "**Research prototype — not an operational fire detection system.**"
)


# ---------------------------------------------------------------------------
# TOP-LEVEL KPIs
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

n_facilities = len(filtered_facilities)
n_detections = len(filtered_detections)
n_anomalous = (filtered_facilities["classification"].isin(["YELLOW", "RED"])).sum()
n_red = (filtered_facilities["classification"] == "RED").sum()

kpi1.metric("Facilities monitored", n_facilities)
kpi2.metric("Thermal detections (in range)", n_detections)
kpi3.metric("Anomalous facilities (Yellow + Red)", int(n_anomalous))
kpi4.metric("Potential abnormal events (Red)", int(n_red))

st.markdown("---")


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------
st.subheader("🗺️ Interactive Map")
st.caption(
    "Circle markers = facilities (color/size shows anomaly severity). "
    "Small dots = individual thermal detections."
)

if len(filtered_facilities) > 0:
    map_center = [filtered_facilities["latitude"].mean(), filtered_facilities["longitude"].mean()]
else:
    map_center = [facilities_df["latitude"].mean(), facilities_df["longitude"].mean()]

m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

# Plot individual thermal detections as small, faint dots.
for _, det in filtered_detections.iterrows():
    folium.CircleMarker(
        location=[det["latitude"], det["longitude"]],
        radius=2,
        color="#e67e22",
        fill=True,
        fill_opacity=0.5,
        weight=0,
        popup=f"FRP: {det['frp']:.1f} MW<br>{det['timestamp']}",
    ).add_to(m)

# Plot facilities as larger circle markers, styled by classification.
for _, fac in filtered_facilities.iterrows():
    label = fac["classification"]
    color = CLASS_COLORS[label]
    radius = CLASS_RADIUS[label]

    popup_html = (
        f"<b>{fac['facility_name']}</b><br>"
        f"Type: {fac['facility_type']}<br>"
        f"Status: <b>{label}</b><br>"
        f"Recent max FRP: {fac['recent_max_frp']:.1f} MW<br>"
        f"Historical mean FRP: {fac['hist_mean_frp']:.1f} MW"
    )

    folium.CircleMarker(
        location=[fac["latitude"], fac["longitude"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=2,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{fac['facility_name']} ({label})",
    ).add_to(m)

map_output = st_folium(m, width=1200, height=520, returned_objects=["last_object_clicked_tooltip"])

st.markdown("---")


# ---------------------------------------------------------------------------
# FACILITY SELECTION
# ---------------------------------------------------------------------------
st.subheader("🏭 Facility Intelligence Panel")

facility_options = filtered_facilities.sort_values(
    ["classification", "facility_name"],
    key=lambda col: col.map({"RED": 0, "YELLOW": 1, "GREEN": 2}) if col.name == "classification" else col,
)["facility_name"].tolist()

if not facility_options:
    st.info("No facilities match the current filters. Try adjusting the sidebar options.")
    st.stop()

# If the user clicked a facility on the map, try to preselect it.
default_index = 0
if map_output and map_output.get("last_object_clicked_tooltip"):
    clicked_name = map_output["last_object_clicked_tooltip"].split(" (")[0]
    if clicked_name in facility_options:
        default_index = facility_options.index(clicked_name)

selected_name = st.selectbox(
    "Select a facility to inspect in detail",
    options=facility_options,
    index=default_index,
)

facility_row = filtered_facilities[filtered_facilities["facility_name"] == selected_name].iloc[0]
facility_id = facility_row["facility_id"]

label = facility_row["classification"]
color = CLASS_COLORS[label]
status_text = {
    "GREEN": "Routine thermal activity",
    "YELLOW": "Unusual thermal activity",
    "RED": "Potential abnormal industrial thermal event",
}[label]

info_col, chart_col = st.columns([1, 2])

with info_col:
    st.markdown(f"### {facility_row['facility_name']}")
    st.write(f"**Type:** {facility_row['facility_type']}")
    st.write(f"**Location:** {facility_row['latitude']:.4f}, {facility_row['longitude']:.4f}")

    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background-color:{color}22;"
        f"border:1px solid {color};'>"
        f"<b>Status: <span style='color:{color}'>{label}</span></b><br>{status_text}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Key numbers")
    st.write(f"- Historical detections: **{int(facility_row['hist_total_detections'])}**")
    st.write(f"- Recent detections (last {RECENT_WINDOW_DAYS} days): **{int(facility_row['recent_total_detections'])}**")
    st.write(f"- Historical mean FRP: **{facility_row['hist_mean_frp']:.1f} MW**")
    st.write(f"- Recent max FRP: **{facility_row['recent_max_frp']:.1f} MW**")
    st.write(f"- FRP z-score (recent vs. historical): **{facility_row['frp_zscore']:.1f}**")
    st.write(f"- Historical persistence: **{facility_row['hist_persistence']:.0%}** of days active")
    st.write(f"- Recent persistence: **{facility_row['recent_persistence']:.0%}** of days active")
    st.write(f"- Distance of latest detection to facility: **{facility_row['latest_distance_m']:.0f} m**")

    st.markdown("#### Why this classification?")
    if facility_row["explanation_list"]:
        for point in facility_row["explanation_list"]:
            st.write(f"- {point}")
    else:
        st.write("- No specific anomaly signals were triggered.")

with chart_col:
    facility_detections = detections_df[
        detections_df["matched_facility_id"] == facility_id
    ].sort_values("timestamp")

    fig = go.Figure()

    if len(facility_detections) > 0:
        # Split into historical vs recent for coloring.
        cutoff_ts = facility_detections["timestamp"].max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
        hist_part = facility_detections[facility_detections["timestamp"] < cutoff_ts]
        recent_part = facility_detections[facility_detections["timestamp"] >= cutoff_ts]

        fig.add_trace(go.Scatter(
            x=hist_part["timestamp"], y=hist_part["frp"],
            mode="markers", name="Historical detections",
            marker=dict(color="#3498db", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=recent_part["timestamp"], y=recent_part["frp"],
            mode="markers", name="Recent detections (evaluated window)",
            marker=dict(color=color, size=10, symbol="diamond",
                        line=dict(width=1, color="black")),
        ))

        # Historical mean line for reference.
        fig.add_hline(
            y=facility_row["hist_mean_frp"],
            line_dash="dash", line_color="gray",
            annotation_text="Historical mean FRP",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"Thermal activity (FRP) over time — {facility_row['facility_name']}",
        xaxis_title="Date",
        yaxis_title="Fire Radiative Power (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=460,
        margin=dict(t=60),
    )

    st.plotly_chart(fig, width="stretch")

    if len(facility_detections) == 0:
        st.info("This facility has no recorded thermal detections in the dataset.")

st.markdown("---")


# ---------------------------------------------------------------------------
# ALL FACILITIES TABLE (quick overview)
# ---------------------------------------------------------------------------
with st.expander("📋 View all monitored facilities (table)"):
    display_cols = [
        "facility_name", "facility_type", "classification",
        "hist_mean_frp", "recent_max_frp", "frp_zscore",
        "recent_persistence", "hist_persistence",
    ]
    st.dataframe(
        filtered_facilities[display_cols].sort_values(
            "classification", key=lambda c: c.map({"RED": 0, "YELLOW": 1, "GREEN": 2})
        ),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# SECTION 13 - HOW TO CONNECT REAL DATA (informational)
# ---------------------------------------------------------------------------
with st.expander("🔌 How this prototype connects to REAL data sources"):
    st.markdown(
        """
By default the app runs entirely on **synthetic CSV data** stored in the
`data/` folder, generated automatically the first time you run the app.

Toggling **"Use live FIRMS/OSM data"** in the sidebar switches to
`src/real_data_loader.py`, which:

- Fetches real thermal detections from the **NASA FIRMS** Area API (VIIRS
  SNPP/NOAA-20 + MODIS) for the bounding box in `config.py`.
- Fetches real industrial facilities from **OpenStreetMap** via the
  Overpass API, guessing a facility type from each element's OSM tags.
- Spatially matches the two using the same `spatial_matching.py` logic
  used for the synthetic data.

Because the rest of the pipeline (`feature_engineering.py`,
`anomaly_detection.py`, `classification.py`) only depends on the CSV
schema — not on where the data came from — it works identically either
way. If the live fetch fails for any reason (bad/missing key, network
error, no detections in range), the app **automatically falls back** to
the synthetic dataset and shows a warning instead of crashing.

**Sentinel/Landsat imagery** could later be added as a secondary visual
verification layer, but is intentionally out of scope for this MVP.
        """
    )


# ---------------------------------------------------------------------------
# SCIENTIFIC LIMITATIONS (always visible, important for responsible framing)
# ---------------------------------------------------------------------------
with st.expander("⚠️ Important scientific limitations — please read"):
    st.markdown(
        """
- FIRMS-style satellite thermal detections indicate **thermal anomalies**,
  not confirmed fires. Many industrial processes are legitimately hot.
- Satellite observations have **spatial and temporal limitations** (revisit
  time, pixel size, cloud cover) and can miss or mis-locate events.
- **Persistent industrial heat can be completely normal** — this is exactly
  why the system compares each facility to its *own* history rather than
  applying one global threshold.
- A flagged thermal anomaly **does not prove** an industrial accident,
  fire, or safety incident occurred.
- This prototype's classification (GREEN/YELLOW/RED) is a **research
  signal for further investigation**, not an operational emergency warning
  system, and should never be used as the sole basis for a real-world
  safety decision.
- The synthetic dataset (used unless live data is enabled) was generated
  to validate the overall approach before connecting real NASA FIRMS / OSM
  data.
        """
    )
