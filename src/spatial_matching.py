"""
spatial_matching.py
--------------------
This module implements the SPATIAL MATCHING step of the pipeline:

    FIRMS thermal detections  --->  nearest industrial facility

Even though our synthetic detections already carry a `facility_id` (because
we generated them that way), a real-world FIRMS feed would NOT come with a
facility_id attached - it would just be raw lat/lon hotspot points. So this
module implements the REAL spatial-matching logic using GeoPandas + Shapely,
exactly as we would need to do with real FIRMS + OSM data.

Steps:
1. Convert both facilities and detections into GeoDataFrames (WGS84 / EPSG:4326).
2. Re-project both into a metric (metre-based) CRS so that distance
   calculations are accurate. We pick a local UTM zone automatically based on
   the data's location, instead of using degrees (which are not a constant
   distance unit).
3. For each detection, find the NEAREST facility using geopandas' built-in
   `sjoin_nearest`.
4. Compute the distance (in metres) between the detection and its nearest
   facility.
5. Reject (unmatch) detections that are too far from any facility - this
   avoids incorrectly associating a stray hotspot with a facility that is
   actually kilometres away.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Maximum distance (in metres) a detection can be from a facility and still
# be considered "belonging" to it. Real VIIRS pixels are ~375m and MODIS
# pixels are ~1km, so we allow a bit of margin above that.
MAX_MATCH_DISTANCE_METERS = 2000


def _estimate_utm_epsg(lon, lat):
    """
    Given a longitude/latitude, estimate the correct UTM EPSG code.
    UTM zones are 6 degrees wide; this is the standard formula to find
    which zone a coordinate falls into, and whether it's northern or
    southern hemisphere.
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number  # Northern hemisphere UTM
    else:
        return 32700 + zone_number  # Southern hemisphere UTM


def match_detections_to_facilities(detections_df, facilities_df,
                                    max_distance_m=MAX_MATCH_DISTANCE_METERS):
    """
    Performs nearest-facility spatial matching.

    Parameters
    ----------
    detections_df : pd.DataFrame
        Must contain 'latitude' and 'longitude' columns for each detection.
    facilities_df : pd.DataFrame
        Must contain 'facility_id', 'latitude', 'longitude' for each facility.
    max_distance_m : float
        Detections farther than this from every facility are considered
        "unmatched" (matched_facility_id becomes NaN).

    Returns
    -------
    pd.DataFrame
        A copy of detections_df with two new columns added:
        - 'matched_facility_id' : the facility_id of the nearest facility
        - 'distance_to_facility_m' : distance in metres to that facility
    """

    # --- 1. Build GeoDataFrames in WGS84 (standard lat/lon) ---
    facilities_gdf = gpd.GeoDataFrame(
        facilities_df.copy(),
        geometry=gpd.points_from_xy(facilities_df["longitude"], facilities_df["latitude"]),
        crs="EPSG:4326",
    )

    detections_gdf = gpd.GeoDataFrame(
        detections_df.copy(),
        geometry=gpd.points_from_xy(detections_df["longitude"], detections_df["latitude"]),
        crs="EPSG:4326",
    )

    # --- 2. Re-project to a local metric CRS (UTM) so distances are in metres ---
    # We use the mean facility location to pick one representative UTM zone,
    # which is perfectly fine for a small region (a single city/industrial belt).
    mean_lon = facilities_df["longitude"].mean()
    mean_lat = facilities_df["latitude"].mean()
    utm_epsg = _estimate_utm_epsg(mean_lon, mean_lat)

    facilities_proj = facilities_gdf.to_crs(epsg=utm_epsg)
    detections_proj = detections_gdf.to_crs(epsg=utm_epsg)

    # --- 3. Nearest-neighbour spatial join ---
    # sjoin_nearest attaches, to every detection row, the columns of the
    # single nearest facility row, plus a 'distance' column (in the CRS's
    # units, which are metres because we're in UTM).
    joined = gpd.sjoin_nearest(
        detections_proj,
        facilities_proj[["facility_id", "geometry"]],
        distance_col="distance_to_facility_m",
        lsuffix="det",
        rsuffix="fac",
    )

    # sjoin_nearest can occasionally return duplicate rows if two facilities
    # are exactly equidistant. We keep only the first (closest) match per
    # detection to be safe.
    joined = joined[~joined.index.duplicated(keep="first")]

    # --- 4. Reject matches that are too far away ---
    joined["matched_facility_id"] = joined["facility_id_fac"]
    too_far = joined["distance_to_facility_m"] > max_distance_m
    joined.loc[too_far, "matched_facility_id"] = None

    # --- 5. Return a clean, plain pandas DataFrame (drop geometry columns) ---
    result = pd.DataFrame(joined.drop(columns=["geometry"]))

    # Rename the original facility_id column (from detections_df, if present)
    # back cleanly, and keep our new matched columns clearly named.
    if "facility_id_det" in result.columns:
        result = result.rename(columns={"facility_id_det": "facility_id"})
    if "facility_id_fac" in result.columns:
        result = result.drop(columns=["facility_id_fac"])

    return result.reset_index(drop=True)


if __name__ == "__main__":
    # Small self-test using the data_loader module.
    from data_loader import load_or_generate_data

    facilities, detections = load_or_generate_data()
    matched = match_detections_to_facilities(detections, facilities)
    print(matched[["facility_id", "matched_facility_id", "distance_to_facility_m"]].head(10))

    # Sanity check: how many detections matched the SAME facility they were
    # originally generated for? (Should be almost all of them, since our
    # synthetic jitter is small.)
    agreement = (matched["facility_id"] == matched["matched_facility_id"]).mean()
    print(f"Spatial-match agreement with ground truth: {agreement:.2%}")
