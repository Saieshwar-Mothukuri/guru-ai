"""
GURU MARS Gridlock 2.0 -- Data Foundation Layer
=================================================
Loads the raw ASTraM event export, cleans it, backfills zone/junction via a
spatial KNN on lat/lon (the "spatial join... worth an hour's work" item from
process.txt), engineers the corridor/hour/dow/month features every downstream
model depends on, and writes a single processed parquet that all five models
read from.

Run: python3 src/data_foundation.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "astram_events.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")

LOCAL_TZ = "Asia/Kolkata"

# event_cause has a couple of casing duplicates in the raw export
CAUSE_NORMALIZATION = {
    "Debris": "debris",
    "Fog / Low Visibility": "fog_low_visibility",
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    dt_cols = ["start_datetime", "end_datetime", "modified_datetime", "created_date",
               "closed_datetime", "resolved_datetime"]
    for c in dt_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        # convert to Bengaluru local time -- traffic patterns (rush hour, etc.)
        # only make sense in local time, not UTC
        df[c + "_local"] = df[c].dt.tz_convert(LOCAL_TZ)
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["event_cause"] = df["event_cause"].replace(CAUSE_NORMALIZATION)
    df["event_cause"] = df["event_cause"].str.strip().str.lower()
    df["event_type"] = df["event_type"].str.strip().str.lower()
    df["corridor"] = df["corridor"].fillna("Unknown")
    df["priority"] = df["priority"].fillna("Low")  # 2 nulls, default conservative
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    local = df["start_datetime_local"]
    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek  # 0=Monday
    df["dow_name"] = local.dt.day_name()
    df["month"] = local.dt.month
    df["is_weekend"] = df["dow"].isin([5, 6])
    # cyclical encodings, used by the KNN planned-event model (Model 5)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def backfill_zone_junction(df: pd.DataFrame) -> pd.DataFrame:
    """
    zone is 57.9% null, junction is 69.3% null in the raw export. We do not have
    an official BBMP ward/zone shapefile in this environment, so true spatial-join
    backfill isn't available here. As a practical proxy, we train a KNN classifier
    on (lat, lon) -> zone using the ~42% of rows that DO have a labeled zone, then
    predict zone for every other row that has valid coordinates. Same approach for
    junction at a tighter radius. This is flagged in the README as an approximation
    that should be swapped for a real shapefile spatial join in production.
    """
    has_coords = df["latitude"].notna() & df["longitude"].notna() & (df["latitude"] != 0)

    for col, k in [("zone", 7), ("junction", 5)]:
        labeled = df[col].notna() & has_coords
        unlabeled = df[col].isna() & has_coords
        if labeled.sum() < 10 or unlabeled.sum() == 0:
            df[col + "_backfilled"] = df[col]
            continue
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(df.loc[labeled, ["latitude", "longitude"]], df.loc[labeled, col])
        preds = knn.predict(df.loc[unlabeled, ["latitude", "longitude"]])
        backfilled = df[col].copy()
        backfilled.loc[unlabeled] = preds
        df[col + "_backfilled"] = backfilled
        n_filled = unlabeled.sum()
        n_total_missing = df[col].isna().sum()
        print(f"  {col}: backfilled {n_filled}/{n_total_missing} missing values "
              f"({n_filled / max(n_total_missing,1):.1%}) via spatial KNN (k={k})")

    return df


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duration = closed_datetime - start_datetime for status == 'closed' rows with a
    valid closed_datetime. Matches process.txt's Model 2 spec: 2,997 closed events
    used for training after dropping negative/invalid durations.
    """
    closed_mask = (df["status"] == "closed") & df["closed_datetime"].notna() & df["start_datetime"].notna()
    dur_min = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds() / 60.0
    df["duration_min"] = np.where(closed_mask & (dur_min > 0), dur_min, np.nan)
    df["is_closed_with_duration"] = df["duration_min"].notna()
    df["duration_class"] = np.where(df["duration_min"] <= 120, "fast", "slow")
    df.loc[df["duration_min"].isna(), "duration_class"] = np.nan
    df["log_duration_min"] = np.log1p(df["duration_min"])

    # modified_datetime correlates at r=0.998 with closed_datetime wherever both
    # exist (median relative diff ~0), so it's a reliable fallback duration proxy
    # for rows missing closed_datetime -- this matters a lot for Model 5, where
    # only 28/467 planned events have a closed_datetime-based duration but 365/467
    # have a usable modified_datetime-based one.
    proxy_min = (df["modified_datetime"] - df["start_datetime"]).dt.total_seconds() / 60.0
    df["duration_proxy_min"] = np.where(df["duration_min"].notna(), df["duration_min"],
                                         np.where(proxy_min > 0, proxy_min, np.nan))
    df["duration_proxy_is_estimated"] = df["duration_min"].isna() & df["duration_proxy_min"].notna()
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Per process.txt: too sparse to use directly (kept available, just flagged)
    sparse_cols = ["direction", "end_datetime", "assigned_to_police_id",
                   "resolved_datetime", "map_file", "comment", "meta_data"]
    df.attrs["dropped_for_modeling"] = sparse_cols
    return df


def build():
    print("[1/6] Loading raw CSV...")
    df = load_raw()
    print(f"      {len(df)} rows, {len(df.columns)} columns")

    print("[2/6] Parsing datetimes (UTC -> Asia/Kolkata)...")
    df = parse_datetimes(df)

    print("[3/6] Cleaning categoricals (cause casing/dupes, corridor/priority nulls)...")
    df = clean_categoricals(df)

    print("[4/6] Engineering time features (hour/dow/month + cyclical encodings)...")
    df = engineer_time_features(df)

    print("[5/6] Spatial KNN backfill for zone/junction...")
    df = backfill_zone_junction(df)

    print("[6/6] Computing event durations for closed events...")
    df = compute_duration(df)
    df = drop_unusable_columns(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved processed dataset -> {OUT_PATH}  ({len(df)} rows, {len(df.columns)} cols)")

    # quick sanity summary
    print("\n--- Sanity checks vs process.txt ---")
    print("total rows:", len(df), "(expect 8173)")
    print("planned events:", (df.event_type == "planned").sum(), "(expect 467)")
    print("requires_road_closure True:", df.requires_road_closure.sum(), "(expect 676)")
    print("closed w/ valid duration:", df.is_closed_with_duration.sum(), "(expect ~2997-3124)")
    print("fast/slow split:", df.duration_class.value_counts().to_dict())
    return df


if __name__ == "__main__":
    build()
