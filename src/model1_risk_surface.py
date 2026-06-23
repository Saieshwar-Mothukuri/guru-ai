"""
GURU MARS Gridlock 2.0 -- MODEL 1: Risk Surface
=================================================
Expected incident rate per corridor-hour(-dow). XGBoost regressor on AGGREGATED
counts, not raw rows (critical constraint from process.txt: a raw-row model would
have no negative samples and would just memorize "this corridor had an event").

Grouping key: (corridor, hour, dow) -> count = target.
~49% of the 3,696 possible (corridor x hour x dow) slots are observed in the
8,173-event dataset; XGBoost is used specifically because a flat lookup table
can't say anything about the other ~51% of slots that have zero history.

NOTE ON "WEATHER": process.txt's marketing copy mentions weather as a risk
factor ("corridor x hour x weather"), but the actual MODEL 1 feature spec and
the raw ASTraM export have no weather column. This implementation is honest
about that: it trains on corridor/hour/dow/police_station only, and documents
weather as a production add-on (pull from an external API, e.g. IMD/OpenWeather)
rather than fabricating a weather signal.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOOKUP_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "risk_surface_lookup.json")


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["corridor"] != "Unknown"].copy()

    # most-frequent police_station per corridor -> used as a categorical feature
    # (police jurisdictions roughly track corridors, so this lets the model pick
    # up station-level enforcement/reporting differences without exploding the
    # combo space by grouping on police_station directly)
    station_mode = (
        sub.groupby("corridor")["police_station"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("dominant_police_station")
    )

    agg = (
        sub.groupby(["corridor", "hour", "dow"])
        .size()
        .rename("count")
        .reset_index()
    )
    agg = agg.merge(station_mode, on="corridor", how="left")
    agg["is_weekend"] = agg["dow"].isin([5, 6]).astype(int)
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24)
    agg["dow_sin"] = np.sin(2 * np.pi * agg["dow"] / 7)
    agg["dow_cos"] = np.cos(2 * np.pi * agg["dow"] / 7)
    return agg


def make_full_grid(df: pd.DataFrame, station_mode: pd.Series) -> pd.DataFrame:
    """Every possible (corridor, hour, dow) slot -- including the ~51% with zero
    observed history -- so the trained model can score slots that never had an
    incident logged. This is the whole point of using XGBoost over a lookup table."""
    corridors = df[df["corridor"] != "Unknown"]["corridor"].unique()
    grid = pd.MultiIndex.from_product(
        [corridors, range(24), range(7)], names=["corridor", "hour", "dow"]
    ).to_frame(index=False)
    grid = grid.merge(station_mode, on="corridor", how="left")
    grid["is_weekend"] = grid["dow"].isin([5, 6]).astype(int)
    grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24)
    grid["dow_sin"] = np.sin(2 * np.pi * grid["dow"] / 7)
    grid["dow_cos"] = np.cos(2 * np.pi * grid["dow"] / 7)
    return grid


def train():
    df = pd.read_parquet(PROCESSED_PATH)
    agg = build_training_table(df)
    print(f"Training table: {len(agg)} (corridor,hour,dow) combos "
          f"out of {df[df.corridor!='Unknown'].corridor.nunique()*24*7} possible "
          f"({len(agg)/(df[df.corridor!='Unknown'].corridor.nunique()*24*7):.1%})")

    cat_cols = ["corridor", "dominant_police_station"]
    for c in cat_cols:
        agg[c] = agg[c].astype("category")

    feature_cols = ["corridor", "dominant_police_station", "hour", "dow",
                     "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    X = agg[feature_cols]
    y = agg["count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective="count:poisson",   # target is a count -> Poisson loss fits better than squared error
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.3f} incidents | R2: {r2:.3f} | baseline (mean) MAE: "
          f"{mean_absolute_error(y_test, [y_train.mean()]*len(y_test)):.3f}")

    # refit on ALL observed combos before scoring the full grid (more signal = better)
    model_full = xgb.XGBRegressor(**model.get_params())
    model_full.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model_full, "feature_cols": feature_cols,
                 "categories": {c: agg[c].cat.categories.tolist() for c in cat_cols}},
                os.path.join(MODEL_DIR, "model1_risk_surface.joblib"))

    # score the FULL grid (observed + never-seen slots) for the demo lookup / API
    station_mode = agg[["corridor", "dominant_police_station"]].drop_duplicates().set_index("corridor")["dominant_police_station"]
    grid = make_full_grid(df, station_mode)
    for c in cat_cols:
        grid[c] = pd.Categorical(grid[c], categories=agg[c].cat.categories)
    grid["expected_count"] = model_full.predict(grid[feature_cols])
    grid["expected_count"] = grid["expected_count"].clip(lower=0)

    # risk score 0-100 = percentile rank of expected_count within the whole grid
    grid["risk_score"] = (grid["expected_count"].rank(pct=True) * 100).round(1)
    grid["risk_tier"] = pd.cut(grid["risk_score"], [0, 40, 70, 90, 100],
                                labels=["Low", "Moderate", "High", "Critical"], include_lowest=True)

    grid["observed"] = grid.set_index(["corridor", "hour", "dow"]).index.isin(
        agg.set_index(["corridor", "hour", "dow"]).index
    )

    lookup = {}
    for _, row in grid.iterrows():
        key = f"{row['corridor']}|{int(row['hour'])}|{int(row['dow'])}"
        lookup[key] = {
            "expected_count": round(float(row["expected_count"]), 2),
            "risk_score": float(row["risk_score"]),
            "risk_tier": str(row["risk_tier"]),
            "observed_in_training": bool(row["observed"]),
        }
    os.makedirs(os.path.dirname(LOOKUP_OUT), exist_ok=True)
    with open(LOOKUP_OUT, "w") as f:
        json.dump(lookup, f)
    print(f"Saved full risk-surface lookup ({len(lookup)} slots) -> {LOOKUP_OUT}")

    # sanity: show Mysore Road Thursday 21:00 -- the demo scenario from process.txt
    demo_key = "Mysore Road|21|3"  # dow 3 = Thursday (Mon=0)
    if demo_key in lookup:
        print("Demo scenario (Mysore Road, Thursday, 9PM):", lookup[demo_key])

    return model_full, grid


if __name__ == "__main__":
    train()
