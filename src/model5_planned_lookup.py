"""
GURU MARS Gridlock 2.0 -- MODEL 5: Planned Event Lookup
==========================================================
467 planned events (construction/public_event/procession/vip_movement/protest/
etc.) -- too few for gradient boosting, so this is K-Nearest-Neighbours (K=5)
returning actual past events as evidence: "here are 5 similar past processions
and what happened to them."

Custom distance = Hamming distance on categoricals (event_cause, corridor,
police_station, priority) + cosine distance on the cyclical (hour, dow)
encoding. Computed as a full 467x467 pairwise matrix (trivial at this size).

Output per query: median duration across the 5 neighbours, typical closure
rate, median priority, and the reference event IDs themselves.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import joblib

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CAT_FEATURES = ["event_cause", "corridor", "police_station", "priority"]
CYCLICAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
K = 5


def hamming_block(df: pd.DataFrame) -> np.ndarray:
    """Pairwise count of mismatching categorical fields, normalized to [0,1]."""
    n = len(df)
    cat_arr = df[CAT_FEATURES].astype(str).values
    mismatches = np.zeros((n, n))
    for j in range(len(CAT_FEATURES)):
        col = cat_arr[:, j]
        mismatches += (col[:, None] != col[None, :]).astype(float)
    return mismatches / len(CAT_FEATURES)


def cosine_block(df: pd.DataFrame) -> np.ndarray:
    vecs = df[CYCLICAL_FEATURES].values
    dist = squareform(pdist(vecs, metric="cosine"))
    return np.nan_to_num(dist, nan=0.0)


def build():
    df = pd.read_parquet(PROCESSED_PATH)
    planned = df[df["event_type"] == "planned"].reset_index(drop=True).copy()
    print(f"Planned events: {len(planned)}")
    print(planned["event_cause"].value_counts())

    for c in CAT_FEATURES:
        planned[c] = planned[c].fillna("Unknown").astype(str)

    dist_matrix = hamming_block(planned) + cosine_block(planned)
    np.fill_diagonal(dist_matrix, np.inf)  # never match an event to itself

    # leave-one-out validation: for each planned event, predict via its 5 NN
    # and compare predicted vs actual duration/closure (where ground truth exists)
    knn_idx = np.argsort(dist_matrix, axis=1)[:, :K]

    rows = []
    for i in range(len(planned)):
        neighbours = planned.iloc[knn_idx[i]]
        pred_duration = neighbours["duration_proxy_min"].median()
        pred_closure_rate = neighbours["requires_road_closure"].mean()
        actual_duration = planned.iloc[i]["duration_proxy_min"]
        actual_closure = planned.iloc[i]["requires_road_closure"]
        rows.append({"pred_duration": pred_duration, "actual_duration": actual_duration,
                      "pred_closure_rate": pred_closure_rate, "actual_closure": actual_closure})
    val = pd.DataFrame(rows)
    has_actual = val["actual_duration"].notna() & val["pred_duration"].notna()
    mae = (val.loc[has_actual, "pred_duration"] - val.loc[has_actual, "actual_duration"]).abs().median()
    print(f"\nLeave-one-out validation (n={has_actual.sum()}): "
          f"median absolute duration error = {mae:.1f} min")
    # closure-rate calibration: does avg predicted closure rate roughly match actual closure rate?
    print(f"Mean predicted closure rate: {val['pred_closure_rate'].mean():.3f} "
          f"vs actual closure rate: {val['actual_closure'].mean():.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "planned_events": planned,
        "cat_features": CAT_FEATURES,
        "cyclical_features": CYCLICAL_FEATURES,
        "k": K,
    }, os.path.join(MODEL_DIR, "model5_planned_lookup.joblib"))
    print(f"\nSaved -> {MODEL_DIR}/model5_planned_lookup.joblib")

    # demo query: a hypothetical new procession on a known corridor
    demo_query(planned)
    return planned, dist_matrix


def query_neighbors(planned, query: dict, k: int = K):
    """query = {event_cause, corridor, police_station, priority, hour, dow}"""
    qdf = pd.DataFrame([query])
    qdf["hour_sin"] = np.sin(2 * np.pi * qdf["hour"] / 24)
    qdf["hour_cos"] = np.cos(2 * np.pi * qdf["hour"] / 24)
    qdf["dow_sin"] = np.sin(2 * np.pi * qdf["dow"] / 7)
    qdf["dow_cos"] = np.cos(2 * np.pi * qdf["dow"] / 7)
    for c in CAT_FEATURES:
        qdf[c] = qdf[c].fillna("Unknown").astype(str)

    combined = pd.concat([planned[CAT_FEATURES + CYCLICAL_FEATURES], qdf[CAT_FEATURES + CYCLICAL_FEATURES]],
                          ignore_index=True)
    cat_arr = combined[CAT_FEATURES].astype(str).values
    qi = len(combined) - 1
    ham = np.zeros(len(combined))
    for j in range(len(CAT_FEATURES)):
        ham += (cat_arr[:, j] != cat_arr[qi, j]).astype(float)
    ham /= len(CAT_FEATURES)

    from scipy.spatial.distance import cosine
    qvec = combined.iloc[qi][CYCLICAL_FEATURES].values.astype(float)
    cos_dist = np.array([
        cosine(qvec, combined.iloc[r][CYCLICAL_FEATURES].values.astype(float)) if r != qi else np.inf
        for r in range(len(combined) - 1)
    ])
    cos_dist = np.nan_to_num(cos_dist, nan=0.0)

    total_dist = ham[:-1] + cos_dist
    nn_idx = np.argsort(total_dist)[:k]
    neighbours = planned.iloc[nn_idx]
    return {
        "median_duration_min": float(neighbours["duration_proxy_min"].median()) if neighbours["duration_proxy_min"].notna().any() else None,
        "closure_rate": float(neighbours["requires_road_closure"].mean()),
        "median_priority": neighbours["priority"].mode().iloc[0] if not neighbours["priority"].mode().empty else None,
        "reference_event_ids": neighbours["id"].tolist(),
        "reference_distances": total_dist[nn_idx].round(3).tolist(),
    }


def demo_query(planned):
    q = {"event_cause": "procession", "corridor": "Mysore Road", "police_station": "Wilson Garden",
         "priority": "High", "hour": 18, "dow": 5}
    result = query_neighbors(planned, q)
    print("\nDemo KNN query (new procession, Mysore Road, Saturday 6PM):")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    build()
