"""
GURU MARS Gridlock 2.0 -- MODEL 3: Cascade / Domino Detector  (the novel gap)
===============================================================================
Resample incidents to 1-hour counts per corridor over a shared, zero-filled
hourly index spanning the full observation window. For every ordered corridor
pair (A, B) compute Pearson r between A[t] and B[t+tau] for tau = 0,1,2,3 hours.
If the BEST correlation happens at tau > 0 (i.e. B genuinely lags A, it isn't
just simultaneous co-incidence at tau=0) and r > 0.25, we register a directed
cascade edge A -> B with that lag. Edges are assembled into a NetworkX digraph
that the recommendation layer queries at alert time: "corridor X just spiked,
who should be pre-alerted, and in how many minutes?"

Why Pearson over Granger causality: Granger needs longer, denser, stationary
series. With sparse hourly counts (~1 incident/hour on average per corridor)
fixed-lag Pearson is far more stable on this much data. Granger is left as a
documented validation step for the top corridors only.
"""
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr
import joblib

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CASCADE_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cascade_edges.json")

EXCLUDE_CORRIDORS = {"Unknown", "Non-corridor"}
LAGS = [0, 1, 2, 3]
R_THRESHOLD = 0.25
MIN_OVERLAP = 200  # minimum paired observations required for a correlation to count


def build_hourly_matrix(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[~df["corridor"].isin(EXCLUDE_CORRIDORS)].copy()
    sub = sub.dropna(subset=["start_datetime_local"])
    sub["hour_bucket"] = sub["start_datetime_local"].dt.floor("h")

    full_range = pd.date_range(sub["hour_bucket"].min(), sub["hour_bucket"].max(), freq="h")
    pivot = (
        sub.groupby(["hour_bucket", "corridor"]).size().unstack(fill_value=0)
        .reindex(full_range, fill_value=0)
    )
    pivot.index.name = "hour_bucket"
    return pivot


def time_lagged_pearson(a: pd.Series, b: pd.Series, lag: int):
    """corr(A[t], B[t+lag]) -- does a spike in A now predict a spike in B `lag` hours later."""
    if lag == 0:
        x, y = a, b
    else:
        x = a.iloc[:-lag]
        y = b.shift(-lag).iloc[:-lag]
    mask = x.notna() & y.notna()
    if mask.sum() < MIN_OVERLAP:
        return None, None, int(mask.sum())
    r, p = pearsonr(x[mask], y[mask])
    return r, p, int(mask.sum())


def build_cascade_graph(pivot: pd.DataFrame):
    corridors = pivot.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(corridors)

    edges = []
    for a in corridors:
        for b in corridors:
            if a == b:
                continue
            results = {}
            for lag in LAGS:
                r, p, n = time_lagged_pearson(pivot[a], pivot[b], lag)
                results[lag] = (r, p, n)
            # best lag = highest correlation among the computed lags
            valid = {l: v for l, v in results.items() if v[0] is not None}
            if not valid:
                continue
            best_lag = max(valid, key=lambda l: valid[l][0])
            best_r, best_p, best_n = valid[best_lag]
            if best_lag > 0 and best_r is not None and best_r > R_THRESHOLD:
                edges.append({
                    "from": a, "to": b, "lag_hours": best_lag,
                    "pearson_r": round(float(best_r), 4),
                    "p_value": float(best_p),
                    "n_obs": best_n,
                })
                G.add_edge(a, b, lag_hours=best_lag, r=best_r, p=best_p)

    edges.sort(key=lambda e: -e["pearson_r"])
    return G, edges


def train():
    df = pd.read_parquet(PROCESSED_PATH)
    pivot = build_hourly_matrix(df)
    print(f"Hourly matrix: {pivot.shape[0]} hourly buckets x {pivot.shape[1]} corridors")
    nonzero_counts = (pivot > 0).sum()
    avg_when_active = pivot.replace(0, np.nan).mean()
    print("Non-zero slots per corridor (range):", int(nonzero_counts.min()), "-", int(nonzero_counts.max()))
    print(f"Avg incidents per active slot (range): {avg_when_active.min():.2f} - {avg_when_active.max():.2f}")

    G, edges = build_cascade_graph(pivot)
    print(f"\nCascade edges found (r > {R_THRESHOLD}, lag > 0): {len(edges)}")
    for e in edges[:15]:
        print(f"  {e['from']:25s} -> {e['to']:25s}  lag={e['lag_hours']}h  r={e['pearson_r']:.3f}  "
              f"p={e['p_value']:.2e}  n={e['n_obs']}")

    # explicit check on the demo narrative pair
    mysore_magadi = [e for e in edges if e["from"] == "Mysore Road" and e["to"] == "Magadi Road"]
    if mysore_magadi:
        print("\n>>> Mysore Road -> Magadi Road cascade CONFIRMED:", mysore_magadi[0])
    else:
        all_pairs = [(l, r, p, n) for l in LAGS
                     for r, p, n in [time_lagged_pearson(pivot.get("Mysore Road", pd.Series(dtype=float)),
                                                          pivot.get("Magadi Road", pd.Series(dtype=float)), l)]]
        print("\n>>> Mysore Road -> Magadi Road NOT above threshold in this data. "
              "Raw lag correlations (lag, r, p, n):", all_pairs)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"graph": G, "edges": edges, "corridors": pivot.columns.tolist()},
                os.path.join(MODEL_DIR, "model3_cascade.joblib"))
    with open(CASCADE_OUT, "w") as f:
        json.dump(edges, f, indent=2)
    print(f"\nSaved cascade graph + edges -> {MODEL_DIR}/model3_cascade.joblib, {CASCADE_OUT}")
    return G, edges


if __name__ == "__main__":
    train()
