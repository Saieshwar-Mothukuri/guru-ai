"""
GURU MARS Gridlock 2.0 -- Prediction Ledger (Gap 4: post-event learning loop)
================================================================================
Every closed incident logs predicted-vs-actual. This module does two things:

  1. `log_prediction` / `log_outcome` -- the live SQLite ledger API the FastAPI
     backend calls at inference time and at incident-close time.

  2. `run_chronological_backtest` -- a REAL backtest, not a scripted number.
     Closed events are sorted by start_datetime and split into 6 sequential
     chunks. Stage-1 (fast/slow) classifier is retrained on each growing
     window and evaluated on the next chunk only (genuine walk-forward
     validation), so the "accuracy trending up" chart in the demo is an
     actual measurement of this dataset, not a hardcoded 68% -> 81%.
"""
import os
import sqlite3
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prediction_ledger.db")
BACKTEST_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "learning_loop_backtest.json")

CAT_FEATURES = ["event_cause", "corridor", "priority", "police_station",
                 "requires_road_closure", "veh_type", "zone_backfilled"]
NUM_FEATURES = ["hour"]
FEATURE_COLS = CAT_FEATURES + NUM_FEATURES

RETRAIN_TRIGGER_THRESHOLD = 0.75  # if rolling accuracy dips below this, flag for retrain


def init_db(path: str = DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            event_id TEXT PRIMARY KEY,
            predicted_at TEXT,
            corridor TEXT,
            predicted_risk_score REAL,
            predicted_duration_class TEXT,
            predicted_duration_min REAL,
            predicted_closure INTEGER,
            actual_duration_min REAL,
            actual_closure INTEGER,
            outcome_logged_at TEXT,
            duration_class_match INTEGER,
            closure_match INTEGER
        )
    """)
    conn.commit()
    return conn


def log_prediction(conn, event_id, corridor, predicted_risk_score, predicted_duration_class,
                    predicted_duration_min, predicted_closure):
    conn.execute("""
        INSERT OR REPLACE INTO predictions
        (event_id, predicted_at, corridor, predicted_risk_score, predicted_duration_class,
         predicted_duration_min, predicted_closure)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (event_id, datetime.now(timezone.utc).isoformat(), corridor, predicted_risk_score,
          predicted_duration_class, predicted_duration_min, int(predicted_closure)))
    conn.commit()


def log_outcome(conn, event_id, actual_duration_min, actual_closure):
    row = conn.execute("SELECT predicted_duration_class, predicted_closure FROM predictions WHERE event_id=?",
                        (event_id,)).fetchone()
    if row is None:
        return False
    pred_class, pred_closure = row
    actual_class = "slow" if actual_duration_min > 120 else "fast"
    duration_match = int(actual_class == pred_class)
    closure_match = int(int(actual_closure) == int(pred_closure))
    conn.execute("""
        UPDATE predictions SET actual_duration_min=?, actual_closure=?,
        outcome_logged_at=?, duration_class_match=?, closure_match=?
        WHERE event_id=?
    """, (actual_duration_min, int(actual_closure), datetime.now(timezone.utc).isoformat(),
          duration_match, closure_match, event_id))
    conn.commit()
    return True


def get_accuracy_trend(conn) -> dict:
    df = pd.read_sql("SELECT * FROM predictions WHERE outcome_logged_at IS NOT NULL ORDER BY outcome_logged_at", conn)
    if df.empty:
        return {"n_logged": 0, "duration_class_accuracy": None, "closure_accuracy": None, "retrain_needed": False}
    duration_acc = df["duration_class_match"].mean()
    closure_acc = df["closure_match"].mean()
    return {
        "n_logged": len(df),
        "duration_class_accuracy": round(float(duration_acc), 3),
        "closure_accuracy": round(float(closure_acc), 3),
        "retrain_needed": bool(duration_acc < RETRAIN_TRIGGER_THRESHOLD),
    }


def run_chronological_backtest(n_chunks: int = 6):
    """Walk-forward validation: retrain on everything BEFORE chunk i, evaluate on chunk i.
    This is the real number behind 'accuracy trending up as the model learns'."""
    df = pd.read_parquet(PROCESSED_PATH)
    sub = df[df["is_closed_with_duration"]].copy().sort_values("start_datetime")
    for c in CAT_FEATURES:
        sub[c] = sub[c].astype(str)

    sub["duration_class_bin"] = (sub["duration_class"] == "slow").astype(int)
    n = len(sub)
    chunk_size = n // (n_chunks + 1)  # first chunk reserved as the initial training seed
    boundaries = [chunk_size * i for i in range(1, n_chunks + 2)]
    boundaries[-1] = n

    results = []
    for i in range(1, len(boundaries)):
        train_end = boundaries[i - 1]
        test_end = boundaries[i]
        train = sub.iloc[:train_end]
        test = sub.iloc[train_end:test_end]
        if len(test) == 0 or train["duration_class_bin"].nunique() < 2:
            continue

        Xtr = pd.get_dummies(train[FEATURE_COLS], dummy_na=True)
        Xte = pd.get_dummies(test[FEATURE_COLS], dummy_na=True).reindex(columns=Xtr.columns, fill_value=0)
        ytr = train["duration_class_bin"]
        yte = test["duration_class_bin"]

        n_pos, n_neg = ytr.sum(), len(ytr) - ytr.sum()
        spw = n_neg / max(n_pos, 1)
        model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08,
                                   scale_pos_weight=spw, random_state=42, eval_metric="logloss")
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        acc = accuracy_score(yte, preds)

        results.append({
            "chunk": i,
            "train_size": len(train),
            "test_size": len(test),
            "period_start": str(test["start_datetime"].min()),
            "period_end": str(test["start_datetime"].max()),
            "duration_class_accuracy": round(float(acc), 3),
        })
        print(f"Chunk {i}: trained on {len(train)} events, tested on {len(test)} events "
              f"({test['start_datetime'].min():%Y-%m-%d} to {test['start_datetime'].max():%Y-%m-%d}) "
              f"-> accuracy {acc:.3f}")

    with open(BACKTEST_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved walk-forward backtest -> {BACKTEST_OUT}")
    if len(results) >= 2:
        delta = results[-1]["duration_class_accuracy"] - results[0]["duration_class_accuracy"]
        print(f"Accuracy moved {delta:+.1%} from first retrain window to last "
              f"({results[0]['duration_class_accuracy']:.1%} -> {results[-1]['duration_class_accuracy']:.1%})")
    return results


def seed_ledger_from_history():
    """Populate the ledger with real historical events + their real outcomes so the
    /ledger endpoint has something genuine to show on first run.

    Both predictions are genuine naive-baseline rules evaluated against the same
    rows' real outcomes -- NOT the ground truth echoed back as the "prediction"
    (an earlier version of this function did that for closure, which produced a
    meaningless 100% accuracy. Fixed here to use the actual cause-rate tier rule
    from Model 4, the same rule the recommendation engine uses client-side)."""
    conn = init_db()
    backtest_path = BACKTEST_OUT
    if not os.path.exists(backtest_path):
        run_chronological_backtest()

    rates_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "barricade_cause_rates.json")
    with open(rates_path) as f:
        cause_rates = json.load(f)

    df = pd.read_parquet(PROCESSED_PATH)
    sub = df[df["is_closed_with_duration"]].sample(min(300, df["is_closed_with_duration"].sum()), random_state=42)
    for _, row in sub.iterrows():
        # naive duration prediction = median duration for that cause (simulates what Model 2 would have said)
        pred_dur = df[df.event_cause == row.event_cause]["duration_min"].median()
        pred_class = "slow" if pred_dur > 120 else "fast"

        # naive closure prediction = the cause-rate tier rule (HIGH tier -> always
        # recommend closure, else threshold on historical closure_rate) -- a real
        # rule-based prediction, not ground truth echoed back
        info = cause_rates.get(row.event_cause, {"closure_rate": 0.083, "tier": "LOW"})
        pred_closure = True if info["tier"] == "HIGH" else info["closure_rate"] > 0.15

        log_prediction(conn, row["id"], row["corridor"], None, pred_class, pred_dur, pred_closure)
        log_outcome(conn, row["id"], row["duration_min"], row["requires_road_closure"])
    trend = get_accuracy_trend(conn)
    print("Seeded ledger from history. Current accuracy snapshot:", trend)
    conn.close()
    return trend


if __name__ == "__main__":
    print("=== Walk-forward backtest (real post-event learning curve) ===")
    run_chronological_backtest()
    print("\n=== Seeding SQLite ledger with historical predicted-vs-actual ===")
    seed_ledger_from_history()
