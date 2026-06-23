"""
GURU MARS Gridlock 2.0 -- MODEL 4: Barricade Classifier
==========================================================
Predicts requires_road_closure. Baseline accuracy from always predicting False
is 91.7% (676/8173 = 8.3% positive rate), so accuracy is a useless metric here
-- everything is judged on F1/recall for the True class.

Hybrid design:
  1. Cause-rate lookup table (closure rate per event_cause) -- a hard rule for
     HIGH-tier causes (vip_movement 80%, public_event 46%, protest 40%,
     tree_fall 39%) where near-perfect recall matters more than a learned
     boundary.
  2. XGBoost with scale_pos_weight=11.1 (= 7497 neg / 676 pos) for the
     MED-tier causes (construction, procession, road_conditions) where the
     decision depends on corridor/police_station/time-of-day context, not
     just the cause alone.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, accuracy_score
import joblib

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOOKUP_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "barricade_cause_rates.json")

CAT_FEATURES = ["event_cause", "event_type", "priority", "corridor", "police_station"]
NUM_FEATURES = ["hour"]
FEATURE_COLS = CAT_FEATURES + NUM_FEATURES

HIGH_TIER = 0.30
MED_TIER = 0.10


def build_cause_rate_table(df: pd.DataFrame) -> dict:
    rates = df.groupby("event_cause")["requires_road_closure"].mean().sort_values(ascending=False)
    table = {}
    for cause, rate in rates.items():
        tier = "HIGH" if rate > HIGH_TIER else ("MED" if rate >= MED_TIER else "LOW")
        table[cause] = {"closure_rate": round(float(rate), 3), "tier": tier}
    return table


def train():
    df = pd.read_parquet(PROCESSED_PATH).copy()
    print(f"Rows: {len(df)}  pos(closure)={int(df.requires_road_closure.sum())}  "
          f"neg={int((~df.requires_road_closure).sum())}")

    cause_rates = build_cause_rate_table(df)
    os.makedirs(os.path.dirname(LOOKUP_OUT), exist_ok=True)
    with open(LOOKUP_OUT, "w") as f:
        json.dump(cause_rates, f, indent=2)
    print("\nCause-rate lookup table:")
    for cause, info in cause_rates.items():
        print(f"  {cause:20s} {info['closure_rate']*100:5.1f}%  [{info['tier']}]")

    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    X = df[FEATURE_COLS]
    y = df["requires_road_closure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos
    print(f"\nscale_pos_weight = {spw:.2f}")

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\nXGBoost-only -- accuracy: {accuracy_score(y_test, pred):.3f}  "
          f"(baseline always-False accuracy: {1 - y_test.mean():.3f})")
    print(classification_report(y_test, pred, target_names=["no_closure", "closure"]))

    # --- hybrid rule: HIGH-tier causes always flagged, XGBoost decides the rest ---
    test_df = df.loc[X_test.index].copy()
    test_df["xgb_proba"] = proba
    test_df["hybrid_pred"] = np.where(
        test_df["event_cause"].map(lambda c: cause_rates.get(c, {}).get("tier")) == "HIGH",
        1,
        (test_df["xgb_proba"] >= 0.5).astype(int),
    )
    print("\nHybrid (lookup-table hard rule for HIGH tier + XGBoost elsewhere):")
    print(classification_report(y_test, test_df["hybrid_pred"], target_names=["no_closure", "closure"]))
    print(f"Hybrid F1(closure): {f1_score(y_test, test_df['hybrid_pred']):.3f} "
          f"vs XGBoost-only F1(closure): {f1_score(y_test, pred):.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS,
                 "categories": {c: df[c].cat.categories.tolist() for c in CAT_FEATURES},
                 "cause_rates": cause_rates},
                os.path.join(MODEL_DIR, "model4_barricade.joblib"))
    print(f"\nSaved -> {MODEL_DIR}/model4_barricade.joblib")
    return clf, cause_rates


if __name__ == "__main__":
    train()
