"""
GURU MARS Gridlock 2.0 -- MODEL 2: Duration (2-stage)
=======================================================
Stage 1: XGBoost classifier -> fast (<=120min) vs slow (>120min)
Stage 2: Random Forest regressor on log1p(duration), one fit per class
         (RF chosen over a second XGBoost specifically for the slow class:
         only ~1,006 samples, RF's bagging is more stable than boosting at
         small n and gives free OOB error without burning a holdout split)

Kruskal-Wallis test on log-duration by event_cause confirms cause is a very
strong duration signal (H ~ 1219-1234, p ~ 1e-257 depending on exact grouping/
filtering) -- this justifies splitting fast/slow by cause-dominated behavior
before regressing, rather than regressing duration directly across all causes.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_absolute_error
from scipy.stats import kruskal
import joblib

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "events_processed.parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CAT_FEATURES = ["event_cause", "corridor", "priority", "police_station",
                 "requires_road_closure", "veh_type", "zone_backfilled"]
NUM_FEATURES = ["hour"]
FEATURE_COLS = CAT_FEATURES + NUM_FEATURES


def prep(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["is_closed_with_duration"]].copy()
    sub["requires_road_closure"] = sub["requires_road_closure"].astype(str)
    for c in CAT_FEATURES:
        sub[c] = sub[c].astype("category")
    return sub


def train():
    df = pd.read_parquet(PROCESSED_PATH)
    sub = prep(df)
    print(f"Closed events with valid duration: {len(sub)}  "
          f"(fast={int((sub.duration_class=='fast').sum())}, slow={int((sub.duration_class=='slow').sum())})")

    # --- statistical justification for the 2-stage split ---
    groups = [g["log_duration_min"].values for _, g in sub.groupby("event_cause") if len(g) >= 5]
    H, p = kruskal(*groups)
    print(f"Kruskal-Wallis (event_cause -> log_duration): H={H:.1f}, p={p:.3e}")

    y_class = (sub["duration_class"] == "slow").astype(int)  # 1 = slow
    X = sub[FEATURE_COLS]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_class, sub.index, test_size=0.2, random_state=42, stratify=y_class
    )

    # ---------------- Stage 1: fast vs slow classifier ----------------
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        scale_pos_weight=n_neg / n_pos,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    pred_class = clf.predict(X_test)
    acc = accuracy_score(y_test, pred_class)
    f1 = f1_score(y_test, pred_class)
    print(f"\nStage 1 (fast/slow classifier) -- test accuracy: {acc:.3f}, F1(slow): {f1:.3f}")
    print(classification_report(y_test, pred_class, target_names=["fast", "slow"]))

    # ---------------- Stage 2: RF regressors on log-duration, per class ----------------
    sub_train = sub.loc[idx_train]
    sub_test = sub.loc[idx_test]

    rf_models = {}
    for cls_name, cls_val in [("fast", 0), ("slow", 1)]:
        mask = sub_train["duration_class"] == cls_name
        Xc = pd.get_dummies(sub_train.loc[mask, FEATURE_COLS], dummy_na=True)
        yc = sub_train.loc[mask, "log_duration_min"]
        rf = RandomForestRegressor(
            n_estimators=400, max_depth=10, min_samples_leaf=3,
            random_state=42, oob_score=True, n_jobs=-1,
        )
        rf.fit(Xc, yc)
        rf_models[cls_name] = {"model": rf, "columns": Xc.columns.tolist()}
        print(f"Stage 2 RF ({cls_name}-class) -- OOB R2: {rf.oob_score_:.3f}, n={mask.sum()}")

    # ---------------- End-to-end evaluation on held-out test set ----------------
    def predict_duration(X_row_df, predicted_class_arr):
        preds = []
        for i, cls_val in enumerate(predicted_class_arr):
            cls_name = "slow" if cls_val == 1 else "fast"
            cols = rf_models[cls_name]["columns"]
            xi = pd.get_dummies(X_row_df.iloc[[i]][FEATURE_COLS], dummy_na=True)
            xi = xi.reindex(columns=cols, fill_value=0)
            log_pred = rf_models[cls_name]["model"].predict(xi)[0]
            preds.append(np.expm1(log_pred))
        return np.array(preds)

    final_preds_min = predict_duration(sub_test, pred_class)
    actual_min = sub_test["duration_min"].values
    mae_all = mean_absolute_error(actual_min, final_preds_min)
    # median-relative MAE is more honest than raw MAE given pothole-class 200k-min outliers
    print(f"\nEnd-to-end duration MAE (all test rows, raw minutes): {mae_all:.1f} min "
          f"(skewed by long-tail causes like pot_holes/Debris -- see median below)")
    print(f"Median abs error: {np.median(np.abs(actual_min - final_preds_min)):.1f} min")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "stage1_classifier": clf,
        "stage2_rf_models": rf_models,
        "feature_cols": FEATURE_COLS,
        "cat_features": CAT_FEATURES,
    }, os.path.join(MODEL_DIR, "model2_duration.joblib"))
    print(f"Saved -> {MODEL_DIR}/model2_duration.joblib")

    # median duration by cause (for the recommendation engine + judge story)
    median_by_cause = sub.groupby("event_cause")["duration_min"].median().sort_values()
    median_by_cause.to_json(os.path.join(os.path.dirname(PROCESSED_PATH), "duration_median_by_cause.json"))
    print("\nMedian resolution time by cause (minutes):")
    print(median_by_cause)

    return clf, rf_models


if __name__ == "__main__":
    train()
