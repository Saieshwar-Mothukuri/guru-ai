"""
GURU MARS Gridlock 2.0 -- Recommendation Engine (Gap 2: data-driven deployment)
=================================================================================
Ties Models 1-4 together into the single recommendation card the controller
sees and approves with one click: unit count, barricade flag, diversion route.

Two honesty notes, stated up front rather than hidden in code comments:

  1. Manpower allocation here is a transparent RULE on top of model outputs
     (risk tier, predicted closure, priority), not a learned "manpower model" --
     because the raw export has no historical "units dispatched" column to
     train one against. The rule is documented below so it can be replaced
     with a learned allocator the moment that data exists (e.g. from GURU
     MARS's existing geo-tagged officer e-attendance log).
  2. "Nearest geo-tagged officer" and "MapmyIndia diversion route" both
     require live external feeds (GURU MARS's own officer GPS system, and a
     MapmyIndia API key) that aren't available in this offline environment.
     Both are implemented as clearly-labeled stubs with the real integration
     point documented, not faked as if they were live data.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model5_planned_lookup import query_neighbors  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class GuruMarsRecommendationEngine:
    def __init__(self):
        with open(os.path.join(PROCESSED_DIR, "risk_surface_lookup.json")) as f:
            self.risk_lookup = json.load(f)
        with open(os.path.join(PROCESSED_DIR, "cascade_edges.json")) as f:
            self.cascade_edges = json.load(f)
        with open(os.path.join(PROCESSED_DIR, "barricade_cause_rates.json")) as f:
            self.cause_rates = json.load(f)
        with open(os.path.join(PROCESSED_DIR, "duration_median_by_cause.json")) as f:
            self.median_duration_by_cause = json.load(f)

        self.model4 = joblib.load(os.path.join(MODEL_DIR, "model4_barricade.joblib"))
        self.model5_data = joblib.load(os.path.join(MODEL_DIR, "model5_planned_lookup.joblib"))

        # Officer IDs are the real top-4 most-frequently-assigned officers from the
        # assigned_to_police_id column in the ASTraM export (FKUSR01441 has 52
        # assignments, FKUSR00292 has 4, etc.). Names and live ETAs are NOT in the
        # dataset -- production reads those from GURU MARS's geo-tagged e-attendance.
        self.MOCK_OFFICER_ROSTER = [
            {"officer_id": "FKUSR01441", "name": "FKUSR01441", "corridor_zone": "West Zone 1", "eta_min": 6},
            {"officer_id": "FKUSR00292", "name": "FKUSR00292", "corridor_zone": "West Zone 2", "eta_min": 9},
            {"officer_id": "FKUSR01429", "name": "FKUSR01429", "corridor_zone": "Central Zone 1", "eta_min": 11},
            {"officer_id": "FKUSR01540", "name": "FKUSR01540", "corridor_zone": "North Zone 1", "eta_min": 14},
        ]

    # ---------- Gap 1: risk score ----------
    def get_risk(self, corridor: str, hour: int, dow: int) -> dict:
        key = f"{corridor}|{hour}|{dow}"
        result = self.risk_lookup.get(key)
        if result is None:
            return {"error": f"No data for corridor '{corridor}' -- check corridor name"}
        return {"corridor": corridor, "hour": hour, "dow": dow, "dow_name": DOW_NAMES[dow], **result}

    # ---------- duration estimate ----------
    # Cross-cause fallback map for causes with zero closed events in the 6-month window.
    # Based on event profile similarity:
    #   public_event  → procession (both are planned gatherings, similar resolution dynamics)
    #   vip_movement  → protest    (both require road holding, similar short-window durations)
    # Overall planned/unplanned medians used as final backstop.
    DURATION_FALLBACK = {
        "public_event":     ("procession",  "cross-cause estimate: public events profile like processions in this dataset"),
        "vip_movement":     ("protest",     "cross-cause estimate: VIP movements profile like protests in this dataset"),
        "fog_low_visibility":("accident",   "cross-cause estimate: visibility incidents resolve similarly to accidents"),
    }

    def get_duration_estimate(self, event_cause: str) -> dict:
        med = self.median_duration_by_cause.get(event_cause)
        if med is not None:
            return {"event_cause": event_cause, "median_duration_min": round(med, 1),
                    "median_duration_human": self._humanize_minutes(med),
                    "note": f"direct median from {event_cause} closed events in dataset"}

        # Cross-cause fallback for causes with 0 closed events
        fallback = self.DURATION_FALLBACK.get(event_cause)
        if fallback:
            proxy_cause, note = fallback
            proxy_med = self.median_duration_by_cause.get(proxy_cause)
            if proxy_med is not None:
                return {"event_cause": event_cause, "median_duration_min": round(proxy_med, 1),
                        "median_duration_human": self._humanize_minutes(proxy_med),
                        "note": note, "proxy_cause": proxy_cause}

        # Final backstop: overall median from all closed events (~64 min)
        all_medians = list(self.median_duration_by_cause.values())
        backstop = sorted(all_medians)[len(all_medians)//2]
        return {"event_cause": event_cause, "median_duration_min": round(backstop, 1),
                "median_duration_human": self._humanize_minutes(backstop),
                "note": "insufficient closed-event history for this cause — using overall median as backstop"}

    @staticmethod
    def _humanize_minutes(m: float) -> str:
        if m < 60:
            return f"{m:.0f} min"
        if m < 1440:
            return f"{m/60:.1f} hr"
        return f"{m/1440:.1f} days"

    # ---------- Gap 4 (barricade half of Gap 2): closure prediction ----------
    def get_barricade_recommendation(self, event_cause: str, corridor: str = None,
                                      police_station: str = None, priority: str = "High",
                                      hour: int = 12) -> dict:
        cause_info = self.cause_rates.get(event_cause, {"closure_rate": 0.083, "tier": "LOW"})
        if cause_info["tier"] == "HIGH":
            recommend_closure = True
            basis = f"hard rule: {event_cause} closes roads {cause_info['closure_rate']*100:.0f}% of the time historically"
        else:
            # fall back to the trained XGBoost for MED/LOW tier causes, where context
            # (corridor, station, time) matters more than the cause alone
            try:
                model = self.model4["model"]
                cats = self.model4["categories"]
                row = pd.DataFrame([{
                    "event_cause": event_cause,
                    "event_type": "unplanned",
                    "priority": priority,
                    "corridor": corridor or "Unknown",
                    "police_station": police_station or cats["police_station"][0],
                    "hour": hour,
                }])
                for c in self.model4["feature_cols"][:-1]:
                    valid_cats = cats.get(c, list(row[c].unique()))
                    val = row.at[0, c] if row.at[0, c] in valid_cats else valid_cats[0]
                    row[c] = pd.Categorical([val], categories=valid_cats)
                proba = float(model.predict_proba(row[self.model4["feature_cols"]])[0, 1])
                recommend_closure = proba >= 0.5
                basis = f"XGBoost model: {proba:.0%} closure probability given context"
            except Exception as e:
                recommend_closure = cause_info["closure_rate"] > 0.15
                basis = f"fallback cause-rate threshold ({e})"
        return {
            "event_cause": event_cause,
            "recommend_barricade": bool(recommend_closure),
            "historical_closure_rate": cause_info["closure_rate"],
            "tier": cause_info["tier"],
            "basis": basis,
        }

    # ---------- Gap 3: cascade ----------
    def get_cascade_alerts(self, corridor: str) -> list:
        return [e for e in self.cascade_edges if e["from"] == corridor]

    # ---------- Gap 2: manpower allocator (rule-based, documented above) ----------
    def allocate_manpower(self, risk_tier: str, recommend_barricade: bool, priority: str) -> dict:
        units = 1
        reasons = ["baseline 1 unit for any logged incident"]
        if risk_tier in ("High", "Critical"):
            units += 1
            reasons.append(f"+1 unit: {risk_tier} risk corridor-hour")
        if recommend_barricade:
            units += 1
            reasons.append("+1 unit: barricade/closure expected, needs a unit to manage diversion at the barrier")
        if priority == "High":
            units += 1
            reasons.append("+1 unit: High priority incident")
        units = min(units, 5)

        assigned = self.MOCK_OFFICER_ROSTER[:units]
        return {
            "recommended_units": units,
            "reasoning": reasons,
            "officers_to_dispatch": assigned,
            "note": "Officer roster is MOCKED for this demo -- production reads live "
                    "geo-tagged positions from GURU MARS's existing e-attendance feed.",
        }

    # ---------- diversion stub ----------
    def suggest_diversion(self, corridor: str) -> dict:
        cascade_targets = self.get_cascade_alerts(corridor)
        suggested_avoid = [e["to"] for e in cascade_targets]
        return {
            "diversion_api": "MapmyIndia Directions API (not connected in this offline build)",
            "corridors_to_pre_alert": suggested_avoid,
            "note": "In production this calls the MapmyIndia Directions API for a live "
                    "alternate route; here we surface the cascade-predicted downstream "
                    "corridor(s) so the controller knows which routes to avoid recommending.",
        }

    # ---------- full recommendation card (the demo's "Step 4") ----------
    def recommend(self, corridor: str, hour: int, dow: int, event_cause: str = None,
                   priority: str = "High", police_station: str = None) -> dict:
        risk = self.get_risk(corridor, hour, dow)
        cascade = self.get_cascade_alerts(corridor)
        barricade = None
        duration = None
        if event_cause:
            barricade = self.get_barricade_recommendation(event_cause, corridor, police_station, priority, hour)
            duration = self.get_duration_estimate(event_cause)

        risk_tier = risk.get("risk_tier", "Low")
        manpower = self.allocate_manpower(
            risk_tier, recommend_barricade=bool(barricade and barricade["recommend_barricade"]), priority=priority
        )
        diversion = self.suggest_diversion(corridor)

        return {
            "query": {"corridor": corridor, "hour": hour, "dow": dow, "dow_name": DOW_NAMES[dow],
                      "event_cause": event_cause, "priority": priority},
            "risk": risk,
            "duration": duration,
            "barricade": barricade,
            "cascade_alerts": cascade,
            "manpower": manpower,
            "diversion": diversion,
        }

    # ---------- planned events (Model 5) ----------
    def recommend_planned(self, event_cause: str, corridor: str, police_station: str,
                           priority: str, hour: int, dow: int) -> dict:
        planned_df = self.model5_data["planned_events"]
        result = query_neighbors(planned_df, {
            "event_cause": event_cause, "corridor": corridor, "police_station": police_station,
            "priority": priority, "hour": hour, "dow": dow,
        })
        manpower = self.allocate_manpower(
            risk_tier="High" if result["closure_rate"] > 0.5 else "Moderate",
            recommend_barricade=result["closure_rate"] >= 0.5,
            priority=priority,
        )
        return {"query": {"event_cause": event_cause, "corridor": corridor, "dow_name": DOW_NAMES[dow], "hour": hour},
                "knn_analog_result": result, "manpower": manpower}


if __name__ == "__main__":
    engine = GuruMarsRecommendationEngine()
    print("=" * 70)
    print("DEMO: Mysore Road, Thursday, 9PM (unplanned vehicle_breakdown)")
    print("=" * 70)
    rec = engine.recommend(corridor="Mysore Road", hour=21, dow=3,
                            event_cause="vehicle_breakdown", priority="High",
                            police_station="Halasuru Gate")
    print(json.dumps(rec, indent=2, default=str))

    print("\n" + "=" * 70)
    print("DEMO: real cascade pair found in data -- Mysore Road -> ORR East 1")
    print("=" * 70)
    print(json.dumps(engine.get_cascade_alerts("Mysore Road"), indent=2))
