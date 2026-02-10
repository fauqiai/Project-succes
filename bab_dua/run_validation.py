# run_validation.py

import pickle

from core.edge_validator import walkforward_edge_validation, edge_health
from core.regime_validator import validate_regimes, regime_health_score
from core.uncertainty_engine import bootstrap_confidence
from core.regime_shift_engine import detect_regime_shift, regime_shift_alert


def main():

    print("\n🔥 STEP 2 — VALIDATING EDGE\n")

    # =====================
    # LOAD RESEARCH
    # =====================

    with open("research_output.pkl", "rb") as f:
        data = pickle.load(f)

    df = data["df"]
    state = data["state"]

    # =====================
    # WALKFORWARD
    # =====================

    wf = walkforward_edge_validation(df, state)
    wf_health = edge_health(wf)

    print("\nWalkforward Health:", wf_health)

    # =====================
    # REGIME VALIDATION
    # =====================

    regime_report = validate_regimes(df, state)
    regime_health = regime_health_score(regime_report)

    print("Regime Quality:", regime_health)

    # =====================
    # CONFIDENCE
    # =====================

    confidence = bootstrap_confidence(wf["avg_edge"])

    print("Confidence:", round(confidence["confidence"], 3))

    # =====================
    # SHIFT DETECTION
    # =====================

    shift = detect_regime_shift(df.close.pct_change())
    shift_status = regime_shift_alert(shift)

    print("Market Shift:", shift_status)

    # =====================
    # FINAL DECISION
    # =====================

    if (
        wf_health == "REAL EDGE 🔥"
        and regime_health in ["STRONG", "DECENT"]
        and confidence["confidence"] > 0.8
        and shift_status == "STABLE"
    ):
        verdict = "✅ EDGE APPROVED"

    elif wf_health in ["POSSIBLE EDGE", "WEAK EDGE"]:
        verdict = "⚠️ EDGE WEAK"

    else:
        verdict = "⛔ EDGE REJECTED"

    print("\n============================")
    print("FINAL VERDICT:", verdict)
    print("============================")


if __name__ == "__main__":
    main()
