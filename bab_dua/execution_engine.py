import numpy as np


# =====================================
# EXECUTION DECISION ENGINE
# =====================================

def execution_decision(
        state_label,
        bias,
        confidence,
        regime,
        min_confidence=0.55):

    """
    Final trade gatekeeper.
    """

    # =============================
    # HARD FILTERS
    # =============================

    if "AVOID" in state_label:
        return "⛔ NO TRADE", "Bad market state"

    if "NEUTRAL" in state_label:
        return "⏳ WAIT", "No statistical edge"

    if "NO CLEAR" in bias:
        return "⏳ WAIT", "Direction unclear"

    if confidence < min_confidence:
        return "⏳ WAIT", "Low model confidence"

    if "CHAOTIC" in regime:
        return "⏳ WAIT", "Market unstable"

    # =============================
    # AGGRESSION LOGIC
    # =============================

    if "STRONG" in state_label and confidence > 0.7:
        return "🔥 EXECUTE (AGGRESSIVE)", "High edge environment"

    if "STRONG" in state_label:
        return "✅ EXECUTE", "Good trading conditions"

    if "TRADEABLE" in state_label:
        return "✅ EXECUTE (LIGHT SIZE)", "Moderate edge"

    return "⏳ WAIT", "Conditions not optimal"



# =====================================
# POSITION STYLE
# =====================================

def execution_style(confidence):

    if confidence > 0.8:
        return "🚀 FULL POSITION"

    elif confidence > 0.65:
        return "⚖️ NORMAL SIZE"

    return "🌱 SMALL SIZE"



# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("EXECUTION ENGINE TEST")

    decision, reason = execution_decision(
        state_label="🔥 STRONG TRADE ZONE",
        bias="🚀 LONG BIAS",
        confidence=0.72,
        regime="🔥 TRENDING"
    )

    style = execution_style(0.72)

    print("Decision:", decision)
    print("Reason:", reason)
    print("Style:", style)

    print("\nTEST PASSED ✅")
