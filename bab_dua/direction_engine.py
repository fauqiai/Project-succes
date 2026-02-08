import pandas as pd
import numpy as np


# =====================================
# HELPER
# =====================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =====================================
# BUILD DIRECTION SCORE
# =====================================

def compute_direction(features):

    df = features.copy()

    # Core drivers
    momentum = df["momentum"]
    trend = df.get("trend_strength", momentum)
    volatility = df["volatility"]
    rejection = df["rejection"]

    # Interaction drivers (kalau ada)
    trend_quality = df.get("trend_quality", momentum)
    chaos = df.get("chaos_index", rejection)

    # ===============================
    # RAW PRESSURE
    # ===============================

    directional_pressure = (
        momentum * 0.35 +
        trend * 0.25 +
        trend_quality * 0.25 -
        chaos * 0.15
    )

    # normalize
    score = sigmoid(directional_pressure * 3)

    return score



# =====================================
# INTERPRET DIRECTION
# =====================================

def interpret_direction(score):

    latest = score.iloc[-1]

    if latest > 0.60:
        return "🚀 LONG BIAS", latest

    elif latest < 0.40:
        return "🔻 SHORT BIAS", latest

    else:
        return "⚖️ NO CLEAR EDGE", latest



# =====================================
# MARKET PRESSURE TYPE
# =====================================

def pressure_regime(features):

    vol = features["volatility"].iloc[-1]
    chaos = features.get("chaos_index", pd.Series([0])).iloc[-1]

    if chaos > vol:
        return "🌪 CHAOTIC"

    elif vol > chaos * 1.5:
        return "🔥 TRENDING"

    else:
        return "🌊 BALANCED"



# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("DIRECTION ENGINE TEST")

    size = 1500
    rng = np.random.randn(size)

    df = pd.DataFrame({
        "momentum": rng,
        "trend_strength": rng * 0.8,
        "volatility": abs(rng),
        "rejection": abs(np.random.randn(size)),
        "trend_quality": rng,
        "chaos_index": abs(np.random.randn(size))
    })

    score = compute_direction(df)

    bias, val = interpret_direction(score)
    regime = pressure_regime(df)

    print("Bias:", bias)
    print("Confidence:", round(val, 3))
    print("Regime:", regime)

    print("\nTEST PASSED ✅")
