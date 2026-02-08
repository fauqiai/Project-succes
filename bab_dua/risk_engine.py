import numpy as np
import pandas as pd


# =====================================
# VOLATILITY BASED STOP
# =====================================

def compute_volatility_stop(features, atr_multiple=2.5):

    vol = features["volatility"].iloc[-1]

    stop_distance = vol * atr_multiple

    return stop_distance


# =====================================
# POSITION SIZING (VERY PRO)
# =====================================

def position_size(
        account_size,
        risk_per_trade,
        stop_distance):

    dollar_risk = account_size * risk_per_trade

    if stop_distance == 0:
        return 0

    size = dollar_risk / stop_distance

    return round(size, 4)


# =====================================
# CONFIDENCE WEIGHTING
# =====================================

def confidence_risk_adjustment(base_risk, confidence):

    """
    Scale risk based on model confidence.
    """

    if confidence > 0.75:
        return base_risk * 1.5

    elif confidence > 0.6:
        return base_risk * 1.2

    elif confidence < 0.4:
        return base_risk * 0.5

    return base_risk


# =====================================
# REGIME RISK FILTER
# =====================================

def regime_risk(regime):

    if "CHAOTIC" in regime:
        return 0.5

    if "TRENDING" in regime:
        return 1.2

    return 1.0


# =====================================
# MASTER RISK ENGINE
# =====================================

def build_risk_model(
        features,
        confidence,
        regime,
        account_size=10000,
        base_risk=0.01):

    stop = compute_volatility_stop(features)

    adjusted_risk = confidence_risk_adjustment(
        base_risk,
        confidence
    )

    adjusted_risk *= regime_risk(regime)

    size = position_size(
        account_size,
        adjusted_risk,
        stop
    )

    take_profit = stop * 2.5   # asymmetric payoff

    rr = take_profit / stop if stop != 0 else 0

    return {
        "stop_distance": round(stop, 5),
        "take_profit_distance": round(take_profit, 5),
        "risk_percent": round(adjusted_risk * 100, 2),
        "position_size": size,
        "risk_reward": round(rr, 2)
    }


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("RISK ENGINE TEST")

    size = 1500
    rng = np.random.randn(size)

    df = pd.DataFrame({
        "volatility": abs(rng) + 0.5
    })

    result = build_risk_model(
        df,
        confidence=0.68,
        regime="🔥 TRENDING",
        account_size=10000
    )

    print(result)

    print("\nTEST PASSED ✅")
