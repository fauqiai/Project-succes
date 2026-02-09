import numpy as np


# =====================================
# MAIN FUNCTION (IMPORTANT)
# =====================================

def compute_exit(
    df,
    features,
    state_label,
    bias
):

    current_price = df["close"].iloc[-1]

    volatility = features["volatility"].iloc[-1]

    # behavior-based move estimate
    avg_move = abs(features["momentum"].rolling(50).mean().iloc[-1]) * 10000

    if np.isnan(avg_move) or avg_move == 0:
        avg_move = current_price * 0.001   # fallback


    vol_multiplier = np.clip(volatility * 10, 0.8, 2.5)

    tp_distance = avg_move * vol_multiplier
    sl_distance = tp_distance * 0.6


    if bias == "LONG":

        tp = current_price + tp_distance
        sl = current_price - sl_distance

    elif bias == "SHORT":

        tp = current_price - tp_distance
        sl = current_price + sl_distance

    else:
        tp = None
        sl = None


    return {
        "tp": round(tp, 3) if tp else None,
        "sl": round(sl, 3) if sl else None,
        "tp_distance": round(tp_distance, 3),
        "sl_distance": round(sl_distance, 3)
    }



# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    import pandas as pd

    print("\nEXIT ENGINE TEST\n")

    df = pd.DataFrame({
        "close": np.linspace(1900, 1950, 200)
    })

    features = pd.DataFrame({
        "volatility": np.random.rand(200) * 0.3,
        "momentum": np.random.randn(200) * 0.0001
    })

    result = compute_exit(
        df,
        features,
        "STRONG_TRADE",
        "LONG"
    )

    print(result)

    print("\n✅ TEST PASSED\n")
