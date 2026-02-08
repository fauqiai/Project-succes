import pandas as pd
import numpy as np


# =====================================
# NORMALIZATION HELPER
# =====================================

def zscore(series, window=100):

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    return (series - mean) / std.replace(0, np.nan)



# =====================================
# BUILD INTERACTIONS (MONSTER)
# =====================================

def build_interactions(features):

    inter = pd.DataFrame(index=features.index)

    vol = features["volatility"]
    comp = features["compression"]
    liq = features["liquidity"]
    mom = features["momentum"]
    rej = features["rejection"]

    # =====================================
    # EXPLOSION DETECTOR
    # compression + rising volatility
    # =====================================

    inter["vol_compression_break"] = zscore(vol * comp)

    # =====================================
    # LIQUIDITY EXPANSION
    # sweep + movement
    # =====================================

    inter["liquidity_expansion"] = zscore(liq * vol)

    # =====================================
    # TREND IGNITION
    # momentum confirmed by volatility
    # =====================================

    inter["trend_ignition"] = zscore(mom * vol)

    # =====================================
    # POSSIBLE EXHAUSTION
    # rejection inside high vol
    # =====================================

    inter["exhaustion_risk"] = zscore(rej * vol)

    # =====================================
    # TRAP DETECTOR
    # liquidity spike but weak momentum
    # =====================================

    trap_raw = liq * (1 / (abs(mom) + 0.0001))
    inter["trap_risk"] = zscore(trap_raw)

    # =====================================
    # CHAOS INDEX
    # volatility + rejection
    # =====================================

    chaos = vol * rej
    inter["chaos_index"] = zscore(chaos)

    # =====================================
    # TREND QUALITY
    # momentum without chaos
    # =====================================

    trend_quality = mom / (rej + 0.0001)
    inter["trend_quality"] = zscore(trend_quality)

    # =====================================
    # COMPRESSION PRESSURE
    # market storing energy
    # =====================================

    pressure = comp * (1 / (vol + 0.0001))
    inter["compression_pressure"] = zscore(pressure)

    return inter.replace([np.inf, -np.inf], np.nan).fillna(0)



# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("MONSTER INTERACTION ENGINE TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix

    size = 1500
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    f = build_feature_matrix(df)

    inter = build_interactions(f)

    print(inter.head())

    print("\nFeatures created:", len(inter.columns))

    print("PASSED ✅")
