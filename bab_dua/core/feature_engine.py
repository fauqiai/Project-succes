import pandas as pd
import numpy as np
from .measures import *


def build_feature_matrix(df):

    f = pd.DataFrame(index=df.index)

    # =====================
    # ORIGINAL MICRO FEATURES (tetap dipakai)
    # =====================
    f["volatility"] = volatility_ratio(df)
    f["expansion"] = expansion_ratio(df)
    f["momentum"] = momentum(df)
    f["liquidity"] = liquidity_penetration(df)
    f["rejection"] = wick_pressure(df)

    # hindari division zero
    f["compression"] = 1 / f["expansion"].replace(0, 1)

    # =====================
    # 🔥 CONTEXT FEATURES (multi-candle regime)
    # =====================

    close = df["close"]

    # --- Trend strength (EMA distance normalized)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    f["trend_fast"] = (close - ema20) / ema20.replace(0, 1)
    f["trend_slow"] = (ema20 - ema50) / ema50.replace(0, 1)

    # --- Rolling volatility regime
    ret = close.pct_change().fillna(0)
    f["vol_regime_20"] = ret.rolling(20).std().fillna(0)
    f["vol_regime_50"] = ret.rolling(50).std().fillna(0)

    # --- Range position in recent window (structure awareness)
    high_roll = df["high"].rolling(20).max()
    low_roll = df["low"].rolling(20).min()
    rng = (high_roll - low_roll).replace(0, 1)
    f["range_pos_20"] = (close - low_roll) / rng

    # --- Candle size relative to ATR-like range
    atr_like = (df["high"] - df["low"]).rolling(20).mean().replace(0, 1)
    f["candle_size_rel"] = (df["high"] - df["low"]) / atr_like

    # --- Momentum persistence (short vs medium return)
    f["ret_5"] = close.pct_change(5)
    f["ret_15"] = close.pct_change(15)

    # =====================
    # CLEANUP
    # =====================
    f = f.replace([np.inf, -np.inf], 0)
    return f.fillna(0)


if __name__ == "__main__":

    print("FEATURE ENGINE TEST")

    size = 500
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    print(build_feature_matrix(df).head())
    print("PASSED ✅")
