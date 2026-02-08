import pandas as pd
import numpy as np


# ============================================================
# HELPERS
# ============================================================

def compute_atr(data, period=14):
    high = data["high"]
    low = data["low"]
    close = data["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    return atr


# ============================================================
# IMPULSE (EXPANSION CANDLE)
# ============================================================

def detect_impulse(data, atr=None, multiplier=1.5):

    if atr is None:
        atr = compute_atr(data)

    range_ = data["high"] - data["low"]

    impulse = range_ > atr * multiplier

    return impulse.fillna(False)


# ============================================================
# RETRACEMENT (PULLBACK)
# ============================================================

def detect_retracement(data, atr=None):

    if atr is None:
        atr = compute_atr(data)

    close = data["close"]

    pullback = close.diff()

    retrace = (
        (pullback.abs() < atr * 0.8) &
        (pullback < 0)
    )

    return retrace.fillna(False)


# ============================================================
# CONSOLIDATION (COMPRESSION)
# ============================================================

def detect_consolidation(data, atr=None, threshold=0.6):

    if atr is None:
        atr = compute_atr(data)

    range_ = data["high"] - data["low"]

    consolidation = range_ < atr * threshold

    return consolidation.fillna(False)


# ============================================================
# VOLATILITY SPIKE  ⭐ NEW FEATURE
# ============================================================

def detect_volatility_spike(data, atr=None):

    if atr is None:
        atr = compute_atr(data)

    range_ = data["high"] - data["low"]

    spike = range_ > atr * 2.5

    return spike.fillna(False)


# ============================================================
# LIQUIDITY SWEEP ⭐ NEW FEATURE
# ============================================================

def detect_liquidity_sweep(data, window=20):

    high = data["high"]
    low = data["low"]
    close = data["close"]

    prev_high = high.rolling(window).max().shift(1)
    prev_low = low.rolling(window).min().shift(1)

    sweep_high = (high > prev_high) & (close < prev_high)
    sweep_low = (low < prev_low) & (close > prev_low)

    sweep = sweep_high | sweep_low

    return sweep.fillna(False)


# ============================================================
# EXHAUSTION ⭐ NEW FEATURE
# ============================================================

def detect_exhaustion(data, atr=None):

    if atr is None:
        atr = compute_atr(data)

    range_ = data["high"] - data["low"]
    body = (data["close"] - data["open"]).abs()

    exhaustion = (
        (range_ > atr * 2) &
        (body < range_ * 0.3)   # long wick
    )

    return exhaustion.fillna(False)


# ============================================================
# BUILD EVENT SEQUENCE (MULTI-DIMENSIONAL)
# ============================================================

def build_event_sequence(data):

    atr = compute_atr(data)

    impulse = detect_impulse(data, atr)
    retrace = detect_retracement(data, atr)
    consolidation = detect_consolidation(data, atr)
    spike = detect_volatility_spike(data, atr)
    sweep = detect_liquidity_sweep(data)
    exhaustion = detect_exhaustion(data, atr)

    events = []

    for i in range(len(data)):

        if spike.iloc[i]:
            events.append("volatility_spike")

        elif sweep.iloc[i]:
            events.append("liquidity_sweep")

        elif exhaustion.iloc[i]:
            events.append("exhaustion")

        elif impulse.iloc[i]:
            events.append("impulse")

        elif retrace.iloc[i]:
            events.append("retracement")

        elif consolidation.iloc[i]:
            events.append("consolidation")

        else:
            events.append("neutral")

    return pd.Series(events, index=data.index)


# ============================================================
# SELF TEST (WAJIB ADA)
# ============================================================

if __name__ == "__main__":

    print("Running event_detection self-test...")

    np.random.seed(1)
    size = 500

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.2
    })

    events = build_event_sequence(df)

    print(events.value_counts())
    print("SELF TEST PASSED ✅")
