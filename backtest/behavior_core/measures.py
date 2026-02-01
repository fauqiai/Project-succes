import pandas as pd
import numpy as np

# ============================================================
# 1. VOLATILITY MEASURES
# ============================================================

def atr(data, period=14):
    high = data["high"]
    low = data["low"]
    close = data["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def rolling_std(data, period=20):
    return data["close"].rolling(period).std()


def volatility_score(data):
    atr_val = atr(data, 14)
    std_val = rolling_std(data, 20)
    rng = data["high"] - data["low"]

    atr_n = atr_val / atr_val.rolling(50).mean()
    std_n = std_val / std_val.rolling(50).mean()
    rng_n = rng / rng.rolling(50).mean()

    return (atr_n.fillna(0) + std_n.fillna(0) + rng_n.fillna(0)) / 3.0

# ============================================================
# 2. CANDLE AND PRICE RANGE MEASURES
# ============================================================

def body_size(data):
    return (data["close"] - data["open"]).abs()


def wick_size(data):
    upper = data["high"] - data[["open", "close"]].max(axis=1)
    lower = data[["open", "close"]].min(axis=1) - data["low"]
    return upper.abs() + lower.abs()


def candle_range(data):
    return data["high"] - data["low"]


def range_ratio(data):
    rng = candle_range(data)
    body = body_size(data)
    return body / rng.replace(0, np.nan)

# ============================================================
# 3. RANGE COMPRESSION OR SQUEEZE MEASURES
# ============================================================

def range_compression(data, window=5):
    atr_val = atr(data, 14)
    body = body_size(data)
    rng = candle_range(data)

    small_body = body < body.rolling(window).mean() * 0.7
    small_range = rng < rng.rolling(window).mean() * 0.7
    atr_down = atr_val < atr_val.rolling(window).mean()

    return (small_body & small_range & atr_down).astype(int)


def small_candle_cluster(data, threshold=0.3, window=4):
    rng = candle_range(data)
    body = body_size(data)

    ratio = body / rng.replace(0, np.nan)
    small = (ratio < threshold).astype(int)

    cluster = small.rolling(window).sum() == window
    return cluster.astype(int)


def atr_compression(data, lookback=10):
    atr_val = atr(data, 14)
    slope = atr_val.diff().rolling(lookback).sum()
    return (slope < 0).astype(int)

# ============================================================
# 4. IMPULSE SCORE MEASURES
# ============================================================

def impulse_score(data, multiplier=1.5):
    body = body_size(data)
    atr_val = atr(data, 14)
    ratio = body / atr_val.replace(0, np.nan)
    return (ratio > multiplier).astype(int)

# ============================================================
# 5. RETRACEMENT MEASURES
# ============================================================

def retrace_amount(high_point, low_point, current_price):
    total = high_point - low_point
    if total == 0:
        return 0.0
    return ((high_point - current_price) / total) * 100.0


def retrace_depth(data, window=10):
    high_roll = data["high"].rolling(window).max()
    low_roll = data["low"].rolling(window).min()
    total = high_roll - low_roll
    return (high_roll - data["close"]) / total.replace(0, np.nan)

# ============================================================
# 6. COMBINED MEASURES
# ============================================================

def compression_score(data):
    rc = range_compression(data)
    ac = atr_compression(data)
    sc = small_candle_cluster(data)
    return (rc + ac + sc) / 3.0


def momentum_score(data):
    body = body_size(data)
    rng = candle_range(data)
    direction = np.where(data["close"] > data["open"], 1, -1)

    body_n = body / body.rolling(20).mean()
    rng_n = rng / rng.rolling(20).mean()

    return (body_n.fillna(0) + rng_n.fillna(0)) * direction

# ============================================================
# MAIN (SELF TEST)
# ============================================================

if __name__ == "__main__":
    print("Running measures.py self-test...")

    data = pd.DataFrame({
        "open":  np.random.rand(100) * 100,
        "high":  np.random.rand(100) * 100 + 1,
        "low":   np.random.rand(100) * 100,
        "close": np.random.rand(100) * 100
    })

    atr(data)
    rolling_std(data)
    volatility_score(data)
    body_size(data)
    wick_size(data)
    candle_range(data)
    range_ratio(data)
    range_compression(data)
    small_candle_cluster(data)
    atr_compression(data)
    impulse_score(data)
    retrace_depth(data)
    compression_score(data)
    momentum_score(data)

    print("OK - measures.py clean, no error")
