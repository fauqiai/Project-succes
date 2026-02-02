"""
microstructure.py
-----------------
Module untuk mendeteksi microstructure berbasis OHLC.
Semua komentar ASCII agar aman untuk Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. LIQUIDITY SWEEP (STOP HUNT)
# ============================================================

def detect_sweep(data, wick_ratio=0.6):
    """
    Deteksi liquidity sweep.
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]
    open_ = data["open"]
    close = data["close"]

    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    range_ = high - low

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    sweep_high = (
        (high > prev_high) &
        (upper_wick / range_ > wick_ratio) &
        (close < prev_high)
    )

    sweep_low = (
        (low < prev_low) &
        (lower_wick / range_ > wick_ratio) &
        (close > prev_low)
    )

    return (sweep_high | sweep_low).fillna(False)


def sweep_points(data, wick_ratio=0.6):
    mask = detect_sweep(data, wick_ratio)
    return list(data.index[mask])


# ============================================================
# 2. IMBALANCE CANDLE
# ============================================================

def detect_imbalance(data, body_threshold=0.6):
    """
    Output: Series boolean
    """

    open_ = data["open"]
    close = data["close"]
    high = data["high"]
    low = data["low"]

    body = (close - open_).abs()
    range_ = high - low

    body_ratio = body / range_

    imbalance = body_ratio > body_threshold
    return imbalance.fillna(False)


def imbalance_zones(data):
    zones = []
    mask = detect_imbalance(data)

    for i in data.index[mask]:
        zones.append((i, i))

    return zones


# ============================================================
# 3. FAIR VALUE GAP (FVG)
# ============================================================

def detect_fvg(data):
    """
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]

    fvg_bullish = high.shift(2) < low
    fvg_bearish = low.shift(2) > high

    return (fvg_bullish | fvg_bearish).fillna(False)


def fvg_zones(data):
    zones = []

    high = data["high"]
    low = data["low"]

    for i in range(2, len(data)):
        if high.iloc[i - 2] < low.iloc[i]:
            zones.append((data.index[i - 2], data.index[i]))
        elif low.iloc[i - 2] > high.iloc[i]:
            zones.append((data.index[i - 2], data.index[i]))

    return zones


# ============================================================
# 4. DISPLACEMENT CANDLE
# ============================================================

def detect_displacement(data, multiplier=1.5):
    """
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]

    range_ = high - low
    avg_range = range_.rolling(20, min_periods=1).mean()

    displacement = range_ > avg_range * multiplier
    return displacement.fillna(False)


def displacement_points(data, multiplier=1.5):
    mask = detect_displacement(data, multiplier)
    return list(data.index[mask])


# ============================================================
# 5. SWING FAILURE PATTERN (SFP)
# ============================================================

def detect_sfp(data, lookback=5):
    """
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]
    close = data["close"]

    prev_high = high.rolling(lookback, min_periods=1).max().shift(1)
    prev_low = low.rolling(lookback, min_periods=1).min().shift(1)

    sfp_high = (high > prev_high) & (close < prev_high)
    sfp_low = (low < prev_low) & (close > prev_low)

    return (sfp_high | sfp_low).fillna(False)


def sfp_points(data, lookback=5):
    mask = detect_sfp(data, lookback)
    return list(data.index[mask])


# ============================================================
# 6. ORDERFLOW SHIFT (SIMPLE VERSION)
# ============================================================

def detect_orderflow_shift(data, lookback=3):
    """
    Output: Series boolean
    """

    open_ = data["open"]
    close = data["close"]

    body = close - open_
    direction = np.sign(body)

    prev_dir = (
        direction.rolling(lookback, min_periods=1)
        .apply(lambda x: np.sign(x[:-1].sum()), raw=False)
    )

    shift = (direction != prev_dir) & (body.abs() > body.abs().rolling(lookback).mean())
    return shift.fillna(False)


def orderflow_shift_points(data, lookback=3):
    mask = detect_orderflow_shift(data, lookback)
    return list(data.index[mask])


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    size = 100

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    print("Sweep points:", sweep_points(df))
    print("Imbalance zones:", imbalance_zones(df)[:5])
    print("FVG zones:", fvg_zones(df)[:5])
    print("Displacement points:", displacement_points(df))
    print("SFP points:", sfp_points(df))
    print("Orderflow shift points:", orderflow_shift_points(df))

    print("microstructure.py self-test OK")

