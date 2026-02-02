"""
regime_detection.py
-------------------
Module untuk mendeteksi regime atau keadaan market:
1. Trend
2. Range
3. Squeeze (volatilitas rendah)
Semua komentar ASCII agar aman untuk Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. TREND DETECTION
# ============================================================

def detect_trend(data, ma_period=20, threshold=0.7):
    """
    Deteksi kondisi trending.
    Output: Series boolean
    """

    close = data["close"]

    ma = close.rolling(ma_period, min_periods=1).mean()
    slope = ma.diff()

    direction = np.sign(close.diff())
    same_dir_ratio = (
        direction.rolling(ma_period, min_periods=1)
        .apply(lambda x: (x == x.iloc[-1]).mean(), raw=False)
    )

    distance = (close - ma).abs() / ma

    trend = (
        (slope.abs() > slope.abs().rolling(ma_period).mean()) &
        (same_dir_ratio > threshold) &
        (distance > 0.001)
    )

    return trend.fillna(False)


def trend_strength(data, ma_period=20):
    """
    Mengukur kekuatan trend menggunakan slope MA
    dan continuity candle.
    Output: Series float
    """

    close = data["close"]
    ma = close.rolling(ma_period, min_periods=1).mean()

    slope = ma.diff()
    direction = np.sign(close.diff())

    continuity = (
        direction.rolling(ma_period, min_periods=1)
        .apply(lambda x: (x != 0).mean(), raw=False)
    )

    strength = slope.abs() * continuity
    return strength.fillna(0.0)


# ============================================================
# 2. RANGE DETECTION
# ============================================================

def detect_range(data, window=20, threshold=0.3):
    """
    Deteksi kondisi ranging atau sideways.
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]
    close = data["close"]

    rolling_high = high.rolling(window, min_periods=1).max()
    rolling_low = low.rolling(window, min_periods=1).min()

    width = rolling_high - rolling_low
    avg_width = width.rolling(window, min_periods=1).mean()

    price_drift = close.diff().abs().rolling(window, min_periods=1).mean()

    range_cond = (
        (width < avg_width * (1 + threshold)) &
        (price_drift < avg_width * threshold)
    )

    return range_cond.fillna(False)


def range_width(data, window=20):
    """
    Mengukur lebar range.
    Output: Series float
    """

    high = data["high"]
    low = data["low"]

    rolling_high = high.rolling(window, min_periods=1).max()
    rolling_low = low.rolling(window, min_periods=1).min()

    return (rolling_high - rolling_low).fillna(0.0)


# ============================================================
# 3. SQUEEZE DETECTION
# ============================================================

def detect_squeeze(data, atr_series=None, lookback=20):
    """
    Deteksi squeeze.
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]

    candle_range = high - low

    if atr_series is None:
        atr_series = candle_range.rolling(lookback, min_periods=1).mean()

    atr_mean = atr_series.rolling(lookback, min_periods=1).mean()

    squeeze = atr_series < atr_mean * 0.7
    return squeeze.fillna(False)


def squeeze_strength(data, atr_series=None, lookback=20):
    """
    Mengukur seberapa kuat squeeze terjadi.
    Output: Series float
    """

    high = data["high"]
    low = data["low"]

    candle_range = high - low

    if atr_series is None:
        atr_series = candle_range.rolling(lookback, min_periods=1).mean()

    atr_mean = atr_series.rolling(lookback, min_periods=1).mean()

    strength = (atr_mean - atr_series) / atr_mean
    return strength.fillna(0.0)


# ============================================================
# 4. REGIME CLASSIFIER
# ============================================================

def classify_regime(data):
    """
    Mengembalikan label regime untuk setiap candle.
    Output: list of strings
    """

    trend = detect_trend(data)
    range_ = detect_range(data)
    squeeze = detect_squeeze(data)

    labels = []

    for i in range(len(data)):
        if squeeze.iloc[i]:
            labels.append("squeeze")
        elif trend.iloc[i]:
            labels.append("trend")
        elif range_.iloc[i]:
            labels.append("range")
        else:
            labels.append("neutral")

    return labels


def regime_summary(regime_labels):
    """
    Ringkasan frekuensi masing-masing regime.
    Output: dict
    """

    counts = {}
    for label in regime_labels:
        counts[label] = counts.get(label, 0) + 1

    return counts


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    size = 120

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    labels = classify_regime(df)

    print("Regime labels sample:", labels[:20])
    print("Regime summary:", regime_summary(labels))

    print("regime_detection.py self-test OK")

