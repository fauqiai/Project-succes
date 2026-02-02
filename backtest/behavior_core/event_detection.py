"""
event_detection.py
------------------
Module untuk mendeteksi event utama dalam Quant Behavior:
1. Impulse (gerakan besar)
2. Retracement (tarikan balik)
3. Consolidation (konsolidasi atau range sempit)
Semua komentar menggunakan ASCII agar aman disimpan di Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. DETEKSI IMPULSE
# ============================================================

def detect_impulse(data, atr_series=None, multiplier=1.5):
    """
    Mendeteksi candle impulsif.
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]
    open_ = data["open"]
    close = data["close"]

    body = (close - open_).abs()
    candle_range = high - low

    # Fallback ATR sederhana jika tidak disediakan
    if atr_series is None:
        atr_series = candle_range.rolling(14, min_periods=1).mean()

    prev_range = candle_range.shift(1)

    impulse = (
        (candle_range > atr_series * multiplier) &
        (candle_range > prev_range * 1.2)
    )

    return impulse.fillna(False)


def impulse_points(data, atr_series=None, multiplier=1.5):
    """
    Mengembalikan indeks titik-titik di mana impuls terjadi.
    """
    impulse_mask = detect_impulse(data, atr_series, multiplier)
    return list(data.index[impulse_mask])


# ============================================================
# 2. DETEKSI RETRACEMENT
# ============================================================

def detect_retracement(data, window=10):
    """
    Mendeteksi retracement setelah impuls.
    Output: Series boolean
    """

    close = data["close"]
    high = data["high"]
    low = data["low"]

    rolling_high = high.rolling(window, min_periods=1).max()
    rolling_low = low.rolling(window, min_periods=1).min()

    range_ = rolling_high - rolling_low
    pullback = close.diff()

    retracement = (
        (pullback < 0) &
        (close > rolling_low + range_ * 0.3)
    )

    return retracement.fillna(False)


def retracement_points(data, window=10):
    """
    Mengembalikan indeks titik retracement yang valid.
    """
    mask = detect_retracement(data, window)
    return list(data.index[mask])


# ============================================================
# 3. DETEKSI CONSOLIDATION
# ============================================================

def detect_consolidation(data, window=5, threshold=0.3):
    """
    Mendeteksi konsolidasi.
    Output: Series boolean
    """

    high = data["high"]
    low = data["low"]

    candle_range = high - low
    avg_range = candle_range.rolling(window, min_periods=1).mean()

    rolling_high = high.rolling(window, min_periods=1).max()
    rolling_low = low.rolling(window, min_periods=1).min()

    total_range = rolling_high - rolling_low

    consolidation = (
        (avg_range < total_range * threshold)
    )

    return consolidation.fillna(False)


def consolidation_zones(data, window=5, threshold=0.3):
    """
    Mengembalikan zona konsolidasi berupa tuples (start, end).
    """

    mask = detect_consolidation(data, window, threshold)
    zones = []

    start = None
    for idx, val in mask.items():
        if val and start is None:
            start = idx
        elif not val and start is not None:
            zones.append((start, idx))
            start = None

    if start is not None:
        zones.append((start, mask.index[-1]))

    return zones


# ============================================================
# 4. EVENT SEQUENCE BUILDER
# ============================================================

def build_event_sequence(data):
    """
    Menghasilkan urutan event seperti:
    impulse -> retrace -> consolidation -> impulse ...
    Output: list of strings
    """

    impulse = detect_impulse(data)
    retrace = detect_retracement(data)
    consolidate = detect_consolidation(data)

    sequence = []

    for i in range(len(data)):
        if impulse.iloc[i]:
            sequence.append("impulse")
        elif retrace.iloc[i]:
            sequence.append("retracement")
        elif consolidate.iloc[i]:
            sequence.append("consolidation")
        else:
            sequence.append("neutral")

    return sequence


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    # Simple synthetic data for safety test
    np.random.seed(42)
    size = 100

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.2,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.2,
    })

    print("Impulse points:", impulse_points(df))
    print("Retracement points:", retracement_points(df))
    print("Consolidation zones:", consolidation_zones(df))
    print("Event sequence sample:", build_event_sequence(df)[:20])

    print("event_detection.py self-test OK")

