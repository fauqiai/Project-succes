"""
entry_engine.py
Behavior-Based Trade Executor (FINAL)

NO ATR
NO RR
NO GUESSING

Uses:
- Strategy Finder
- Expectancy (MFE/MAE)
- Transitions (optional ready)

Builds Trade Blueprint.
"""

import pandas as pd
import numpy as np

# ✅ IMPORT SAFE
from behavior_core.strategy_finder import find_best_strategies
from behavior_core.expectancy import compute_excursions


# ============================================================
# 1. SIGNAL OBJECT
# ============================================================

def create_signal(
        index,
        signal_type,
        entry,
        sl,
        tp,
        strategy,
        confidence,
        expected_move):

    return {
        "index": index,
        "type": signal_type,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "strategy": strategy,
        "confidence": float(confidence),
        "expected_move": float(expected_move)
    }


# ============================================================
# 2. BUILD SL / TP FROM BEHAVIOR
# ============================================================

def build_trade_levels(entry, avg_up, avg_down, direction):

    if direction == "BUY":

        tp = entry * (1 + avg_up)
        sl = entry * (1 - avg_down * 1.1)  # wick buffer

    else:

        tp = entry * (1 - avg_up)
        sl = entry * (1 + avg_down * 1.1)

    return sl, tp


# ============================================================
# 3. STRATEGY → ENTRY
# (NO ATR, PURE BEHAVIOR)
# ============================================================

def detect_entries(data, strategy_name, strategy_stats):

    signals = []

    avg_up = strategy_stats["avg_up_move"]
    avg_down = strategy_stats["avg_down_move"]
    winrate = strategy_stats["winrate"]

    confidence = winrate  # transition bisa ditambah nanti

    for i in range(1, len(data)):

        open_price = data["open"].iloc[i]
        close_price = data["close"].iloc[i]

        signal_type = None

        # SAME RULES as strategy finder
        if strategy_name == "bullish_candle":
            if close_price > open_price:
                signal_type = "BUY"

        elif strategy_name == "bearish_candle":
            if close_price < open_price:
                signal_type = "SELL"

        elif strategy_name == "large_range":
            if (data["high"].iloc[i] - data["low"].iloc[i]) > \
               ((data["high"].iloc[i] - data["low"].iloc[i]) * 0.7):
                signal_type = "BUY"

        elif strategy_name == "small_body":
            if abs(close_price - open_price) < \
               ((data["high"].iloc[i] - data["low"].iloc[i]) * 0.3):
                signal_type = "BUY"

        if signal_type:

            entry = close_price

            sl, tp = build_trade_levels(
                entry,
                avg_up,
                avg_down,
                signal_type
            )

            signals.append(
                create_signal(
                    index=i,
                    signal_type=signal_type,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    strategy=strategy_name,
                    confidence=confidence,
                    expected_move=avg_up
                )
            )

    return signals


# ============================================================
# 4. MAIN GENERATOR
# ============================================================

def generate_entry_signals(data):

    print("Scanning market with behavior engine...")

    best = find_best_strategies(data)

    if not best:
        print("No positive EV strategy.")
        return []

    strategy_name, stats = best[0]

    print(f"Selected strategy: {strategy_name}")
    print(f"Expected move: {round(stats['avg_up_move']*100, 3)}%")

    signals = detect_entries(
        data,
        strategy_name,
        stats
    )

    print(f"Generated {len(signals)} trade blueprints.")

    return signals


# ============================================================
# 5. TO DATAFRAME
# ============================================================

def signals_to_dataframe(signals):

    if not signals:
        return pd.DataFrame()

    return pd.DataFrame(signals)


# ============================================================
# 6. QUICK STATS
# ============================================================

def entry_summary(signals):

    if not signals:
        return {"total": 0}

    df = signals_to_dataframe(signals)

    return {
        "total_signals": len(df),
        "buy_signals": int((df["type"] == "BUY").sum()),
        "sell_signals": int((df["type"] == "SELL").sum()),
        "avg_confidence": round(df["confidence"].mean(), 3)
    }


# ============================================================
# QUANT DECISION ENGINE (FROM REPORT)
# ============================================================

def quant_decision(row):
    """
    Mengubah behavior menjadi keputusan BUY / SELL.
    Berdasarkan hasil Quant Report:
    - Bearish candle = edge tertinggi
    """

    open_price = row["open"]
    close_price = row["close"]

    body = close_price - open_price

    # ===== PRIORITY EDGE =====
    # Bearish candle -> SELL
    if body < 0:
        return {
            "type": "SELL",
            "confidence": 0.55,   # dari winrate ~51-53%
        }

    # Bullish -> BUY tapi confidence lebih kecil
    if body > 0:
        return {
            "type": "BUY",
            "confidence": 0.51,
        }

    return None


# ============================================================
# GENERATE SIGNALS FROM QUANT
# ============================================================

def generate_quant_signals(data):

    signals = []

    for i in range(len(data)):

        decision = quant_decision(data.iloc[i])

        if decision is None:
            continue

        signals.append({
            "index": i,
            "price": data.iloc[i]["close"],
            "type": decision["type"],
            "confidence": decision["confidence"]
        })

    return signals


# ============================================================
# SELF TEST (SAFE — NO VISUALIZER)
# ============================================================

if __name__ == "__main__":

    print("Running Entry Engine self test...")

    np.random.seed(42)
    size = 400

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    signals = generate_entry_signals(df)

    print("\nEntry summary:")
    print(entry_summary(signals))

    print("\nEntry Engine OK ✅")
