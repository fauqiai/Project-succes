"""
entry_engine.py
---------------
Generate trading entries berdasarkan strategi terbaik
yang ditemukan oleh Quant Behavior engine.

Features:
- Auto detect best strategy
- Generate BUY / SELL
- ATR-based Stop Loss
- Risk Reward Take Profit
- Structured signal object
"""

import pandas as pd
import numpy as np

from behavior_core.strategy_finder import find_best_strategies
from config import ATR_PERIOD


# ============================================================
# 1. SIGNAL OBJECT
# ============================================================

def create_signal(index, signal_type, entry, sl, tp, strategy):

    return {
        "index": index,
        "type": signal_type,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "strategy": strategy
    }


# ============================================================
# 2. ATR CALCULATION (SAFETY)
# ============================================================

def ensure_atr(data):

    if "atr" not in data.columns:

        data["range"] = data["high"] - data["low"]
        data["atr"] = data["range"].rolling(
            ATR_PERIOD, min_periods=1).mean()

    return data


# ============================================================
# 3. STOP LOSS
# ============================================================

def calculate_stop_loss(data, i, signal_type, atr_multiplier=1.5):

    atr = data["atr"].iloc[i]

    if signal_type == "BUY":
        return data["close"].iloc[i] - atr * atr_multiplier

    return data["close"].iloc[i] + atr * atr_multiplier


# ============================================================
# 4. TAKE PROFIT
# ============================================================

def calculate_take_profit(entry, sl, rr_ratio=2.0):

    risk = abs(entry - sl)

    if entry > sl:
        return entry + risk * rr_ratio

    return entry - risk * rr_ratio


# ============================================================
# 5. STRATEGY → ENTRY TRANSLATOR
# ============================================================

def detect_entry_from_strategy(data, strategy_name):

    data = ensure_atr(data)

    signals = []

    for i in range(ATR_PERIOD, len(data)):

        open_price = data["open"].iloc[i]
        close_price = data["close"].iloc[i]

        signal_type = None

        # ====================================================
        # STRATEGY RULES
        # ====================================================

        # Strategy 1: Bullish candle continuation
        if strategy_name == "bullish_candle":

            if close_price > open_price:
                signal_type = "BUY"

        # Strategy 2: Bearish candle continuation
        elif strategy_name == "bearish_candle":

            if close_price < open_price:
                signal_type = "SELL"

        # Strategy fallback (VERY IMPORTANT)
        else:

            # Generic momentum fallback
            prev_close = data["close"].iloc[i-1]

            if close_price > prev_close:
                signal_type = "BUY"

            elif close_price < prev_close:
                signal_type = "SELL"

        # ====================================================

        if signal_type:

            entry = close_price

            sl = calculate_stop_loss(data, i, signal_type)

            tp = calculate_take_profit(entry, sl)

            signals.append(
                create_signal(
                    index=i,
                    signal_type=signal_type,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    strategy=strategy_name
                )
            )

    return signals


# ============================================================
# 6. MAIN ENTRY GENERATOR
# ============================================================

def generate_entry_signals(data):

    print("Detecting best strategy for entries...")

    best_strategies = find_best_strategies(data)

    if not best_strategies:
        print("No profitable strategy detected.")
        return []

    top_strategy = best_strategies[0]["strategy"]

    print(f"Top strategy: {top_strategy}")

    signals = detect_entry_from_strategy(data, top_strategy)

    print(f"Generated {len(signals)} signals.")

    return signals


# ============================================================
# 7. TO DATAFRAME
# ============================================================

def signals_to_dataframe(signals):

    if not signals:
        return pd.DataFrame()

    return pd.DataFrame(signals)


# ============================================================
# 8. QUICK STATS
# ============================================================

def entry_summary(signals):

    if not signals:
        return {"total": 0}

    df = signals_to_dataframe(signals)

    return {
        "total_signals": len(df),
        "buy_signals": int((df["type"] == "BUY").sum()),
        "sell_signals": int((df["type"] == "SELL").sum()),
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("Entry Engine Ready.")
