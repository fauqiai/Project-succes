"""
entry_engine.py
TRUE Quant Execution Engine

- Direct sync with Strategy Finder
- No manual candle logic
- Behavior driven entries
"""

import numpy as np

from behavior_core.strategy_finder import find_best_strategies


# ============================================================
# SIGNAL OBJECT
# ============================================================

def create_signal(index, signal_type, entry, sl, tp, strategy, confidence):

    return {
        "index": index,
        "type": signal_type,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "strategy": strategy,
        "confidence": float(confidence),
    }


# ============================================================
# BUILD LEVELS FROM EXPECTANCY
# ============================================================

def build_trade_levels(entry, avg_up, avg_down, direction):

    if direction == "BUY":
        tp = entry * (1 + avg_up)
        sl = entry * (1 - avg_down * 1.2)

    else:
        tp = entry * (1 - avg_up)
        sl = entry * (1 + avg_down * 1.2)

    return sl, tp


# ============================================================
# STRATEGY → MASK (SUPER IMPORTANT)
# ============================================================

def build_mask_from_strategy(data, strategy_name):

    regime = data["regime"]

    impulse = data["impulse"]
    retracement = data["retracement"]
    consolidation = data["consolidation"]

    bullish = data["close"] > data["open"]
    bearish = data["close"] < data["open"]

    if strategy_name == "trend_impulse_buy":
        return impulse & (regime == "trend") & bullish

    if strategy_name == "trend_impulse_sell":
        return impulse & (regime == "trend") & bearish

    if strategy_name == "range_retrace_buy":
        return retracement & (regime == "range") & bullish

    if strategy_name == "range_retrace_sell":
        return retracement & (regime == "range") & bearish

    if strategy_name == "breakout_buy":
        return consolidation & bullish

    if strategy_name == "breakout_sell":
        return consolidation & bearish

    return None


# ============================================================
# GENERATE TRUE QUANT ENTRIES
# ============================================================

def generate_entry_signals(data):

    print("Running TRUE quant execution...")

    best = find_best_strategies(data)

    if not best:
        print("No positive EV strategy.")
        return []

    strategy_name, stats = best[0]

    print("SELECTED STRATEGY:", strategy_name)

    mask = build_mask_from_strategy(data, strategy_name)

    indices = np.where(mask)[0]

    signals = []

    avg_up = stats["avg_up_move"]
    avg_down = stats["avg_down_move"]
    confidence = stats["winrate"]

    direction = "BUY" if "buy" in strategy_name else "SELL"

    for i in indices:

        entry = data["close"].iloc[i]

        sl, tp = build_trade_levels(
            entry,
            avg_up,
            avg_down,
            direction
        )

        signals.append(
            create_signal(
                index=i,
                signal_type=direction,
                entry=entry,
                sl=sl,
                tp=tp,
                strategy=strategy_name,
                confidence=confidence
            )
        )

    print(f"Generated {len(signals)} quant entries.")

    return signals
