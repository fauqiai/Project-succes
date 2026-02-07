"""
strategy_finder.py
Strategy Finder V3 (FULL FIX - Expectancy V3 Compatible)

- No legacy indices
- No dict excursions
- Fully aligned with compute_excursions()
"""

import pandas as pd
import numpy as np

from .event_detection import detect_impulse, detect_retracement, detect_consolidation
from .regime_detection import classify_regime
from .expectancy import compute_excursions


MIN_SAMPLES = 200
MIN_WINRATE = 0.52


# ============================================================
# BUILD CONDITIONS
# ============================================================

def build_conditions(data):

    atr = (data["high"] - data["low"]).rolling(14, min_periods=1).mean()

    impulse = detect_impulse(data, atr)
    retracement = detect_retracement(data)
    consolidation = detect_consolidation(data)

    regime = pd.Series(classify_regime(data), index=data.index)

    conditions = {

        # TREND CONTINUATION
        "trend_impulse_buy":
            impulse & (regime == "trend") & (data["close"] > data["open"]),

        "trend_impulse_sell":
            impulse & (regime == "trend") & (data["close"] < data["open"]),

        # RANGE MEAN REVERSION
        "range_retrace_buy":
            retracement & (regime == "range") & (data["close"] > data["open"]),

        "range_retrace_sell":
            retracement & (regime == "range") & (data["close"] < data["open"]),

        # BREAKOUT
        "breakout_buy":
            consolidation & (data["close"] > data["open"]),

        "breakout_sell":
            consolidation & (data["close"] < data["open"]),
    }

    return conditions


# ============================================================
# EVALUATE STRATEGY (FIXED)
# ============================================================

def evaluate_strategy(mask, data, forward_points):

    indices = np.where(mask.to_numpy())[0]

    if len(indices) < MIN_SAMPLES:
        return None

    # ✅ Expectancy V3 format
    mfe, mae = compute_excursions(data, forward_points)

    forward_returns = data["close"].shift(-forward_points)
    forward_returns = (forward_returns - data["close"]) / data["close"]

    returns = forward_returns.iloc[indices]
    mfe_vals = mfe.iloc[indices]
    mae_vals = mae.iloc[indices]

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    if len(returns) == 0:
        return None

    winrate = len(wins) / len(returns)

    if winrate < MIN_WINRATE:
        return None

    avg_win = wins.mean() if len(wins) else 0
    avg_loss = abs(losses.mean()) if len(losses) else 0

    EV = (winrate * avg_win) - ((1 - winrate) * avg_loss)

    return {
        "EV": float(EV),
        "avg_up_move": float(mfe_vals.mean()),
        "avg_down_move": float(abs(mae_vals.mean())),
        "winrate": float(winrate),
        "samples": int(len(indices))
    }


# ============================================================
# FIND BEST STRATEGIES
# ============================================================

def find_best_strategies(data, forward_points=10):

    print("Scanning strategies (V3)...")

    conditions = build_conditions(data)

    results = []

    for name, mask in conditions.items():

        stats = evaluate_strategy(mask, data, forward_points)

        if stats:
            results.append((name, stats))

    if not results:
        print("No strategy with positive expectancy.")
        return []

    results.sort(key=lambda x: x[1]["EV"], reverse=True)

    print(f"Found {len(results)} valid strategies.")
    print("Top strategy:", results[0][0])

    return results


# ============================================================
# SUMMARY
# ============================================================

def strategy_summary(best):

    if not best:
        return "No valid strategy found."

    strategy, stats = best[0]

    return {
        "strategy": strategy,
        "EV": stats["EV"],
        "winrate": stats["winrate"],
        "samples": stats["samples"]
    }
