"""
strategy_finder.py
Behavior-Based Strategy Finder (FINAL IMPORT SAFE)

Upgrades:
- Uses MFE / MAE from expectancy engine
- Calculates TRUE Expected Value
- Avoids winrate trap
- Scores real opportunity vs risk
ASCII safe.
"""

import pandas as pd
import numpy as np

# ✅ IMPORT FIX (VERY IMPORTANT)
from backtest.behavior_core.expectancy import compute_excursions


# ============================================================
# 1. STRATEGY RULE BUILDER
# ============================================================

def build_strategy_rules():

    rules = {}

    rules["bullish_candle"] = lambda r: r["close"] > r["open"]

    rules["bearish_candle"] = lambda r: r["close"] < r["open"]

    rules["large_range"] = lambda r: (r["high"] - r["low"]) > (
        (r["high"] - r["low"]) * 0.7
    )

    rules["small_body"] = lambda r: abs(r["close"] - r["open"]) < (
        (r["high"] - r["low"]) * 0.3
    )

    return rules


# ============================================================
# 2. MASK BUILDER
# ============================================================

def build_mask(data, rule_fn):
    return data.apply(rule_fn, axis=1).fillna(False)


# ============================================================
# 3. TRUE EV CALCULATION (CORE)
# ============================================================

def calculate_true_ev(mfe, mae, mask):

    mfe_f = mfe[mask]
    mae_f = mae[mask]

    samples = len(mfe_f)

    if samples == 0:
        return None

    avg_up = mfe_f.mean()
    avg_down = abs(mae_f.mean())

    winrate = (mfe_f > abs(mae_f)).mean()

    # TRUE EXPECTED VALUE
    ev = (avg_up * winrate) - (avg_down * (1 - winrate))

    return {
        "EV": ev,
        "avg_up_move": avg_up,
        "avg_down_move": avg_down,
        "winrate": winrate,
        "samples": samples
    }


# ============================================================
# 4. STRATEGY EVALUATION ENGINE
# ============================================================

def evaluate_strategies(data, strategy_rules, forward_points=20):

    mfe, mae = compute_excursions(data, forward_points)

    results = {}

    for name, rule_fn in strategy_rules.items():

        mask = build_mask(data, rule_fn)

        stats = calculate_true_ev(mfe, mae, mask)

        if stats is None:
            continue

        # kill low sample strategies
        if stats["samples"] < 30:
            continue

        results[name] = stats

    return results


# ============================================================
# 5. SMART RANKING
# ============================================================

def rank_strategies(strategy_results):

    ranked = sorted(
        strategy_results.items(),
        key=lambda x: x[1]["EV"],
        reverse=True
    )

    return ranked


# ============================================================
# 6. BEST STRATEGY SELECTOR
# ============================================================

def find_best_strategies(data, forward_points=20):

    rules = build_strategy_rules()

    evaluated = evaluate_strategies(
        data,
        rules,
        forward_points
    )

    ranked = rank_strategies(evaluated)

    best = [(name, stats) for name, stats in ranked if stats["EV"] > 0]

    return best


# ============================================================
# 7. STRATEGY SUMMARY
# ============================================================

def strategy_summary(best_strategies):

    summary = []

    for name, stats in best_strategies:

        summary.append({
            "strategy": name,
            "EV": stats["EV"],
            "avg_up_move": stats["avg_up_move"],
            "avg_down_move": stats["avg_down_move"],
            "winrate": stats["winrate"],
            "samples": stats["samples"]
        })

    return summary


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)
    size = 300

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    best = find_best_strategies(df, forward_points=20)
    summary = strategy_summary(best)

    print("\nBest strategies:\n")

    for row in summary:
        print(row)

    print("\nStrategy Finder Behavior Engine OK")
