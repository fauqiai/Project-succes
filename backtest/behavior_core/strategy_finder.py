"""
strategy_finder.py
Behavior-Based Strategy Finder (QUANT v2)

MAJOR UPGRADE:
- Multi-condition strategies
- Behavior stacking
- True Expected Value
- Sample protection
- Volatility filter
"""

import pandas as pd
import numpy as np

from behavior_core.expectancy import compute_excursions


# ============================================================
# 1. FEATURE PREP
# ============================================================

def prepare_features(data):

    data = data.copy()

    data["range"] = data["high"] - data["low"]
    data["body"] = (data["close"] - data["open"]).abs()

    # volatility baseline
    data["avg_range"] = data["range"].rolling(20, min_periods=5).mean()

    # impulse proxy
    data["impulse"] = data["range"] > data["avg_range"] * 1.5

    # strong body
    data["strong_body"] = data["body"] > data["range"] * 0.6

    return data


# ============================================================
# 2. STRATEGY BUILDER (MULTI CONDITION)
# ============================================================

def build_strategy_rules(data):

    rules = {}

    # 🔥 MUCH smarter than bearish candle

    rules["bearish_impulse"] = lambda r: (
        (r["close"] < r["open"]) and
        (r["impulse"])
    )

    rules["bullish_impulse"] = lambda r: (
        (r["close"] > r["open"]) and
        (r["impulse"])
    )

    rules["strong_bearish_break"] = lambda r: (
        (r["close"] < r["open"]) and
        (r["strong_body"]) and
        (r["range"] > r["avg_range"])
    )

    rules["volatility_expansion_buy"] = lambda r: (
        (r["close"] > r["open"]) and
        (r["range"] > r["avg_range"] * 1.3)
    )

    rules["volatility_expansion_sell"] = lambda r: (
        (r["close"] < r["open"]) and
        (r["range"] > r["avg_range"] * 1.3)
    )

    return rules


# ============================================================
# 3. MASK BUILDER
# ============================================================

def build_mask(data, rule_fn):
    return data.apply(rule_fn, axis=1).fillna(False)


# ============================================================
# 4. TRUE EV
# ============================================================

def calculate_true_ev(mfe, mae, mask):

    mfe_f = mfe[mask]
    mae_f = mae[mask]

    samples = len(mfe_f)

    if samples < 40:   # 🔥 sample protection
        return None

    avg_up = mfe_f.mean()
    avg_down = abs(mae_f.mean())

    winrate = (mfe_f > abs(mae_f)).mean()

    ev = (avg_up * winrate) - (avg_down * (1 - winrate))

    return {
        "EV": ev,
        "avg_up_move": avg_up,
        "avg_down_move": avg_down,
        "winrate": winrate,
        "samples": samples
    }


# ============================================================
# 5. EVALUATION ENGINE
# ============================================================

def evaluate_strategies(data, forward_points=20):

    data = prepare_features(data)

    mfe, mae = compute_excursions(data, forward_points)

    rules = build_strategy_rules(data)

    results = {}

    for name, rule_fn in rules.items():

        mask = build_mask(data, rule_fn)

        stats = calculate_true_ev(mfe, mae, mask)

        if stats is None:
            continue

        results[name] = stats

    return results


# ============================================================
# 6. SMART RANKING
# ============================================================

def rank_strategies(results):

    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["EV"],
        reverse=True
    )

    return ranked


# ============================================================
# 7. BEST SELECTOR
# ============================================================

def find_best_strategies(data, forward_points=20):

    evaluated = evaluate_strategies(
        data,
        forward_points
    )

    ranked = rank_strategies(evaluated)

    best = [(name, stats) for name, stats in ranked if stats["EV"] > 0]

    return best


# ============================================================
# 8. SUMMARY
# ============================================================

def strategy_summary(best):

    summary = []

    for name, stats in best:

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
    size = 500

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    best = find_best_strategies(df)

    print("\nBEST STRATEGIES:\n")
    for name, stats in best:
        print(name, "->", stats)

    print("\nStrategy Finder QUANT v2 READY")
