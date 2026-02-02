"""
strategy_finder.py
------------------
Module untuk menemukan strategi terbaik berdasarkan Quant Behavior.
Semua komentar menggunakan ASCII agar aman untuk Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. STRATEGY RULE BUILDER
# ============================================================

def build_strategy_rules():
    """
    Mengembalikan dictionary:
    { strategy_name: rule_function }
    """

    rules = {}

    # Simple behavior based rules (baseline)
    rules["bullish_candle"] = lambda r: r["close"] > r["open"]

    rules["bearish_candle"] = lambda r: r["close"] < r["open"]

    rules["large_range"] = lambda r: (r["high"] - r["low"]) > (
        (r["high"] - r["low"]).mean() if hasattr(r, "mean") else 0
    )

    rules["small_body"] = lambda r: abs(r["close"] - r["open"]) < (
        (r["high"] - r["low"]) * 0.3
    )

    return rules


# ============================================================
# 2. STRATEGY TESTING ENGINE
# ============================================================

def test_strategy(data, rule_fn, forward_points=10):
    """
    Menguji rule dan mengembalikan:
    mask, outcomes
    """

    close = data["close"]
    future_close = close.shift(-forward_points)
    outcomes = (future_close - close) / close

    mask = data.apply(rule_fn, axis=1).fillna(False)

    return mask, outcomes.fillna(0.0)


# ============================================================
# 3. STRATEGY EXPECTANCY CALCULATION
# ============================================================

def calculate_strategy_expectancy(data, strategy_rules, forward_points=10):
    """
    Output:
    dict[strategy_name] -> metrics
    """

    results = {}

    for name, rule_fn in strategy_rules.items():
        mask, outcomes = test_strategy(data, rule_fn, forward_points)

        samples = int(mask.sum())
        if samples == 0:
            continue

        filtered = outcomes[mask]

        ce = filtered.mean()
        winrate = (filtered > 0).mean()

        results[name] = {
            "CE": ce,
            "winrate": winrate,
            "samples": samples
        }

    return results


# ============================================================
# 4. STRATEGY RANKING
# ============================================================

def rank_strategies(expectancy_results):
    """
    Mengurutkan strategi berdasarkan CE tertinggi.
    Output: list of tuples
    """

    ranked = sorted(
        expectancy_results.items(),
        key=lambda x: x[1]["CE"],
        reverse=True
    )

    return ranked


# ============================================================
# 5. BEST STRATEGY SELECTOR
# ============================================================

def find_best_strategies(data, forward_points=10):
    """
    Pipeline utama strategy finder.
    """

    rules = build_strategy_rules()
    expectancy = calculate_strategy_expectancy(
        data, rules, forward_points
    )

    ranked = rank_strategies(expectancy)

    best = []
    for name, stats in ranked:
        if stats["CE"] > 0:
            best.append((name, stats))

    return best


# ============================================================
# 6. STRATEGY SUMMARY OUTPUT
# ============================================================

def strategy_summary(best_strategies):
    """
    Output: list of dict
    """

    summary = []

    for name, stats in best_strategies:
        summary.append({
            "strategy": name,
            "CE": stats["CE"],
            "winrate": stats["winrate"],
            "samples": stats["samples"]
        })

    return summary


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    size = 200

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    best = find_best_strategies(df, forward_points=5)
    summary = strategy_summary(best)

    print("Best strategies:")
    for row in summary:
        print(row)

    print("strategy_finder.py self-test OK")

