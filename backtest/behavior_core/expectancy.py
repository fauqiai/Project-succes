"""
expectancy.py
-------------
Module untuk menghitung Conditional Expectancy (CE) dalam Quant Behavior.
Semua komentar menggunakan ASCII agar aman untuk Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. CONDITION FILTERING
# ============================================================

def filter_conditions(data, condition_fn):
    """
    Memfilter data berdasarkan kondisi tertentu.
    Output: Series boolean
    """

    mask = data.apply(condition_fn, axis=1)
    return mask.fillna(False)


# ============================================================
# 2. OUTCOME CALCULATION
# ============================================================

def compute_outcomes(data, forward_points=5):
    """
    Menghitung outcome berupa return ke depan.
    Output: Series float
    """

    close = data["close"]
    future_close = close.shift(-forward_points)

    outcome = (future_close - close) / close
    return outcome.fillna(0.0)


# ============================================================
# 3. CONDITIONAL EXPECTANCY
# ============================================================

def calculate_expectancy(conditions, outcomes):
    """
    CE = rata-rata outcome saat condition True
    """

    if len(conditions) == 0:
        return 0.0

    filtered = outcomes[conditions]

    if len(filtered) == 0:
        return 0.0

    return filtered.mean()


# ============================================================
# 4. CE TABLE GENERATOR
# ============================================================

def generate_ce_table(condition_dict, data, forward_points=5):
    """
    Output: DataFrame berisi CE dan winrate per kondisi
    """

    results = []

    outcomes = compute_outcomes(data, forward_points)

    for name, cond_fn in condition_dict.items():
        mask = filter_conditions(data, cond_fn)

        ce = calculate_expectancy(mask, outcomes)

        wins = (outcomes[mask] > 0).sum()
        total = mask.sum()
        winrate = wins / total if total > 0 else 0.0

        results.append({
            "condition": name,
            "ce": ce,
            "winrate": winrate,
            "samples": int(total)
        })

    return pd.DataFrame(results)


# ============================================================
# 5. EXPECTANCY SUMMARY
# ============================================================

def expectancy_summary(ce_table):
    """
    Ringkasan CE terbaik.
    Output: dict
    """

    if ce_table.empty:
        return {}

    best = ce_table.sort_values("ce", ascending=False).iloc[0]

    return {
        "best_condition": best["condition"],
        "best_ce": best["ce"],
        "winrate": best["winrate"],
        "samples": best["samples"]
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    size = 150

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price + np.random.randn(size) * 0.1,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price + np.random.randn(size) * 0.1,
    })

    # Example conditions
    conditions = {
        "bull_candle": lambda r: r["close"] > r["open"],
        "bear_candle": lambda r: r["close"] < r["open"],
        "large_range": lambda r: (r["high"] - r["low"]) > 1.2
    }

    ce_table = generate_ce_table(conditions, df, forward_points=5)
    summary = expectancy_summary(ce_table)

    print("CE Table:")
    print(ce_table)
    print("Summary:", summary)

    print("expectancy.py self-test OK")

