"""
expectancy.py
Behavior-based Expectancy Engine (FINAL STABLE)

Features:
- Forward Conditional Expectancy (legacy safe)
- MFE / MAE excursion engine
- Behavior expectancy
- TP/SL suggestion
- Distribution-ready stats

ASCII safe.
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. CONDITION FILTERING
# ============================================================

def filter_conditions(data, condition_fn):
    mask = data.apply(condition_fn, axis=1)
    return mask.fillna(False)


# ============================================================
# 2. FORWARD RETURN (LEGACY)
# ============================================================

def compute_outcomes(data, forward_points=20):

    close = data["close"]
    future_close = close.shift(-forward_points)

    outcome = (future_close - close) / close
    return outcome.fillna(0.0)


# ============================================================
# 3. CONDITIONAL EXPECTANCY (LEGACY - MUST EXIST)
# ============================================================

def calculate_expectancy(conditions, outcomes):

    if len(conditions) == 0:
        return 0.0

    filtered = outcomes[conditions]

    if len(filtered) == 0:
        return 0.0

    return filtered.mean()


# ============================================================
# 4. MFE / MAE ENGINE
# ============================================================

def compute_excursions(data, forward_points=20):

    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values

    n = len(data)

    mfe = np.zeros(n)
    mae = np.zeros(n)

    for i in range(n):

        end = min(i + forward_points, n - 1)

        if i >= end:
            continue

        future_high = np.max(highs[i+1:end+1])
        future_low = np.min(lows[i+1:end+1])

        entry = closes[i]

        mfe[i] = (future_high - entry) / entry
        mae[i] = (future_low - entry) / entry

    return pd.Series(mfe, index=data.index), pd.Series(mae, index=data.index)


# ============================================================
# 5. EXCURSION STATISTICS
# ============================================================

def excursion_statistics(mfe, mae, mask):

    mfe_f = mfe[mask]
    mae_f = mae[mask]

    if len(mfe_f) == 0:
        return {}

    stats = {

        "avg_mfe": mfe_f.mean(),
        "median_mfe": mfe_f.median(),

        "avg_mae": mae_f.mean(),
        "median_mae": mae_f.median(),

        "max_mfe": mfe_f.max(),
        "max_mae": mae_f.min(),

        # TP zones
        "tp_70": mfe_f.quantile(0.70),
        "tp_50": mfe_f.quantile(0.50),

        # SL zones (negative values)
        "sl_30": mae_f.quantile(0.30),
        "sl_10": mae_f.quantile(0.10),

        # probabilities
        "prob_tp50_hit": (mfe_f > mfe_f.quantile(0.50)).mean(),
        "prob_tp70_hit": (mfe_f > mfe_f.quantile(0.70)).mean(),

        "samples": int(mask.sum())
    }

    return stats


# ============================================================
# 6. BEHAVIOR EXPECTANCY
# ============================================================

def behavior_expectancy(stats):

    if not stats:
        return 0.0

    avg_win = stats["median_mfe"]
    avg_loss = abs(stats["median_mae"])

    p_win = stats["prob_tp50_hit"]
    p_loss = 1 - p_win

    expectancy = (p_win * avg_win) - (p_loss * avg_loss)

    return expectancy


# ============================================================
# 7. CE TABLE (UPGRADED NON-BREAKING)
# ============================================================

def generate_ce_table(condition_dict, data, forward_points=20):

    results = []

    outcomes = compute_outcomes(data, forward_points)
    mfe, mae = compute_excursions(data, forward_points)

    for name, cond_fn in condition_dict.items():

        mask = filter_conditions(data, cond_fn)

        ce_forward = calculate_expectancy(mask, outcomes)

        wins = (outcomes[mask] > 0).sum()
        total = mask.sum()
        winrate = wins / total if total > 0 else 0.0

        stats = excursion_statistics(mfe, mae, mask)

        behavior_ce = behavior_expectancy(stats) if stats else 0.0

        results.append({
            "condition": name,
            "ce_forward": ce_forward,
            "ce_behavior": behavior_ce,
            "winrate": winrate,
            "samples": int(total),
            "tp_70": stats.get("tp_70", 0),
            "sl_30": stats.get("sl_30", 0),
            "median_mfe": stats.get("median_mfe", 0),
            "median_mae": stats.get("median_mae", 0),
        })

    return pd.DataFrame(results)


# ============================================================
# 8. EXPECTANCY SUMMARY
# ============================================================

def expectancy_summary(ce_table):

    if ce_table.empty:
        return {}

    best = ce_table.sort_values("ce_behavior", ascending=False).iloc[0]

    return {
        "best_condition": best["condition"],
        "behavior_ce": best["ce_behavior"],
        "forward_ce": best["ce_forward"],
        "winrate": best["winrate"],
        "samples": best["samples"],
        "suggested_tp": best["tp_70"],
        "suggested_sl": best["sl_30"]
    }


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

    conditions = {
        "bull_candle": lambda r: r["close"] > r["open"],
        "bear_candle": lambda r: r["close"] < r["open"],
        "wide_range": lambda r: (r["high"] - r["low"]) > 1.2
    }

    ce_table = generate_ce_table(conditions, df, forward_points=20)
    summary = expectancy_summary(ce_table)

    print("\nCE Table:")
    print(ce_table)

    print("\nSummary:")
    print(summary)

    print("\nBehavior Expectancy Engine OK")
