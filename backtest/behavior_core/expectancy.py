"""
expectancy.py
-------------
Quant Expectancy Engine (V3)

Features:
- Forward return
- MFE / MAE
- Winrate
- Avg win / loss
- True Expectancy (EV)

ASCII safe.
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. EXCURSIONS (CORE ENGINE)
# ============================================================

def compute_excursions(data, forward_points=20):
    """
    Menghitung MFE dan MAE untuk setiap candle.
    Tidak pakai indices lagi -> biar sinkron V3.
    """

    closes = data["close"]
    highs = data["high"]
    lows = data["low"]

    mfe = []
    mae = []

    size = len(data)

    for i in range(size):

        end = min(i + forward_points, size - 1)

        future_high = highs.iloc[i:end].max()
        future_low = lows.iloc[i:end].min()

        entry = closes.iloc[i]

        mfe.append((future_high - entry) / entry)
        mae.append((future_low - entry) / entry)

    return pd.Series(mfe), pd.Series(mae)


# ============================================================
# 2. FORWARD RETURNS
# ============================================================

def compute_forward_returns(data, forward_points=20):

    closes = data["close"]

    forward_returns = closes.shift(-forward_points)
    forward_returns = (forward_returns - closes) / closes

    return forward_returns.fillna(0)


# ============================================================
# 3. EXPECTANCY CALCULATION
# ============================================================

def calculate_expectancy(mask, data, forward_points=20):
    """
    Hitung statistik expectancy berdasarkan kondisi / strategy mask.
    """

    indices = np.where(mask)[0]

    if len(indices) < 30:   # hindari noise
        return None

    forward_returns = compute_forward_returns(data, forward_points)
    mfe, mae = compute_excursions(data, forward_points)

    returns = forward_returns.iloc[indices]
    mfe_vals = mfe.iloc[indices]
    mae_vals = mae.iloc[indices]

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    winrate = len(wins) / len(returns)

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    # ⭐ TRUE EXPECTANCY
    expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)

    stats = {
        "samples": len(indices),
        "winrate": winrate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "EV": expectancy,
        "avg_up_move": mfe_vals.mean(),
        "avg_down_move": abs(mae_vals.mean()),
    }

    return stats


# ============================================================
# 4. CONDITIONAL EXPECTANCY TABLE
# ============================================================

def generate_ce_table(condition_dict, data, forward_points=20):

    table = {}

    for name, condition in condition_dict.items():

        mask = data.apply(condition, axis=1).values

        stats = calculate_expectancy(
            mask,
            data,
            forward_points
        )

        if stats:
            table[name] = stats

    return table


# ============================================================
# 5. SUMMARY
# ============================================================

def expectancy_summary(table):

    if not table:
        return {}

    ranked = sorted(
        table.items(),
        key=lambda x: x[1]["EV"],
        reverse=True
    )

    return ranked


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)
    size = 500

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    condition_dict = {
        "bullish": lambda r: r["close"] > r["open"],
        "bearish": lambda r: r["close"] < r["open"],
    }

    table = generate_ce_table(condition_dict, df)

    print("\nEXPECTANCY SUMMARY:\n")
    print(expectancy_summary(table))

    print("\nQuant Expectancy Engine V3 OK")
