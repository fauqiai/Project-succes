"""
quant_backtest.py
-----------------
File utama untuk menjalankan pipeline Quant Behavior.
Semua komentar ASCII agar aman untuk Notepad Windows.
"""

import pandas as pd
import numpy as np

# Import modul dari folder behavior_core
from behavior_core.measures import *
from behavior_core.event_detection import *
from behavior_core.regime_detection import *
from behavior_core.microstructure import *
from behavior_core.transitions import *
from behavior_core.expectancy import *
from behavior_core.strategy_finder import *
from behavior_core.report_generator import *

from data_loader import load_and_prepare
from config import *

# ============================================================
# 1. PIPELINE UTAMA
# ============================================================

def run_quant_backtest(path_to_csv, timeframe=None, forward_points=10):
    """
    Pipeline utama untuk menjalankan seluruh modul Quant Behavior.
    """

    print("Loading data...")
    data = load_and_prepare(path_to_csv, timeframe)
    if data is None or len(data) == 0:
        print("Gagal load data.")
        return

    # RESET INDEX BIAR NUMERIK (INI PENTING)
    data = data.reset_index(drop=True)

    print("Calculating measures...")
    data["range"] = data["high"] - data["low"]
    data["body"] = (data["close"] - data["open"]).abs()
    data["atr"] = data["range"].rolling(ATR_PERIOD, min_periods=1).mean()

    print("Detecting events...")
    impulse_mask = detect_impulse(data, data["atr"])
    retrace_mask = detect_retracement(data)
    cons_mask = detect_consolidation(data)

    event_sequence = []
    for i in range(len(data)):
        if impulse_mask.iloc[i]:
            event_sequence.append("impulse")
        elif retrace_mask.iloc[i]:
            event_sequence.append("retracement")
        elif cons_mask.iloc[i]:
            event_sequence.append("consolidation")
        else:
            event_sequence.append("neutral")

    event_summary = pd.Series(event_sequence).value_counts().to_dict()

    print("Detecting regime...")
    regime_sequence = classify_regime(data)
    regime_summary = pd.Series(regime_sequence).value_counts().to_dict()

    print("Detecting microstructure...")
    micro_flags = {
        "sweep": detect_sweep(data),
        "imbalance": detect_imbalance(data),
        "fvg": detect_fvg(data),
        "displacement": detect_displacement(data),
        "sfp": detect_sfp(data),
        "orderflow_shift": detect_orderflow_shift(data),
    }

    micro_summary = {k: int(v.sum()) for k, v in micro_flags.items()}

    print("Building transition matrix...")
    combined_sequence = generate_state_sequence(
        events=event_sequence,
        regimes=regime_sequence
    )

    transition_matrix = build_transition_matrix(combined_sequence)
    transition_summary_data = transition_summary(transition_matrix)

    print("Calculating Conditional Expectancy...")

    condition_dict = {
        "impulse": lambda r: event_sequence[r.name] == "impulse",
        "retracement": lambda r: event_sequence[r.name] == "retracement",
        "consolidation": lambda r: event_sequence[r.name] == "consolidation",
    }

    ce_table = generate_ce_table(
        condition_dict,
        data,
        forward_points=forward_points
    )

    ce_summary = expectancy_summary(ce_table)

    print("Finding best strategies...")
    best_strategies = find_best_strategies(
        data,
        forward_points=forward_points
    )

    best_strategy_summary = strategy_summary(best_strategies)

    print("Generating report...")
    report_text = build_report(
        event_summary=event_summary,
        regime_summary=regime_summary,
        transition_summary=transition_summary_data,
        ce_table=ce_table,
        best_strategies=best_strategy_summary
    )

    save_report(report_text)

    print("Quant Behavior analysis complete.")
    return report_text


# ============================================================
# 2. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Quant Behavior Backtest Engine Running...")

    DATA_PATH = "xauusd_m1_cleaned.csv"

    run_quant_backtest(
        path_to_csv=DATA_PATH,
        timeframe=DEFAULT_TIMEFRAME,
        forward_points=FORWARD_POINTS
    )
    
