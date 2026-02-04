"""
transitions.py
Behavior-Aware Transition Engine (FINAL)

Upgrades:
- Transition probability
- Avg move after transition
- MFE / MAE tracking
- Opportunity vs risk insight
ASCII safe.
"""

import pandas as pd
import numpy as np

# ✅ IMPORT FIX
from behavior_core.expectancy import compute_excursions


# ============================================================
# 1. TRANSITION COUNT
# ============================================================

def count_transitions(sequence):

    counts = {}

    for i in range(len(sequence) - 1):
        frm = sequence[i]
        to = sequence[i + 1]

        key = (frm, to)
        counts[key] = counts.get(key, 0) + 1

    return counts


# ============================================================
# 2. STATE SEQUENCE
# ============================================================

def generate_state_sequence(events=None, regimes=None):

    if events is None and regimes is None:
        return []

    if events is None:
        return list(regimes)

    if regimes is None:
        return list(events)

    size = min(len(events), len(regimes))

    return [
        f"{events[i]}_{regimes[i]}"
        for i in range(size)
    ]


# ============================================================
# 3. BEHAVIOR TRANSITION BUILDER
# ============================================================

def build_behavior_transitions(
        sequence,
        data,
        forward_points=20):

    mfe, mae = compute_excursions(data, forward_points)
    closes = data["close"]

    transitions = {}

    for i in range(len(sequence) - 1):

        frm = sequence[i]
        to = sequence[i + 1]

        key = (frm, to)

        if key not in transitions:
            transitions[key] = {
                "count": 0,
                "moves": [],
                "mfe": [],
                "mae": []
            }

        entry = closes.iloc[i]
        future = closes.iloc[min(i + forward_points, len(closes)-1)]

        move = (future - entry) / entry

        transitions[key]["count"] += 1
        transitions[key]["moves"].append(move)
        transitions[key]["mfe"].append(mfe.iloc[i])
        transitions[key]["mae"].append(mae.iloc[i])

    return transitions


# ============================================================
# 4. NORMALIZE + STATS
# ============================================================

def compute_transition_stats(transitions):

    stats = {}

    total_from = {}

    # count totals
    for (frm, _), data in transitions.items():
        total_from[frm] = total_from.get(frm, 0) + data["count"]

    for key, data in transitions.items():

        frm, to = key
        probability = data["count"] / total_from[frm]

        moves = np.array(data["moves"])
        mfe = np.array(data["mfe"])
        mae = np.array(data["mae"])

        stats[key] = {

            "probability": probability,
            "samples": data["count"],

            "avg_move": moves.mean(),
            "median_move": np.median(moves),

            "max_move": moves.max(),
            "min_move": moves.min(),

            "avg_mfe": mfe.mean(),
            "avg_mae": mae.mean()
        }

    return stats


# ============================================================
# 5. FLOW MATRIX (SMART VIEW)
# ============================================================

def build_flow_matrix(transition_stats):

    flow = {}

    for (frm, to), stats in transition_stats.items():

        if frm not in flow:
            flow[frm] = {}

        flow[frm][to] = stats

    return flow


# ============================================================
# 6. TRANSITION SUMMARY
# ============================================================

def transition_summary(flow_matrix):

    summary = {}

    for frm, targets in flow_matrix.items():

        ranked = sorted(
            targets.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )

        summary[frm] = ranked

    return summary


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)
    size = 300

    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    events = np.random.choice(
        ["impulse", "retracement", "consolidation"],
        size
    )

    regimes = np.random.choice(
        ["trend", "range"],
        size
    )

    sequence = generate_state_sequence(events, regimes)

    transitions = build_behavior_transitions(
        sequence,
        df,
        forward_points=20
    )

    stats = compute_transition_stats(transitions)
    flow = build_flow_matrix(stats)
    summary = transition_summary(flow)

    print("\nFLOW SUMMARY:\n")

    for state, targets in summary.items():
        print(state, "->", targets[:2])

    print("\nBehavior Transition Engine OK")
