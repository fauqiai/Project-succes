"""
transitions.py
Behavior Flow Engine (FINAL STABLE)

- No legacy structures
- No int transitions
- Always dict
- Probability safe
"""

import pandas as pd
import numpy as np

# ✅ IMPORT BENAR (sesuai struktur kamu)
from behavior_core.expectancy import compute_excursions


# ============================================================
# 1. STATE SEQUENCE
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
# 2. BUILD BEHAVIOR TRANSITIONS
# ============================================================

def build_behavior_transitions(sequence, data, forward_points=20):

    mfe, mae = compute_excursions(data, forward_points)
    closes = data["close"]

    transitions = {}

    for i in range(len(sequence) - 1):

        frm = sequence[i]
        to = sequence[i + 1]

        key = (frm, to)

        # ALWAYS DICT
        if key not in transitions:
            transitions[key] = {
                "count": 0,
                "moves": [],
                "mfe": [],
                "mae": []
            }

        entry = closes.iloc[i]
        future_index = min(i + forward_points, len(closes)-1)
        future_price = closes.iloc[future_index]

        move = (future_price - entry) / entry

        transitions[key]["count"] += 1
        transitions[key]["moves"].append(move)
        transitions[key]["mfe"].append(mfe.iloc[i])
        transitions[key]["mae"].append(mae.iloc[i])

    return transitions


# ============================================================
# 3. COMPUTE TRANSITION STATS
# ============================================================

def compute_transition_stats(transitions):

    stats = {}
    total_from = {}

    # hitung total origin state
    for (frm, _), data in transitions.items():
        total_from[frm] = total_from.get(frm, 0) + data["count"]

    for key, data in transitions.items():

        frm, to = key
        total = total_from.get(frm, 1)  # anti div zero

        probability = data["count"] / total

        moves = np.array(data["moves"])
        mfe = np.array(data["mfe"])
        mae = np.array(data["mae"])

        stats[key] = {
            "probability": float(probability),
            "samples": int(data["count"]),

            "avg_move": float(moves.mean()),
            "median_move": float(np.median(moves)),

            "max_move": float(moves.max()),
            "min_move": float(moves.min()),

            "avg_mfe": float(mfe.mean()),
            "avg_mae": float(mae.mean())
        }

    return stats


# ============================================================
# 4. FLOW MATRIX
# ============================================================

def build_flow_matrix(transition_stats):

    flow = {}

    for (frm, to), stats in transition_stats.items():

        if frm not in flow:
            flow[frm] = {}

        flow[frm][to] = stats

    return flow


# ============================================================
# 5. SAFE SUMMARY (ANTI TYPEERROR)
# ============================================================

def transition_summary(flow_matrix):

    summary = {}

    for frm, targets in flow_matrix.items():

        ranked = sorted(
            targets.items(),
            key=lambda x: x[1].get("probability", 0),
            reverse=True
        )

        summary[frm] = ranked

    return summary


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)
    size = 400

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
