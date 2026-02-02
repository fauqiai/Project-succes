"""
transitions.py
--------------
Module untuk membangun transition matrix pada Quant Behavior.
Semua komentar menggunakan ASCII agar aman disimpan di Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. TRANSITION COUNT BUILDER
# ============================================================

def count_transitions(sequence):
    """
    Menghitung jumlah perpindahan antar state.
    Output: dict {(from, to): count}
    """

    counts = {}

    for i in range(len(sequence) - 1):
        frm = sequence[i]
        to = sequence[i + 1]

        key = (frm, to)
        counts[key] = counts.get(key, 0) + 1

    return counts


# ============================================================
# 2. TRANSITION MATRIX BUILDER
# ============================================================

def build_transition_matrix(sequence):
    """
    Membangun transition matrix dari sequence state.
    Output: dict of dict
    matrix[state_from][state_to] = probability (belum dinormalisasi)
    """

    counts = count_transitions(sequence)
    matrix = {}

    for (frm, to), cnt in counts.items():
        if frm not in matrix:
            matrix[frm] = {}
        matrix[frm][to] = cnt

    return normalize_matrix(matrix)


# ============================================================
# 3. NORMALIZATION
# ============================================================

def normalize_matrix(matrix):
    """
    Menormalisasi matrix count menjadi probabilitas.
    """

    norm_matrix = {}

    for frm, targets in matrix.items():
        total = sum(targets.values())
        norm_matrix[frm] = {}

        for to, cnt in targets.items():
            if total > 0:
                norm_matrix[frm][to] = cnt / total
            else:
                norm_matrix[frm][to] = 0.0

    return norm_matrix


# ============================================================
# 4. STATE SEQUENCE GENERATOR
# ============================================================

def generate_state_sequence(events=None, regimes=None):
    """
    Menggabungkan event dan regime menjadi satu state.
    Contoh:
    impulse + trend -> impulse_trend
    """

    if events is None and regimes is None:
        return []

    if events is None:
        return list(regimes)

    if regimes is None:
        return list(events)

    sequence = []
    size = min(len(events), len(regimes))

    for i in range(size):
        sequence.append(f"{events[i]}_{regimes[i]}")

    return sequence


# ============================================================
# 5. TRANSITION SUMMARY
# ============================================================

def transition_summary(matrix):
    """
    Ringkasan transition matrix.
    Output: dict
    """

    summary = {}

    for frm, targets in matrix.items():
        summary[frm] = sorted(
            targets.items(),
            key=lambda x: x[1],
            reverse=True
        )

    return summary


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    # Simple synthetic sequence test
    events = [
        "impulse", "retracement", "consolidation",
        "impulse", "retracement", "impulse",
        "consolidation", "consolidation", "impulse"
    ]

    regimes = [
        "trend", "trend", "range",
        "trend", "range", "trend",
        "range", "range", "trend"
    ]

    state_sequence = generate_state_sequence(events, regimes)

    matrix = build_transition_matrix(state_sequence)
    summary = transition_summary(matrix)

    print("State sequence:", state_sequence)
    print("Transition matrix:", matrix)
    print("Transition summary:", summary)

    print("transitions.py self-test OK")

