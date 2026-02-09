import numpy as np


class ExitEngine:

    def __init__(self):
        pass

    def compute_behavior_targets(
        self,
        current_price,
        state_stats,
        direction_bias,
        volatility
    ):
        """
        Behavior-based TP & SL
        """

        avg_move = abs(state_stats["edge"]) * 10000

        # adjust by volatility
        vol_multiplier = np.clip(volatility * 10, 0.8, 2.5)

        tp_distance = avg_move * vol_multiplier
        sl_distance = tp_distance * 0.6   # asymmetric risk (behavioral)

        if direction_bias == "LONG":
            tp = current_price + tp_distance
            sl = current_price - sl_distance
        else:
            tp = current_price - tp_distance
            sl = current_price + sl_distance

        return {
            "tp": tp,
            "sl": sl,
            "tp_distance": tp_distance,
            "sl_distance": sl_distance
        }

    # ===============================
    # SMART EXIT LOGIC
    # ===============================

    def should_exit(
        self,
        entry_state,
        current_state,
        momentum_score,
        regime,
        unrealized_pnl
    ):
        """
        Decide whether thesis is broken
        """

        # regime shift = danger
        if regime == "RANGING" and entry_state == "TRENDING":
            return True, "Regime shift detected"

        # momentum collapse
        if momentum_score < 0.2:
            return True, "Momentum died"

        # protect profit
        if unrealized_pnl > 2:
            return True, "Locking profit"

        return False, "Hold trade"


# ===============================
# SELF TEST
# ===============================

if __name__ == "__main__":

    engine = ExitEngine()

    state_stats = {
        "edge": 0.00012
    }

    result = engine.compute_behavior_targets(
        current_price=1950,
        state_stats=state_stats,
        direction_bias="LONG",
        volatility=0.3
    )

    print("\nEXIT ENGINE TEST\n")
    print(result)

    decision = engine.should_exit(
        entry_state="TRENDING",
        current_state="RANGING",
        momentum_score=0.1,
        regime="RANGING",
        unrealized_pnl=3
    )

    print("\nEXIT DECISION:\n", decision)
    print("\nTEST PASSED\n")
