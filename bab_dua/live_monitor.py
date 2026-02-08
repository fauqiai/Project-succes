import pandas as pd
import time

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.state_engine import build_state_matrix, cluster_states
from core.interaction_engine import build_interactions
from core.expectancy_engine import state_edge

from interpreter_engine import (
    interpret_states,
    interpret_current_state
)

from direction_engine import (
    compute_direction,
    interpret_direction,
    pressure_regime
)

from risk_engine import build_risk_model
from execution_engine import execution_decision, execution_style


# =====================================
# SETTINGS
# =====================================

CSV_PATH = "xauusd_m1_cleaned.csv"

WINDOW = 1200        # last candles to analyze
SLEEP_SECONDS = 60   # check every 1 minute



# =====================================
# LOAD DATA
# =====================================

def load_latest_data():

    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.lower().str.strip()

    df = df[["open", "high", "low", "close"]].dropna()

    return df.tail(WINDOW)



# =====================================
# SINGLE LIVE PASS
# =====================================

def run_live_pass():

    df = load_latest_data()

    features = build_feature_matrix(df)
    regimes = build_regime_matrix(df)

    interactions = build_interactions(features)
    features = pd.concat([features, interactions], axis=1)

    state, scaled, _ = build_state_matrix(features, regimes)
    state, _ = cluster_states(state, scaled, k=8)

    edge_table = state_edge(df, state)

    interpretation = interpret_states(edge_table)

    cluster, state_label = interpret_current_state(
        state,
        interpretation
    )

    direction_score = compute_direction(features)
    bias, confidence = interpret_direction(direction_score)
    regime = pressure_regime(features)

    risk = build_risk_model(
        features,
        confidence,
        regime,
        account_size=10000,
        base_risk=0.01
    )

    decision, reason = execution_decision(
        state_label,
        bias,
        confidence,
        regime
    )

    style = execution_style(confidence)

    print("\n==============================")
    print("📡 LIVE MARKET SNAPSHOT")
    print("==============================")

    print(f"STATE: {state_label} (Cluster {cluster})")
    print(f"BIAS: {bias}")
    print(f"CONFIDENCE: {round(confidence,3)}")
    print(f"REGIME: {regime}")

    print("\n💰 RISK:")
    for k, v in risk.items():
        print(f"{k}: {v}")

    print("\n🎯 DECISION:")
    print("Action:", decision)
    print("Reason:", reason)
    print("Position:", style)

    print("\n⏰ Waiting next candle...")


# =====================================
# LIVE LOOP
# =====================================

def start_live_monitor():

    print("\n🚀 STARTING LIVE MONITOR")
    print("Press CTRL+C to stop.\n")

    while True:

        try:

            run_live_pass()

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:

            print("\n🛑 Live monitor stopped.")
            break

        except Exception as e:

            print("ERROR:", e)
            print("Retrying in 10 seconds...")
            time.sleep(10)



# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("LIVE MONITOR TEST RUN...\n")

    run_live_pass()

    print("\n✅ TEST PASSED")
