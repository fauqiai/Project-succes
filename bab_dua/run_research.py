import pandas as pd

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.state_engine import build_state_matrix, cluster_states
from core.interaction_engine import build_interactions
from core.transition_engine import transition_matrix, transition_expectancy
from core.expectancy_engine import state_edge

from interpreter_engine import (
    interpret_states,
    interpret_current_state,
    print_interpretation
)

from direction_engine import (
    compute_direction,
    interpret_direction,
    pressure_regime
)

from risk_engine import build_risk_model

from execution_engine import (
    execution_decision,
    execution_style
)

# ✅ NEW
from exit_engine import compute_exit



def load_csv_data(path):

    print("\nLoading CSV data...")

    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.strip()
    df = df[["open", "high", "low", "close"]].dropna()

    print(f"✅ Loaded {len(df):,} rows")

    return df



def run():

    print("\n🔥 RUNNING ULTRA QUANT ENGINE\n")

    df = load_csv_data("xauusd_m1_cleaned.csv")

    print("\nBuilding features...")
    features = build_feature_matrix(df)

    print("Building regimes...")
    regimes = build_regime_matrix(df)

    print("Building interactions...")
    interactions = build_interactions(features)

    features = pd.concat([features, interactions], axis=1)

    print("Clustering states...")
    state, scaled, _ = build_state_matrix(features, regimes)
    state, _ = cluster_states(state, scaled, k=8)

    edge_table = state_edge(df, state)

    print("\n🔥 TOP STATES:")
    print(edge_table.head(10))

    interpretation = interpret_states(edge_table)
    print_interpretation(interpretation)

    cluster, state_label = interpret_current_state(
        state,
        interpretation
    )

    print(f"\n📍 CURRENT STATE → Cluster {cluster}")
    print(f"BOT: {state_label}")

    # =============================
    # DIRECTION
    # =============================

    direction_score = compute_direction(features)
    bias, confidence = interpret_direction(direction_score)
    regime = pressure_regime(features)

    print("\n🧭 DIRECTION:")
    print("Bias:", bias)
    print("Confidence:", round(confidence, 3))
    print("Market Pressure:", regime)

    # =============================
    # RISK
    # =============================

    risk = build_risk_model(
        features,
        confidence,
        regime,
        account_size=10000,
        base_risk=0.01
    )

    print("\n💰 RISK MODEL:")
    for k, v in risk.items():
        print(f"{k}: {v}")

    # =============================
    # EXECUTION
    # =============================

    decision, reason = execution_decision(
        state_label,
        bias,
        confidence,
        regime
    )

    style = execution_style(confidence)

    print("\n🎯 EXECUTION ENGINE:")
    print("Decision:", decision)
    print("Reason:", reason)
    print("Position Style:", style)

    # =============================
    # 🔥 EXIT ENGINE (NEW)
    # =============================

    exit_plan = compute_exit(
        df,
        features,
        state_label,
        bias
    )

    print("\n🚪 EXIT ENGINE:")
    for k, v in exit_plan.items():
        print(f"{k}: {v}")

    # =============================
    print("\n🔥 TRANSITIONS:")
    print(transition_matrix(state))

    print("\n🔥 TRANSITION EXPECTANCY:")
    print(transition_expectancy(df, state).head(10))

    print("\n✅ ULTRA ENGINE COMPLETE\n")



if __name__ == "__main__":
    run()
