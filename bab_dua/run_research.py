import pandas as pd
import numpy as np

# ================= CORE =================

from core.feature_engine import build_feature_matrix
from core.feature_discovery import discover_features
from core.regime_engine import build_regime_matrix
from core.regime_shift_engine import detect_regime_shift, regime_shift_alert
from core.regime_validator import validate_regimes, regime_health_score
from core.state_engine import build_state_matrix, cluster_states
from core.interaction_engine import build_interactions
from core.transition_engine import transition_matrix, transition_expectancy
from core.expectancy_engine import state_edge
from core.uncertainty_engine import bootstrap_confidence

# ================= HAND =================

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

from exit_engine import compute_exit

# ======================================================

# LOAD DATA

# ======================================================

def load_csv_data(path):

```
print("\nLoading CSV data...")

df = pd.read_csv(path)

df.columns = df.columns.str.lower().str.strip()
df = df[["open", "high", "low", "close"]].dropna()

print(f"✅ Loaded {len(df):,} rows")

return df
```

# ======================================================

# MAIN ENGINE

# ======================================================

def run():

```
print("\n🔥 RUNNING FULL QUANT RESEARCH ENGINE\n")

df = load_csv_data("xauusd_m1_cleaned.csv")

# ================= FEATURES =================

print("\nBuilding features...")
features = build_feature_matrix(df)

print("Discovering hidden features...")
discovered, _ = discover_features(features)
features = pd.concat([features, discovered], axis=1)

print("Building interactions...")
interactions = build_interactions(features)
features = pd.concat([features, interactions], axis=1)

# ================= REGIMES =================

print("\nBuilding regimes...")
regimes = build_regime_matrix(df)

returns = df["close"].pct_change()

shift_series = detect_regime_shift(returns)
shift_status = regime_shift_alert(shift_series)

print("\n🚨 REGIME SHIFT:", shift_status)

# ================= STATE =================

print("\nClustering states...")
state, scaled, _, _, _ = build_state_matrix(features, regimes)
state, _ = cluster_states(state, scaled, k=8)

# ================= EDGE =================

edge_table = state_edge(df, state)

print("\n🔥 TOP STATES:")
print(edge_table.head(10))

# ================= REGIME VALIDATION =================

regime_report = validate_regimes(df, state)
health = regime_health_score(regime_report)

print("\n🧠 REGIME HEALTH:", health)
print(regime_report.head())

# ================= UNCERTAINTY =================

print("\nCalculating statistical confidence...")

forward_returns = (df["close"].shift(-20) - df["close"]) / df["close"]

uncertainty_map = {}

for cluster in state["cluster"].unique():

    idx = state[state["cluster"] == cluster].index
    sample = forward_returns.loc[idx].dropna()

    if len(sample) > 50:
        uncertainty_map[cluster] = bootstrap_confidence(sample)
    else:
        uncertainty_map[cluster] = {
            "mean": None,
            "confidence": 0,
            "range": (None, None)
        }

print("\n🔬 UNCERTAINTY SAMPLE:")
for k, v in list(uncertainty_map.items())[:5]:
    print(f"Cluster {k} → {v}")

# ================= INTERPRET =================

interpretation = interpret_states(edge_table)
print_interpretation(interpretation)

cluster, state_label = interpret_current_state(
    state,
    interpretation
)

print(f"\n📍 CURRENT STATE → Cluster {cluster}")
print(f"BOT: {state_label}")

# ================= TRANSITIONS =================

print("\n🔥 TRANSITION MATRIX:")
print(transition_matrix(state))

print("\n🔥 TRANSITION EXPECTANCY:")
print(transition_expectancy(df, state).head(10))

# ================= DIRECTION =================

direction_score = compute_direction(features)
bias, confidence = interpret_direction(direction_score)
regime = pressure_regime(features)

print("\n🧭 DIRECTION:")
print("Bias:", bias)
print("Confidence:", round(confidence, 3))
print("Market Pressure:", regime)

# ================= RISK =================

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

# ================= EXECUTION =================

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

# ================= EXIT =================

exit_plan = compute_exit(
    df,
    features,
    state_label,
    bias
)

print("\n🚪 EXIT ENGINE:")
for k, v in exit_plan.items():
    print(f"{k}: {v}")

print("\n✅ FULL QUANT ENGINE COMPLETE\n")
```

if **name** == "**main**":
run()
