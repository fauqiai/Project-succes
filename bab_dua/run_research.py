import pandas as pd

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.state_engine import build_state_matrix, cluster_states
from core.interaction_engine import build_interactions
from core.transition_engine import transition_matrix, transition_expectancy
from core.expectancy_engine import state_edge


# ======================================
# CSV LOADER (PRO VERSION)
# ======================================

def load_csv_data(path):

    print("\nLoading CSV data...")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    required_cols = ["open", "high", "low", "close"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"\nCSV ERROR → Missing column: '{col}'\n"
                f"Columns found: {list(df.columns)}"
            )

    df = df[required_cols].copy()

    df = df.dropna()

    print(f"✅ Loaded {len(df):,} rows")

    if len(df) < 5000:
        print("⚠️ WARNING: Dataset kecil — clustering kurang optimal.")

    print(df.head())

    return df


# ======================================
# ENGINE RUNNER
# ======================================

def run():

    print("\n🔥 RUNNING ULTRA EDGE ENGINE (REAL DATA)\n")

    # 👉 GANTI NAMA FILE DI SINI kalau beda
    df = load_csv_data("xauusd_m1_cleaned.csv")

    # =========================
    # BUILD FEATURES
    # =========================
    print("\nBuilding features...")
    features = build_feature_matrix(df)

    # =========================
    # REGIME
    # =========================
    print("Building regimes...")
    regimes = build_regime_matrix(df)

    # =========================
    # INTERACTIONS (EDGE BOOSTER)
    # =========================
    print("Building feature interactions...")
    interactions = build_interactions(features)

    features = pd.concat([features, interactions], axis=1)

    # =========================
    # STATE SPACE
    # =========================
    print("Clustering market states...")

    state, scaled, _ = build_state_matrix(features, regimes)

    # 🔥 cluster 8 = sweet spot
    state, _ = cluster_states(state, scaled, k=8)

    # =========================
    # EDGE TABLE
    # =========================
    print("\n🔥 TOP STATES (EDGE):")
    print(state_edge(df, state).head(10))

    # =========================
    # TRANSITIONS
    # =========================
    print("\n🔥 TRANSITION MATRIX:")
    print(transition_matrix(state))

    # =========================
    # TRANSITION EXPECTANCY
    # =========================
    print("\n🔥 TRANSITION EXPECTANCY:")
    print(transition_expectancy(df, state).head(10))

    print("\n✅ RESEARCH COMPLETE\n")


if __name__ == "__main__":
    run()
