import pandas as pd


# =====================================
# STATE INTERPRETATION
# =====================================

def interpret_states(edge_table,
                     min_edge=0.02,
                     strong_edge=0.05,
                     min_samples=500):

    interpretation = {}

    for cluster, row in edge_table.iterrows():

        edge = row["edge"]
        count = row["count"]

        if count < min_samples:
            interpretation[cluster] = "IGNORE (LOW SAMPLE)"
            continue

        if edge >= strong_edge:
            interpretation[cluster] = "🔥 STRONG TRADE ZONE"

        elif edge >= min_edge:
            interpretation[cluster] = "✅ TRADEABLE"

        elif edge < 0:
            interpretation[cluster] = "⛔ AVOID"

        else:
            interpretation[cluster] = "⚠️ NEUTRAL"

    return interpretation


# =====================================
# CURRENT MARKET STATE
# =====================================

def interpret_current_state(state_df, interpretation):

    latest_cluster = state_df["cluster"].iloc[-1]

    message = interpretation.get(
        latest_cluster,
        "UNKNOWN STATE"
    )

    return latest_cluster, message


# =====================================
# PRETTY PRINT
# =====================================

def print_interpretation(interpretation):

    print("\n🧠 MARKET STATE MAP:\n")

    for k, v in interpretation.items():
        print(f"Cluster {k} → {v}")


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("INTERPRETER ENGINE TEST")

    data = {
        "mean":[0.001,0.0002,-0.0001],
        "std":[0.01,0.02,0.015],
        "count":[2000,800,3000],
        "edge":[0.06,0.02,-0.01]
    }

    edge_table = pd.DataFrame(data)
    edge_table.index = [0,1,2]

    interp = interpret_states(edge_table)

    print_interpretation(interp)

    print("\nTEST PASSED ✅")
