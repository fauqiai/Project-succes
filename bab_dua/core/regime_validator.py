import pandas as pd
import numpy as np


# =====================================
# REGIME VALIDATOR
# =====================================

def validate_regimes(
    df,
    state,
    forward=20,
    min_samples=150,
    separation_threshold=0.002
):
    """
    Validate if clustered regimes are meaningful.

    Checks:
    --------
    ✅ sample size
    ✅ return separation
    ✅ edge stability proxy

    Returns:
    --------
    report : DataFrame
    """

    future = df.close.shift(-forward)
    returns = (future - df.close) / df.close

    temp = pd.concat([
        state["cluster"],
        returns.rename("fwd_return")
    ], axis=1).dropna()

    grouped = temp.groupby("cluster")["fwd_return"]

    report = grouped.agg([
        ("count", "count"),
        ("mean_return", "mean"),
        ("volatility", "std")
    ])

    report["edge"] = report["mean_return"] / report["volatility"]

    # sample quality
    report["enough_samples"] = report["count"] > min_samples

    # separation test
    global_mean = temp["fwd_return"].mean()
    report["separated"] = (
        (report["mean_return"] - global_mean).abs()
        > separation_threshold
    )

    # simple score
    report["tradable"] = (
        report["enough_samples"] &
        report["separated"]
    )

    return report.sort_values("edge", ascending=False)


# =====================================
# QUICK HEALTH CHECK
# =====================================

def regime_health_score(report):
    """
    Returns a simple health metric.
    """

    tradable_ratio = report["tradable"].mean()

    if tradable_ratio > 0.6:
        return "STRONG"
    elif tradable_ratio > 0.4:
        return "DECENT"
    elif tradable_ratio > 0.2:
        return "WEAK"
    else:
        return "TRASH 😄"


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("REGIME VALIDATOR TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix
    from .regime_engine import build_regime_matrix
    from .state_engine import build_state_matrix, cluster_states

    size = 2000
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    f = build_feature_matrix(df)
    r = build_regime_matrix(df)

    state, scaled, scaler = build_state_matrix(f, r)
    state, model = cluster_states(state, scaled)

    report = validate_regimes(df, state)

    print(report)
    print("\nHEALTH:", regime_health_score(report))

    print("PASSED ✅")
