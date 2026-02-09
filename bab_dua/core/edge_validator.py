import pandas as pd
import numpy as np


# =====================================
# WALK FORWARD EDGE VALIDATION
# =====================================

def walkforward_edge_validation(
    df,
    state,
    forward=20,
    windows=6
):
    """
    Splits data into time windows
    and checks if edge survives.
    """

    future = df.close.shift(-forward)
    ret = (future - df.close) / df.close

    temp = pd.concat([
        state["cluster"],
        ret.rename("return")
    ], axis=1).dropna()

    size = len(temp)
    step = size // windows

    reports = []

    for i in range(windows):

        chunk = temp.iloc[i*step:(i+1)*step]

        stats = chunk.groupby("cluster")["return"].mean()

        reports.append(stats)

    result = pd.concat(reports, axis=1)
    result.columns = [f"window_{i}" for i in range(windows)]

    # stability score
    result["stability"] = result.std(axis=1)

    # mean edge
    result["avg_edge"] = result.mean(axis=1)

    # tradable filter
    result["tradable"] = (
        (result["avg_edge"].abs() > 0.001)
        & (result["stability"] < result["avg_edge"].abs())
    )

    return result.sort_values(
        "avg_edge",
        ascending=False
    )


# =====================================
# GLOBAL EDGE SCORE
# =====================================

def edge_health(result):

    tradable_ratio = result["tradable"].mean()

    if tradable_ratio > 0.5:
        return "REAL EDGE 🔥"
    elif tradable_ratio > 0.3:
        return "POSSIBLE EDGE"
    elif tradable_ratio > 0.15:
        return "WEAK EDGE"
    else:
        return "ILLUSION ❌"


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("EDGE VALIDATOR TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix
    from .regime_engine import build_regime_matrix
    from .state_engine import build_state_matrix, cluster_states

    size = 3000
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    f = build_feature_matrix(df)
    r = build_regime_matrix(df)

    state, scaled, *_ = build_state_matrix(f, r)

    state, _ = cluster_states(
        state,
        scaled,
        method="hdbscan"
    )

    report = walkforward_edge_validation(
        df,
        state
    )

    print(report)

    print("\nEDGE HEALTH:", edge_health(report))

    print("PASSED ✅")
