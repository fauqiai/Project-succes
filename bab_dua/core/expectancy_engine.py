import pandas as pd


def forward_returns(df, n=20):

    future = df.close.shift(-n)

    return (future - df.close) / df.close


def state_edge(df, state):

    fwd = forward_returns(df)

    table = (
        pd.concat([state.cluster, fwd], axis=1)
        .dropna()
        .groupby("cluster")[fwd.name]
        .agg(["mean","std","count"])
    )

    table["edge"] = table["mean"] / table["std"]

    return table.sort_values("edge", ascending=False)



if __name__ == "__main__":

    print("EXPECTANCY ENGINE TEST")

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

    state, scaled, _ = build_state_matrix(f,r)
    state, _ = cluster_states(state, scaled)

    print(state_edge(df, state).head())

    print("PASSED ✅")
