import pandas as pd


def transition_matrix(states):

    cur = states.cluster
    nxt = cur.shift(-1)

    return pd.crosstab(cur, nxt, normalize="index")


def transition_expectancy(df, states, forward=20):

    future = df.close.shift(-forward)
    ret = (future - df.close) / df.close

    temp = pd.concat([
        states.cluster,
        states.cluster.shift(-1).rename("next"),
        ret
    ], axis=1).dropna()

    result = temp.groupby(["cluster","next"])[ret.name].mean()

    return result.unstack().fillna(0)



if __name__ == "__main__":

    print("TRANSITION ENGINE TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix
    from .regime_engine import build_regime_matrix
    from .state_engine import build_state_matrix, cluster_states

    size = 1500
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

    print(transition_expectancy(df, state).head())

    print("PASSED ✅")
