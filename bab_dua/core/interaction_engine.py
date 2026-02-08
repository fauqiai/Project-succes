import pandas as pd


def build_interactions(features):

    inter = pd.DataFrame(index=features.index)

    inter["compression_x_vol"] = features["compression"] * features["volatility"]
    inter["liq_x_vol"] = features["liquidity"] * features["volatility"]
    inter["momentum_x_vol"] = features["momentum"] * features["volatility"]
    inter["rejection_x_liq"] = features["rejection"] * features["liquidity"]

    return inter.fillna(0)



if __name__ == "__main__":

    print("INTERACTION ENGINE TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix

    size = 700
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    f = build_feature_matrix(df)

    print(build_interactions(f).head())

    print("PASSED ✅")
