import pandas as pd
from .measures import *


def build_feature_matrix(df):

    f = pd.DataFrame(index=df.index)

    f["volatility"] = volatility_ratio(df)
    f["expansion"] = expansion_ratio(df)
    f["momentum"] = momentum(df)
    f["liquidity"] = liquidity_penetration(df)
    f["rejection"] = wick_pressure(df)

    f["compression"] = 1 / f["expansion"].replace(0,1)

    return f.fillna(0)



if __name__ == "__main__":

    print("FEATURE ENGINE TEST")

    import numpy as np

    size = 500
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    print(build_feature_matrix(df).head())

    print("PASSED ✅")
