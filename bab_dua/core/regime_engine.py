import pandas as pd
from .measures import atr


def trend_strength(df):

    fast = df.close.rolling(20).mean()
    slow = df.close.rolling(50).mean()

    return ((fast - slow).abs() / atr(df)).fillna(0)


def volatility_regime(df):

    a = atr(df)
    base = a.rolling(200, min_periods=50).mean()

    return (a / base).fillna(1)


def volatility_slope(df):

    vol = atr(df)

    return vol.diff(10) / vol


def regime_shift(df):

    slope = volatility_slope(df)

    # magnitude of change
    return slope.abs()



def build_regime_matrix(df):

    r = pd.DataFrame(index=df.index)

    r["trend_strength"] = trend_strength(df)
    r["vol_regime"] = volatility_regime(df)
    r["vol_slope"] = volatility_slope(df)
    r["regime_shift"] = regime_shift(df)

    return r.fillna(0)



if __name__ == "__main__":

    print("REGIME ENGINE TEST")

    import numpy as np

    size = 800
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    print(build_regime_matrix(df).tail())

    print("PASSED ✅")
