import numpy as np
import pandas as pd


def atr(df, period=14):

    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=1).mean()


def volatility_ratio(df, lookback=100):

    a = atr(df)
    base = a.rolling(lookback, min_periods=20).mean()

    return (a / base).fillna(1)


def expansion_ratio(df):

    return (df.high - df.low) / atr(df)


def momentum(df, n=5):

    return (df.close - df.close.shift(n)) / atr(df)


def liquidity_penetration(df, window=20):

    prev_high = df.high.rolling(window).max().shift(1)
    prev_low = df.low.rolling(window).min().shift(1)

    up = (df.high - prev_high).clip(lower=0)
    down = (prev_low - df.low).clip(lower=0)

    return ((up + down) / atr(df)).fillna(0)


def wick_pressure(df):

    rng = df.high - df.low
    upper = df.high - df[['open','close']].max(axis=1)
    lower = df[['open','close']].min(axis=1) - df.low

    return ((upper + lower) / rng.replace(0, np.nan)).fillna(0)



if __name__ == "__main__":

    print("MEASURES SELF TEST")

    import numpy as np

    size = 1000
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    print(volatility_ratio(df).tail())
    print("PASSED ✅")
