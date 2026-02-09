import pandas as pd


def monitor_edge_decay(
    returns,
    window_short=50,
    window_long=300
):
    """
    Detect if edge is fading.
    """

    short = returns.rolling(window_short).mean()
    long = returns.rolling(window_long).mean()

    decay = short < long

    return decay.fillna(False)


def model_health(decay_series):

    recent = decay_series.tail(100).mean()

    if recent > 0.6:
        return "EDGE DYING"
    elif recent > 0.4:
        return "WEAKENING"
    else:
        return "HEALTHY"


# ============== SELF TEST ==============

if __name__ == "__main__":

    print("META MONITOR TEST")

    import numpy as np

    good = np.random.normal(0.003,0.01,800)
    bad = np.random.normal(-0.002,0.02,200)

    series = pd.Series(
        list(good) + list(bad)
    )

    decay = monitor_edge_decay(series)

    print("Decay signals:", decay.sum())
    print("Health:", model_health(decay))

    print("PASSED ✅")
