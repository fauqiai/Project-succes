import numpy as np
import pandas as pd


def detect_regime_shift(
    returns,
    window=200,
    z_threshold=2.5
):
    """
    Detect volatility / return distribution shift.
    """

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    zscore = (
        (returns - rolling_mean) /
        rolling_std.replace(0, np.nan)
    )

    shift = zscore.abs() > z_threshold

    return shift.fillna(False)


def regime_shift_alert(shift_series):
    """
    Simple health signal.
    """

    recent = shift_series.tail(50).mean()

    if recent > 0.3:
        return "HIGH SHIFT"
    elif recent > 0.15:
        return "MODERATE SHIFT"
    else:
        return "STABLE"


# ================= SELF TEST =================

if __name__ == "__main__":

    print("REGIME SHIFT ENGINE TEST")

    np.random.seed(42)

    normal = np.random.normal(0,1,1000)
    crash = np.random.normal(-3,2,200)

    series = pd.Series(
        np.concatenate([normal, crash])
    )

    shift = detect_regime_shift(series)

    print("Shift detected:", shift.sum())
    print("Health:", regime_shift_alert(shift))

    print("PASSED ✅")
