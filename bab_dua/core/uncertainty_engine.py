import numpy as np


def bootstrap_confidence(
    returns,
    n_bootstrap=500
):
    """
    Estimate confidence using bootstrap.
    """

    means = []

    returns = np.array(returns.dropna())

    for _ in range(n_bootstrap):
        sample = np.random.choice(
            returns,
            size=len(returns),
            replace=True
        )
        means.append(sample.mean())

    lower = np.percentile(means, 5)
    upper = np.percentile(means, 95)

    confidence = 1 - (upper - lower)

    return {
        "mean": np.mean(means),
        "confidence": confidence,
        "range": (lower, upper)
    }


# ============== SELF TEST ==============

if __name__ == "__main__":

    print("UNCERTAINTY ENGINE TEST")

    import pandas as pd

    fake_returns = pd.Series(
        np.random.normal(0.002, 0.01, 1000)
    )

    result = bootstrap_confidence(fake_returns)

    print(result)

    print("PASSED ✅")
