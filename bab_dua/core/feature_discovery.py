import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# =====================================
# AUTO FEATURE DISCOVERY
# =====================================

def discover_features(
    features,
    variance_keep=0.80,
    model=None
):
    """
    Creates hidden features using PCA.

    Think of this as:
    extracting latent market forces.
    """

    X = features.values

    # reuse model (live future)
    if model is not None:
        comps = model.transform(X)
    else:
        pca_full = PCA()
        pca_full.fit(X)

        cumulative = np.cumsum(
            pca_full.explained_variance_ratio_
        )

        n = np.searchsorted(
            cumulative,
            variance_keep
        ) + 1

        model = PCA(n_components=n)
        comps = model.fit_transform(X)

        print(f"[Discovery] Created {n} hidden features")

    cols = [
        f"latent_{i}"
        for i in range(comps.shape[1])
    ]

    discovered = pd.DataFrame(
        comps,
        index=features.index,
        columns=cols
    )

    return discovered, model


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("FEATURE DISCOVERY TEST")

    import numpy as np
    import pandas as pd

    np.random.seed(42)

    fake = pd.DataFrame(
        np.random.normal(0,1,(1500,10)),
        columns=[f"f{i}" for i in range(10)]
    )

    d, model = discover_features(fake)

    print(d.head())

    print("Discovered shape:", d.shape)

    print("PASSED ✅")
