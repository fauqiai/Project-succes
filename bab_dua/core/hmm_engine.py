import numpy as np

def hmm_cluster(
    X,
    n_states=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
):
    """
    Hidden Markov Model clustering.

    Parameters
    ----------
    X : np.array
        Scaled feature matrix (output state_engine)
    n_states : int
        Number of hidden states (start with 2 or 3)
    covariance_type : str
        "full", "diag", "tied", "spherical"
    n_iter : int
        Training iterations
    random_state : int

    Returns
    -------
    labels : np.array
    model  : trained HMM model
    """

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError(
            "hmmlearn belum terinstall. Jalankan:\n"
            "pip install hmmlearn"
        )

    # HMM model
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )

    # fit model
    model.fit(X)

    # predict hidden states per candle
    labels = model.predict(X)

    print(f"HMM states found: {np.unique(labels)}")

    return labels, model
