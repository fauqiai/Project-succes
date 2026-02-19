import numpy as np

def hmm_cluster(
    X,
    n_states=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
):
    """
    Hidden Markov Model clustering with FULL REPORT
    """

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError(
            "hmmlearn belum terinstall. Jalankan:\n"
            "pip install hmmlearn"
        )

    print("\n==============================")
    print("🧠 HMM TRAINING STARTED")
    print("==============================")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"States requested: {n_states}")
    print(f"Covariance: {covariance_type}")
    print(f"Iterations: {n_iter}")

    # build model
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )

    # train
    model.fit(X)

    # predict states
    labels = model.predict(X)

    print("\n✅ HMM TRAINED")

    unique, counts = np.unique(labels, return_counts=True)

    print("\n📊 STATE DISTRIBUTION:")
    for u, c in zip(unique, counts):
        print(f"State {u}: {c} candles ({c/len(labels)*100:.2f}%)")

    print("\n📉 LOG LIKELIHOOD:", model.score(X))

    print("\n🔢 TRANSITION MATRIX:")
    print(model.transmat_)

    print("\n📌 MEANS PER STATE:")
    print(model.means_)

    print("==============================\n")

    return labels, model
