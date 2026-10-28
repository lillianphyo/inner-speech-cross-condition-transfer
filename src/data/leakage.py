"""Guards against train/test overlap."""
import hashlib

import numpy as np


def assert_disjoint(X_train: np.ndarray, X_test: np.ndarray) -> None:
    """Raise if any trial is present in both sets (train-on-test leakage)."""
    def digest(trial: np.ndarray) -> bytes:
        return hashlib.sha1(np.ascontiguousarray(trial).tobytes()).digest()

    train_digests = {digest(trial) for trial in X_train}
    shared = sum(digest(trial) in train_digests for trial in X_test)
    if shared:
        raise ValueError(f"{shared} trials appear in both the train and the test set")
