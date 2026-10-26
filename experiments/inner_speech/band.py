"""Zero-phase band-pass of epoched EEG, for the band-limited (8-30 Hz) arms on ds003626."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(X: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Butterworth band-pass of `X` (..., n_times) between `low` and `high` Hz, forward and backward."""
    nyquist = sfreq / 2.0
    if not 0.0 < low < high < nyquist:
        raise ValueError(f"band {low}-{high} Hz must lie strictly inside (0, Nyquist={nyquist} Hz)")
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(X.dtype)
