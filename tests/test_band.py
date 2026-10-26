"""Band-pass helper for the band-limited (8-30 Hz) transfer arms."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.band import bandpass

SFREQ = 256.0


def _power_at(X: np.ndarray, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(X, axis=-1))
    freqs = np.fft.rfftfreq(X.shape[-1], 1 / SFREQ)
    return float(spectrum[..., np.argmin(np.abs(freqs - freq))].mean())


def test_bandpass_keeps_in_band_and_removes_out_of_band_components():
    t = np.arange(4 * int(SFREQ)) / SFREQ
    slow, mid, fast = np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 20 * t), np.sin(2 * np.pi * 60 * t)
    X = np.tile(slow + mid + fast, (3, 5, 1)).astype(np.float32)
    Y = bandpass(X, SFREQ, 8.0, 30.0)
    assert Y.shape == X.shape and Y.dtype == np.float32
    assert _power_at(Y, 20) > 0.9 * _power_at(X, 20)
    assert _power_at(Y, 2) < 0.05 * _power_at(X, 2)
    assert _power_at(Y, 60) < 0.05 * _power_at(X, 60)


def test_bandpass_refuses_a_band_the_sampling_rate_cannot_hold():
    X = np.zeros((1, 2, 256), dtype=np.float32)
    try:
        bandpass(X, SFREQ, 8.0, 200.0)
    except ValueError as error:
        assert "Nyquist" in str(error)
    else:
        raise AssertionError("a band above Nyquist must be refused")
