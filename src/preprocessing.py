"""Signal preprocessing and feature extraction for single-lead ECG.

Pipeline: band-pass filter -> normalize -> fixed-length window -> log-Mel
spectrogram. Augmentation (additive noise, time shift) is applied at training
time only.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, filtfilt

FS = 300  # PhysioNet/CinC 2017 sampling rate (Hz)


def bandpass_filter(signal: np.ndarray, fs: int = FS,
                    lowcut: float = 0.5, highcut: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass. Removes baseline wander and HF noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance (robust to constant signals)."""
    std = signal.std()
    if std < 1e-8:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def fix_length(signal: np.ndarray, length: int) -> np.ndarray:
    """Crop (centre) or zero-pad a 1-D signal to exactly `length` samples."""
    n = len(signal)
    if n == length:
        return signal
    if n > length:
        start = (n - length) // 2
        return signal[start:start + length]
    pad = length - n
    return np.pad(signal, (pad // 2, pad - pad // 2))


def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise at a target signal-to-noise ratio."""
    power = np.mean(signal ** 2)
    if power < 1e-12:
        return signal
    noise_power = power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0.0, np.sqrt(noise_power), size=signal.shape)


def time_shift(signal: np.ndarray, max_frac: float = 0.1) -> np.ndarray:
    """Circular shift by up to `max_frac` of the length."""
    shift = np.random.randint(-int(max_frac * len(signal)), int(max_frac * len(signal)) + 1)
    return np.roll(signal, shift)


def log_mel(signal: np.ndarray, fs: int = FS, n_mels: int = 64,
            n_fft: int = 256, hop: int = 64, n_frames: int = 250) -> np.ndarray:
    """Log-Mel spectrogram, returned as a fixed (n_mels, n_frames) float32 image."""
    mel = librosa.feature.melspectrogram(
        y=signal.astype(np.float32), sr=fs, n_fft=n_fft,
        hop_length=hop, n_mels=n_mels, power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # pad / crop the time axis to a fixed width
    if logmel.shape[1] < n_frames:
        logmel = np.pad(logmel, ((0, 0), (0, n_frames - logmel.shape[1])),
                        mode="constant", constant_values=logmel.min())
    else:
        logmel = logmel[:, :n_frames]
    # scale to roughly [0, 1]
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)
    return logmel.astype(np.float32)


def make_features(signal: np.ndarray, *, window_sec: float = 9.0,
                  augment: bool = False, **mel_kwargs) -> np.ndarray:
    """Full pipeline: filter -> normalize -> (augment) -> window -> log-Mel.

    Returns an array of shape (n_mels, n_frames, 1).
    """
    x = bandpass_filter(signal)
    x = normalize(x)
    if augment:
        if np.random.rand() < 0.5:
            x = add_noise(x, snr_db=np.random.uniform(15, 30))
        if np.random.rand() < 0.5:
            x = time_shift(x)
    x = fix_length(x, int(window_sec * FS))
    feat = log_mel(x, **mel_kwargs)
    return feat[..., np.newaxis]
